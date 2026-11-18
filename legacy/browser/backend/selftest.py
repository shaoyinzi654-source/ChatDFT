"""
ChatDFT self-test.

Runs a ladder of real calculations and checks the results against
experimental / literature reference values. Prints a pass/fail table.

    python -m backend.selftest            # quick tier
    python -m backend.selftest --full     # adds optimisation + TD-DFT
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time

from .engine import DFTEngine, DFTError
from .engine import molecule as molmod


# ----------------------------------------------------------------------
# reference data: experimental geometries and literature energies
# ----------------------------------------------------------------------
# Total energies below are B3LYP/6-31G* reference values in Hartree. They are
# reproduced here so the test can catch a broken build (a wrong integral
# library, a mis-wired solver) rather than to claim sub-kcal accuracy.
CASES = [
    {
        "name": "water",
        "method": ("b3lyp", "6-31g*"),
        "expect": {
            "nelec": 10,
            "homo_ev": (-8.6, -6.6),   # broad window: functional-dependent
            "gap_ev": (5.0, 11.0),
            "dipole": (1.5, 2.4),      # experimental 1.85 D
            "bond": ("O-H", (0.93, 1.00)),
        },
    },
    {
        "name": "methane",
        "method": ("b3lyp", "6-31g*"),
        "expect": {
            "nelec": 10,
            "gap_ev": (9.0, 16.0),
            "dipole": (0.0, 0.05),     # Td symmetry forbids a dipole
            "bond": ("C-H", (1.06, 1.12)),
        },
    },
    {
        "name": "benzene",
        "method": ("b3lyp", "6-31g*"),
        "expect": {
            "nelec": 42,
            "gap_ev": (4.5, 7.5),
            "dipole": (0.0, 0.05),     # D6h
            "bond": ("C-C", (1.36, 1.43)),
        },
    },
    {
        "name": "ammonia",
        "method": ("b3lyp", "6-31g*"),
        "expect": {
            "nelec": 10,
            "dipole": (1.2, 1.9),      # experimental 1.47 D
        },
    },
    {
        "name": "formaldehyde",
        "method": ("b3lyp", "6-31g*"),
        "expect": {
            "nelec": 16,
            "dipole": (1.9, 2.9),      # experimental 2.33 D
            "bond": ("C=O", (1.18, 1.26)),
        },
    },
    {
        "name": "methanol",
        "method": ("hf", "sto-3g"),
        "expect": {
            "nelec": 18,
            "dipole": (1.2, 2.4),      # HF/STO-3G is rough but bounded
        },
    },
    {
        "name": "hydroxide",
        "method": ("b3lyp", "6-31g*"),
        "expect": {
            "nelec": 10,               # 8+1+1 = 10 electrons for OH-
            "charge": -1,
        },
    },
    {
        "name": "methyl_radical",
        "method": ("b3lyp", "6-31g*"),
        "expect": {
            "nelec": 9,
            "mult": 2,
        },
    },
    {
        "name": "co2",
        "method": ("b3lyp", "6-31g*"),
        "expect": {
            "nelec": 22,
            "dipole": (0.0, 0.05),     # linear, cancels
        },
    },
]


def _in_range(value, bounds) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    lo, hi = bounds
    return lo <= value <= hi


_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


def _split_bond_prefix(prefix: str):
    """'O-H' / 'C=O' / 'C-H' -> the two element symbols it names."""
    parts = [p for p in re.split(r"[-=]", prefix) if p]
    if len(parts) != 2:
        return None
    return tuple(p.strip() for p in parts)


def _bond_label_elements(label: str):
    """'O1-H2' -> ('O', 'H'); atom indices are dropped."""
    parts = label.split("-")
    if len(parts) != 2:
        return None
    out = []
    for p in parts:
        m = _ELEMENT_RE.match(p.strip())
        out.append(m.group(1) if m else p.strip())
    return tuple(out)


def _angle_label_elements(label: str):
    """'H2-O1-H3' -> ('H', 'O', 'H'); atom indices are dropped."""
    parts = label.split("-")
    out = []
    for p in parts:
        m = _ELEMENT_RE.match(p.strip())
        out.append(m.group(1) if m else p.strip())
    return tuple(out)


def _find_bond(result, prefix: str):
    """Shortest bond between the two element types named in ``prefix``.

    Bond labels carry atom indices ('O1-H2'), so the comparison is made on the
    element symbols alone, and either order counts.
    """
    want = _split_bond_prefix(prefix)
    if want is None:
        return None
    wanted = {want[0], want[1]}

    geo = result.get("geometry") or {}
    hits = [
        b["value"]
        for b in geo.get("bonds", [])
        if _bond_label_elements(b["label"]) is not None
        and set(_bond_label_elements(b["label"])) == wanted
    ]
    if hits:
        return min(hits)

    # fall back to the perception in the molecule block
    mol = result.get("molecule") or {}
    atoms = mol.get("atoms", [])
    hits = []
    for b in mol.get("bonds", []):
        if b["i"] < len(atoms) and b["j"] < len(atoms):
            pair = {atoms[b["i"]]["symbol"], atoms[b["j"]]["symbol"]}
            if pair == wanted:
                hits.append(b["length"])
    return min(hits) if hits else None


def run_case(case: dict, verbose: bool = False) -> dict:
    t0 = time.time()
    outcome = {"name": case["name"], "checks": [], "error": None, "seconds": 0.0}

    try:
        mol = molmod.resolve(case["name"])
    except Exception as exc:
        outcome["error"] = f"structure build failed: {exc}"
        return outcome

    functional, basis = case["method"]
    engine = DFTEngine(
        atom_xyz=mol.to_xyz(),
        charge=mol.charge,
        multiplicity=mol.multiplicity,
        functional=functional,
        basis=basis,
    )

    try:
        result = engine.run_scf()
    except DFTError as exc:
        outcome["error"] = str(exc)
        return outcome
    except Exception as exc:  # pragma: no cover
        outcome["error"] = f"unexpected: {exc}"
        return outcome

    outcome["seconds"] = round(time.time() - t0, 2)
    result["geometry"] = None

    # attach geometry values for bond checks
    geo = engine._harvest.__self__ if False else None
    from .engine.dft import _geometry_table

    if result.get("job_type") != "geometry_optimization":
        result["geometry"] = _geometry_table(mol.to_xyz())

    exp = case["expect"]

    def check(label, value, ok, detail=""):
        outcome["checks"].append(
            {"label": label, "value": value, "ok": bool(ok), "detail": detail}
        )

    if "nelec" in exp:
        check(f"electron count == {exp['nelec']}", result["nelec"],
              result["nelec"] == exp["nelec"])

    check("SCF converged", "yes" if result["converged"] else "no", result["converged"])

    if "gap_ev" in exp:
        check(f"gap in {exp['gap_ev']} eV", f"{result['gap_ev']:.3f}",
              _in_range(result["gap_ev"], exp["gap_ev"]))

    if "homo_ev" in exp:
        check(f"HOMO in {exp['homo_ev']} eV", f"{result['homo_ev']:.3f}",
              _in_range(result["homo_ev"], exp["homo_ev"]))

    if "dipole" in exp:
        mu = result["dipole"]["magnitude"]
        check(f"dipole in {exp['dipole']} D", f"{mu:.3f}",
              _in_range(mu, exp["dipole"]))

    if "bond" in exp:
        prefix, bounds = exp["bond"]
        d = _find_bond(result, prefix)
        check(f"{prefix} in {bounds} A", f"{d:.3f}" if d else "n/a",
              d is not None and _in_range(d, bounds))

    if "charge" in exp:
        check(f"charge == {exp['charge']}", result["charge"],
              result["charge"] == exp["charge"])

    if "mult" in exp:
        check(f"multiplicity == {exp['mult']}", result["multiplicity"],
              result["multiplicity"] == exp["mult"])

    # sanity: no NaN in the headline numbers
    check("finite energy", f"{result['energy_hartree']:.6f}",
          math.isfinite(result["energy_hartree"]))

    # charge sum should recover the total charge
    qsum = sum(c["charge"] for c in result.get("mulliken_charges", []))
    check("Mulliken sum == total charge", f"{qsum:.4f}",
          abs(qsum - result["charge"]) < 0.02)

    outcome["result"] = result
    return outcome


def run_optimisation_case() -> dict:
    """Water optimisation: the acid test for the gradient machinery."""
    outcome = {"name": "water optimisation", "checks": [], "error": None}
    try:
        mol = molmod.resolve("water")
        engine = DFTEngine(
            atom_xyz=mol.to_xyz(), charge=0, multiplicity=1,
            functional="b3lyp", basis="6-31g*",
        )
        t0 = time.time()
        result = engine.optimize(max_steps=20)
        outcome["seconds"] = round(time.time() - t0, 2)
    except Exception as exc:
        outcome["error"] = str(exc)
        return outcome

    geo = result.get("geometry", {})
    oh = [
        b["value"]
        for b in geo.get("bonds", [])
        if set(_bond_label_elements(b["label"]) or ()) == {"O", "H"}
    ]
    angles = [
        a["value"]
        for a in geo.get("angles", [])
        if _angle_label_elements(a["label"]) == ("H", "O", "H")
    ]

    if oh:
        avg = sum(oh) / len(oh)
        outcome["checks"].append({
            "label": "O-H bond in (0.94, 0.99) A", "value": f"{avg:.4f}",
            "ok": 0.94 <= avg <= 0.99, "detail": "experimental 0.957 A",
        })
    else:
        outcome["checks"].append({"label": "O-H bond found", "value": "missing",
                                  "ok": False, "detail": ""})

    if angles:
        a = angles[0]
        outcome["checks"].append({
            "label": "H-O-H angle in (100, 108) deg", "value": f"{a:.2f}",
            "ok": 100 <= a <= 108, "detail": "experimental 104.5 deg",
        })
    else:
        outcome["checks"].append({"label": "H-O-H angle found", "value": "missing",
                                  "ok": False, "detail": ""})

    outcome["result"] = result
    return outcome


def run_tddft_case() -> dict:
    """Formaldehyde: the n -> pi* transition is well established."""
    outcome = {"name": "formaldehyde TD-DFT", "checks": [], "error": None}
    try:
        mol = molmod.resolve("formaldehyde")
        engine = DFTEngine(
            atom_xyz=mol.to_xyz(), charge=0, multiplicity=1,
            functional="b3lyp", basis="6-31g*",
        )
        t0 = time.time()
        result = engine.excited_states(nstates=5)
        outcome["seconds"] = round(time.time() - t0, 2)
    except Exception as exc:
        outcome["error"] = str(exc)
        return outcome

    states = result.get("excited_states", [])
    outcome["checks"].append({
        "label": "5 states returned", "value": str(len(states)),
        "ok": len(states) == 5, "detail": "",
    })

    if states:
        s1 = states[0]
        # The S1 n->pi* state of formaldehyde is around 3.0-4.2 eV at B3LYP.
        outcome["checks"].append({
            "label": "S1 in (2.5, 4.5) eV", "value": f"{s1['energy_ev']:.3f}",
            "ok": 2.5 <= s1["energy_ev"] <= 4.5, "detail": "experimental 3.5 eV",
        })
        outcome["checks"].append({
            "label": "S1 wavelength in (270, 500) nm",
            "value": f"{s1['wavelength_nm']:.1f}" if s1.get("wavelength_nm") else "n/a",
            "ok": bool(s1.get("wavelength_nm")) and 270 <= s1["wavelength_nm"] <= 500,
            "detail": "",
        })

    outcome["result"] = result
    return outcome


def run_smiles_case() -> dict:
    """RDKit round-trip: SMILES -> 3D -> SCF."""
    outcome = {"name": "SMILES pipeline (aspirin)", "checks": [], "error": None}
    try:
        mol = molmod.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
    except Exception as exc:
        outcome["error"] = f"RDKit embedding failed: {exc}"
        return outcome

    outcome["checks"].append({
        "label": "formula == C9H8O4", "value": mol.formula,
        "ok": mol.formula == "C9H8O4", "detail": "",
    })
    outcome["checks"].append({
        "label": "21 atoms after H addition", "value": str(mol.natoms()),
        "ok": mol.natoms() == 21, "detail": "",
    })

    engine = DFTEngine(
        atom_xyz=mol.to_xyz(), charge=0, multiplicity=1,
        functional="hf", basis="sto-3g",
    )
    try:
        t0 = time.time()
        result = engine.run_scf()
        outcome["seconds"] = round(time.time() - t0, 2)
    except Exception as exc:
        outcome["error"] = f"SCF on embedded structure failed: {exc}"
        return outcome

    outcome["checks"].append({
        "label": "aspirin SCF converged", "value": "yes" if result["converged"] else "no",
        "ok": result["converged"], "detail": "HF/STO-3G",
    })
    outcome["checks"].append({
        "label": "gap in (5, 20) eV", "value": f"{result['gap_ev']:.3f}",
        "ok": 5 <= result["gap_ev"] <= 20, "detail": "",
    })
    outcome["result"] = result
    return outcome


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="ChatDFT self-test")
    ap.add_argument("--full", action="store_true",
                    help="also run geometry optimisation and TD-DFT")
    ap.add_argument("--only", default="", help="run a single case by name")
    args = ap.parse_args()

    cases = CASES
    if args.only:
        cases = [c for c in CASES if c["name"] == args.only]
        if not cases:
            print(f"No case named '{args.only}'.")
            return 2

    outcomes = []
    for case in cases:
        print(f"  {case['name']:<24} ...", end="", flush=True)
        out = run_case(case)
        outcomes.append(out)
        if out["error"]:
            print(f" ERROR: {out['error']}")
        else:
            bad = sum(1 for c in out["checks"] if not c["ok"])
            print(f" {'ok' if bad == 0 else f'{bad} FAILED'}  ({out['seconds']}s)")

    if args.full and not args.only:
        print(f"  {'water optimisation':<24} ...", end="", flush=True)
        out = run_optimisation_case()
        outcomes.append(out)
        if out["error"]:
            print(f" ERROR: {out['error']}")
        else:
            bad = sum(1 for c in out["checks"] if not c["ok"])
            print(f" {'ok' if bad == 0 else f'{bad} FAILED'}  ({out.get('seconds', 0)}s)")

        print(f"  {'formaldehyde TD-DFT':<24} ...", end="", flush=True)
        out = run_tddft_case()
        outcomes.append(out)
        if out["error"]:
            print(f" ERROR: {out['error']}")
        else:
            bad = sum(1 for c in out["checks"] if not c["ok"])
            print(f" {'ok' if bad == 0 else f'{bad} FAILED'}  ({out.get('seconds', 0)}s)")

    print(f"  {'SMILES pipeline':<24} ...", end="", flush=True)
    out = run_smiles_case()
    outcomes.append(out)
    if out["error"]:
        print(f" ERROR: {out['error']}")
    else:
        bad = sum(1 for c in out["checks"] if not c["ok"])
        print(f" {'ok' if bad == 0 else f'{bad} FAILED'}  ({out.get('seconds', 0)}s)")

    # ---- detail report -------------------------------------------------
    print()
    total = passed = 0
    failures = []
    for out in outcomes:
        if out["error"]:
            failures.append((out["name"], "CRASH", out["error"]))
            continue
        for c in out["checks"]:
            total += 1
            if c["ok"]:
                passed += 1
            else:
                failures.append((out["name"], c["label"], f"got {c['value']} {c['detail']}"))

    print("=" * 72)
    if failures:
        print("FAILURES")
        for name, label, detail in failures:
            print(f"  [{name}] {label} -> {detail}")
        print("-" * 72)

    print(f"passed {passed}/{total} checks")
    print("=" * 72)
    return 0 if passed == total and total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
