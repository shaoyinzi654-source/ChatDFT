"""Decide, per library entry, what its geometry actually is -- and fix it.

Every number this program reports is computed *on* a geometry, so the report
is only as good as the claim it makes about that geometry.  For years the
library stamped one string on all 62 entries -- ``MMFF94 (RDKit ETKDGv3)`` --
and three probes showed that claim was false for the majority of them:

* probe 29: only 29 of 62 entries reproduce their own internal distances when
  re-embedded by the pipeline the string credits;
* probe 30: 29 entries are not MMFF94 stationary points at all -- the MMFF94
  gradient evaluated *at* their coordinates is 1.2 to 886 kcal/mol/A, against
  0.05 for a genuine minimum -- and for 7 more MMFF94 has no parameters, so
  ``from_smiles`` was handing out unrelaxed embeddings under an MMFF94 label;
* probe 35: two entries are not molecules.  The stored methanol closes an
  H-C-H angle of 62 degrees and methanethiol 62.0; trifluoroacetic acid puts
  F-C-F at 78.5; ozone has O-O = 1.4064 A against an experimental 1.2717 A.

Worse provenance is one thing; geometry that cannot exist is another, because
the energy, the spectrum and the shieldings computed from it describe nothing.

So this module stops guessing where the coordinates came from and instead
holds each entry to one of two standards it can actually meet:

1. **reproducible** -- the entry's internal geometry is rebuilt, to within
   ``TOL``, by the pipeline this program runs for any SMILES
   (ETKDG + MMFF94, or UFF where MMFF94 has no parameters).  Then the label
   is that pipeline's, and anyone can regenerate the coordinates.
2. **relaxed here** -- the geometry is a stationary point of B3LYP/6-31G*,
   the level this program optimises at, checked with one analytic gradient
   against the same thresholds the optimiser uses.  Then the label is
   ``DFT-optimised (B3LYP / 6-31G*)``.

An entry meeting neither is not merely mislabelled: nothing vouches for its
coordinates.  ``--apply`` relaxes those at B3LYP/6-31G* with this program's
own optimiser and rewrites the entry, so the library ends up containing only
geometries this program can produce and attest to.

Usage::

    python -m backend.verify_library_geometry            # report, change nothing
    python -m backend.verify_library_geometry --apply    # relax the failures
    python -m backend.verify_library_geometry --only methanol,tfa

The cheap half (1) is importable and is what ``backend/contract.py`` asserts
on every run; the DFT half (2) costs a gradient per entry and is exercised by
its own gate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backend.engine.dft import DFTEngine
from backend.engine.molecule import (
    DFT_BASIS,
    DFT_GEOMETRY,
    DFT_METHOD,
    LIBRARY_UNVERIFIED,
    UNRELAXED_GEOMETRY,
    _load_library,
    from_smiles,
    parse_xyz,
)

# How far two internal geometries may differ and still be called the same.
# ETKDG+MMFF94 is deterministic, so a genuine product of the pipeline
# reproduces to ~1e-6 A -- and the entries this tool accepts do exactly that,
# while the nearest miss (water) is 4.6e-4 A away, 450 times further.  The
# threshold sits in that gap rather than at a round number: at 1e-3 water
# would be labelled MMFF94, and although its coordinates are only half a
# thousandth of an Angstrom from that minimum, 17O shieldings move at
# 539 ppm/A here, so a geometry that is *nearly* the one claimed is still a
# different measurement.
TOL = 1e-5

# Inherited from the optimiser rather than restated: if this tool used its own
# thresholds it would be vouching for a different standard than the one that
# produced the coordinates.
OPT_CRITERIA = DFTEngine.OPT_CONVERGENCE

LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "molecules",
    "library.json",
)


# ----------------------------------------------------------------------
# geometry comparison
# ----------------------------------------------------------------------
def internal_fingerprint(atoms: Sequence) -> List[Tuple[Tuple[str, str], float]]:
    """Multiset of (element pair, distance) -- invariant to how the molecule
    is placed, oriented, mirrored or ordered, which a coordinate comparison
    is not.  The first version of the probe behind this module compared
    coordinates and declared dihydrogen wrong by 0.74 A, its own bond
    length, because ETKDG had simply put the molecule somewhere else.
    """
    pos = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
    syms = [a.symbol for a in atoms]
    out = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            out.append((tuple(sorted((syms[i], syms[j]))), round(d, 6)))
    return sorted(out)


def fingerprint_deviation(atoms_a: Sequence, atoms_b: Sequence) -> float:
    """Largest difference between two internal geometries, in Angstrom."""
    fa, fb = internal_fingerprint(atoms_a), internal_fingerprint(atoms_b)
    if len(fa) != len(fb):
        return float("inf")
    return max(abs(da - db) for (_, da), (_, db) in zip(fa, fb))


# ----------------------------------------------------------------------
# standard 1: is it what the SMILES pipeline produces?
# ----------------------------------------------------------------------
def force_field_verdict(entry: dict, key: str = "") -> dict:
    """Re-embed the entry from its own SMILES and compare.

    Cheap: no quantum chemistry.  Returns the provenance the entry has earned
    from this test, or ``None`` when it has earned none.
    """
    out = {"ok": False, "dev": None, "source": None, "reason": ""}
    smiles = (entry.get("smiles") or "").strip()
    if not smiles:
        out["reason"] = "the entry has no SMILES, so nothing can rebuild it"
        return out
    try:
        lib_atoms = parse_xyz(entry["xyz"], name=key).atoms
        fresh = from_smiles(smiles)
    except Exception as exc:
        out["reason"] = f"could not be rebuilt ({type(exc).__name__}: {exc})"
        return out
    if sorted(a.symbol for a in lib_atoms) != sorted(a.symbol for a in fresh.atoms):
        out["reason"] = ("the stored composition does not match the SMILES, so "
                         "the two geometries cannot be compared")
        return out

    dev = fingerprint_deviation(lib_atoms, fresh.atoms)
    out["dev"] = dev
    if dev > TOL:
        out["reason"] = (f"re-embedding from its SMILES moves it {dev:.4f} A, "
                         f"so these coordinates are not from that pipeline")
        return out
    # A geometry that matches an *unrelaxed* embedding has earned nothing:
    # it is the output of ETKDG's distance-geometry guess, with bond lengths
    # no force field ever looked at.
    if fresh.geometry_source == UNRELAXED_GEOMETRY:
        out["reason"] = ("it reproduces an unrelaxed ETKDG embedding exactly, "
                         "which means no force field ever relaxed it")
        return out
    out["ok"] = True
    out["source"] = fresh.geometry_source
    return out


# ----------------------------------------------------------------------
# standard 2: is it a B3LYP/6-31G* stationary point?
# ----------------------------------------------------------------------
def dft_gradient(atom_xyz: str, charge: int, multiplicity: int) -> dict:
    """One analytic gradient at the given geometry, in hartree/bohr.

    A stationary point is a point whose gradient is below the thresholds the
    optimiser actually uses -- the same test ``DFTEngine.optimize`` applies to
    its own output, reused here so the two cannot drift apart.
    """
    out = {"grms": None, "gmax": None, "converged": None, "error": ""}
    try:
        eng = DFTEngine(
            atom_xyz=atom_xyz,
            charge=int(charge),
            multiplicity=int(multiplicity),
            functional=DFT_METHOD,
            basis=DFT_BASIS,
        )
        res = eng.run_scf()
        if eng.mf is None or not getattr(eng.mf, "converged", False):
            out["error"] = "the SCF did not converge, so the gradient is meaningless"
            return out
        g = np.asarray(eng.mf.nuc_grad_method().kernel(), dtype=float)
        out["grms"] = float(np.sqrt((g ** 2).mean()))
        out["gmax"] = float(np.abs(g).max())
        out["converged"] = bool(
            out["grms"] <= OPT_CRITERIA["convergence_grms"]
            and out["gmax"] <= OPT_CRITERIA["convergence_gmax"])
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def relax(atom_xyz: str, charge: int, multiplicity: int, max_steps: int = 60) -> dict:
    """Optimise a geometry at B3LYP/6-31G* with this program's optimiser."""
    eng = DFTEngine(
        atom_xyz=atom_xyz,
        charge=int(charge),
        multiplicity=int(multiplicity),
        functional=DFT_METHOD,
        basis=DFT_BASIS,
    )
    return eng.optimize(max_steps=max_steps)


def xyz_block(atoms: Sequence, title: str) -> str:
    """An XYZ block in the format this program writes."""
    lines = [str(len(atoms)), title]
    for a in atoms:
        lines.append(f"{a.symbol:<3s} {a.x:16.8f} {a.y:16.8f} {a.z:16.8f}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# per-entry classification
# ----------------------------------------------------------------------
def classify(entry: dict, key: str = "") -> dict:
    """What this entry's geometry is, and what should be done about it."""
    info = {"key": key, "verdict": "", "source": None, "detail": ""}
    ff = force_field_verdict(entry, key)
    if ff["ok"]:
        info["verdict"] = "reproducible"
        info["source"] = ff["source"]
        info["detail"] = f"re-embeds to {ff['dev']:.2e} A"
        return info
    info["detail"] = ff["reason"]

    grad = dft_gradient(entry["xyz"], entry.get("charge", 0),
                        entry.get("multiplicity", 1))
    if grad["error"]:
        info["verdict"] = "unverifiable"
        info["detail"] += f"; the gradient could not be computed: {grad['error']}"
        return info
    if grad["converged"]:
        info["verdict"] = "dft"
        info["source"] = DFT_GEOMETRY
        info["detail"] += (f"; a B3LYP/6-31G* gradient at these coordinates is "
                           f"|g|rms {grad['grms']:.1e}, |g|max {grad['gmax']:.1e} "
                           f"hartree/bohr -- a stationary point")
        return info
    info["verdict"] = "foreign"
    info["detail"] += (f"; |g|rms {grad['grms']:.1e} hartree/bohr at B3LYP/6-31G*, "
                       f"so nothing here produced these coordinates")
    return info


# ----------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------
def _write_library(lib: dict, path: str) -> None:
    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(lib, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="relax the entries nothing can vouch for, in place")
    ap.add_argument("--only", default="",
                    help="comma-separated entry keys to look at")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero unless every entry meets one of the "
                         "two standards (for use as a gate)")
    args = ap.parse_args(argv)

    with open(LIB_PATH, "r", encoding="utf-8") as fh:
        lib = json.load(fh)
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    print("=== what is each library geometry, really? ===\n")
    print(f"reproducible: re-embedding from the SMILES rebuilds it to {TOL:g} A")
    print(f"dft         : it is a {DFT_METHOD}/{DFT_BASIS} stationary point")
    print("foreign     : nothing here produced these coordinates\n")
    print(f"{'entry':20s} {'verdict':14s} {'source':46s}")
    print("-" * 82)

    counts: Dict[str, int] = {}
    applied: List[str] = []
    updated: List[str] = []
    for key, entry in lib.items():
        if wanted and key not in wanted:
            continue
        info = classify(entry, key)
        counts[info["verdict"]] = counts.get(info["verdict"], 0) + 1
        print(f"{key:20s} {info['verdict']:14s} {str(info['source'] or '-')[:46]:46s}")
        if info["detail"]:
            print(f"{'':20s}   {info['detail']}")
        sys.stdout.flush()

        if info["verdict"] == "reproducible" or info["verdict"] == "dft":
            if entry.get("geometry_source") != info["source"]:
                entry["geometry_source"] = info["source"]
                if args.apply:
                    updated.append(key)
            continue
        if not args.apply:
            continue

        # Nothing vouches for the coordinates: produce ones that can be
        # vouched for, from this geometry where possible and from a fresh
        # embedding where the stored one is too broken to start from.
        starts = [("stored", entry["xyz"])]
        try:
            starts.append(("fresh embedding", xyz_block(
                from_smiles(entry["smiles"]).atoms, entry.get("display_name", key))))
        except Exception:
            pass
        for label, start in starts:
            try:
                res = relax(start, entry.get("charge", 0),
                            entry.get("multiplicity", 1))
            except Exception as exc:
                print(f"{'':20s}   optimisation from {label} failed: "
                      f"{type(exc).__name__}: {str(exc)[:80]}")
                continue
            if not res.get("opt_converged"):
                print(f"{'':20s}   optimisation from {label} did not converge "
                      f"(|g|rms {res.get('opt_grms')}) - not storing it")
                continue
            new_xyz = res["optimized_xyz"].rstrip("\n")
            # Re-check with our own gradient: trust the physics, not the flag.
            check = dft_gradient(new_xyz, entry.get("charge", 0),
                                 entry.get("multiplicity", 1))
            if not check.get("converged"):
                print(f"{'':20s}   the relaxed geometry fails the gradient "
                      f"test (|g|rms {check.get('grms')}) - not storing it")
                continue
            atoms = parse_xyz(new_xyz, name=key).atoms
            entry["xyz"] = xyz_block(atoms, entry.get("display_name", key))
            entry["geometry_source"] = DFT_GEOMETRY
            applied.append(key)
            print(f"{'':20s}   relaxed from the {label} geometry and stored "
                  f"(|g|rms {check['grms']:.1e} hartree/bohr)")
            break

    print()
    for verdict, n in sorted(counts.items()):
        print(f"{verdict:14s}: {n}")

    if args.check:
        # Counted during the pass above, so this costs nothing extra.
        n_foreign = counts.get("foreign", 0) + counts.get("unverifiable", 0)
        if n_foreign:
            print(f"\nFAIL: {n_foreign} entr(y/ies) carry coordinates that "
                  "nothing here can vouch for")
            return 1
        print("\nPASS: every library geometry is either reproducible from its "
              "own SMILES or a verified B3LYP/6-31G* stationary point")
        return 0
    if args.apply:
        touched = sorted(set(applied) | set(updated))
        if touched:
            _write_library(lib, LIB_PATH)
            print(f"\nrewrote {len(touched)} entries in data/molecules/library.json")
            if applied:
                print(f"   relaxed at {DFT_METHOD}/{DFT_BASIS}: " + ", ".join(sorted(applied)))
            if updated:
                print(f"   provenance recorded: " + ", ".join(sorted(updated)))
        else:
            print("\nnothing needed changing")
    else:
        print("\n(re-run with --apply to relax the entries nothing can vouch for)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
