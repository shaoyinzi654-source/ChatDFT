"""Geometric validation of the built-in molecule library.

A corrupt entry is the worst kind of bug here: the geometry resolves, the SCF
converges, and every number downstream is quietly wrong.  (Naphthalene shipped
with two hydrogens sitting 0.43 A from a carbon and a ring carbon with no
hydrogen at all, which drove its HOMO-LUMO gap down to 2.3 eV.)

Checks, per entry:
  * no two nuclei closer than 0.70 A (they must not overlap)
  * every H has a heavy neighbour inside its element's bond limit
  * every heavy atom's nearest-neighbour count is chemically plausible
  * all bond lengths fall in a sane window for the element pair
  * the stored formula matches the atoms actually present

Usage:  python -m backend.check_library
"""

from __future__ import annotations

import math
import re
import sys

from backend.engine import molecule as molmod

# Generous upper bound for a covalent bond, by element pair (Angstrom).
BOND_MAX = {
    ("H", "H"): 0.90, ("C", "H"): 1.25, ("N", "H"): 1.20, ("O", "H"): 1.15,
    ("S", "H"): 1.50, ("C", "C"): 1.70, ("C", "N"): 1.60, ("C", "O"): 1.55,
    ("C", "F"): 1.50, ("C", "S"): 1.90, ("C", "Cl"): 1.85, ("N", "N"): 1.55,
    ("N", "O"): 1.55, ("O", "O"): 1.60, ("O", "S"): 1.75,
    # B, Si, F and Cl used to be absent from this table, which did not make
    # the checker strict -- it made it blind: an H-F or Si-H bond was not a
    # recognised pair, so HF and silane were reported as "no bonded
    # neighbours".  Silane's real Si-H is 1.48 A, not a carbon's 1.09 A.
    ("B", "H"): 1.35, ("Si", "H"): 1.65, ("F", "H"): 1.10, ("Cl", "H"): 1.45,
    ("Br", "H"): 1.60, ("I", "H"): 1.80, ("P", "H"): 1.60, ("Se", "H"): 1.65,
    ("B", "N"): 1.80, ("B", "O"): 1.65, ("B", "C"): 1.75,
    ("Si", "C"): 2.05, ("Si", "O"): 1.85, ("Si", "N"): 1.90,
    ("Si", "Si"): 2.55, ("Si", "Cl"): 2.20,
    ("P", "O"): 1.80, ("P", "C"): 1.90, ("S", "S"): 2.30, ("N", "S"): 1.75,
    ("C", "Br"): 2.15, ("C", "I"): 2.40, ("F", "F"): 1.50, ("Cl", "Cl"): 2.10,
}
# Lower bound for a real bond, by element pair.  H2 really is 0.74 A, so a
# single global floor would flag a perfectly good molecule.
BOND_MIN = {
    ("H", "H"): 0.70, ("C", "H"): 1.00, ("N", "H"): 0.95, ("O", "H"): 0.90,
    ("S", "H"): 1.25, ("C", "C"): 1.15, ("C", "N"): 1.10, ("C", "O"): 1.10,
    ("C", "F"): 1.25, ("C", "S"): 1.60, ("C", "Cl"): 1.60, ("N", "N"): 1.05,
    ("N", "O"): 1.10, ("O", "O"): 1.10, ("O", "S"): 1.20,
    ("B", "H"): 1.00, ("Si", "H"): 1.30, ("F", "H"): 0.80, ("Cl", "H"): 1.10,
    ("Br", "H"): 1.25, ("I", "H"): 1.45, ("P", "H"): 1.25, ("Se", "H"): 1.30,
    ("B", "N"): 1.35, ("B", "O"): 1.25, ("B", "C"): 1.35,
    ("Si", "C"): 1.65, ("Si", "O"): 1.45, ("Si", "N"): 1.50,
    ("Si", "Si"): 2.10, ("Si", "Cl"): 1.90,
    ("P", "O"): 1.40, ("P", "C"): 1.50, ("S", "S"): 1.90, ("N", "S"): 1.50,
    ("C", "Br"): 1.75, ("C", "I"): 1.95, ("F", "F"): 1.30, ("Cl", "Cl"): 1.85,
}
# How far an H may sit from its nearest heavy atom.  A single global limit
# would flag a perfectly ordinary S-H bond (1.34 A), so it is per element.
H_NEIGHBOUR_MAX = {
    "B": 1.35, "C": 1.25, "N": 1.20, "O": 1.15, "F": 1.15, "S": 1.55,
    "Cl": 1.45, "P": 1.60,
    # Si-H is 1.48 A.  The 1.30 default below is a carbon number and was
    # rejecting a correct silane geometry as a missing hydrogen.
    "Si": 1.65, "Se": 1.60, "Br": 1.55, "I": 1.75,
}
H_NEIGHBOUR_DEFAULT = 1.30
CLASH = 0.70             # no two nuclei can be closer than this

# Covalent radii (Cordero et al., Dalton Trans. 2008, 2832), used only for
# element pairs the tables above do not cover.  Without a fallback an unknown
# pair silently counts as "not bonded", and a molecule made of unfamiliar
# elements is reported as a pile of unconnected atoms.
COVALENT_RADII = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "K": 2.03, "Ca": 1.76, "Fe": 1.32, "Cu": 1.32, "Zn": 1.22,
    "Se": 1.20, "Br": 1.20, "I": 1.39,
}
RADIUS_FALLBACK_MAX = 1.30   # bond if d < 1.30 * (r1 + r2)
RADIUS_FALLBACK_MIN = 0.65   # too short if d < 0.65 * (r1 + r2)


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if (a, b) in BOND_MAX else (b, a)


def _bond_max(a: str, b: str) -> float:
    p = _pair(a, b)
    if p in BOND_MAX:
        return BOND_MAX[p]
    ra, rb = COVALENT_RADII.get(a), COVALENT_RADII.get(b)
    if ra is None or rb is None:
        return 0.0        # unknown element: do not guess a bond
    return RADIUS_FALLBACK_MAX * (ra + rb)


def _bond_min(a: str, b: str) -> float:
    p = _pair(a, b)
    if p in BOND_MIN:
        return BOND_MIN[p]
    ra, rb = COVALENT_RADII.get(a), COVALENT_RADII.get(b)
    if ra is None or rb is None:
        return 0.0
    return RADIUS_FALLBACK_MIN * (ra + rb)


def _parse_formula(f: str) -> dict:
    """'C2H4O2' -> {'C': 2, 'H': 4, 'O': 2}, so ordering does not matter."""
    return {el: int(cnt or 1) for el, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", f) if el}


def _dist(a, b) -> float:
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def check_entry(key: str) -> list[str]:
    problems: list[str] = []
    try:
        mol = molmod.from_name(key)
    except Exception as exc:  # noqa: BLE001
        return [f"could not build: {exc}"]

    atoms = mol.atoms
    n = len(atoms)
    if n == 0:
        return ["no atoms"]

    has_heavy = any(a.symbol != "H" for a in atoms)

    # ---- overlapping nuclei -----------------------------------------
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist(atoms[i], atoms[j])
            if d < CLASH:
                problems.append(
                    f"{atoms[i].symbol}{i + 1}-{atoms[j].symbol}{j + 1} "
                    f"are {d:.3f} A apart (nuclei overlap)"
                )

    # ---- every H needs exactly one heavy neighbour -------------------
    if has_heavy:
        for i, a in enumerate(atoms):
            if a.symbol != "H":
                continue
            heavy = sorted(
                (_dist(a, b), j) for j, b in enumerate(atoms) if b.symbol != "H"
            )
            d1, j1 = heavy[0]
            limit = H_NEIGHBOUR_MAX.get(atoms[j1].symbol, H_NEIGHBOUR_DEFAULT)
            if d1 > limit:
                problems.append(
                    f"H{i + 1} is {d1:.3f} A from the nearest heavy atom "
                    f"({atoms[j1].symbol}{j1 + 1}; limit {limit:.2f})"
                )

    # ---- bond lengths in a sane window -------------------------------
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = atoms[i].symbol, atoms[j].symbol
            bmax = _bond_max(si, sj)
            if bmax <= 0.0:
                continue
            d = _dist(atoms[i], atoms[j])
            if d <= bmax:
                bmin = _bond_min(si, sj)
                if d < bmin:
                    problems.append(
                        f"{si}{i + 1}-{sj}{j + 1} bond is {d:.3f} A "
                        f"(below {bmin:.2f})"
                    )

    # ---- coordination numbers ---------------------------------------
    for i, a in enumerate(atoms):
        if a.symbol == "H":
            continue
        nb = 0
        for j, b in enumerate(atoms):
            if j == i:
                continue
            bmax = _bond_max(a.symbol, b.symbol)
            if bmax > 0.0 and _dist(a, b) <= bmax:
                nb += 1
        if nb == 0:
            problems.append(f"{a.symbol}{i + 1} has no bonded neighbours")
        elif nb > 4:
            problems.append(f"{a.symbol}{i + 1} has {nb} neighbours (max 4)")

    # ---- formula consistency ----------------------------------------
    if mol.formula and _parse_formula(mol.formula) != _parse_formula(mol.compute_formula()):
        problems.append(
            f"stored formula {mol.formula} != atoms present {mol.compute_formula()}"
        )

    return problems


def main() -> int:
    names = molmod.library_names()
    keys = [n["key"] if isinstance(n, dict) else n for n in names]
    print(f"validating {len(keys)} library entries\n")

    total = 0
    for key in keys:
        problems = check_entry(key)
        if problems:
            total += len(problems)
            print(f"[FAIL] {key}")
            for p in problems:
                print(f"         {p}")
        else:
            print(f"[PASS] {key}")

    print()
    if total:
        print(f"=== FAIL: {total} geometric problem(s) in the library ===")
        return 1
    print("=== PASS: every library geometry is chemically sane ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
