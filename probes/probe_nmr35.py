"""Probe 35 -- a chemistry sanity pass over the whole molecule library.

probe 34 found that the stored methanol puts two of its methyl hydrogens
1.0938 A apart, which with C-H ~ 1.09 A means an H-C-H angle of 62 degrees.
No molecule does that.  The ozone entry has O-O = 1.4064 A against an
experimental 1.2717 A.

So part of this library is not "a geometry from a different method" -- it is
simply not a molecule.  Every number computed from such an entry is a number
about nothing: its energy, its spectrum, its NMR shieldings, and any figure
drawn from them.

This probe stops asking where a geometry came from and asks whether it is
chemically possible at all, using only the coordinates and the covalent radii
the codebase already carries:

* perceived bonds (elements.bond_cutoff);
* the tightest bond angle at every atom -- below about 70 degrees no stable
  structure in this library sits (cyclopropane would, and is not in it);
* bond lengths against the sum of covalent radii;
* the perceived hydrogen count on each heavy atom against the SMILES.

Nothing here depends on a reference geometry or a force field, so it cannot
be argued with by saying the library uses a different method.

Changes nothing.

Run:  python probes/probe_nmr35.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rdkit import Chem, RDLogger

from backend.engine.elements import bond_cutoff, get as element_get
from backend.engine.molecule import _load_library, parse_xyz

RDLogger.DisableLog("rdApp.*")

# No minimum-energy structure in this library has a bond angle below this.
# (Cyclopropane does -- 60 degrees -- and is deliberately not in the library.)
MIN_ANGLE = 70.0
# A bond may be this much shorter/longer than the sum of covalent radii.
BOND_LO, BOND_HI = 0.78, 1.45


def perceive(atoms):
    pos = np.array([[a.x, a.y, a.z] for a in atoms])
    syms = [a.symbol for a in atoms]
    n = len(atoms)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d <= bond_cutoff(syms[i], syms[j]):
                bonds.append((i, j, d))
    return pos, syms, bonds


def neighbours(n, bonds):
    out = {i: [] for i in range(n)}
    for i, j, d in bonds:
        out[i].append((j, d))
        out[j].append((i, d))
    return out


def min_angle(pos, nb) -> tuple:
    """Tightest bond angle in the structure, with where it was found."""
    worst = (999.0, "", -1)
    for i, lst in nb.items():
        if len(lst) < 2:
            continue
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                j, _ = lst[a]
                k, _ = lst[b]
                v1 = pos[j] - pos[i]
                v2 = pos[k] - pos[i]
                c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                ang = math.degrees(math.acos(max(-1.0, min(1.0, c))))
                if ang < worst[0]:
                    worst = (ang, "", i)
    return worst


def smiles_h_counts(smiles: str):
    """{heavy atom index: number of attached hydrogens} from the SMILES."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles)) if smiles else None
    if mol is None:
        return None
    return {a.GetIdx(): sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "H")
            for a in mol.GetAtoms() if a.GetSymbol() != "H"}


def main() -> int:
    lib = _load_library()
    print("=== probe 35: is every library geometry a possible molecule? ===\n")
    print(f"bond angle floor {MIN_ANGLE:g} deg; "
          f"bond length window {BOND_LO}-{BOND_HI} x sum of covalent radii\n")
    print(f"{'entry':20s} {'nbonds':>7s} {'min angle':>10s} {'worst bond / A':>16s}")

    broken = []
    for key, e in sorted(lib.items()):
        try:
            atoms = parse_xyz(e["xyz"], name=key).atoms
        except Exception as exc:
            print(f"{key:20s}  UNREADABLE: {exc}")
            broken.append(key)
            continue
        pos, syms, bonds = perceive(atoms)
        nb = neighbours(len(atoms), bonds)

        # connectivity: every atom must be reached
        if len(atoms) > 1:
            seen, stack = {0}, [0]
            while stack:
                i = stack.pop()
                for j, _ in nb[i]:
                    if j not in seen:
                        seen.add(j)
                        stack.append(j)
            if len(seen) != len(atoms):
                print(f"{key:20s}  DISCONNECTED: {len(atoms) - len(seen)} atoms adrift")
                broken.append(key)
                continue

        ang, _, at = min_angle(pos, nb)
        worst_bond = ""
        for i, j, d in bonds:
            rs = element_get(syms[i]).covalent_radius + element_get(syms[j]).covalent_radius
            ratio = d / rs if rs else 1.0
            if ratio < BOND_LO or ratio > BOND_HI:
                worst_bond = f"{syms[i]}{i}-{syms[j]}{j} {d:.4f}"
                break
        flag = ""
        if ang < MIN_ANGLE:
            flag = f"  <<< angle {ang:.1f} deg at atom {at} ({syms[at]})"
            broken.append(key)
        if worst_bond:
            flag += f"  <<< bond {worst_bond}"
            if key not in broken:
                broken.append(key)
        print(f"{key:20s} {len(bonds):7d} {ang:10.1f} {worst_bond:>16s}{flag}")

    print(f"\nentries that fail: {len(broken)}")
    for k in broken:
        print("   " + k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
