"""Probe -- build the 77Se reference geometry with this program, not by hand.

77Se's IUPAC reference compound is neat dimethyl selenide, Me2Se (confirmed
against the HUJI nucleus table, which also gives the shift range as 3000 ppm,
-1000 to 2000 -- larger than 33S, so "this is an absolute shielding, not a
shift" matters even more here).

The other references carry experimental r_e geometries with the bond length
and angle written into the string.  I am not going to type one for Me2Se from
memory: this project has already been burned once by a provenance string
nobody checked, and a reference geometry that is wrong moves the entire scale.
So the geometry is produced the way the program produces every other geometry
it trusts -- RDKit embeds it, then this program relaxes it at B3LYP/6-31G* and
checks the gradient -- and whatever comes out is what gets written down.

Changes nothing.

Run:  python -u probes/probe_se1.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import molecule as molmod
from backend.engine import nmr
from backend import verify_library_geometry as V


def main() -> int:
    m = molmod.from_smiles("C[Se]C", name="dimethyl selenide")
    atoms = [(a.symbol, a.x, a.y, a.z) for a in m.atoms]
    start = nmr._xyz(atoms)
    print(f"=== Me2Se: {m.natoms()} atoms, {m.compute_formula()} ===")
    print(f"    embedding geometry source: {m.geometry_source}\n")

    t0 = time.time()
    res = V.relax(start, 0, 1, max_steps=80)
    dt = time.time() - t0
    print(f"    relaxation {dt:.1f} s, converged: {res.get('opt_converged')}")
    print(f"    gradient rms {res.get('grms')}, max {res.get('gmax')}")
    if res.get("error"):
        print(f"    ERROR {res['error']}")
    xyz = res.get("xyz") or start

    mol = nmr._build_mol(xyz, "6-31g*", 0, 1)
    nao = int(mol.nao_nr())
    print(f"    nao {nao}, memory estimate "
          f"{nmr.memory_estimate_mb(nao) / 1024:.2f} GB "
          f"(budget {nmr.MEMORY_BUDGET_MB / 1024:.1f} GB)")

    # Describe it the way reference_geometry_note does, so the hand-written
    # string can be made to agree with the derived one instead of competing.
    from backend.engine import elements
    syms = [l.split()[0] for l in xyz.splitlines()[2:] if l.strip()]
    coords = [[float(x) for x in l.split()[1:4]] for l in xyz.splitlines()[2:]
              if l.strip()]
    import numpy as np
    c = np.array(coords)
    print("\n    bonds:")
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            d = float(np.linalg.norm(c[i] - c[j]))
            if 0.4 < d <= elements.bond_cutoff(syms[i], syms[j]):
                print(f"      {syms[i]:2s}-{syms[j]:2s} {d:.4f} A")
    se = syms.index("Se")
    nb = [j for j in range(len(syms)) if j != se
          and np.linalg.norm(c[se] - c[j]) <= elements.bond_cutoff("Se", syms[j])]
    if len(nb) >= 2:
        v1, v2 = c[nb[0]] - c[se], c[nb[1]] - c[se]
        ang = np.degrees(np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        print(f"\n    C-Se-C {ang:.2f} deg")
        print(f"    Se-C   {np.linalg.norm(v1):.4f} A")

    print("\n    xyz block for _builtin_reference_geometry:")
    for s, xyz3 in zip(syms, coords):
        print(f"        (\"{s}\", {xyz3[0]:.6f}, {xyz3[1]:.6f}, "
              f"{xyz3[2]:.6f}),")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
