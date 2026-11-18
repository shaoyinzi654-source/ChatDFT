"""Probe 30 -- which library geometries are really MMFF94 minima?

probe 29 showed that 33 of the 62 library entries do not reproduce their own
internal distances when re-embedded by the pipeline the code credits
(ETKDG + MMFF94).  Two readings are possible:

* they are MMFF94 geometries from an older RDKit, converged a hair
  differently -- in which case the label is essentially right;
* they came from somewhere else (a DFT relaxation, a textbook) -- in which
  case the label is a false claim.

Re-embedding cannot tell those apart: a geometry 0.0005 A away from the MMFF94
minimum looks the same as a foreign geometry that happens to sit close by.
Water is exactly that trap -- MMFF94 gives O-H 0.9690 / 104.0 deg and the
library says 0.96857 / 104.00 deg, and B3LYP/6-31G* would give almost the same
numbers.  Distance is not evidence of parentage.

What does settle it: MMFF94 is a fixed published force field, so its stationary
points are the same in every RDKit version.  If a geometry is an MMFF94
minimum, the MMFF94 gradient there is zero; if it came from a DFT relaxation,
the gradient is not.  So this probe evaluates |dE/dx| of MMFF94 *at* each
library geometry and compares it with the gradient at the freshly embedded
minimum of the same molecule.  No distance thresholds are involved.

It also checks a suspicion about ``from_smiles`` itself: when MMFF94 has no
parameters for a molecule the code silently falls back to UFF but still stamps
the result ``MMFF94 (RDKit ETKDGv3)``.  If that is true, every such molecule
carries a provenance string naming a force field that was never used.

Changes nothing.

Run:  python probes/probe_nmr30.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from backend.engine.molecule import _load_library, from_smiles, parse_xyz

RDLogger.DisableLog("rdApp.*")


def gradient_norm(mol: Chem.Mol, xyz: np.ndarray) -> float:
    """|dE/dx| of MMFF94 at ``xyz``, in kcal/mol/A, summed over atoms.

    Returns NaN when MMFF94 has no parameters for this molecule.
    """
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return float("nan")
    if mol.GetNumConformers() == 0:
        p = AllChem.ETKDGv3()
        p.randomSeed = 0xC0FFEE
        if AllChem.EmbedMolecule(mol, p) != 0:
            p.useRandomCoords = True
            AllChem.EmbedMolecule(mol, p)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    if ff is None:
        return float("nan")
    g = np.array(ff.CalcGrad()).reshape(-1, 3)
    return float(np.sqrt((g ** 2).sum(axis=1)).sum())


def mapping(smi_mol: Chem.Mol, xyz_text: str):
    """Map smi_mol atom index -> library atom index, or None."""
    n = smi_mol.GetNumAtoms()
    lib = parse_xyz(xyz_text, name="x")
    if len(lib.atoms) != n:
        return None, None
    if sorted(a.symbol for a in lib.atoms) != sorted(
            a.GetSymbol() for a in smi_mol.GetAtoms()):
        return None, None

    # Perceive bonds from the coordinates so the two graphs can be matched on
    # bond order, not just on connectivity.
    xyz_mol = None
    try:
        block = f"{n}\nlib\n" + "".join(
            f"{a.symbol} {a.x:.6f} {a.y:.6f} {a.z:.6f}\n" for a in lib.atoms)
        raw = Chem.MolFromXYZBlock(block)
        if raw is not None:
            rdDetermineBonds.DetermineBonds(raw, charge=int(lib.charge))
            xyz_mol = raw
    except Exception:
        xyz_mol = None

    if xyz_mol is not None and xyz_mol.GetNumAtoms() == n:
        try:
            match = tuple(xyz_mol.GetSubstructMatch(smi_mol))
            if len(match) == n and -1 not in match:
                return lib, list(match)
        except Exception:
            pass

    # Fall back: match on element plus the sorted multiset of distances to
    # every other element.  Good enough for the small, mostly rigid entries.
    lib_pos = np.array([[a.x, a.y, a.z] for a in lib.atoms])
    dmat = np.linalg.norm(lib_pos[:, None, :] - lib_pos[None, :, :], axis=-1)
    lib_elems = [a.symbol for a in lib.atoms]
    smi_elems = [a.GetSymbol() for a in smi_mol.GetAtoms()]
    sig = {}
    for i in range(n):
        key = [lib_elems[i]]
        for e in sorted(set(lib_elems)):
            ds = sorted(round(float(dmat[i, j]), 3)
                        for j in range(n) if j != i and lib_elems[j] == e)
            key.append((e, tuple(ds)))
        sig[i] = str(key)
    used, order = set(), []
    for i, e in enumerate(smi_elems):
        best, bestsig = None, None
        for j in range(n):
            if j in used or lib_elems[j] != e:
                continue
            if best is None or sig[j] < sig[best]:
                best = j
        if best is None:
            return None, None
        used.add(best)
        order.append(best)
    return lib, order


def mmff_or_uff(smiles: str) -> str:
    """Which force field the pipeline actually applies to this SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparsed"
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, p) != 0:
            return "embed-failed"
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        return "MMFF94"
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
            return "UFF (MMFF94 failed)"
        except Exception:
            return "neither"


def main() -> int:
    lib = _load_library()
    print("=== probe 30: is each library geometry an MMFF94 stationary point? ===\n")
    print("g(lib) = |dE/dx| of MMFF94 evaluated AT the stored coordinates")
    print("g(fresh) = the same at the freshly embedded minimum\n")
    print(f"{'entry':22s} {'force field used':22s} {'g(lib)':>10s} {'g(fresh)':>10s}")

    rows = []
    for key, e in sorted(lib.items()):
        smi = (e.get("smiles") or "").strip()
        if not smi:
            rows.append((key, "no smiles", float("nan"), float("nan")))
            continue
        smi_mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        if smi_mol is None:
            rows.append((key, "unparsed", float("nan"), float("nan")))
            continue
        lib_m, order = mapping(smi_mol, e["xyz"])
        if lib_m is None:
            rows.append((key, "cannot map", float("nan"), float("nan")))
            continue
        xyz = np.array([[lib_m.atoms[j].x, lib_m.atoms[j].y, lib_m.atoms[j].z] for j in order])
        glib = gradient_norm(smi_mol, xyz)
        fresh = from_smiles(smi)
        gfresh = gradient_norm(
            smi_mol, np.array([[a.x, a.y, a.z] for a in fresh.atoms]))
        ff = mmff_or_uff(smi)
        rows.append((key, ff, glib, gfresh))

    for key, ff, glib, gfresh in rows:
        fmt = lambda v: "     n/a" if v != v else f"{v:9.4f}"
        print(f"{key:22s} {ff:22s} {fmt(glib)} {fmt(gfresh)}")

    vals = sorted(r[3] for r in rows if r[3] == r[3])
    if vals:
        print(f"\nfresh-minimum gradients: max {max(vals):.4f}")
    bad = [(k, g) for k, ff, g, _ in rows if g == g and g > 1.0]
    print(f"\nlibrary geometries with g(lib) > 1.0 kcal/mol/A : {len(bad)}")
    for k, g in sorted(bad, key=lambda t: -t[1])[:40]:
        print(f"   {k:22s} {g:10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
