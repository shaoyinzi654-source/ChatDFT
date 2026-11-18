"""Probe 31 -- does ``from_smiles`` really run MMFF94 on every molecule?

probe 30 could not evaluate an MMFF94 gradient for borane, carbon monoxide,
hydrogen, hydrogen chloride, hydrogen fluoride, oxygen and ozone: RDKit has no
MMFF94 parameters for them.  ``from_smiles`` calls ``MMFFOptimizeMolecule``
inside a ``try``, so if that call reports failure by returning -1 instead of
raising, the molecule keeps its raw ETKDG coordinates -- no force field
relaxation at all -- while being stamped ``MMFF94 (RDKit ETKDGv3)``.

That would be a provenance string naming a force field that was never applied,
on every molecule MMFF94 cannot describe, which includes anything the
auto-designer invents with boron or a bare ion.

This probe asks RDKit directly and prints the return code.  Changes nothing.

Run:  python probes/probe_nmr31.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

CASES = [
    ("O", "water"),
    ("CCO", "ethanol"),
    ("B", "borane"),
    ("[C-]#[O+]", "carbon monoxide"),
    ("[HH]", "hydrogen"),
    ("Cl", "hydrogen chloride"),
    ("F", "hydrogen fluoride"),
    ("O=O", "oxygen"),
    ("O=[O+][O-]", "ozone"),
    ("[NH4+]", "ammonium"),
    ("[OH-]", "hydroxide"),
    ("[O-][N+](=O)[O-]", "nitrate"),
    ("[SiH4]", "silane"),
]


def main() -> int:
    print("=== probe 31: does MMFF94 actually run for every SMILES? ===\n")
    print(f"{'molecule':20s} {'embed':>6s} {'mmff props':>11s} {'opt ret':>8s}  raised")
    for smi, label in CASES:
        raw = Chem.MolFromSmiles(smi)
        if raw is None:
            print(f"{label:20s} {'--':>6s} {'--':>11s} {'--':>8s}  SMILES did not parse")
            continue
        mol = Chem.AddHs(raw)
        p = AllChem.ETKDGv3()
        p.randomSeed = 0xC0FFEE
        rc = AllChem.EmbedMolecule(mol, p)
        if rc != 0:
            p.useRandomCoords = True
            rc = AllChem.EmbedMolecule(mol, p)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        raised = ""
        try:
            ret: object = AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        except Exception as exc:
            ret = "raised"
            raised = f"{type(exc).__name__}: {exc}"
        has = "yes" if props is not None else "NONE"
        print(f"{label:20s} {rc:>6d} {has:>11s} {str(ret):>8s}  {raised}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
