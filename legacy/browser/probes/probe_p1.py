"""Probe -- which phosphorus molecule should the library gain?

The library has 62 entries and not one contains phosphorus, so the 31P path
has never been exercised by a real job: only the reference layer is checked,
by the contract, in isolation.  Adding a molecule fixes that, but which one
matters.

Phosphine itself is the wrong choice.  It is the 31P *reference* compound, and
the library-geometry rule says that when the library ships a reference
compound the reference is computed at the library's geometry -- so adding PH3
would replace the experimental r_e reference with whatever RDKit relaxed, and
move the whole 31P scale.  This build already measured that 31P shielding
moves by 2.819 ppm between two geometries of PH3, so that is not a rounding
error.

That leaves a non-reference phosphine.  Size is the binding constraint: the
NMR memory guard, the 20-atom cap, and how long a gate can afford to wait.
This measures the candidates rather than guessing.

Changes nothing.

Run:  python probes/probe_p1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto

from backend.engine import molecule as molmod
from backend.engine import nmr

CANDIDATES = [
    ("methylphosphine", "CP", "primary phosphine, CH3PH2"),
    ("dimethylphosphine", "CPC", "secondary phosphine, (CH3)2PH"),
    ("trimethylphosphine", "CP(C)C", "the common ligand PMe3"),
    ("phosphine_oxide", "O=P", "H3PO"),
    ("trifluorophosphine", "FP(F)F", "PF3"),
]


def main() -> int:
    print("=== phosphorus candidates ===\n")
    print(f"  {'key':22s} {'atoms':>5s} {'nao':>4s} {'estimate':>10s} "
          f"{'fits':>5s}  geometry source")
    for key, smi, note in CANDIDATES:
        try:
            m = molmod.from_smiles(smi, name=key)
        except Exception as exc:                          # noqa: BLE001
            print(f"  {key:22s}  RDKit refused: {type(exc).__name__}: "
                  f"{str(exc)[:44]}")
            continue
        # `m.to_xyz()` carries a title line, and PySCF's atom parser reads the
        # first line as an atom and raises "Zmatrix format error at L1
        # methylphosphine".  nmr._xyz is the project's own conversion from an
        # atoms sequence to a PySCF block, so it is the one to use here.
        # _xyz emits the XYZ *file* layout (count, title, atoms), which PySCF's
        # atom= does not accept -- it reads the count line as an atom and
        # raises "Zmatrix format error at L1".  _build_mol is the function that
        # parses that layout, so go through it rather than around it.
        atoms = [(a.symbol, a.x, a.y, a.z) for a in m.atoms]
        mol = nmr._build_mol(nmr._xyz(atoms), "6-31g*",
                             int(m.charge), int(m.multiplicity))
        nao = int(mol.nao_nr())
        need = nmr.memory_estimate_mb(nao)
        fits = need <= nmr.MEMORY_BUDGET_MB
        print(f"  {key:22s} {m.natoms():5d} {nao:4d} "
              f"{need / 1024:8.2f} GB {'yes' if fits else 'NO':>5s}  "
              f"{m.geometry_source}   ({note})")
    print()
    print("  the 20-atom cap is on NMR, so atom count matters less than nao")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
