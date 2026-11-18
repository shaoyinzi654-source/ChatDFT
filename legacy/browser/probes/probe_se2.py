"""Probe -- which selenium molecule should the library gain?

Adding a 77Se reference means the coverage rule I added last round kicks in:
every referenced nucleus must be carried by at least one library entry.  So a
selenium molecule has to exist, and it must not be the reference compound
itself -- shipping Me2Se would make the reference use the library geometry and
move the 77Se scale, exactly the trap phosphine would have been.

Size decides the rest, so this measures rather than guesses.

Changes nothing.

Run:  python probes/probe_se2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import molecule as molmod
from backend.engine import nmr

CANDIDATES = [
    ("hydrogen_selenide", "[SeH2]", "H2Se, the group-16 hydride series"),
    ("dimethyl_selenide", "C[Se]C", "Me2Se -- THE REFERENCE, do not ship"),
    ("selenophene", "c1cc[se]c1", "the selenium analogue of thiophene"),
    ("dimethyl_diselenide", "C[Se][Se]C", "a diselenide, Se-Se bond"),
    ("selenourea", "NC(=[Se])N", "selenourea, a common Se donor"),
]


def main() -> int:
    print("=== selenium candidates ===\n")
    print(f"  {'key':22s} {'atoms':>5s} {'nao':>4s} {'estimate':>10s} "
          f"{'fits':>5s}  source")
    for key, smi, note in CANDIDATES:
        try:
            m = molmod.from_smiles(smi, name=key)
        except Exception as exc:                          # noqa: BLE001
            print(f"  {key:22s}  RDKit refused: {type(exc).__name__}: "
                  f"{str(exc)[:40]}")
            continue
        atoms = [(a.symbol, a.x, a.y, a.z) for a in m.atoms]
        try:
            mol = nmr._build_mol(nmr._xyz(atoms), "6-31g*", int(m.charge),
                                 int(m.multiplicity))
        except Exception as exc:                          # noqa: BLE001
            print(f"  {key:22s}  basis failed: {str(exc)[:50]}")
            continue
        nao = int(mol.nao_nr())
        need = nmr.memory_estimate_mb(nao)
        print(f"  {key:22s} {m.natoms():5d} {nao:4d} {need / 1024:8.2f} GB "
              f"{'yes' if need <= nmr.MEMORY_BUDGET_MB else 'NO':>5s}  "
              f"{m.geometry_source}")
        print(f"  {'':22s}   {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
