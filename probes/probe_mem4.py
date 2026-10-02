"""Probe -- did raising the constant to its measured value break TMS?

The budget check is `need > MEMORY_BUDGET_MB`, and the budget was 6000 MB.
The counted constant put TMS at 5165 MB, which passed with 16% to spare.  The
measured constant puts the same reference at 6365 MB -- over the budget.  The
free-memory guard is not involved in that comparison at all, so the refusal
would happen on a 128 GB machine too.

Every gate runs with the TMS cache warm, and `reference_shieldings` reads the
cache before it ever calls check_memory, so no gate would notice: the failure
only appears on a cold cache, which is a fresh install.  That is the whole
shape of the regression, so it is worth seeing the number rather than
reasoning about it.

Changes nothing.

Run:  python probes/probe_mem4.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr


def main() -> int:
    print("=== does the budget still admit every reference? ===\n")
    print(f"budget {nmr.MEMORY_BUDGET_MB} MB, "
          f"constant {nmr._BYTES_PER_NAO4} bytes/nao^4\n")
    bad = 0
    for key in sorted(nmr.REFERENCES):
        atoms = nmr.reference_geometry(key)
        mol = nmr._build_mol(nmr._xyz(atoms), "6-31g*", 0, 1)
        nao = int(mol.nao_nr())
        need = nmr.memory_estimate_mb(nao)
        over = need > nmr.MEMORY_BUDGET_MB
        bad += 1 if over else 0
        print(f"  {key:14s} nao={nao:3d}  need {need:8.1f} MB  "
              f"{'REFUSED' if over else 'fits'}")
    print()
    if bad:
        print(f"!! {bad} reference(s) cannot be computed at all: the budget "
              "refuses them whatever the machine has free.")
        print("   With a cold cache that removes every shift quoted against "
              "them.")
    else:
        print("all references fit the budget")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
