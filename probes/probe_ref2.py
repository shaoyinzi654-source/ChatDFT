"""Probe -- how big is BF3.OEt2, the 11B reference, in this basis?

The 11B standard is boron trifluoride etherate, and the only question that
decides whether this build can ever compute it is its size.  nao is read from
PySCF for the real molecule rather than added up by hand from shell counts,
and the memory estimate comes from the guard itself so the two numbers cannot
disagree.

Changes nothing.

Run:  python probes/probe_ref2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto

from backend.engine import nmr

# BF3.O(C2H5)2: the boron carries three F, the oxygen carries two ethyls.
XYZ = """B 0.000 0.000 0.000
F 1.310 0.000 0.000
F -0.655 1.134 0.000
F -0.655 -1.134 0.000
O 0.000 0.000 1.590
C 0.760 1.240 2.100
C 1.520 1.120 3.400
C -1.230 0.480 2.100
C -2.400 1.200 3.400
H 0.060 2.130 2.100
H 1.480 1.360 1.400
H 1.130 0.180 3.800
H 2.280 1.860 3.300
H 1.040 1.100 4.100
H -1.750 -0.400 1.400
H -0.860 0.180 3.800
H -2.050 1.900 3.300
H -3.100 0.550 3.600
H -2.950 1.480 4.100"""


def main() -> int:
    print("=== 11B reference BF3.OEt2 at 6-31G* ===\n")
    mol = gto.M(atom=XYZ, basis="6-31g*", charge=0, spin=0, verbose=0)
    nao = int(mol.nao_nr())
    need = nmr.memory_estimate_mb(nao)
    print(f"  atoms  {mol.natm}")
    print(f"  nao    {nao}")
    print(f"  estimate {need / 1024:.1f} GB   "
          f"(budget {nmr.MEMORY_BUDGET_MB / 1024:.1f} GB)")
    print(f"  fits?  {'yes' if need <= nmr.MEMORY_BUDGET_MB else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
