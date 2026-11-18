"""probe_nmr2.py -- pin down the exact names/shapes of the GIAO integrals.

probe_nmr1 showed the ig* family exists.  The reference formulation also names
int1e_giao_irjxp, int1e_ia01p, int1e_giao_a11part and int1e_a01gp.  Find out
which spellings this libcint answers to, and what each one is, before writing a
line of product code.
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto

mol = gto.M(
    atom=[("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587), ("H", 0.0, 0.757, 0.587)],
    basis="sto-3g", unit="Angstrom", verbose=0)
nao = mol.nao_nr()
print(f"nao = {nao}\n")

NAMES = [
    # field derivative of the core Hamiltonian (GIAO)
    "int1e_giao_irjxp", "int1e_igirjxp", "int1e_giao_irjxp",
    # GIAO phase derivatives
    "int1e_igovlp", "int1e_igkin", "int1e_ignuc",
    # nuclear magnetic dipole (orbital Zeeman / PSO)
    "int1e_ia01p", "int1e_pnucxp", "int1e_prinvxp",
    # second order B x mu
    "int1e_giao_a11part", "int1e_a01gp", "int1e_giao_a11",
    # ERI field derivative
    "int2e_ig1", "int2e_giao",
    # overlap / momentum
    "int1e_ovlp", "int1e_ipovlp",
]

good = {}
for name in NAMES:
    try:
        with np.errstate(all="ignore"):
            v = mol.intor(name)
        good[name] = v.shape
        print(f"  OK    {name:24s} shape={str(v.shape):16s} comp_guess={v.size//(nao*nao)}")
    except Exception as exc:
        print(f"  --    {name:24s} {type(exc).__name__}: {str(exc).splitlines()[0][:55]}")

print()
print("=" * 74)
print("  with_rinv_orig: does the PSO operator move with the nucleus?")
print("=" * 74)
h_at = {}
for i in range(mol.natm):
    with mol.with_rinv_orig(mol.atom_coord(i)):
        h_at[i] = mol.intor("int1e_ia01p")
    print(f"  atom {i} ({mol.atom_symbol(i)}): shape={h_at[i].shape} "
          f"norm={np.linalg.norm(h_at[i]):.6e}")

print()
print("=" * 74)
print("  physics checks")
print("=" * 74)

# 1. int1e_giao_irjxp should be the GIAO orbital angular momentum:
#    for the H 1s functions it should be ~0 (s functions have no l), and for O
#    the 2p block should be nonzero and antisymmetric in the right way.
for name in ["int1e_giao_irjxp", "int1e_cg_irxp"]:
    if name not in good:
        continue
    v = mol.intor(name)
    asym = np.max(np.abs(v + np.transpose(v, (0, 2, 1))))
    sym = np.max(np.abs(v - np.transpose(v, (0, 2, 1))))
    print(f"  {name:22s} max|antisym|={asym:.4e}  max|sym|={sym:.4e}")

# 2. int1e_ignuc / int1e_igkin: the GIAO phase derivative.  These should be
#    antisymmetric in the field index pattern and vanish for a field along the
#    bond axis of a diatomic by symmetry.  Check with water, where the O-H
#    direction should give a smaller response than the out-of-plane axis.
if "int1e_ignuc" in good:
    v = mol.intor("int1e_ignuc")
    print(f"  int1e_ignuc per-component norms: "
          f"{[float(np.linalg.norm(v[t])) for t in range(3)]}")

# 3. int1e_a01gp should be the (B x mu) second-order operator, symmetric in t,s
if "int1e_a01gp" in good:
    v = mol.intor("int1e_a01gp")
    n = v.size // (nao * nao)
    if n == 9:
        v = v.reshape(3, 3, nao, nao)
        print(f"  int1e_a01gp as (3,3,nao,nao): "
              f"max|v_ts - v_st| = {np.max(np.abs(v - np.transpose(v, (1, 0, 2, 3)))):.4e}")

# 4. trace of a11part + a01gp should be related to the diamagnetic operator
if "int1e_giao_a11part" in good:
    v = mol.intor("int1e_giao_a11part")
    print(f"  int1e_giao_a11part shape={v.shape} comps={v.size//(nao*nao)}")

print()
print("=" * 74)
print("  gauge-origin independence: are the ig* integrals origin dependent?")
print("=" * 74)
for name in ["int1e_igovlp", "int1e_igkin", "int1e_ignuc", "int1e_giao_irjxp"]:
    if name not in good:
        continue
    a = mol.intor(name)
    with mol.with_common_origin([1.0, 2.0, 3.0]):
        b = mol.intor(name)
    print(f"  {name:22s} max|shift| = {np.max(np.abs(a - b)):.4e}")

print()
print("=" * 74)
print("  int2e_ig1 availability decides whether a *numerical* cross-check")
print("  of the full GIAO Hamiltonian is possible")
print("=" * 74)
try:
    e = mol.intor("int2e_ig1")
    print(f"  int2e_ig1 OK shape={e.shape}")
except Exception as exc:
    print(f"  int2e_ig1 missing: {type(exc).__name__}")
