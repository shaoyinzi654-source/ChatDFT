"""Round-15 probe 2: why is the finite-field polarizability wrong?

Finite field gave alpha_iso = 5.13 a.u. for water; experiment is 9.90 and a
6-31G* basis should land near 8.6.  The tensor was also wrongly anisotropic
(2.74 / 7.32 / 5.32 where the two in-plane components should be within a few
per cent of each other), so it is not a unit slip -- something in the setup
is wrong.  Three checks, each of which isolates one piece:

  A. Does mol.intor('int1e_r') mean what I think it means?
  B. Does the dipole from that operator reproduce mf.dip_moment()?
  C. Does dE/dE_field reproduce -mu?  (Hellmann-Feynman: it must.)

Run:  python probes/probe_raman2.py
"""
from __future__ import annotations

import numpy as np

import backend.bootstrap as bootstrap

bootstrap.setup()

from pyscf import dft, gto

WATER = [
    ("O", 0.000000, 0.000000, 0.119262),
    ("H", 0.000000, 0.763239, -0.477047),
    ("H", 0.000000, -0.763239, -0.477047),
]


def make_mol(basis="6-31g*"):
    return gto.M(atom=WATER, basis=basis, verbose=0)


print("=" * 74)
print("A -- what does int1e_r actually return?")
print("=" * 74)
# A single hydrogen 1s at (0,0,1.0) bohr: <1s|z|1s> must be 1.0
probe = gto.M(atom=[("He", 0.0, 0.0, 1.0)], basis="sto-3g", verbose=0)
r1 = probe.intor("int1e_r")
print(f"  int1e_r shape {r1.shape}  (expect (3, nao, nao))")
z = float(r1[2][0, 0])
print(f"  <1s|z|1s> for He at z = 1.0 bohr: {z:.6f}   (expect 1.000000)")
print(f"  <1s|x|1s>: {float(r1[0][0, 0]):.6f}   (expect 0)")
print(f"  <1s|y|1s>: {float(r1[1][0, 0]):.6f}   (expect 0)")

print()
print("=" * 74)
print("B -- does the operator reproduce mf.dip_moment()?")
print("=" * 74)
mol = make_mol()
mf = dft.RKS(mol)
mf.xc = "b3lyp"
mf.verbose = 0
mf.conv_tol = 1e-11
mol.chkfile = None
mf.chkfile = None
mf.kernel()
dm = mf.make_rdm1()
dip_ao = mol.intor("int1e_r")
elec = -np.einsum("xij,ji->x", dip_ao, dm)
nuc = np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords())
mine = elec + nuc
theirs = mf.dip_moment(unit="AU", verbose=0)
print(f"  from int1e_r : {np.array2string(mine, precision=6)}")
print(f"  mf.dip_moment: {np.array2string(theirs, precision=6)}")
print(f"  max |diff|   : {np.abs(mine - theirs).max():.2e}")

print()
print("=" * 74)
print("C -- is dE/dE_field equal to -mu?")
print("=" * 74)
h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
base = mf.e_tot


def energy(field):
    m2 = gto.M(atom=WATER, basis="6-31g*", verbose=0)
    m2.chkfile = None
    k = dft.RKS(m2)
    k.xc = "b3lyp"
    k.verbose = 0
    k.conv_tol = 1e-11
    k.chkfile = None
    hf_ = h0 + np.einsum("x,xij->ij", np.asarray(field, float), dip_ao)
    k.get_hcore = lambda *a, _h=hf_: _h
    k.kernel()
    return k.e_tot


for h in (1e-3, 1e-4):
    for ax in (0, 1, 2):
        f = np.zeros(3)
        f[ax] = h
        ep = energy(f)
        f[ax] = -h
        em = energy(f)
        d = (ep - em) / (2 * h)
        print(f"  h={h:7.1e} axis {ax}: dE/dE = {d:+12.7f}   "
              f"-mu = {-mine[ax]:+12.7f}   diff {d + mine[ax]:+.2e}")

print()
print("=" * 74)
print("D -- the dipole in a field: does mu change by alpha*E?")
print("=" * 74)
# mu_tot = -<r> + sum Z R; the SCF polarises, so <r> moves with the field.
for h in (1e-3, 1e-2):
    f = np.zeros(3)
    f[2] = h
    m2 = gto.M(atom=WATER, basis="6-31g*", verbose=0)
    m2.chkfile = None
    k = dft.RKS(m2)
    k.xc = "b3lyp"
    k.verbose = 0
    k.conv_tol = 1e-11
    k.chkfile = None
    hf_ = h0 + np.einsum("x,xij->ij", f, dip_ao)
    k.get_hcore = lambda *a, _h=hf_: _h
    k.kernel()
    d2 = k.make_rdm1()
    mu = -np.einsum("xij,ji->x", dip_ao, d2) + nuc
    print(f"  E_z = {h:8.1e}: mu_z = {mu[2]:+.8f}   "
          f"delta mu_z = {mu[2] - mine[2]:+.3e}   "
          f"delta/E = {(mu[2] - mine[2]) / h:+.5f} a.u.")
print("  (a converged SCF must show delta mu / E = -alpha_zz, negative)")

print()
print("=" * 74)
print("E -- is the SCF even re-converging, or does it stay at the field-free dm?")
print("=" * 74)
f = np.zeros(3)
f[2] = 1e-2
m2 = gto.M(atom=WATER, basis="6-31g*", verbose=0)
m2.chkfile = None
k = dft.RKS(m2)
k.xc = "b3lyp"
k.verbose = 0
k.conv_tol = 1e-11
k.chkfile = None
hf_ = h0 + np.einsum("x,xij->ij", f, dip_ao)
k.get_hcore = lambda *a, _h=hf_: _h
k.kernel()
print(f"  e_tot in field  : {k.e_tot:.10f}")
print(f"  e_tot no field  : {base:.10f}")
print(f"  first-order term -mu.E = {-mine[2] * 1e-2:+.3e}")
print(f"  actual difference       {k.e_tot - base:+.3e}")
print(f"  second order from a_zz  {-0.5 * 5.32 * 1e-4:+.3e}")
