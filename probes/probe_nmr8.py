"""probe_nmr8.py -- the 4-point stencil, which is the recipe that was validated
against Gaussian.

probe_nmr7 isolated the trouble: perturbing the overlap alone produces a
spurious linear term dE/dB = -8.72e-2 for B_x (and only B_x), and the resulting
1H tensor is asymmetric by 0.92 ppm where a mixed partial derivative forces
zero.

The documented recipe does not use a 2-point field derivative of a
Hellmann-Feynman expectation value; it uses the 4-point stencil

    sigma_ts = [ E(+B,+mu) - E(-B,+mu) - E(+B,-mu) + E(-B,-mu) ] / (4 dB dmu)

which cancels every term linear in B or linear in mu.  Run it and compare with
the 2-point answer.  Water/STO-3G: 3 directions x 3 atoms x 3 components x 4
SCF runs = 108 runs, a few seconds.
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=170)
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6

mol = gto.M(atom=[("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587),
                  ("H", 0.0, 0.757, 0.587)],
            basis="sto-3g", unit="Angstrom", verbose=0)
nao, natm = mol.nao_nr(), mol.natm
R = mol.atom_coords()

h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
s0 = mol.intor("int1e_ovlp")
eri0 = mol.intor("int2e")
h_b = -1j * (0.5 * mol.intor("int1e_giao_irjxp")
             + mol.intor("int1e_ignuc") + mol.intor("int1e_igkin"))
s_b = -1j * mol.intor("int1e_igovlp")
ig1 = mol.intor("int2e_ig1")
eri_b = -1j * (ig1 + ig1.transpose(0, 3, 4, 1, 2))

h_m = np.zeros((natm, 3, nao, nao), dtype=complex)
h_bm = np.zeros((natm, 3, 3, nao, nao))
for a in range(natm):
    with mol.with_rinv_orig(R[a]):
        h_m[a] = -1j * mol.intor("int1e_ia01p")
        aa = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
        h_bm[a] = aa - np.einsum("ts,uv->tsuv", np.eye(3),
                                 aa.trace(axis1=0, axis2=1))
        h_bm[a] += mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)

NSCF = [0]


def energy(bvec, muop, dm0):
    """Energy with the core Hamiltonian perturbed by bvec (field) and muop
    (a nuclear magnetic moment operator, already including its second order
    piece if the caller built it that way)."""
    NSCF[0] += 1
    bvec = np.asarray(bvec, dtype=float)
    m = scf.RHF(mol)
    m.verbose = 0
    m.chkfile = None
    m.max_cycle = 200
    m.conv_tol = 1e-12
    h1 = h0 - 1j * np.einsum("t,tuv->uv", bvec, h_b) + muop
    m.get_hcore = lambda *a: h1
    m.get_ovlp = lambda *a: s0 - 1j * np.einsum("t,tuv->uv", bvec, s_b)
    m._eri = eri0 - 1j * np.einsum("t,tuvkl->uvkl", bvec, eri_b)
    m.kernel(dm0)
    return m


m0 = energy([0.0] * 3, np.zeros((nao, nao), dtype=complex), None)
dm0 = m0.make_rdm1()
print(f"E0 = {m0.e_tot.real:.12f}   nao={nao} natm={natm}")

dB = 3e-4
dmu = 1e-4

print()
print("=" * 74)
print("  4-point stencil over (B, mu)")
print("=" * 74)
sigma = np.zeros((natm, 3, 3))
for t in range(3):
    for a in range(natm):
        for s in range(3):
            acc = 0.0
            for sb in (+1, -1):
                bv = [0.0, 0.0, 0.0]
                bv[t] = sb * dB
                for sm in (+1, -1):
                    # the mu operator at this field: h^mu + sum_t B_t Q_ts
                    muop = sm * dmu * (h_m[a][s]
                                       + sum(bv[k] * h_bm[a][k][s] for k in range(3)))
                    e = energy(bv, muop, dm0).e_tot.real
                    acc += sb * sm * e
            sigma[a, t, s] = acc / (4 * dB * dmu)

print(f"  SCF runs used: {NSCF[0]}")
for a in range(natm):
    sg = sigma[a]
    print(f"  {mol.atom_symbol(a)}{a}: iso = {-np.trace(sg) / 3 * PPM:+10.4f}   "
          f"max|asym| = {np.max(np.abs(sg - sg.T)) * PPM:9.5f} ppm   "
          f"whole {np.linalg.norm(sg - sg.T) * PPM:.5f}")
print(f"  H1 tensor (ppm):\n{np.array2string(sigma[1] * PPM, prefix='    ')}")

print()
print("=" * 74)
print("  compare: 2-point field derivative of Tr(D h^mu)")
print("=" * 74)
sig2 = np.zeros((natm, 3, 3))
for t in range(3):
    bp = [0.0, 0.0, 0.0]; bp[t] = dB
    bm = [0.0, 0.0, 0.0]; bm[t] = -dB
    mp_ = energy(bp, np.zeros((nao, nao), dtype=complex), dm0)
    mm_ = energy(bm, np.zeros((nao, nao), dtype=complex), dm0)
    dp, dm_ = mp_.make_rdm1(), mm_.make_rdm1()
    for a in range(natm):
        for s in range(3):
            hp = h_m[a][s] + sum(bp[k] * h_bm[a][k][s] for k in range(3))
            hm = h_m[a][s] + sum(bm[k] * h_bm[a][k][s] for k in range(3))
            sig2[a, t, s] = (np.einsum("uv,vu->", dp, hp).real
                             - np.einsum("uv,vu->", dm_, hm).real) / (2 * dB)
print(f"  SCF runs used (total): {NSCF[0]}")
for a in range(natm):
    sg = sig2[a]
    print(f"  {mol.atom_symbol(a)}{a}: iso = {-np.trace(sg) / 3 * PPM:+10.4f}   "
          f"max|asym| = {np.max(np.abs(sg - sg.T)) * PPM:9.5f} ppm   "
          f"whole {np.linalg.norm(sg - sg.T) * PPM:.5f}")

print()
print("=" * 74)
print("  do the two routes agree?")
print("=" * 74)
for a in range(natm):
    d = np.max(np.abs(sigma[a] - sig2[a])) * PPM
    print(f"  {mol.atom_symbol(a)}{a}: max|4-point - 2-point| = {d:.5f} ppm")
