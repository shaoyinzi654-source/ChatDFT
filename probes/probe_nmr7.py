"""probe_nmr7.py -- the field-perturbed SCF must actually converge.

Diagnosis so far: results are dB-stable, the sign is confirmed on four nuclei,
but every field-perturbed run reports converged=False at conv_tol=1e-12, and
there is a 0.92 ppm asymmetry in the water 1H tensor where the mixed partial
derivative forces zero.  Transposing the second-order integral changes nothing,
so the asymmetry is in the field response D^B, i.e. in an unconverged SCF.

Test the obvious cure: a second-order (Newton) SCF, which converges
quadratically, and compare the tensor symmetry.
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
        h_bm[a] += mol.intor("int1e_a01p" if False else "int1e_a01gp").reshape(3, 3, nao, nao)


def make(bvec, dm0, newton):
    bvec = np.asarray(bvec, dtype=float)
    m = scf.RHF(mol)
    m.verbose = 0
    m.chkfile = None
    m.max_cycle = 300
    m.conv_tol = 1e-12
    m.diis_space = 12
    m.get_hcore = lambda *a: h0 - 1j * np.einsum("t,tuv->uv", bvec, h_b)
    m.get_ovlp = lambda *a: s0 - 1j * np.einsum("t,tuv->uv", bvec, s_b)
    m._eri = eri0 - 1j * np.einsum("t,tuvkl->uvkl", bvec, eri_b)
    if newton:
        m = m.newton()
        m.max_cycle = 60
        m.conv_tol = 1e-12
    m.kernel(dm0)
    return m


m0 = make([0.0] * 3, None, False)
dm0 = m0.make_rdm1()
print(f"unperturbed: converged={m0.converged}  E={m0.e_tot.real:.12f}")
print()

print("=" * 74)
print("  convergence of the field-perturbed SCF")
print("=" * 74)
dB = 3e-4
for newton in (False, True):
    for t in range(3):
        bp = [0.0, 0.0, 0.0]; bp[t] = dB
        bm = [0.0, 0.0, 0.0]; bm[t] = -dB
        mp_ = make(bp, dm0, newton)
        mm_ = make(bm, dm0, newton)
        lin = (mp_.e_tot.real - mm_.e_tot.real) / (2 * dB)
        print(f"  newton={str(newton):5s} B_{'xyz'[t]}: "
              f"conv {mp_.converged}/{mm_.converged}  "
              f"cycles {getattr(mp_, 'cycles', -1)}/{getattr(mm_, 'cycles', -1)}  "
              f"dE/dB = {lin:+.3e}  (must be 0)")
print()

print("=" * 74)
print("  shieldings, both solvers")
print("=" * 74)
for newton in (False, True):
    sig = np.zeros((natm, 3, 3))
    for t in range(3):
        bp = [0.0, 0.0, 0.0]; bp[t] = dB
        bm = [0.0, 0.0, 0.0]; bm[t] = -dB
        dp = make(bp, dm0, newton).make_rdm1()
        dm_ = make(bm, dm0, newton).make_rdm1()
        for a in range(natm):
            for s in range(3):
                hp = h_m[a][s] + sum(bp[k] * h_bm[a][k][s] for k in range(3))
                hm = h_m[a][s] + sum(bm[k] * h_bm[a][k][s] for k in range(3))
                gp = np.einsum("uv,vu->", dp, hp).real
                gm = np.einsum("uv,vu->", dm_, hm).real
                sig[a, t, s] = (gp - gm) / (2 * dB)
    print(f"  --- newton={newton} ---")
    for a in range(natm):
        s = sig[a]
        print(f"    {mol.atom_symbol(a)}{a}: iso = {-np.trace(s) / 3 * PPM:+10.4f}  "
              f"max|asym| = {np.max(np.abs(s - s.T)) * PPM:9.5f} ppm   "
              f"whole {np.linalg.norm(s - s.T) * PPM:.5f}")
    print(f"    tensor H1 (ppm):\n{np.array2string(sig[1] * PPM, prefix='      ')}")

print()
print("=" * 74)
print("  if newton converges, rescan dB to confirm the asymmetry is gone")
print("=" * 74)
for dB2 in (1e-3, 3e-4, 1e-4, 3e-5):
    sig = np.zeros((natm, 3, 3))
    for t in range(3):
        bp = [0.0, 0.0, 0.0]; bp[t] = dB2
        bm = [0.0, 0.0, 0.0]; bm[t] = -dB2
        dp = make(bp, dm0, True).make_rdm1()
        dm_ = make(bm, dm0, True).make_rdm1()
        for a in range(natm):
            for s in range(3):
                hp = h_m[a][s] + sum(bp[k] * h_bm[a][k][s] for k in range(3))
                hm = h_m[a][s] + sum(bm[k] * h_bm[a][k][s] for k in range(3))
                sig[a, t, s] = (np.einsum("uv,vu->", dp, hp).real
                                - np.einsum("uv,vu->", dm_, hm).real) / (2 * dB2)
    print(f"    dB={dB2:8.1e}  1H iso = {-np.trace(sig[1]) / 3 * PPM:+10.5f}  "
          f"17O iso = {-np.trace(sig[0]) / 3 * PPM:+10.4f}  "
          f"max|asym| = {max(np.max(np.abs(s - s.T)) for s in sig) * PPM:9.5f} ppm")
