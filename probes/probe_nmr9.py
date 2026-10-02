"""probe_nmr9.py -- is the 0.92 ppm tensor asymmetry a basis-set artifact?

The 2-point route reproduces the literature 1H shielding of water (30.675 vs
30.7) and of ammonia (30.64 vs 30.7-31.0), and its tensor diagonal agrees with
the 4-point stencil to 0.002 ppm.  What is left is a 0.92 ppm asymmetry on the
hydrogen atoms, where the mixed partial derivative forces exact zero, and a
spurious linear term dE/dB = -8.7e-2 along B_x that the overlap perturbation
alone produces.

A field-dependent *basis* is an approximation, so a residual gauge artifact is
plausible.  If so it must shrink as the basis improves.  Measure it across five
bases.  Either way this fixes the recommended basis for the product.
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=170)
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6

BASES = ["sto-3g", "3-21g", "6-31g*", "cc-pvdz", "aug-cc-pvdz"]


def run(basis):
    mol = gto.M(atom=[("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587),
                      ("H", 0.0, 0.757, 0.587)],
                basis=basis, unit="Angstrom", verbose=0)
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

    def scf_at(bvec, dm0):
        bvec = np.asarray(bvec, dtype=float)
        m = scf.RHF(mol)
        m.verbose = 0
        m.chkfile = None
        m.max_cycle = 300
        m.conv_tol = 1e-11
        m.get_hcore = lambda *a: h0 - 1j * np.einsum("t,tuv->uv", bvec, h_b)
        m.get_ovlp = lambda *a: s0 - 1j * np.einsum("t,tuv->uv", bvec, s_b)
        m._eri = eri0 - 1j * np.einsum("t,tuvkl->uvkl", bvec, eri_b)
        m.kernel(dm0)
        return m

    m0 = scf_at([0.0] * 3, None)
    dm0 = m0.make_rdm1()
    dB = 3e-4
    sig = np.zeros((natm, 3, 3))
    lin = []
    for t in range(3):
        bp = [0.0, 0.0, 0.0]; bp[t] = dB
        bm = [0.0, 0.0, 0.0]; bm[t] = -dB
        mp_ = scf_at(bp, dm0)
        mm_ = scf_at(bm, dm0)
        lin.append((mp_.e_tot.real - mm_.e_tot.real) / (2 * dB))
        dp, dm_ = mp_.make_rdm1(), mm_.make_rdm1()
        for a in range(natm):
            for s in range(3):
                hp = h_m[a][s] + sum(bp[k] * h_bm[a][k][s] for k in range(3))
                hm = h_m[a][s] + sum(bm[k] * h_bm[a][k][s] for k in range(3))
                sig[a, t, s] = (np.einsum("uv,vu->", dp, hp).real
                                - np.einsum("uv,vu->", dm_, hm).real) / (2 * dB)
    return mol, sig, lin, m0


print(f"{'basis':12s} {'nao':>4s} {'1H iso':>9s} {'17O iso':>9s} "
      f"{'H asym':>8s} {'O asym':>8s} {'dE/dB(x)':>11s}")
print("-" * 70)
store = {}
for basis in BASES:
    mol, sig, lin, m0 = run(basis)
    h1 = -np.trace(sig[1]) / 3 * PPM
    o17 = -np.trace(sig[0]) / 3 * PPM
    ha = np.max(np.abs(sig[1] - sig[1].T)) * PPM
    oa = np.max(np.abs(sig[0] - sig[0].T)) * PPM
    store[basis] = (h1, o17, ha)
    print(f"{basis:12s} {mol.nao_nr():4d} {h1:9.4f} {o17:9.4f} "
          f"{ha:8.4f} {oa:8.4f} {lin[0]:+11.3e}")

print()
print("=" * 74)
print("  literature:  1H 30.7   17O 344  (gas phase, experiment)")
print("=" * 74)
for b, (h, o, a) in store.items():
    print(f"  {b:12s}  1H {h - 30.7:+7.3f}   17O {o - 344:+7.1f}   "
          f"asym {a:.4f}")
