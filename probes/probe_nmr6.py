"""probe_nmr6.py -- GIAO shielding with the SCF actually converged.

probe_nmr5 showed the field-perturbed SCF was hitting max_cycle with
conv_tol=1e-13, so probe_nmr3's 1.7e-2 tensor asymmetry was computed from
unconverged densities.  Redo it properly:

  * conv_tol 1e-12, and *report* the convergence instead of assuming it
  * the unperturbed density as the initial guess for both +/- runs
  * a step-size scan, because the whole method is a finite difference
  * two molecules, because a sign determined from one number is a fit

Established conventions (measured, see probe_nmr4/5):
  int1e_pnucxp  = -sum_A Z_A int1e_ia01p(origin R_A)      ratio -1.000000
  int1e_igovlp  = -1/2 eps_tab (R_u-R_v)_a <u|r_b|v>       residual 8.3e-17
  => dS/dB_t = -1j * int1e_igovlp[t]     (the GIAO phase is exp(-i/2 (B x R).r))
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=170)
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6

H2O = gto.M(atom=[("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587),
                  ("H", 0.0, 0.757, 0.587)],
            basis="sto-3g", unit="Angstrom", verbose=0)
# NH3, C3v: N-H 1.012 A, H-N-H 106.7 deg -> polar angle 67.89 deg from +z
NH3 = gto.M(atom=[("N", 0.0, 0.0, 0.0)] + [
    ("H", 0.93748 * np.cos(2 * np.pi * k / 3),
     0.93748 * np.sin(2 * np.pi * k / 3), 0.38101) for k in range(3)],
    basis="sto-3g", unit="Angstrom", verbose=0)


def build(mol):
    nao, natm = mol.nao_nr(), mol.natm
    d = {}
    d["h0"] = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    d["s0"] = mol.intor("int1e_ovlp")
    d["eri0"] = mol.intor("int2e")
    d["h_b"] = -1j * (0.5 * mol.intor("int1e_giao_irjxp")
                      + mol.intor("int1e_ignuc") + mol.intor("int1e_igkin"))
    d["s_b"] = -1j * mol.intor("int1e_igovlp")
    ig1 = mol.intor("int2e_ig1")
    d["eri_b"] = -1j * (ig1 + ig1.transpose(0, 3, 4, 1, 2))
    d["h_m"] = np.zeros((natm, 3, nao, nao), dtype=complex)
    d["h_bm"] = np.zeros((natm, 3, 3, nao, nao))
    R = mol.atom_coords()
    for a in range(natm):
        with mol.with_rinv_orig(R[a]):
            d["h_m"][a] = -1j * mol.intor("int1e_ia01p")
            aa = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
            d["h_bm"][a] = aa - np.einsum("ts,uv->tsuv", np.eye(3),
                                          aa.trace(axis1=0, axis2=1))
            d["h_bm"][a] += mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)
    return d


def scf_at(mol, D, bvec, dm0):
    bvec = np.asarray(bvec, dtype=float)
    m = scf.RHF(mol)
    m.verbose = 0
    m.chkfile = None
    m.max_cycle = 200
    m.conv_tol = 1e-12
    m.get_hcore = lambda *a: D["h0"] - 1j * np.einsum("t,tuv->uv", bvec, D["h_b"])
    m.get_ovlp = lambda *a: D["s0"] - 1j * np.einsum("t,tuv->uv", bvec, D["s_b"])
    m._eri = D["eri0"] - 1j * np.einsum("t,tuvkl->uvkl", bvec, D["eri_b"])
    m.kernel(dm0)
    return m


def shieldings(mol, D, dB):
    nao, natm = mol.nao_nr(), mol.natm
    m0 = scf_at(mol, D, [0.0] * 3, None)
    dm0 = m0.make_rdm1()
    assert m0.converged, "the unperturbed SCF did not converge"
    sig = np.zeros((natm, 3, 3))
    diag = []
    for t in range(3):
        bp = [0.0, 0.0, 0.0]; bp[t] = dB
        bm = [0.0, 0.0, 0.0]; bm[t] = -dB
        mp_, mm_ = scf_at(mol, D, bp, dm0), scf_at(mol, D, bm, dm0)
        dp, dm_ = mp_.make_rdm1(), mm_.make_rdm1()
        # E(B) must be even; report both the library energy and a hand-built
        # 1/2 Tr(D(h+F)), since e_tot on a complex Hamiltonian is worth checking
        ep = 0.5 * np.einsum("uv,vu->", dp, mp_.get_hcore() + mp_.get_fock(dp)).real
        em = 0.5 * np.einsum("uv,vu->", dm_, mm_.get_hcore() + mm_.get_fock(dm_)).real
        diag.append((mp_.converged, mm_.converged, mp_.e_tot.real - mm_.e_tot.real,
                     ep - em))
        for a in range(natm):
            for s in range(3):
                hp = D["h_m"][a][s] + sum(bp[k] * D["h_bm"][a][k][s] for k in range(3))
                hm = D["h_m"][a][s] + sum(bm[k] * D["h_bm"][a][k][s] for k in range(3))
                gp = np.einsum("uv,vu->", dp, hp).real
                gm = np.einsum("uv,vu->", dm_, hm).real
                sig[a, t, s] = (gp - gm) / (2 * dB)
    return sig, m0, diag


for tag, m in (("H2O", H2O), ("NH3", NH3)):
    D = build(m)
    print("=" * 74)
    print(f"  {tag}")
    print("=" * 74)
    for dB in (1e-3, 3e-4, 1e-4):
        sig, m0, diag = shieldings(m, D, dB)
        print(f"  dB = {dB:8.1e}   converged = "
              f"{[f'{a}/{b}' for a, b, _, _ in diag]}")
        print(f"                E(+)-E(-) = {[f'{c:+.2e}' for _, _, c, _ in diag]}"
              f"   hand-built = {[f'{d:+.2e}' for _, _, _, d in diag]}")
        for a in range(m.natm):
            s = sig[a]
            iso = -np.trace(s) / 3 * PPM
            asym = np.max(np.abs(s - s.T)) * PPM
            eigs = np.linalg.eigvalsh((s + s.T) / 2) * PPM
            aniso = -(eigs[2] - 0.5 * (eigs[0] + eigs[1]))
            print(f"      {m.atom_symbol(a)}{a}: iso = {iso:+9.4f}   "
                  f"aniso = {aniso:+9.4f}   max|asym| = {asym:8.4f} ppm")
        print()

print("=" * 74)
print("  literature, gas phase, absolute shielding (ppm)")
print("    H2O  1H  30.7        17O  344")
print("    NH3  1H  30.7-31.0   14N  264     RHF/STO-3G 14N ~ 292")
print("=" * 74)
