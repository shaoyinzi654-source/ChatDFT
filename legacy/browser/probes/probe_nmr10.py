"""probe_nmr10.py -- score the eight sign combinations of the GIAO perturbation.

Everything is now measured except one thing: the relative sign of the three
first-order GIAO integrals.  probe_nmr5 pinned the overlap exactly
(int1e_igovlp = -1/2 eps_tab (R_u-R_v)_a <u|r_b|v>, residual 8.3e-17), which
fixes dS/dB = -i int1e_igovlp.  For the ERI there is no equally cheap analytic
identity, and libcint is free to use a different sign for a different function.

A wrong sign is not a small error: it makes the perturbed Hamiltonian
inconsistent, which is exactly what shows up as a spurious dE/dB and a tensor
asymmetry that the mixed partial derivative forbids.

So do what worked for the CPHF prefactor in round 15: enumerate the candidates
and score them against two independent requirements.

    requirement 1:  the shielding tensor must be symmetric
                    (d2E/dB_t dmu_s commutes with d2E/dB_s dmu_t)
    requirement 2:  the isotropic 1H shielding of water must be near +30.7 ppm
                    and of ammonia near +30.7..31.0 ppm

Only one combination can satisfy both.
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=170)
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6

WATER = gto.M(atom=[("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587),
                    ("H", 0.0, 0.757, 0.587)],
              basis="sto-3g", unit="Angstrom", verbose=0)
NH3 = gto.M(atom=[("N", 0.0, 0.0, 0.0)] + [
    ("H", 0.93748 * np.cos(2 * np.pi * k / 3),
     0.93748 * np.sin(2 * np.pi * k / 3), 0.38101) for k in range(3)],
    basis="sto-3g", unit="Angstrom", verbose=0)


def build(mol, sh, ss, se):
    nao, natm = mol.nao_nr(), mol.natm
    R = mol.atom_coords()
    d = {"h0": mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
         "s0": mol.intor("int1e_ovlp"), "eri0": mol.intor("int2e")}
    hb = 0.5 * mol.intor("int1e_giao_irjxp") + mol.intor("int1e_ignuc") + mol.intor("int1e_igkin")
    d["h_b"] = sh * (-1j) * hb
    d["s_b"] = ss * (-1j) * mol.intor("int1e_igovlp")
    ig1 = mol.intor("int2e_ig1")
    d["eri_b"] = se * (-1j) * (ig1 + ig1.transpose(0, 3, 4, 1, 2))
    d["h_m"] = np.zeros((natm, 3, nao, nao), dtype=complex)
    d["h_bm"] = np.zeros((natm, 3, 3, nao, nao))
    for a in range(natm):
        with mol.with_rinv_orig(R[a]):
            d["h_m"][a] = -1j * mol.intor("int1e_ia01p")
            aa = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
            d["h_bm"][a] = aa - np.einsum("ts,uv->tsuv", np.eye(3),
                                          aa.trace(axis1=0, axis2=1))
            d["h_bm"][a] += mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)
    return d


def shieldings(mol, D, dB=3e-4):
    nao, natm = mol.nao_nr(), mol.natm

    def scf_at(bvec, dm0):
        bvec = np.asarray(bvec, dtype=float)
        m = scf.RHF(mol)
        m.verbose = 0
        m.chkfile = None
        m.max_cycle = 300
        m.conv_tol = 1e-11
        m.get_hcore = lambda *a: D["h0"] - 1j * np.einsum("t,tuv->uv", bvec, D["h_b"] * 1j)
        m.get_ovlp = lambda *a: D["s0"] - 1j * np.einsum("t,tuv->uv", bvec, D["s_b"] * 1j)
        m._eri = D["eri0"] - 1j * np.einsum("t,tuvkl->uvkl", bvec, D["eri_b"] * 1j)
        m.kernel(dm0)
        return m

    m0 = scf_at([0.0] * 3, None)
    dm0 = m0.make_rdm1()
    sig = np.zeros((natm, 3, 3))
    lin = []
    for t in range(3):
        bp = [0.0, 0.0, 0.0]; bp[t] = dB
        bm = [0.0, 0.0, 0.0]; bm[t] = -dB
        mp_, mm_ = scf_at(bp, dm0), scf_at(bm, dm0)
        lin.append((mp_.e_tot.real - mm_.e_tot.real) / (2 * dB))
        dp, dm_ = mp_.make_rdm1(), mm_.make_rdm1()
        for a in range(natm):
            for s in range(3):
                hp = D["h_m"][a][s] + sum(bp[k] * D["h_bm"][a][k][s] for k in range(3))
                hm = D["h_m"][a][s] + sum(bm[k] * D["h_bm"][a][k][s] for k in range(3))
                sig[a, t, s] = (np.einsum("uv,vu->", dp, hp).real
                                - np.einsum("uv,vu->", dm_, hm).real) / (2 * dB)
    return sig, lin


print("=" * 100)
print("  scoring the eight sign combinations of (h_b, s_b, eri_b)")
print("  a correct set must give a SYMMETRIC tensor and the literature 1H shift")
print("=" * 100)
print(f"{'sh':>3s} {'ss':>3s} {'se':>3s} | {'H2O 1H':>9s} {'H2O 17O':>9s} {'H asym':>8s} "
      f"{'|dE/dB|max':>11s} | {'NH3 1H':>9s} {'NH3 14N':>9s} {'N asym':>8s} | verdict")
print("-" * 100)
rows = []
for sh in (+1, -1):
    for ss in (+1, -1):
        for se in (+1, -1):
            sw, linw = shieldings(WATER, build(WATER, sh, ss, se))
            sn, linn = shieldings(NH3, build(NH3, sh, ss, se))
            h1 = -np.trace(sw[1]) / 3 * PPM
            o17 = -np.trace(sw[0]) / 3 * PPM
            ha = np.max(np.abs(sw[1] - sw[1].T)) * PPM
            n1 = -np.trace(sn[1]) / 3 * PPM
            n14 = -np.trace(sn[0]) / 3 * PPM
            na = np.max(np.abs(sn[1] - sn[1].T)) * PPM
            lmax = max(abs(x) for x in linw + linn)
            ok = (ha < 0.05 and na < 0.05 and abs(h1 - 30.7) < 2.0
                  and abs(n1 - 30.8) < 2.0)
            rows.append((sh, ss, se, h1, ha, lmax))
            print(f"{sh:+3d} {ss:+3d} {se:+3d} | {h1:9.3f} {o17:9.2f} {ha:8.4f} "
                  f"{lmax:11.3e} | {n1:9.3f} {n14:9.2f} {na:8.4f} | "
                  f"{'*** MATCH ***' if ok else ''}")

print()
print("=" * 100)
print("  ranking by (asymmetry, then |dE/dB|) -- the physics must be both")
print("  symmetric and free of a spurious linear Zeeman term")
print("=" * 100)
for sh, ss, se, h1, ha, lmax in sorted(rows, key=lambda r: (round(r[4], 3), r[5])):
    print(f"  ({sh:+d},{ss:+d},{se:+d})   H asym = {ha:8.4f}   "
          f"|dE/dB|max = {lmax:.3e}   1H = {h1:8.3f}")
