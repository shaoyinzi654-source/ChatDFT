"""probe_nmr16.py -- is the residual H-atom asymmetry numerical or physical?

probe_nmr15 settled the origin question: `int1e_ia01p`, `int1e_giao_a11part`
and `int1e_a01gp` are ALL set by `with_rinv_orig` (|A-B| = 1.27e+1, 4.49e-1,
4.40e-1 under rinv_orig; exactly 0.0 under with_common_origin, which this
libcint build does not consult for them).  Building Q with the common origin
destroys the shieldings -- H2O 1H goes 31.85 -> 154.63 -- so the product's
`with_rinv_orig` is right and the alternative is not a candidate.

That leaves the asymmetry itself.  sigma_ts = d2E/dB_t dmu_s is symmetric by
construction, so a 1.6 ppm asymmetry is either

  (a) numerical: the field-perturbed SCF's imaginary part is not converged
      tightly enough, or the step dB is badly chosen; or
  (b) a wrong operator.

(a) is testable and cheap: sweep dB over five decades and conv_tol over three,
and watch whether the asymmetry moves.  A physical error is invariant under
both; a numerical one is not.

This probe also answers the Pulay question that probe_nmr15 raised.  The GIAO
basis functions depend on B, so dE/dB is NOT simply Tr(D dh/dB) -- there is an
energy-weighted-density term from the normalisation constraint Tr(D S(B)) = N.
Route B in probe_nmr15 omitted it, which is why route B disagreed so badly.
Measuring Tr(D (-i O^t)) against the exact zero tells us how large that term
has to be.

Run:  python probes/probe_nmr16.py
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=200)
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6

WATER = [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.7572, 0.5857),
         ("H", 0.0, -0.7572, 0.5857)]
AMMONIA = [("N", 0.0, 0.0, 0.0), ("H", 0.0, 0.9391, -0.3816),
           ("H", 0.8133, -0.4696, -0.3816), ("H", -0.8133, -0.4696, -0.3816)]
METHANE = [("C", 0.0, 0.0, 0.0),
           ("H", 0.6291, 0.6291, 0.6291), ("H", -0.6291, -0.6291, 0.6291),
           ("H", -0.6291, 0.6291, -0.6291), ("H", 0.6291, -0.6291, -0.6291)]


def build(mol):
    nao, natm = mol.nao_nr(), mol.natm
    R = mol.atom_coords()
    d = {"h0": mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
         "s0": mol.intor("int1e_ovlp"), "eri0": mol.intor("int2e"),
         "natm": natm, "nao": nao}
    d["O_h"] = (0.5 * mol.intor("int1e_giao_irjxp")
                + mol.intor("int1e_ignuc") + mol.intor("int1e_igkin"))
    d["O_S"] = mol.intor("int1e_igovlp")
    ig1 = mol.intor("int2e_ig1")
    d["O_G"] = ig1 + ig1.transpose(0, 3, 4, 1, 2)
    d["P"] = np.zeros((natm, 3, nao, nao))
    d["Q"] = np.zeros((natm, 3, 3, nao, nao))
    for a in range(natm):
        with mol.with_rinv_orig(R[a]):
            d["P"][a] = mol.intor("int1e_ia01p")
            aa = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
            d["Q"][a] = aa - np.einsum("ts,uv->tsuv", np.eye(3),
                                       aa.trace(axis1=0, axis2=1))
            d["Q"][a] += mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)
    return d


def scf_at(mol, D, bvec, dm0, conv_tol):
    bvec = np.asarray(bvec, dtype=float)
    m = scf.RHF(mol)
    m.verbose = 0
    m.chkfile = None
    m.max_cycle = 300
    m.conv_tol = conv_tol
    m.get_hcore = lambda *a: D["h0"] - 1j * np.einsum("t,tuv->uv", bvec, D["O_h"])
    m.get_ovlp = lambda *a: D["s0"] - 1j * np.einsum("t,tuv->uv", bvec, D["O_S"])
    m._eri = D["eri0"] - 1j * np.einsum("t,tuvkl->uvkl", bvec, D["O_G"])
    m.kernel(dm0)
    return m


def shieldings(mol, D, dB, conv_tol, dm0):
    sig = np.zeros((D["natm"], 3, 3))
    lin = []
    for t in range(3):
        bp = [0.0] * 3; bp[t] = dB
        bm = [0.0] * 3; bm[t] = -dB
        mp_, mm_ = scf_at(mol, D, bp, dm0, conv_tol), scf_at(mol, D, bm, dm0, conv_tol)
        lin.append((mp_.e_tot.real - mm_.e_tot.real) / (2 * dB))
        dp, dm_ = mp_.make_rdm1(), mm_.make_rdm1()
        for a in range(D["natm"]):
            for s in range(3):
                hp = -1j * D["P"][a][s] + sum(bp[k] * D["Q"][a][k][s]
                                              for k in range(3))
                hm = -1j * D["P"][a][s] + sum(bm[k] * D["Q"][a][k][s]
                                              for k in range(3))
                sig[a, t, s] = (np.einsum("uv,vu->", dp, hp).real
                                - np.einsum("uv,vu->", dm_, hm).real) / (2 * dB)
    return sig, lin


def main():
    # ---- 0. geometry sanity: what unit did gto.M actually take? ----
    mol = gto.M(atom=WATER, basis="6-31g*", verbose=0)
    R = mol.atom_coords() * 0.52917721092
    print("=" * 78)
    print(" 0. geometry / unit check")
    print("=" * 78)
    oh = np.linalg.norm(R[1] - R[0])
    u1, u2 = R[1] - R[0], R[2] - R[0]
    ang = np.degrees(np.arccos(np.dot(u1, u2)
                               / (np.linalg.norm(u1) * np.linalg.norm(u2))))
    print(f"   O-H = {oh:.4f} A  (expect 0.9572)")
    print(f"   H-O-H = {ang:.2f} deg  (expect 104.52)")

    for name, atoms, nuc in (("H2O", WATER, [(0, "O"), (1, "H")]),
                             ("NH3", AMMONIA, [(0, "N"), (1, "H")]),
                             ("CH4", METHANE, [(0, "C"), (1, "H")])):
        mol = gto.M(atom=atoms, basis="6-31g*", verbose=0)
        D = build(mol)
        m0 = scf_at(mol, D, [0.0] * 3, None, 1e-11)
        dm0 = m0.make_rdm1()
        D0 = m0.make_rdm1()

        print()
        print("=" * 78)
        print(f" {name}   natm={mol.natm}  nao={mol.nao_nr()}")
        print("=" * 78)

        # ---- 1. the Pulay question ----
        print("\n 1. is Tr(D (-i O^t)) zero?  (it must be: no linear Zeeman term)")
        for t in range(3):
            v = np.einsum("uv,vu->", D0, -1j * D["O_h"][t]).real
            print(f"      t={t}  Tr(D (-i O^t)) = {v: .10e}")

        # ---- 2. dB sweep ----
        print("\n 2. dB sweep at conv_tol = 1e-11")
        print(f"    {'dB':>9s}  {'iso(O/C/N)':>11s} {'iso(H)':>10s} "
              f"{'asym(O/C/N)':>12s} {'asym(H)':>10s}")
        for dB in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6):
            sig, lin = shieldings(mol, D, dB, 1e-11, dm0)
            a0, a1 = nuc[0][0], nuc[1][0]
            iso0 = np.trace(sig[a0]) / 3 * PPM
            iso1 = np.trace(sig[a1]) / 3 * PPM
            as0 = np.abs(sig[a0] - sig[a0].T).max() * PPM
            as1 = np.abs(sig[a1] - sig[a1].T).max() * PPM
            print(f"    {dB:9.0e}  {iso0:11.4f} {iso1:10.4f} "
                  f"{as0:12.6f} {as1:10.6f}")

        # ---- 3. conv_tol sweep ----
        print("\n 3. conv_tol sweep at dB = 1e-4")
        print(f"    {'conv_tol':>9s}  {'iso(H)':>10s} {'asym(H)':>10s} "
              f"{'asym diag':>10s} {'asym offdiag':>13s}")
        for tol in (1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13):
            sig, lin = shieldings(mol, D, 1e-4, tol, dm0)
            a1 = nuc[1][0]
            m = sig[a1] * PPM
            asy = m - m.T
            print(f"    {tol:9.0e}  {np.trace(m) / 3:10.4f} "
                  f"{np.abs(asy).max():10.6f} {np.abs(np.diag(asy)).max():10.6f} "
                  f"{np.abs(asy - np.diag(np.diag(asy))).max():13.6f}")

        # ---- 4. symmetrised vs raw ----
        sig, lin = shieldings(mol, D, 1e-4, 1e-11, dm0)
        print("\n 4. does symmetrisation move the isotropic value?")
        for a, sym in nuc:
            raw = np.trace(sig[a]) / 3 * PPM
            sm = np.trace((sig[a] + sig[a].T) / 2) / 3 * PPM
            print(f"      {sym:2s}  raw iso = {raw:10.5f}   "
                  f"sym iso = {sm:10.5f}   delta = {sm - raw: .3e}")
    print()


if __name__ == "__main__":
    main()
