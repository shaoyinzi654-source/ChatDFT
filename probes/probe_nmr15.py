"""probe_nmr15.py -- where does the H-atom tensor asymmetry actually come from?

Round 16 left one open defect: sigma_ts must be symmetric, and in the
consistent imaginary formulation every heavy atom and every nucleus of CH4 /
benzene is symmetric to 0.00000 ppm, but H2O 1H keeps 1.64 ppm and NH3 1H
0.61 ppm.  "Confined to hydrogen" is a description, not a cause.

The mixed derivative d2E/dB_t dmu_s has two equally valid expressions, and
they must agree:

    route A (what the product does today)
        sigma_ts = [ Tr(D(+B_t) M_s(+B_t)) - Tr(D(-B_t) M_s(-B_t)) ] / 2dB
        M_s(B)   = -i P^s + sum_k B_k Q^{ks}
        ==> sigma_ts = Tr(D0 Q^{ts}) + Tr(D1_t^B (-i P^s))

    route B (differentiate the other way: field response of dE/dmu)
        sigma_st = Tr(D0 Q^{st}) + Tr(D1_s^mu (-i O^t))
        where D1_s^mu is the density response to a purely imaginary
        nuclear-moment perturbation -i mu_s P^s, and O^t is the orbital-Zeeman
        operator.

Route A and route B must give the same 3x3 matrix for each nucleus.  If they
do, the residual asymmetry is a property of the operator set and the fix is a
symmetrisation; if they do not, one of the two is wrong and the difference
says which.

The probe also settles a question that is easy to get wrong and impossible to
guess: which integrals depend on which origin.  `int1e_ia01p` needs
`with_rinv_orig(R_A)`, but the two second-order pieces `int1e_giao_a11part`
and `int1e_a01gp` are documented against the GAUGE origin, i.e.
`with_common_origin(R_A)`.  Fitting them with the wrong origin would give an
asymmetric Q and therefore an asymmetric sigma -- which is exactly the
symptom.

Run:  python probes/probe_nmr15.py
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=200)
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6

WATER = [("O", 0.0, 0.0, 0.1173), ("H", 0.0, 0.7572, -0.4692),
         ("H", 0.0, -0.7572, -0.4692)]
METHANE = [("C", 0.0, 0.0, 0.0),
           ("H", 0.6291, 0.6291, 0.6291), ("H", -0.6291, -0.6291, 0.6291),
           ("H", -0.6291, 0.6291, -0.6291), ("H", 0.6291, -0.6291, -0.6291)]


def mk(atoms, basis="6-31g*"):
    return gto.M(atom=atoms, basis=basis, verbose=0)


# ----------------------------------------------------------------------
# 1. which origin does each integral actually need?
# ----------------------------------------------------------------------
def origin_test():
    print("=" * 72)
    print(" 1. origin dependence of the second-order pieces")
    print("=" * 72)
    mol = mk(WATER)
    R = mol.atom_coords()
    nao = mol.nao_nr()

    a = np.array([0.0, 0.0, 0.0])
    b = R[1]                      # a hydrogen, well away from the oxygen

    def grab(kind, origin, how):
        if how == "rinv":
            with mol.with_rinv_orig(origin):
                return mol.intor(kind)
        with mol.with_common_origin(origin):
            return mol.intor(kind)

    for kind in ("int1e_ia01p", "int1e_giao_a11part", "int1e_a01gp"):
        same_rinv = np.abs(grab(kind, a, "rinv") - grab(kind, b, "rinv")).max()
        same_comm = np.abs(grab(kind, a, "common") - grab(kind, b, "common")).max()
        rinv_vs_comm = np.abs(grab(kind, a, "rinv") - grab(kind, a, "common")).max()
        print(f"  {kind:24s} |A-B| via rinv_orig = {same_rinv:12.6e}")
        print(f"  {'':24s} |A-B| via common_orig = {same_comm:12.6e}")
        print(f"  {'':24s} rinv vs common @ A    = {rinv_vs_comm:12.6e}")

    print("\n  Reading: an integral that is origin-INDEPENDENT shows a large")
    print("  |A-B| under the wrong setter and ~0 under the right one.\n")


# ----------------------------------------------------------------------
# 2. the operator bundle, both origin conventions side by side
# ----------------------------------------------------------------------
def build(mol, q_how="rinv"):
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
        if q_how == "rinv":
            with mol.with_rinv_orig(R[a]):
                aa = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
                g = mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)
        else:
            with mol.with_common_origin(R[a]):
                aa = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
                g = mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)
        d["Q"][a] = aa - np.einsum("ts,uv->tsuv", np.eye(3),
                                   aa.trace(axis1=0, axis2=1))
        d["Q"][a] += g
    return d


def scf_at(mol, D, bvec, dm0=None, extra=None):
    """SCF at a field bvec, optionally with an extra one-electron matrix."""
    bvec = np.asarray(bvec, dtype=float)
    m = scf.RHF(mol)
    m.verbose = 0
    m.chkfile = None
    m.max_cycle = 300
    m.conv_tol = 1e-11
    h = D["h0"] - 1j * np.einsum("t,tuv->uv", bvec, D["O_h"])
    if extra is not None:
        h = h + extra
    m.get_hcore = lambda *a: h
    m.get_ovlp = lambda *a: D["s0"] - 1j * np.einsum("t,tuv->uv", bvec, D["O_S"])
    m._eri = D["eri0"] - 1j * np.einsum("t,tuvkl->uvkl", bvec, D["O_G"])
    m.kernel(dm0)
    return m


def route_A(mol, D, dB=1e-4):
    m0 = scf_at(mol, D, [0.0] * 3)
    dm0 = m0.make_rdm1()
    sig = np.zeros((D["natm"], 3, 3))
    lin = []
    for t in range(3):
        bp = [0.0] * 3; bp[t] = dB
        bm = [0.0] * 3; bm[t] = -dB
        mp_, mm_ = scf_at(mol, D, bp, dm0), scf_at(mol, D, bm, dm0)
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
    return sig, lin, m0


def route_B(mol, D, dmu=1e-4, with_q=False):
    """Differentiate the other way: field operator against the mu-response.

    dE/dB_t = Tr(D . (-i O^t + mu_s Q^{ts}))  at a finite mu_s, so
    d2E/dB_t dmu_s = Tr(D1^mu_s . (-i O^t)) + Tr(D0 . Q^{ts}).
    `with_q` adds the second piece, which the mu-difference alone cannot see.
    """
    sig = np.zeros((D["natm"], 3, 3))
    m0 = scf_at(mol, D, [0.0] * 3)
    dm0 = m0.make_rdm1()
    D0 = m0.make_rdm1()
    for a in range(D["natm"]):
        for s in range(3):
            mp_ = scf_at(mol, D, [0.0] * 3, dm0,
                         extra=-1j * dmu * D["P"][a][s])
            mm_ = scf_at(mol, D, [0.0] * 3, dm0,
                         extra=+1j * dmu * D["P"][a][s])
            dp, dm_ = mp_.make_rdm1(), mm_.make_rdm1()
            for t in range(3):
                vp = np.einsum("uv,vu->", dp, -1j * D["O_h"][t]).real
                vm = np.einsum("uv,vu->", dm_, -1j * D["O_h"][t]).real
                sig[a, t, s] = (vp - vm) / (2 * dmu)
                if with_q:
                    sig[a, t, s] += np.einsum("uv,vu->", D0,
                                              D["Q"][a][t, s]).real
    return sig, D0


def report(tag, sig, nuc, extra=""):
    print(f"\n  {tag}")
    for a, sym in nuc:
        m = sig[a] * PPM
        iso = np.trace(m) / 3
        asym = np.abs(m - m.T).max()
        print(f"    {sym:2s} iso = {iso:10.4f} ppm   |M-M^T|max = {asym:10.6f}"
              f"   |M|max = {np.abs(m).max():10.4f}  {extra}")
    tot = np.abs(sig - sig.transpose(0, 2, 1)).max() * PPM
    print(f"    whole-tensor asymmetry = {tot:.6f} ppm")


def main():
    origin_test()

    for name, atoms, nuc in (
            ("H2O", WATER, [(0, "O"), (1, "H")]),
            ("CH4", METHANE, [(0, "C"), (1, "H")])):
        mol = mk(atoms)
        print("=" * 72)
        print(f" {name}   natm={mol.natm}  nao={mol.nao_nr()}")
        print("=" * 72)
        for q_how in ("rinv", "common"):
            D = build(mol, q_how=q_how)
            sigA, lin, _ = route_A(mol, D)
            sigB, D0 = route_B(mol, D)
            sigBQ, _ = route_B(mol, D, with_q=True)
            print(f"\n  --- Q built with {q_how} origin ---")
            print(f"    dE/dB = {['%.2e' % x for x in lin]}")
            report("route A  (field response)", sigA, nuc)
            report("route B  (mu response, no Q)", sigB, nuc)
            report("route B+Q (mu response + Q)", sigBQ, nuc)
            print(f"\n    A vs B+Q elementwise (ppm):")
            for a, sym in nuc:
                diff = (sigA[a] - sigBQ[a]) * PPM
                print(f"      {sym:2s} " + " ".join("%9.4f" % v
                                                     for v in diff.ravel()))
    print()


if __name__ == "__main__":
    main()
