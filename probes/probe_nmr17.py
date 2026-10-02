"""probe_nmr17.py -- is the asymmetry the (t,s) index convention of Q?

probe_nmr16 killed the numerical explanation.  At conv_tol = 1e-11 the H2O
1H asymmetry is 1.648 ppm; it is 1.648322 at dB = 1e-3 and 1.648086 at
conv_tol = 1e-13.  Five decades of dB and three of conv_tol move it by 0.0003
ppm.  It is not noise, and it is not slow convergence.  It is also entirely
OFF-DIAGONAL: the diagonal asymmetry is exactly 0.000000 in every molecule
tested, and the heavy atoms are clean (H2O O 0.000000, NH3 N 0.000434).

The mixed derivative splits into exactly two pieces:

    sigma_ts = B_ts + A_ts
    B_ts = Tr(D0 . Q^{ts})            the "diamagnetic" piece
    A_ts = Tr(D1_t^B . (-i P^s))      the "paramagnetic" piece

By commutativity of partial derivatives sigma must be symmetric.  Neither
piece is symmetric on its own, but their sum has to be.  The transpose
identity that has to hold is

    B_ts + A_ts  ==  B_st + A_st

The one input whose index order was never verified in the *correct* imaginary
convention is Q.  probe_nmr7 tested Q vs Q^T, but that was three probes before
probe_nmr11 fixed the mixed convention, so its answer (whole-tensor asymmetry
0.0343897264069187 either way) is stale and has to be redone.

This probe recomputes A and B separately and tries every combination:
Q, Q^T, sym(Q), and Q = 0 -- then reports which one makes sigma symmetric
WITHOUT breaking the shieldings that were validated against experiment.

Run:  python probes/probe_nmr17.py
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
    d["A11"] = np.zeros((natm, 3, 3, nao, nao))
    d["A01"] = np.zeros((natm, 3, 3, nao, nao))
    for a in range(natm):
        with mol.with_rinv_orig(R[a]):
            d["P"][a] = mol.intor("int1e_ia01p")
            aa = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
            gg = mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)
        d["A11"][a] = aa
        d["A01"][a] = gg
        d["Q"][a] = aa - np.einsum("ts,uv->tsuv", np.eye(3),
                                   aa.trace(axis1=0, axis2=1))
        d["Q"][a] += gg
    return d


def scf_at(mol, D, bvec, dm0, tol=1e-11):
    bvec = np.asarray(bvec, dtype=float)
    m = scf.RHF(mol)
    m.verbose = 0
    m.chkfile = None
    m.max_cycle = 300
    m.conv_tol = tol
    m.get_hcore = lambda *a: D["h0"] - 1j * np.einsum("t,tuv->uv", bvec, D["O_h"])
    m.get_ovlp = lambda *a: D["s0"] - 1j * np.einsum("t,tuv->uv", bvec, D["O_S"])
    m._eri = D["eri0"] - 1j * np.einsum("t,tuvkl->uvkl", bvec, D["O_G"])
    m.kernel(dm0)
    return m


def pieces(mol, D, dB=1e-4):
    """Return the paramagnetic piece A and the diamagnetic piece B, separately."""
    m0 = scf_at(mol, D, [0.0] * 3, None)
    dm0 = m0.make_rdm1()
    D0 = m0.make_rdm1()
    natm = D["natm"]
    A = np.zeros((natm, 3, 3))
    D1 = np.zeros((natm, 3, D["nao"], D["nao"]), dtype=complex)
    for t in range(3):
        bp = [0.0] * 3; bp[t] = dB
        bm = [0.0] * 3; bm[t] = -dB
        mp_, mm_ = scf_at(mol, D, bp, dm0), scf_at(mol, D, bm, dm0)
        D1[:, t] = (mp_.make_rdm1() - mm_.make_rdm1()) / (2 * dB)
    for a in range(natm):
        for t in range(3):
            for s in range(3):
                A[a, t, s] = np.einsum("uv,vu->", D1[a, t],
                                       -1j * D["P"][a][s]).real
    B = np.einsum("uv,atsuv->ats", D0, D["Q"]).real
    return A, B, D0, D1


def show(tag, sig, nuc):
    lines = [f"    {tag:22s}"]
    for a, sym in nuc:
        m = sig[a] * PPM
        lines.append(f" {sym:2s} iso={np.trace(m) / 3:9.4f} "
                     f"asym={np.abs(m - m.T).max():8.5f} ")
    print("".join(lines))
    tot = np.abs(sig - sig.transpose(0, 2, 1)).max() * PPM
    print(f"      -> whole-tensor asymmetry = {tot:.6f} ppm")
    return tot


def main():
    for name, atoms, nuc in (("H2O", WATER, [(0, "O"), (1, "H")]),
                             ("NH3", AMMONIA, [(0, "N"), (1, "H")]),
                             ("CH4", METHANE, [(0, "C"), (1, "H")])):
        mol = gto.M(atom=atoms, basis="6-31g*", verbose=0)
        D = build(mol)
        A, B, D0, D1 = pieces(mol, D)

        print("=" * 78)
        print(f" {name}  natm={mol.natm}  nao={mol.nao_nr()}")
        print("=" * 78)

        # how asymmetric is each piece on its own?
        print("\n  piece-by-piece asymmetry (ppm), and the size of each piece:")
        for a, sym in nuc:
            am, bm = A[a] * PPM, B[a] * PPM
            print(f"    {sym:2s}  A asym={np.abs(am - am.T).max():9.5f} "
                  f"|A|max={np.abs(am).max():9.4f}   "
                  f"B asym={np.abs(bm - bm.T).max():9.5f} "
                  f"|B|max={np.abs(bm).max():9.4f}")

        # the combinations
        print("\n  candidate assemblies:")
        combos = {
            "A + B        (product today)": lambda a: A[a] + B[a],
            "A^T + B^T": lambda a: A[a].T + B[a].T,
            "A + B^T": lambda a: A[a] + B[a].T,
            "A^T + B": lambda a: A[a].T + B[a],
            "A + sym(B)": lambda a: A[a] + (B[a] + B[a].T) / 2,
            "sym(A) + B": lambda a: (A[a] + A[a].T) / 2 + B[a],
            "sym(A) + sym(B)": lambda a: (A[a] + A[a].T) / 2 + (B[a] + B[a].T) / 2,
            "A only (Q dropped)": lambda a: A[a],
            "B only (no response)": lambda a: B[a],
        }
        for tag, fn in combos.items():
            sig = np.array([fn(a) for a in range(D["natm"])])
            show(tag, sig, nuc)
        print()


if __name__ == "__main__":
    main()
