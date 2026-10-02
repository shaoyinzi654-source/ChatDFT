"""probe_nmr19.py -- the dB discontinuity, and whether the perturbed SCF converges.

probe_nmr16's dB sweep on water looked stable and then was not:

    dB        iso(17O)     iso(1H)
    1e-02     329.9624     31.8653
    3e-03     329.9760     31.8659
    1e-03     329.9461     31.8662
    3e-04     330.0585     31.8651
    1e-04     330.0684     31.8650
    3e-05     345.2775     31.6561     <-- 15 ppm jump
    1e-05     345.2775     31.6561     <-- and then bit-identical
    1e-06     358.5528     31.4561

Two things are wrong with that picture.  A 15 ppm jump in a quantity that
should be dB-independent means the finite difference is not measuring the
derivative.  And 3e-05 giving a result bit-identical to 1e-05 to four decimals
is not convergence, it is a threshold being crossed.

The reference shieldings quoted in earlier rounds (H2O 17O 345.046) sit on the
3e-05 side of the jump, which is also the side closer to experiment (344), so
"the bigger number is the better one" cannot be assumed either way.  This has
to be settled by measurement before any of it goes into a product, because the
17O reference enters every 17O chemical shift.

What is measured here:
  * the full dB scan, with the SCF cycle count and the converged flag for
    every perturbed run, so a non-converged run is visible instead of being
    silently averaged in;
  * the same scan with conv_tol tightened to 1e-13, to separate "the SCF
    stopped early" from "the derivative is genuinely non-linear";
  * a second-order check: sigma(dB) and sigma(2 dB) must agree, and a
    Richardson extrapolation sigma(2dB) - (sigma(2dB) - sigma(dB)) tells us
    which side is the extrapolated value.

Run:  python probes/probe_nmr19.py
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=200)
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6

WATER = [("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587),
         ("H", 0.0, 0.757, 0.587)]


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
    eye = np.eye(3)
    for a in range(natm):
        with mol.with_rinv_orig(R[a]):
            d["P"][a] = mol.intor("int1e_ia01p")
            aa = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
            gg = mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)
        d["Q"][a] = aa - np.einsum("ts,uv->tsuv", eye,
                                   aa.trace(axis1=0, axis2=1)) + gg
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
    info = []
    for t in range(3):
        bp = [0.0] * 3; bp[t] = dB
        bm = [0.0] * 3; bm[t] = -dB
        mp_ = scf_at(mol, D, bp, dm0, conv_tol)
        mm_ = scf_at(mol, D, bm, dm0, conv_tol)
        info.append((bool(mp_.converged), int(getattr(mp_, "cycles", -1)),
                     bool(mm_.converged), int(getattr(mm_, "cycles", -1))))
        dp, dm_ = mp_.make_rdm1(), mm_.make_rdm1()
        for a in range(D["natm"]):
            for s in range(3):
                hp = -1j * D["P"][a][s] + sum(bp[k] * D["Q"][a][k][s]
                                              for k in range(3))
                hm = -1j * D["P"][a][s] + sum(bm[k] * D["Q"][a][k][s]
                                              for k in range(3))
                sig[a, t, s] = (np.einsum("uv,vu->", dp, hp).real
                                - np.einsum("uv,vu->", dm_, hm).real) / (2 * dB)
    return sig, info


def main():
    mol = gto.M(atom=WATER, basis="6-31g*", verbose=0)
    D = build(mol)
    m0 = scf_at(mol, D, [0.0] * 3, None, 1e-11)
    dm0 = m0.make_rdm1()
    print(f"water  natm={mol.natm}  nao={mol.nao_nr()}  "
          f"unperturbed converged={m0.converged} cycles="
          f"{getattr(m0, 'cycles', -1)}")

    grid = [3e-3, 1e-3, 3e-4, 1e-4, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 3e-6]
    for tol in (1e-11, 1e-13):
        print()
        print("=" * 92)
        print(f" conv_tol = {tol:.0e}")
        print("=" * 92)
        print(f"  {'dB':>9s} {'iso(17O)':>11s} {'iso(1H)':>10s} "
              f"{'asym(H)':>9s}   perturbed SCF (converged, cycles) x3 dirs")
        for dB in grid:
            sig, info = shieldingings = shieldings(mol, D, dB, tol, dm0)
            isoO = np.trace(sig[0]) / 3 * PPM
            isoH = np.trace(sig[1]) / 3 * PPM
            aH = np.abs(sig[1] - sig[1].T).max() * PPM
            flag = " ".join(("ok" if c else "NO") + f"/{n:3d}" for c, n, _, _ in info)
            print(f"  {dB:9.0e} {isoO:11.4f} {isoH:10.4f} {aH:9.5f}   {flag}")

    print()
    print("=" * 92)
    print(" Richardson check: sigma(dB) and sigma(2dB) must agree if linear")
    print("=" * 92)
    print(f"  {'dB':>9s} {'iso(17O)':>11s} {'iso(17O)@2dB':>13s} {'difference':>11s}")
    for dB in (1e-4, 5e-5, 3e-5, 1e-5, 5e-6):
        s1, _ = shieldings(mol, D, dB, 1e-11, dm0)
        s2, _ = shieldings(mol, D, 2 * dB, 1e-11, dm0)
        a = np.trace(s1[0]) / 3 * PPM
        b = np.trace(s2[0]) / 3 * PPM
        print(f"  {dB:9.0e} {a:11.4f} {b:13.4f} {a - b:11.4f}")
    print()


if __name__ == "__main__":
    main()
