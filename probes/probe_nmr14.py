"""probe_nmr14.py -- proper TMS, and the chemical shifts that a paper reports.

probe_nmr13's "TMS" had nao = 33 at STO-3G, which is 9(Si) + 4*5(C) + 4*1(H):
only ONE hydrogen per carbon.  It was never Si(CH3)4.  Build it properly --
each carbon carries three hydrogens at 109.47 deg to the Si-C bond, spaced 120
deg around it -- and then the reference is what it should be.

With a correct TMS, validate the number a paper actually prints: the chemical
shift, delta = sigma_ref - sigma, against gas-phase experiment.
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=170)
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6


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


def shieldings(mol, D, dB=1e-4):
    nao, natm = D["nao"], D["natm"]

    def scf_at(bvec, dm0):
        bvec = np.asarray(bvec, dtype=float)
        m = scf.RHF(mol)
        m.verbose = 0
        m.chkfile = None
        m.max_cycle = 300
        m.conv_tol = 1e-11
        m.get_hcore = lambda *a: D["h0"] - 1j * np.einsum("t,tuv->uv", bvec, D["O_h"])
        m.get_ovlp = lambda *a: D["s0"] - 1j * np.einsum("t,tuv->uv", bvec, D["O_S"])
        m._eri = D["eri0"] - 1j * np.einsum("t,tuvkl->uvkl", bvec, D["O_G"])
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
                hp = -1j * D["P"][a][s] + sum(bp[k] * D["Q"][a][k][s] for k in range(3))
                hm = -1j * D["P"][a][s] + sum(bm[k] * D["Q"][a][k][s] for k in range(3))
                sig[a, t, s] = (np.einsum("uv,vu->", dp, hp).real
                                - np.einsum("uv,vu->", dm_, hm).real) / (2 * dB)
    return sig, lin


def tms_geometry():
    """Si(CH3)4: Si-C 1.875 A, C-H 1.090 A, H-C-Si 109.47 deg."""
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    tet /= np.linalg.norm(tet, axis=1)[:, None]
    out = [("Si", 0.0, 0.0, 0.0)]
    for u in tet:
        cpos = 1.875 * u
        ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        w1 = np.cross(u, ref); w1 /= np.linalg.norm(w1)
        w2 = np.cross(u, w1)
        out.append(("C", *cpos))
        for k in range(3):
            phi = 2 * np.pi * k / 3
            v = u / 3.0 + (2.0 * np.sqrt(2.0) / 3.0) * (np.cos(phi) * w1
                                                        + np.sin(phi) * w2)
            out.append(("H", *(cpos + 1.090 * v)))
    return out


BASIS = "6-31g*"

xyz_tms = tms_geometry()
mol_tms = gto.M(atom=xyz_tms, basis=BASIS, unit="Angstrom", verbose=0)
sig_tms, lin_tms = shieldings(mol_tms, build(mol_tms))
print(f"TMS  natm={mol_tms.natm}  nao={mol_tms.nao_nr()}  "
      f"(expect 17 atoms, Si + 4 C + 12 H)")
print(f"  dE/dB check: {['%.1e' % x for x in lin_tms]}")
iC = [i for i in range(mol_tms.natm) if mol_tms.atom_symbol(i) == "C"]
iH = [i for i in range(mol_tms.natm) if mol_tms.atom_symbol(i) == "H"]
sigC = float(np.mean([np.trace(sig_tms[i]) / 3 for i in iC])) * PPM
sigH = float(np.mean([np.trace(sig_tms[i]) / 3 for i in iH])) * PPM
sigSi = float(np.trace(sig_tms[0]) / 3) * PPM
print(f"  sigma(13C) = {sigC:9.3f}   sigma(1H) = {sigH:9.3f}   "
      f"sigma(29Si) = {sigSi:9.3f}")
print(f"  literature: 13C 188.1   1H 30.6   29Si 368.5")
print(f"  spread over the 4 C: "
      f"{np.ptp([np.trace(sig_tms[i]) / 3 * PPM for i in iC]):.4f} ppm")
print(f"  spread over the 12 H: "
      f"{np.ptp([np.trace(sig_tms[i]) / 3 * PPM for i in iH]):.4f} ppm")

CASES = [
    ("methane", [("C", 0.0, 0.0, 0.0)] + [
        ("H", *(1.087 * v)) for v in
        (np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
         / np.sqrt(3))], {"C": -2.3, "H": 0.23}),
    ("ethane", [("C", 0.0, 0.0, 0.764), ("C", 0.0, 0.0, -0.764),
                ("H", 1.017, 0.0, 1.163), ("H", -0.509, 0.881, 1.163),
                ("H", -0.509, -0.881, 1.163),
                ("H", -1.017, 0.0, -1.163), ("H", 0.509, 0.881, -1.163),
                ("H", 0.509, -0.881, -1.163)], {"C": 5.7, "H": 0.86}),
    ("ethylene", [("C", 0.0, 0.0, 0.667), ("C", 0.0, 0.0, -0.667),
                  ("H", 0.0, 0.923, 1.238), ("H", 0.0, -0.923, 1.238),
                  ("H", 0.0, 0.923, -1.238), ("H", 0.0, -0.923, -1.238)],
     {"C": 123.5, "H": 5.40}),
    ("acetylene", [("C", 0.0, 0.0, 0.601), ("C", 0.0, 0.0, -0.601),
                   ("H", 0.0, 0.0, 1.663), ("H", 0.0, 0.0, -1.663)],
     {"C": 71.9, "H": 1.80}),
    ("benzene", [("C", *(1.397 * np.array([np.cos(np.pi / 3 * k),
                                           np.sin(np.pi / 3 * k), 0.0]))) for k in range(6)]
                + [("H", *(2.481 * np.array([np.cos(np.pi / 3 * k),
                                             np.sin(np.pi / 3 * k), 0.0]))) for k in range(6)],
     {"C": 128.5, "H": 7.26}),
    ("methanol", [("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 1.43),
                  ("H", 0.895, 0.0, -0.44), ("H", -0.448, 0.775, -0.44),
                  ("H", -0.448, -0.775, -0.44), ("H", 0.885, 0.0, 1.85)],
     {"C": 49.5, "H": 3.35}),
]

print()
print("=" * 90)
print("  chemical shifts against TMS   delta = sigma(TMS) - sigma(molecule), ppm")
print("=" * 90)
print(f"{'molecule':11s} {'atom':>4s} {'sigma':>10s} {'delta calc':>11s} "
      f"{'delta exp':>10s} {'error':>8s}")
print("-" * 90)
for name, xyz, exp in CASES:
    mol = gto.M(atom=xyz, basis=BASIS, unit="Angstrom", verbose=0)
    sig, lin = shieldings(mol, build(mol))
    for el, target in exp.items():
        idx = [a for a in range(mol.natm) if mol.atom_symbol(a) == el]
        s = float(np.mean([np.trace(sig[a]) / 3 for a in idx])) * PPM
        ref = sigC if el == "C" else sigH
        d = ref - s
        print(f"{name:11s} {el:>4s} {s:10.3f} {d:11.3f} {target:10.2f} "
              f"{d - target:+8.2f}")
