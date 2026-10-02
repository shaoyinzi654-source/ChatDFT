"""probe_nmr12.py -- pin the scale and the sign of the GIAO shielding.

probe_nmr11 established that the imaginary (physically correct) perturbation
gives dE/dB = 0.00e+00 in all three directions -- the linear Zeeman term
vanishes, as it must.  So the formulation is right.

What is not yet pinned is the unit and the overall sign.  Decide them by
matching absolute shieldings on four molecules at once, which is not a fit: a
single scale factor and a single sign have to satisfy all of

    TMS     13C 188.1    1H 30.6
    CH4     13C 195.0    1H 30.6
    H2O     17O 344      1H 30.7
    NH3     14N 264      1H 30.7-31.0

(the 1H values are all near 30 because a proton's shielding is dominated by the
local spherical density; the heavy-atom values span 188 to 344.)
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=170)
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6
BASIS = "6-31g*"


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


def run(name, xyz):
    mol = gto.M(atom=xyz, basis=BASIS, unit="Angstrom", verbose=0)
    sig, lin = shieldings(mol, build(mol))
    return mol, sig, lin


# TMS
tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
tet /= np.linalg.norm(tet, axis=1)[:, None]
xyz_tms = [("Si", 0.0, 0.0, 0.0)]
for u in tet:
    cpos = 1.875 * u
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    w = np.cross(u, ref); w /= np.linalg.norm(w)
    v = u / 3.0 + (2.0 * np.sqrt(2.0) / 3.0) * w
    xyz_tms.append(("C", *cpos))
    xyz_tms.append(("H", *(cpos + 1.090 * v)))

h4 = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
h4 /= np.linalg.norm(h4, axis=1)[:, None]

CASES = [
    ("TMS", xyz_tms),
    ("CH4", [("C", 0.0, 0.0, 0.0)] + [("H", *(1.087 * v)) for v in h4]),
    ("H2O", [("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587), ("H", 0.0, 0.757, 0.587)]),
    ("NH3", [("N", 0.0, 0.0, 0.0)] + [
        ("H", 0.93748 * np.cos(2 * np.pi * k / 3),
         0.93748 * np.sin(2 * np.pi * k / 3), 0.38101) for k in range(3)]),
]

LIT = {"TMS": {"C": 188.1, "H": 30.6}, "CH4": {"C": 195.0, "H": 30.6},
       "H2O": {"O": 344.0, "H": 30.7}, "NH3": {"N": 264.0, "H": 30.8}}

print(f"basis = {BASIS}   PPM = alpha^2*1e6 = {PPM:.6f}")
print()
print("=" * 96)
print("  raw isotropic shielding, and the two candidate scalings")
print("=" * 96)
print(f"{'molecule':9s} {'atom':>5s} {'raw':>13s} {'raw*PPM':>12s} {'-raw':>11s} "
      f"{'-raw*PPM':>12s} {'lit':>8s} {'best':>10s}")
print("-" * 96)
for name, xyz in CASES:
    mol, sig, lin = run(name, xyz)
    for a in range(mol.natm):
        el = mol.atom_symbol(a)
        if el not in LIT[name]:
            continue
        raw = float(np.trace(sig[a]) / 3)
        lit = LIT[name][el]
        cands = {"raw": raw, "raw*PPM": raw * PPM, "-raw": -raw, "-raw*PPM": -raw * PPM}
        best = min(cands, key=lambda k: abs(cands[k] - lit))
        print(f"{name:9s} {el:>5s} {raw:13.5f} {raw * PPM:12.3f} {-raw:11.5f} "
              f"{-raw * PPM:12.3f} {lit:8.2f} {best:>10s}")
    print(f"{'':9s} {'dE/dB':>5s} {['%.1e' % x for x in lin]}")

print()
print("=" * 96)
print("  conclusion: which single scaling reproduces all eight literature values?")
print("=" * 96)
for tag in ("raw", "raw*PPM", "-raw", "-raw*PPM"):
    errs = []
    for name, xyz in CASES:
        mol, sig, lin = run(name, xyz)
        for a in range(mol.natm):
            el = mol.atom_symbol(a)
            if el not in LIT[name]:
                continue
            raw = float(np.trace(sig[a]) / 3)
            v = {"raw": raw, "raw*PPM": raw * PPM, "-raw": -raw,
                 "-raw*PPM": -raw * PPM}[tag]
            errs.append(abs(v - LIT[name][el]))
    print(f"  {tag:10s}  mean |error| = {np.mean(errs):9.3f}   max = {np.max(errs):9.3f}")
