"""probe_nmr11.py -- one clean, physically consistent GIAO implementation, judged
by chemical shifts against experiment.

Earlier probes were inconsistent: probe_nmr9 built the field perturbation as
h0 - 1j*B*h_b where h_b already carried a -1j, so the field entered *real*,
while the nuclear moment entered *imaginary*.  That is not a magnetic field.

The physically correct perturbation is imaginary in both, because the magnetic
coupling is -i times a real operator:

    h(B,mu) = h0 - i B_t O^t - i mu_s P^s + B_t mu_s Q^{ts}
    S(B)    = S0 - i B_t S^t
    eri(B)  = eri0 - i B_t G^t

with O, S, G, P, Q all real (measured: int1e_igovlp = -1/2 eps (R_u-R_v) <r>,
residual 8.3e-17).  A correct implementation must then have NO linear Zeeman
term, dE/dB = 0, which is a sharp internal test.

Judge it the way a paper would: 1H and 13C chemical shifts against TMS.
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=170)
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6
BOHR = 0.52917721092

BASIS = "6-31g*"


def build(mol):
    nao, natm = mol.nao_nr(), mol.natm
    R = mol.atom_coords()
    d = {"h0": mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
         "s0": mol.intor("int1e_ovlp"), "eri0": mol.intor("int2e"),
         "natm": natm, "nao": nao, "R": R}
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


def scf_at(mol, D, bvec, dm0):
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


def shieldings(mol, D, dB=1e-4, sign=+1.0):
    nao, natm = D["nao"], D["natm"]
    m0 = scf_at(mol, D, [0.0] * 3, None)
    dm0 = m0.make_rdm1()
    sig = np.zeros((natm, 3, 3))
    lin = []
    for t in range(3):
        bp = [0.0, 0.0, 0.0]; bp[t] = dB
        bm = [0.0, 0.0, 0.0]; bm[t] = -dB
        mp_, mm_ = scf_at(mol, D, bp, dm0), scf_at(mol, D, bm, dm0)
        lin.append((mp_.e_tot.real - mm_.e_tot.real) / (2 * dB))
        dp, dm_ = mp_.make_rdm1(), mm_.make_rdm1()
        for a in range(natm):
            for s in range(3):
                hp = -1j * D["P"][a][s] + sum(bp[k] * D["Q"][a][k][s] for k in range(3))
                hm = -1j * D["P"][a][s] + sum(bm[k] * D["Q"][a][k][s] for k in range(3))
                gp = np.einsum("uv,vu->", dp, hp).real
                gm = np.einsum("uv,vu->", dm_, hm).real
                sig[a, t, s] = sign * (gp - gm) / (2 * dB)
    return sig, lin, m0


def make(name, xyz, basis=BASIS):
    mol = gto.M(atom=xyz, basis=basis, unit="Angstrom", verbose=0)
    sig, lin, m0 = shieldings(mol, build(mol))
    return mol, sig, lin, m0.e_tot.real


# ---------------------------------------------------------------- reference
print(f"basis = {BASIS},  alpha^2*1e6 = {PPM:.6f}")
print()
print("=" * 78)
print("  TMS, Si(CH3)4  -- Si-C 1.875 A, C-H 1.090 A, tetrahedral")
print("=" * 78)
si_c, c_h = 1.875, 1.090
tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
tet /= np.linalg.norm(tet, axis=1)[:, None]
xyz_tms = [("Si", 0.0, 0.0, 0.0)]
# H-C-Si = 109.47 deg, so the C-H direction v obeys v.u = 1/3 with u the Si->C
# unit vector: v = u/3 + (2 sqrt2 / 3) w for any unit w perpendicular to u.
for u in tet:
    cpos = si_c * u
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    w = np.cross(u, ref)
    w /= np.linalg.norm(w)
    v = u / 3.0 + (2.0 * np.sqrt(2.0) / 3.0) * w
    hpos = cpos + c_h * v
    xyz_tms.append(("C", *cpos))
    xyz_tms.append(("H", *hpos))
mol_tms, sig_tms, lin_tms, e_tms = make("tms", xyz_tms)
c_ref = np.mean([np.trace(sig_tms[i]) / 3 * PPM for i in range(1, len(xyz_tms), 2)])
h_ref = np.mean([np.trace(sig_tms[i]) / 3 * PPM for i in range(2, len(xyz_tms), 2)])
print(f"  TMS  sigma(13C) = {c_ref:9.3f} ppm     sigma(1H) = {h_ref:9.3f} ppm")
print(f"  dE/dB check: {['%.2e' % x for x in lin_tms]}")
print(f"  literature TMS: 13C 188.1  1H 30.6 (absolute shielding)")

# ---------------------------------------------------------------- molecules
CASES = [
    ("methane", [("C", 0, 0, 0)] + [("H", *(1.087 * v)) for v in
                                    [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]],
     "C", -2.3, "H", 0.23),
    ("ethylene", [("C", 0, 0, 0.667), ("C", 0, 0, -0.667),
                  ("H", 0, 0.923, 1.238), ("H", 0, -0.923, 1.238),
                  ("H", 0, 0.923, -1.238), ("H", 0, -0.923, -1.238)],
     "C", 123.5, "H", 5.40),
    ("acetylene", [("C", 0, 0, 0.601), ("C", 0, 0, -0.601),
                   ("H", 0, 0, 1.663), ("H", 0, 0, -1.663)],
     "C", 71.9, "H", 1.80),
    ("benzene", [("C", *(1.397 * np.array([np.cos(np.pi / 3 * k),
                                           np.sin(np.pi / 3 * k), 0.0]))) for k in range(6)]
                + [("H", *(2.481 * np.array([np.cos(np.pi / 3 * k),
                                             np.sin(np.pi / 3 * k), 0.0]))) for k in range(6)],
     "C", 128.5, "H", 7.26),
    ("methanol", [("C", 0, 0, 0), ("O", 0, 0, 1.43),
                  ("H", 0.895, 0, -0.44), ("H", -0.448, 0.775, -0.44),
                  ("H", -0.448, -0.775, -0.44), ("H", 0.885, 0, 1.85)],
     "C", 49.5, "H", 3.35),
    ("water", [("O", 0, 0, 0), ("H", 0, -0.757, 0.587), ("H", 0, 0.757, 0.587)],
     "O", None, "H", 4.79),
    ("ammonia", [("N", 0, 0, 0)] + [("H", 0.93748 * np.cos(2 * np.pi * k / 3),
                                     0.93748 * np.sin(2 * np.pi * k / 3), 0.38101)
                                    for k in range(3)],
     "N", None, "H", 1.20),
]

print()
print("=" * 78)
print("  chemical shifts against TMS   (sigma_ref - sigma, ppm)")
print("=" * 78)
print(f"{'molecule':11s} {'atom':>4s} {'sigma':>10s} {'delta calc':>11s} "
      f"{'delta exp':>10s} {'error':>9s}   dE/dB max")
print("-" * 78)
for name, xyz, heavy, d_c_exp, hyd, d_h_exp in CASES:
    mol, sig, lin, e = make(name, xyz)
    iso = np.array([np.trace(sig[a]) / 3 for a in range(mol.natm)]) * PPM
    # heavy atom of the first matching element
    idx_heavy = [a for a in range(mol.natm) if mol.atom_symbol(a) == heavy]
    idx_h = [a for a in range(mol.natm) if mol.atom_symbol(a) == hyd]
    s_heavy = float(np.mean([iso[a] for a in idx_heavy]))
    s_h = float(np.mean([iso[a] for a in idx_h]))
    d_c = c_ref - s_heavy
    d_h = h_ref - s_h
    err_c = f"{d_c - d_c_exp:+8.2f}" if d_c_exp is not None else "       -"
    err_h = f"{d_h - d_h_exp:+8.2f}"
    print(f"{name:11s} {heavy:>4s} {s_heavy:10.3f} {d_c:11.3f} "
          f"{str(d_c_exp):>10s} {err_c:>9s}   {max(abs(x) for x in lin):.2e}")
    print(f"{'':11s} {hyd:>4s} {s_h:10.3f} {d_h:11.3f} {d_h_exp:10.2f} "
          f"{err_h:>9s}")
