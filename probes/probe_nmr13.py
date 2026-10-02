"""probe_nmr13.py -- two loose ends before writing product code.

1. TMS came out ~50x too large (raw 13C 188.1 against CH4's 3.77) while four
   other molecules matched literature to a few percent.  Si carries a d
   function in 6-31G*, so test whether the GIAO machinery is the problem or
   the SCF.

2. The tensor must be symmetric, because sigma_ts = d2E/dB_t dmu_s and mixed
   partial derivatives commute.  The earlier asymmetry was measured with a
   mixed (real field / imaginary moment) convention; re-measure it now that
   the perturbation is consistently imaginary.
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
    conv = []

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
    conv.append(m0.converged)
    dm0 = m0.make_rdm1()
    sig = np.zeros((natm, 3, 3))
    lin = []
    for t in range(3):
        bp = [0.0, 0.0, 0.0]; bp[t] = dB
        bm = [0.0, 0.0, 0.0]; bm[t] = -dB
        mp_, mm_ = scf_at(bp, dm0), scf_at(bm, dm0)
        conv += [mp_.converged, mm_.converged]
        lin.append((mp_.e_tot.real - mm_.e_tot.real) / (2 * dB))
        dp, dm_ = mp_.make_rdm1(), mm_.make_rdm1()
        for a in range(natm):
            for s in range(3):
                hp = -1j * D["P"][a][s] + sum(bp[k] * D["Q"][a][k][s] for k in range(3))
                hm = -1j * D["P"][a][s] + sum(bm[k] * D["Q"][a][k][s] for k in range(3))
                sig[a, t, s] = (np.einsum("uv,vu->", dp, hp).real
                                - np.einsum("uv,vu->", dm_, hm).real) / (2 * dB)
    return sig, lin, conv


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

print("=" * 92)
print("  1.  TMS across bases:  is it the d function on Si, or the SCF?")
print("=" * 92)
print(f"{'basis':12s} {'nao':>4s} {'conv':>18s} {'13C raw':>12s} {'1H raw':>12s} "
      f"{'13C ppm':>10s} {'1H ppm':>10s}")
for basis in ("sto-3g", "3-21g", "6-31g", "6-31g*", "cc-pvdz"):
    mol = gto.M(atom=xyz_tms, basis=basis, unit="Angstrom", verbose=0)
    sig, lin, conv = shieldings(mol, build(mol))
    c = float(np.mean([np.trace(sig[i]) / 3 for i in (1, 3, 5, 7)]))
    h = float(np.mean([np.trace(sig[i]) / 3 for i in (2, 4, 6, 8)]))
    print(f"{basis:12s} {mol.nao_nr():4d} {str(conv):>18s} {c:12.5f} {h:12.5f} "
          f"{c * PPM:10.2f} {h * PPM:10.2f}")
print("   literature: 13C 188.1 ppm, 1H 30.6 ppm")

print()
print("=" * 92)
print("  2.  tensor symmetry in the consistently imaginary formulation")
print("=" * 92)
CASES = [
    ("H2O", [("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587), ("H", 0.0, 0.757, 0.587)]),
    ("CH4", [("C", 0.0, 0.0, 0.0)] + [("H", *(1.087 * v)) for v in tet]),
    ("NH3", [("N", 0.0, 0.0, 0.0)] + [
        ("H", 0.93748 * np.cos(2 * np.pi * k / 3),
         0.93748 * np.sin(2 * np.pi * k / 3), 0.38101) for k in range(3)]),
    ("benzene", [("C", *(1.397 * np.array([np.cos(np.pi / 3 * k), np.sin(np.pi / 3 * k), 0.0])))
                 for k in range(6)]
                + [("H", *(2.481 * np.array([np.cos(np.pi / 3 * k), np.sin(np.pi / 3 * k), 0.0])))
                   for k in range(6)]),
]
print(f"{'molecule':10s} {'atom':>6s} {'iso':>10s} {'max|asym|':>11s} "
      f"{'aniso':>10s} {'dE/dB max':>11s}")
for name, xyz in CASES:
    mol = gto.M(atom=xyz, basis="6-31g*", unit="Angstrom", verbose=0)
    sig, lin, conv = shieldings(mol, build(mol))
    worst = 0.0
    for a in range(mol.natm):
        sg = sig[a]
        asym = np.max(np.abs(sg - sg.T)) * PPM
        worst = max(worst, asym)
        eigs = np.linalg.eigvalsh((sg + sg.T) / 2) * PPM
        aniso = eigs[2] - 0.5 * (eigs[0] + eigs[1])
        if a == 0 or mol.atom_symbol(a) == "H" and a == min(
                i for i in range(mol.natm) if mol.atom_symbol(i) == "H"):
            print(f"{name:10s} {mol.atom_symbol(a) + str(a):>6s} "
                  f"{np.trace(sg) / 3 * PPM:10.3f} {asym:11.5f} {aniso:10.3f} "
                  f"{max(abs(x) for x in lin):11.2e}")
    print(f"{'':10s} {'(all)':>6s} worst asymmetry = {worst:.5f} ppm")
