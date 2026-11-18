"""probe_nmr5.py -- pin the GIAO conventions empirically, and explain the
asymmetry that probe_nmr3 left in the water 1H tensor.

Measured so far:
  * int1e_pnucxp == -sum_A Z_A int1e_ia01p(origin R_A)   (ratio -1.000000)
  * the naive recipe reproduces the *magnitude* of the 1H shielding of water
    (30.675 against a literature 30.7) but with the sign flipped, and leaves a
    1.7e-2 asymmetry in the tensor where an exact calculation has none.

Three things to settle:
  1. the libcint GIAO phase convention (fit the factor on int1e_igovlp)
  2. the component ordering of the 9-component second-order integrals
  3. whether the sign is a genuine convention and not a fitted fudge: it has to
     work for two molecules at once.
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=170)
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6

EPS = np.zeros((3, 3, 3))
EPS[0, 1, 2] = EPS[1, 2, 0] = EPS[2, 0, 1] = 1.0
EPS[0, 2, 1] = EPS[2, 1, 0] = EPS[1, 0, 2] = -1.0


def ao_atom_of(mol):
    sl = mol.aoslice_by_atom()
    return np.concatenate([np.full(int(b) - int(a), i)
                           for i, (a, b) in enumerate(sl[:, 2:4])])


# ============================================================================
# 1.  what is int1e_igovlp, really?
# ============================================================================
print("=" * 74)
print("  1.  fitting the int1e_igovlp convention")
print("      candidate:  eps_tab (R_u - R_v)_a <u|r_b|v>  times a complex c")
print("=" * 74)
mol = gto.M(atom=[("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587),
                  ("H", 0.0, 0.757, 0.587)],
            basis="sto-3g", unit="Angstrom", verbose=0)
nao, natm = mol.nao_nr(), mol.natm
R = mol.atom_coords()
ao_atom = ao_atom_of(mol)
r_int = mol.intor("int1e_r")
igovlp = mol.intor("int1e_igovlp")

for tag, dRsign in (("(R_u - R_v)", +1.0), ("(R_v - R_u)", -1.0)):
    pred = np.zeros((3, nao, nao), dtype=complex)
    for t in range(3):
        for a in range(3):
            dR = dRsign * (R[ao_atom][:, None, a] - R[ao_atom][None, :, a])
            for b in range(3):
                pred[t] += EPS[t, a, b] * dR * r_int[b]
    # least-squares complex factor c with  igovlp ~= c * pred
    num = np.vdot(pred.ravel(), igovlp.ravel())
    den = np.vdot(pred.ravel(), pred.ravel())
    c = num / den
    resid = np.max(np.abs(igovlp - c * pred))
    print(f"      {tag:12s} best c = {c.real:+.6f}{c.imag:+.6f}j   "
          f"resid = {resid:.3e}   (max|igovlp|={np.max(np.abs(igovlp)):.4f})")

# also test a form that uses only ONE of the two centres
for tag, which in (("R_u only", "u"), ("R_v only", "v")):
    pred = np.zeros((3, nao, nao), dtype=complex)
    for t in range(3):
        for a in range(3):
            if which == "u":
                dR = R[ao_atom][:, None, a] * np.ones((1, nao))
            else:
                dR = R[ao_atom][None, :, a] * np.ones((nao, 1))
            for b in range(3):
                pred[t] += EPS[t, a, b] * dR * r_int[b]
    num = np.vdot(pred.ravel(), igovlp.ravel())
    den = np.vdot(pred.ravel(), pred.ravel())
    c = num / den
    print(f"      {tag:12s} best c = {c.real:+.6f}{c.imag:+.6f}j   "
          f"resid = {np.max(np.abs(igovlp - c * pred)):.3e}")

# ============================================================================
# 2.  component ordering of the 9-component integrals
# ============================================================================
print()
print("=" * 74)
print("  2.  is the 9-component ordering (t,s) or (s,t)?")
print("      A symmetric operator stays symmetric under either read; an")
print("      asymmetric one flips.  Look at int1e_a01gp and a11part.")
print("=" * 74)
with mol.with_rinv_orig(R[0]):
    a11 = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
    a01 = mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)
for tag, m in (("a11part", a11), ("a01gp", a01)):
    asym = np.max(np.abs(m - m.transpose(1, 0, 2, 3)))
    asym_fro = np.linalg.norm(m - m.transpose(1, 0, 2, 3))
    sym_fro = np.linalg.norm(m + m.transpose(1, 0, 2, 3))
    print(f"      {tag:8s} ||M - M^T|| = {asym_fro:12.6f}   "
          f"||M + M^T|| = {sym_fro:12.6f}   max|M-M^T| = {asym:.6f}")
    print(f"               -> {'SYMMETRIC' if asym_fro < 0.01 * sym_fro else 'asymmetric'}")

# ============================================================================
# 3.  the full numerical GIAO shielding, with diagnostics
# ============================================================================
print()
print("=" * 74)
print("  3.  numerical GIAO shielding with an SCF-convergence readout")
print("=" * 74)


def build(mol):
    nao, natm = mol.nao_nr(), mol.natm
    h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    s0 = mol.intor("int1e_ovlp")
    eri0 = mol.intor("int2e")
    h_b = -1j * (0.5 * mol.intor("int1e_giao_irjxp")
                 + mol.intor("int1e_ignuc") + mol.intor("int1e_igkin"))
    s_b = -1j * mol.intor("int1e_igovlp")
    ig1 = mol.intor("int2e_ig1")
    eri_b = -1j * (ig1 + ig1.transpose(0, 3, 4, 1, 2))
    h_m = np.zeros((natm, 3, nao, nao), dtype=complex)
    h_bm = np.zeros((natm, 3, 3, nao, nao))
    R = mol.atom_coords()
    for a in range(natm):
        with mol.with_rinv_orig(R[a]):
            h_m[a] = -1j * mol.intor("int1e_ia01p")
            aa = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
            h_bm[a] = aa - np.einsum("ts,uv->tsuv", np.eye(3),
                                     aa.trace(axis1=0, axis2=1))
            h_bm[a] += mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)
    return dict(h0=h0, s0=s0, eri0=eri0, h_b=h_b, s_b=s_b, eri_b=eri_b,
                h_m=h_m, h_bm=h_bm, R=R)


def scf_at(mol, D, bvec, dm0=None):
    bvec = np.asarray(bvec, dtype=float)
    m = scf.RHF(mol)
    m.verbose = 0
    m.chkfile = None
    m.max_cycle = 200
    m.conv_tol = 1e-13
    m.get_hcore = lambda *a: D["h0"] - 1j * np.einsum("t,tuv->uv", bvec, D["h_b"])
    m.get_ovlp = lambda *a: D["s0"] - 1j * np.einsum("t,tuv->uv", bvec, D["s_b"])
    m._eri = D["eri0"] - 1j * np.einsum("t,tuvkl->uvkl", bvec, D["eri_b"])
    if dm0 is not None:
        m.kernel(dm0)
    else:
        m.kernel()
    return m


def shieldings(mol, D, dB=1e-4, verbose=False):
    nao, natm = mol.nao_nr(), mol.natm
    global m0
    m0 = scf_at(mol, D, [0.0, 0.0, 0.0])
    dm0 = m0.make_rdm1()
    sig = np.zeros((natm, 3, 3))
    for t in range(3):
        bp = [0.0, 0.0, 0.0]; bp[t] = dB
        bm = [0.0, 0.0, 0.0]; bm[t] = -dB
        mp_ = scf_at(mol, D, bp, dm0)
        mm_ = scf_at(mol, D, bm, dm0)
        if verbose:
            # E(B) must be even: a closed shell has no linear Zeeman term.
            # Cross-check e_tot against 1/2 Tr(D(h+F)) built by hand, because
            # e_tot for a complex Hamiltonian is worth distrusting.
            dp, dm_ = mp_.make_rdm1(), mm_.make_rdm1()
            ep = 0.5 * np.einsum("uv,vu->", dp, mp_.get_hcore() + mp_.get_fock(dp)).real
            em = 0.5 * np.einsum("uv,vu->", dm_, mm_.get_hcore() + mm_.get_fock(dm_)).real
            print(f"      B_{'xyz'[t]}: converged {mp_.converged}/{mm_.converged}"
                  f"   E(+)-E(-) = {mp_.e_tot.real - mm_.e_tot.real:+.3e}"
                  f"   manual = {ep - em:+.3e}"
                  f"   (E0 ref {m0.e_tot.real:.8f})")
        dmp, dmm = mp_.make_rdm1(), mm_.make_rdm1()
        for a in range(natm):
            for s in range(3):
                hp = D["h_m"][a][s] + sum(bp[k] * D["h_bm"][a][k][s] for k in range(3))
                hm = D["h_m"][a][s] + sum(bm[k] * D["h_bm"][a][k][s] for k in range(3))
                gp = np.einsum("uv,vu->", dmp, hp).real
                gm = np.einsum("uv,vu->", dm, hm).real
                sig[a, t, s] = (gp - gm) / (2 * dB)
    return sig, m0


Dw = build(mol)
sig_w, m0w = shieldings(mol, Dw, dB=1e-4, verbose=True)
print(f"      E0 = {m0w.e_tot.real:.10f}  converged={m0w.converged}")

# ammonia
nh3 = gto.M(atom=[("N", 0.0, 0.0, 0.0)] + [
    ("H", 0.93748 * np.cos(2 * np.pi * k / 3),
     0.93748 * np.sin(2 * np.pi * k / 3), 0.38101) for k in range(3)],
    basis="sto-3g", unit="Angstrom", verbose=0)
Dn = build(nh3)
sig_n, m0n = shieldings(nh3, Dn, dB=1e-4, verbose=True)
print(f"      E0 = {m0n.e_tot.real:.10f}  converged={m0n.converged}")

print()
print("=" * 74)
print("  raw results (recipe as documented) -- ppm")
print("=" * 74)
for tag, m, sig in (("H2O", mol, sig_w), ("NH3", nh3, sig_n)):
    print(f"  --- {tag} ---")
    for a in range(m.natm):
        s = sig[a]
        iso = np.trace(s) / 3 * PPM
        asym = np.max(np.abs(s - s.T))
        eigs = np.linalg.eigvalsh((s + s.T) / 2) * PPM
        print(f"    {m.atom_symbol(a)}{a}: iso = {iso:+9.3f}   "
              f"aniso = {(eigs[2] - 0.5 * (eigs[0] + eigs[1])):+9.3f}   "
              f"max|asym| = {asym:.3e}   (raw {asym * PPM:.3f} ppm)")

print()
print("=" * 74)
print("  literature, gas phase, absolute shielding")
print("    H2O  1H  30.7      17O  344")
print("    NH3  1H  30.7-31.0 14N  264      RHF/STO-3G 14N ~ 292")
print("=" * 74)
print(f"    H2O 1H  -> {-np.trace(sig_w[1]) / 3 * PPM:+9.3f} ppm  (sign flipped)")
print(f"    H2O 17O -> {-np.trace(sig_w[0]) / 3 * PPM:+9.3f} ppm  (sign flipped)")
print(f"    NH3 1H  -> {-np.trace(sig_n[1]) / 3 * PPM:+9.3f} ppm  (sign flipped)")
print(f"    NH3 14N -> {-np.trace(sig_n[0]) / 3 * PPM:+9.3f} ppm  (sign flipped)")

# ============================================================================
# 4.  step-size stability
# ============================================================================
print()
print("=" * 74)
print("  4.  is the result stable against the finite-difference step?")
print("=" * 74)
for dB in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
    sw, _ = shieldings(mol, Dw, dB=dB)
    h1 = -np.trace(sw[1]) / 3 * PPM
    o17 = -np.trace(sw[0]) / 3 * PPM
    asym = np.max(np.abs(sw[1] - sw[1].T)) * PPM
    print(f"    dB={dB:8.1e}   1H = {h1:9.4f}   17O = {o17:9.4f}   "
          f"max|asym| = {asym:.4f} ppm")
