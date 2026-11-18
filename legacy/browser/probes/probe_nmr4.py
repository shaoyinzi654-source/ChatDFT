"""probe_nmr4.py -- measure the GIAO integral conventions instead of guessing.

probe_nmr3 gave water 1H = -30.675 ppm against a literature +30.7: the
magnitude is right, so the chain is right, but there is a sign somewhere.  It
also showed a 1.7e-2 asymmetry in the H tensor and an implausible d2E/dB2,
which is either an SCF convergence problem or a wrong ERI derivative.

Rather than fit a sign to one number, pin down what each libcint GIAO integral
actually is:

  A. int1e_igovlp[t]  vs  the analytic GIAO phase derivative of the overlap
  B. int1e_pnucxp     vs  sum_A Z_A * int1e_ia01p(origin R_A)
  C. int1e_giao_irjxp vs  the GIAO orbital-Zeeman operator
  D. int2e_ig1        symmetry structure
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=8, suppress=True, linewidth=170)

mol = gto.M(
    atom=[("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587), ("H", 0.0, 0.757, 0.587)],
    basis="sto-3g", unit="Angstrom", verbose=0)
nao, natm = mol.nao_nr(), mol.natm
# aoslice_by_atom gives (shell0, shell1, ao0, ao1); use the AO range, because
# ao_loc_nr() is indexed per *shell*, not per atom.
sl = mol.aoslice_by_atom()
ao_atom = np.concatenate([np.full(int(b) - int(a), i)
                          for i, (a, b) in enumerate(sl[:, 2:4])])
R = mol.atom_coords()          # (natm, 3) in bohr
print(f"nao={nao} natm={natm}")
print(f"ao_atom = {ao_atom}")

eps = np.zeros((3, 3, 3))
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

print()
print("=" * 74)
print("  A.  is  int1e_igovlp[t]  ==  the analytic GIAO phase derivative?")
print("      S_uv(B) = <chi_u| exp(i/2 B.(R_u-R_v) x r) |chi_v>")
print("      dS/dB_t(0) = i/2 eps_tab (R_u-R_v)_a <chi_u| r_b |chi_v>")
print("=" * 74)
igovlp = mol.intor("int1e_igovlp")
r_int = mol.intor("int1e_r")
for sign, tag in ((+1.0, "+i/2"), (-1.0, "-i/2")):
    pred = np.zeros((3, nao, nao), dtype=complex)
    for t in range(3):
        for a in range(3):
            dR = (R[ao_atom][:, None, :] - R[ao_atom][None, :, :])[:, :, a]
            for b in range(3):
                pred[t] += (sign * 0.5j * eps[t, a, b] * dR * r_int[b])
    err = np.max(np.abs(igovlp - pred))
    print(f"      {tag}: max|int1e_igovlp - pred| = {err:.3e}")
print(f"      scale check: max|igovlp| = {np.max(np.abs(igovlp)):.6f}")

print()
print("=" * 74)
print("  B.  is  int1e_pnucxp  ==  sum_A Z_A * int1e_ia01p(origin R_A)?")
print("=" * 74)
pnucxp = mol.intor("int1e_pnucxp")
acc = np.zeros((3, nao, nao), dtype=complex)
for a in range(natm):
    with mol.with_rinv_orig(R[a]):
        ia01p = mol.intor("int1e_ia01p")
    acc += mol.atom_charge(a) * ia01p
print(f"      max|pnucxp - sum Z_A ia01p| = {np.max(np.abs(pnucxp - acc)):.3e}")
print(f"      max|pnucxp| = {np.max(np.abs(pnucxp)):.6f}  "
      f"max|sum| = {np.max(np.abs(acc)):.6f}")
ratio = pnucxp / np.where(np.abs(acc) > 1e-12, acc, 1.0)
print(f"      ratio on large elements: "
      f"{np.median(ratio[np.abs(acc) > 1e-3].real):.6f}")

print()
print("=" * 74)
print("  C.  what is int1e_giao_irjxp?  compare with int1e_cg_irxp and")
print("      the origin-shifted (r-R_u) x p construction")
print("=" * 74)
irjxp = mol.intor("int1e_giao_irjxp")
cg = mol.intor("int1e_cg_irxp")
print(f"      max|giao_irjxp| = {np.max(np.abs(irjxp)):.6f}")
print(f"      max|cg_irxp|     = {np.max(np.abs(cg)):.6f}")
print(f"      max|giao_irjxp - cg_irxp| = {np.max(np.abs(irjxp - cg)):.3e}")
# <u| (r - R_u) x p |v>  built from int1e_cg_irxp and int1e_r and p
# p = -i d/dr ; <u|r x p|v> = int1e_cg_irxp  (real).  Shifting r by R_u
# gives <u|R_u x p|v> which needs the momentum integral.
p_int = mol.intor("int1e_ipovlp")           # d/dr overlap, (3, nao, nao)
for t in range(3):
    for a in range(3):
        for b in range(3):
            pass
# <chi_u| R_u x p |chi_v>_t = eps_tab R_u,a p_b   -- but p acts on the ket,
# so the AO-basis matrix is  -i * int1e_ipovlp  with R_u of the ROW index.
shift = np.zeros((3, nao, nao), dtype=complex)
for t in range(3):
    for a in range(3):
        for b in range(3):
            shift[t] += eps[t, a, b] * R[ao_atom][:, None] * (-1j * p_int[b])
print(f"      max|cg_irxp - shift| = {np.max(np.abs(cg - shift)):.3e}")
print(f"      max|giao_irjxp - (cg - shift)| = "
      f"{np.max(np.abs(irjxp - (cg - shift))):.3e}")
print(f"      max|giao_irjxp - (cg + shift)| = "
      f"{np.max(np.abs(irjxp - (cg + shift))):.3e}")

print()
print("=" * 74)
print("  D.  int2e_ig1 symmetry")
print("=" * 74)
ig1 = mol.intor("int2e_ig1")
print(f"      shape = {ig1.shape}")
print(f"      max|ig1[t,uvkl] - ig1[t,vukl]|  (uv swap) = "
      f"{np.max(np.abs(ig1 - ig1.transpose(0, 2, 1, 3, 4))):.3e}")
print(f"      max|ig1[t,uvkl] - ig1[t,kluv]|  (pair swap) = "
      f"{np.max(np.abs(ig1 - ig1.transpose(0, 3, 4, 1, 2))):.3e}")
print(f"      max|ig1[t,uvkl] - ig1[t,uvlk]|  (kl swap) = "
      f"{np.max(np.abs(ig1 - ig1.transpose(0, 1, 2, 4, 3))):.3e}")
# the derivative of the GIAO ERI must be symmetric under u<->v and k<->l
eri_b = -1j * (ig1 + ig1.transpose(0, 3, 4, 1, 2))
print(f"      assembled eri_b: max|uv swap| = "
      f"{np.max(np.abs(eri_b - eri_b.transpose(0, 2, 1, 3, 4))):.3e}  "
      f"max|pair swap| = "
      f"{np.max(np.abs(eri_b - eri_b.transpose(0, 3, 4, 1, 2))):.3e}")

print()
print("=" * 74)
print("  E.  does int2e_ig1 match a finite-difference of the GIAO ERI?")
print("      The GIAO ERI is  (uv|kl) with each chi carrying exp(i/2 B.R x r).")
print("      Build the phase explicitly on a grid for the smallest basis")
print("      block and compare.  (Only the 1s block of the two H atoms.)")
print("=" * 74)
# Use a two-atom H2 so that everything is 1s: the phase is then exactly
# exp(i/2 B.(R_u - R_v).r) and the derivative is analytic.
h2 = gto.M(atom=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
           basis="sto-3g", unit="Angstrom", verbose=0)
b1 = h2.intor("int1e_ovlp")
r1 = h2.intor("int1e_r")
Rh = h2.atom_coords()
ig1h = h2.intor("int1e_igovlp")
slh = h2.aoslice_by_atom()
ao_atom_h = np.concatenate([np.full(int(b) - int(a), i)
                            for i, (a, b) in enumerate(slh[:, 2:4])])
pred_h = np.zeros((3, 2, 2), dtype=complex)
for t in range(3):
    for a in range(3):
        dR = (Rh[ao_atom_h][:, None, :] - Rh[ao_atom_h][None, :, :])[:, :, a]
        for b in range(3):
            pred_h[t] += 0.5j * eps[t, a, b] * dR * r1[b]
print(f"      H2 overlap: max|igovlp - pred(+i/2)| = "
      f"{np.max(np.abs(ig1h - pred_h)):.3e}")
print(f"                  max|igovlp| = {np.max(np.abs(ig1h)):.6f}")
print(f"                  max|pred|   = {np.max(np.abs(pred_h)):.6f}")

# Now the ERI: for H2/sto-3g all four basis functions are 1s.
# (uv|kl) with GIAO phases = <u v| 1/r12 |k l> with phases on each chi.
# d/dB_t at 0  =  i/2 [ (R_u - R_k - R_l ... ) ] -- messy but the KEY point is
# that the correct derivative must vanish when all four centres coincide.
# Test the atomic limit instead: a single atom has no GIAO phase at all, so
# int2e_ig1 must be zero there.
he = gto.M(atom=[("He", 0.0, 0.0, 0.0)], basis="sto-3g", verbose=0)
print(f"      single-centre int2e_ig1 max|.| = "
      f"{np.max(np.abs(he.intor('int2e_ig1'))):.3e}  (must be 0)")
