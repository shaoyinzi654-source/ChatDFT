"""probe_nmr3.py -- can this build run the GIAO field-perturbed SCF?

The GIAO field perturbation is *imaginary*: h(B) = h0 - i B_t O^t, with S and the
ERIs perturbed the same way.  A numerical GIAO shielding only needs 6 SCF runs
(3 field directions x +/-) plus a Hellmann-Feynman expectation value, which is
cheap and -- crucially -- handles the overlap derivative and the ERI derivative
automatically, because they are simply part of the Hamiltonian.

But it needs complex arithmetic.  Find out whether PySCF's RHF will do it here.

Recipe taken from the validated derivation at
ajz34.readthedocs.io/QC_Notes/Prop_Series/NMR_GIAO_NumDeriv (agrees with
Gaussian to 0.1 ppm on NH3/STO-3G).
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf

np.set_printoptions(precision=6, suppress=True, linewidth=160)

# ----------------------------------------------------------------------------
# molecule
# ----------------------------------------------------------------------------
mol = gto.M(
    atom=[("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587), ("H", 0.0, 0.757, 0.587)],
    basis="sto-3g", unit="Angstrom", verbose=0)
nao = mol.nao_nr()
natm = mol.natm
print(f"water  nao={nao}  natm={natm}")

# ----------------------------------------------------------------------------
# unperturbed reference
# ----------------------------------------------------------------------------
h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
s0 = mol.intor("int1e_ovlp")
eri0 = mol.intor("int2e")

mf = scf.RHF(mol)
mf.verbose = 0
mf.chkfile = None
e0 = mf.kernel()
print(f"E0 = {e0:.10f}   converged={mf.converged}")

# ----------------------------------------------------------------------------
# the GIAO first-order integrals
# ----------------------------------------------------------------------------
# field derivative of the core Hamiltonian
h_b = -1j * (0.5 * mol.intor("int1e_giao_irjxp")
             + mol.intor("int1e_ignuc")
             + mol.intor("int1e_igkin"))
# field derivative of the overlap
s_b = -1j * mol.intor("int1e_igovlp")
# field derivative of the ERIs:  (uv|kl)^t = ig1[t,uvkl] + ig1[t,kluv]
ig1 = mol.intor("int2e_ig1")
eri_b = -1j * (ig1 + np.einsum("tkluv->tuvkl", ig1))

# nuclear magnetic moment derivative, one origin per nucleus
h_m = np.zeros((natm, 3, nao, nao), dtype=complex)
h_bm = np.zeros((natm, 3, 3, nao, nao), dtype=float)
for a in range(natm):
    with mol.with_rinv_orig(mol.atom_coord(a)):
        h_m[a] = -1j * mol.intor("int1e_ia01p")
        a11 = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
        h_bm[a] = a11 - np.einsum("ts,uv->tsuv", np.eye(3), a11.trace(axis1=0, axis2=1))
        h_bm[a] += mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)

print(f"\nh_b   hermiticity max|h_b - h_b^H| = "
      f"{np.max(np.abs(h_b - h_b.conj().transpose(0, 2, 1))):.3e}")
print(f"s_b   hermiticity max|s_b - s_b^H| = "
      f"{np.max(np.abs(s_b - s_b.conj().transpose(0, 2, 1))):.3e}")
print(f"h_m[O] hermiticity                  = "
      f"{np.max(np.abs(h_m[0] - h_m[0].conj().transpose(0, 2, 1))):.3e}")
print(f"h_bm  is real: max|imag| = {np.max(np.abs(h_bm.imag)) if np.iscomplexobj(h_bm) else 0.0:.3e}")


# ----------------------------------------------------------------------------
# a field-perturbed SCF
# ----------------------------------------------------------------------------
def scf_at_field(bvec):
    """Converge RHF with h, S and the ERIs perturbed by the field bvec."""
    bvec = np.asarray(bvec, dtype=float)

    def get_hcore(mol_=None):
        return h0 - 1j * np.einsum("t,tuv->uv", bvec, h_b)

    def get_ovlp(mol_=None):
        return s0 - 1j * np.einsum("t,tuv->uv", bvec, s_b)

    m = scf.RHF(mol)
    m.verbose = 0
    m.chkfile = None
    m.max_cycle = 100
    m.conv_tol = 1e-12
    m.get_hcore = get_hcore
    m.get_ovlp = get_ovlp
    m._eri = eri0 - 1j * np.einsum("t,tuvkl->uvkl", bvec, eri_b)
    m.kernel()
    return m


print()
print("=" * 74)
print("  does a complex SCF converge at a finite field?")
print("=" * 74)
dB = 1e-4
try:
    mz = scf_at_field([0.0, 0.0, dB])
    print(f"  converged = {mz.converged}   E = {mz.e_tot!r}")
    print(f"  |Im E| = {abs(mz.e_tot.imag):.3e}   (must be 0)")
    dm = mz.make_rdm1()
    print(f"  dm shape={dm.shape} dtype={dm.dtype}  "
          f"hermitian max|dm - dm^H| = {np.max(np.abs(dm - dm.conj().T)):.3e}")
except Exception as exc:
    print(f"  FAILED: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

print()
print("=" * 74)
print("  E(B) must be even in B: no linear Zeeman term for a closed shell")
print("=" * 74)
for t, name in enumerate("xyz"):
    bv = [0.0, 0.0, 0.0]
    bv[t] = dB
    ep = scf_at_field(bv).e_tot.real
    bv[t] = -dB
    em = scf_at_field(bv).e_tot.real
    bv[t] = 2 * dB
    e2 = scf_at_field(bv).e_tot.real
    lin = (ep - em) / (2 * dB)
    quad = (ep + em - 2 * e0) / (2 * dB ** 2)
    quad2 = (e2 + em - 2 * e0) / (2 * (2 * dB) ** 2)
    print(f"  B_{name}: dE/dB = {lin:+.3e} (want 0)   "
          f"d2E/dB2 = {quad:.6f} / {quad2:.6f} (want equal)")

print()
print("=" * 74)
print("  shielding by numerical field derivative of the Hellmann-Feynman")
print("  expectation value  g_A_s(B) = Re Tr(D(B) h^mu_As(B))")
print("=" * 74)
sigma = np.zeros((natm, 3, 3))
for t in range(3):
    bv = [0.0, 0.0, 0.0]
    bv[t] = dB
    mp_ = scf_at_field(bv)
    dm_p = mp_.make_rdm1()
    bv[t] = -dB
    mm_ = scf_at_field(bv)
    dm_m = mm_.make_rdm1()
    for a in range(natm):
        for s in range(3):
            # h^mu_As(B) = h_m[a][s] + sum_t B_t h_bm[a][t][s]
            hp = h_m[a][s] + sum(bv[k] * h_bm[a][k][s] for k in range(3))
            bv_m = [-x for x in bv]
            hm = h_m[a][s] + sum(bv_m[k] * h_bm[a][k][s] for k in range(3))
            gp = np.einsum("uv,vu->", dm_p, hp).real
            gm = np.einsum("uv,vu->", dm_m, hm).real
            sigma[a, t, s] = (gp - gm) / (2 * dB)

alpha = 1.0 / 137.035999084
print()
for a in range(natm):
    sig = sigma[a]
    iso = np.trace(sig) / 3.0
    eigs = np.linalg.eigvalsh((sig + sig.T) / 2)
    print(f"  {mol.atom_symbol(a)}{a}:  raw iso = {iso:+.6f}   "
          f"-> {iso * alpha ** 2 * 1e6:9.3f} ppm")
    print(f"      tensor (raw):\n{np.array2string(sig, prefix=' ' * 8)}")
    print(f"      eigenvalues: {eigs}  -> ppm {eigs * alpha ** 2 * 1e6}")
    print(f"      max|asym| = {np.max(np.abs(sig - sig.T)):.3e}")

print()
print("=" * 74)
print("  literature (gas phase, absolute shielding):")
print("    H2O  1H  ~ 30.7 ppm      17O  ~ 344 ppm   (experiment)")
print("=" * 74)
for a in range(natm):
    iso = np.trace(sigma[a]) / 3.0 * alpha ** 2 * 1e6
    print(f"    {mol.atom_symbol(a)}{a}: {iso:9.3f} ppm")
