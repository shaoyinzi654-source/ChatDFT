"""Round-15 probe: how do we get a polarizability out of this PySCF build?

`pyscf.prop` does not exist here and `dft.RKS` has no `polarizability()`, so
the Raman spectrum -- which needs da/dQ for every normal mode -- has to be
built from something lower level.  Two candidate routes:

  (A) CPHF.  pyscf.scf.cphf.solve exists.  Cheap: three solves per geometry.
  (B) Finite field.  Add E.r to the core Hamiltonian and difference the SCF
      energy.  Expensive (18 SCFs per geometry for the full tensor) but it
      cannot be wrong in a subtle way, because it is the definition.

They are computed by completely different machinery, so agreeing to four
digits is a real cross-check rather than a consistency check.

The literature anchor: water's isotropic polarizability.  Experimental
1.47 A^3 = 9.90 a.u.; B3LYP/aug-cc-pVTZ about 9.7 a.u.; a 6-31G* basis is
polarization-limited and should come in a little low.

Run:  python probes/probe_raman.py
"""
from __future__ import annotations

import math
import time

import numpy as np

import backend.bootstrap as bootstrap

bootstrap.setup()

from pyscf import dft, gto

WATER = [
    ("O", 0.000000, 0.000000, 0.119262),
    ("H", 0.000000, 0.763239, -0.477047),
    ("H", 0.000000, -0.763239, -0.477047),
]

BOHR3_TO_A3 = 0.52917721092 ** 3          # 0.148185


def make_mol(xyz, basis="6-31g*"):
    return gto.M(atom=xyz, basis=basis, verbose=0)


def scf(mol, xc="b3lyp", field=None, dm0=None):
    """One SCF, optionally in a uniform electric field (atomic units)."""
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.conv_tol = 1e-11
    mol.chkfile = None
    mf.chkfile = None
    if field is not None:
        h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        dip = mol.intor("int1e_r")        # <mu| r |nu>, (3, nao, nao)
        # V = -mu.E with mu_el = -sum r  =>  H gains +E.sum r
        hf_ = h0 + np.einsum("x,xij->ij", np.asarray(field, float), dip)
        mf.get_hcore = lambda *a, _h=hf_: _h
    if dm0 is not None:
        mf.kernel(dm0=dm0)
    else:
        mf.kernel()
    return mf


def alpha_finite_field(mol, xc="b3lyp", h=1e-3, dm0=None):
    """alpha_ij = -d2E/dE_i dE_j, central differences.

    The nuclear term -E.sum(Z_A R_A) is linear in E, so it contributes
    nothing to the second derivative and mf.e_tot can be used directly.
    """
    a = np.zeros((3, 3))
    e0 = scf(mol, xc, None, dm0).e_tot
    for i in range(3):
        for j in range(i, 3):
            pts = []
            for si in (1, -1):
                for sj in (1, -1):
                    f = np.zeros(3)
                    f[i] += si * h
                    f[j] += sj * h
                    pts.append((si, sj, scf(mol, xc, f, dm0).e_tot))
            d = {(si, sj): e for si, sj, e in pts}
            a[i, j] = a[j, i] = -(
                d[(1, 1)] - d[(1, -1)] - d[(-1, 1)] + d[(-1, -1)]
            ) / (4 * h * h)
    return a, e0


def alpha_cphf(mol, xc="b3lyp", dm0=None):
    """alpha from the coupled-perturbed Hartree-Fock/Kohn-Sham equations.

    For a closed-shell SCF the dipole perturbation obeys
        (A + B) x_i = -h_i^{ov}
    and the polarizability is the induced dipole per unit field,
        alpha_ij = -2 sum_{ia} h_i^{ia} x_j^{ia} / ... (see the docstring of
    pyscf.prop.polarizability, which is not in this build).
    """
    from pyscf.scf import cphf

    mf = scf(mol, xc, None, dm0)
    mo_e = mf.mo_energy
    mo_c = mf.mo_coeff
    mo_occ = mf.mo_occ
    occ = mo_occ > 0
    vir = mo_occ == 0
    nocc, nvir = int(occ.sum()), int(vir.sum())
    if mf.mo_coeff.ndim != 2:
        raise RuntimeError("open shell: CPHF polarizability not implemented")

    dip_ao = mol.intor("int1e_r")                     # (3, nao, nao)
    dip_mo = np.einsum("xip,pq,qj->xij", mo_c[:, occ].T, dip_ao,
                       mo_c[:, vir], optimize=True)
    # h^{ia}: the field couples as +E.r, so the perturbation is +dip
    h1 = dip_mo.reshape(3, nocc * nvir)

    def fvind(x):
        x = x.reshape(3, nocc, nvir)
        v = mf.get_veff(mol, mf.make_rdm1(mo_c, mo_occ)) * 0
        dm = np.einsum("xij,jp,qi->xpq", x, mo_c[:, vir], mo_c[:, occ],
                       optimize=True)
        dm = dm + dm.transpose(0, 2, 1)
        v = mf.get_veff(mol, dm)
        v = np.einsum("qi,xpq,pj->xij", mo_c[:, occ], v, mo_c[:, vir],
                      optimize=True)
        return v.reshape(3, nocc * nvir)

    x = cphf.solve(fvind, mo_e, mo_occ, h1, max_cycle=100,
                   tol=1e-10, verbose=0)[0]
    x = x.reshape(3, nocc, nvir)
    a = -2.0 * np.einsum("xia,xia->ij", dip_mo, x, optimize=True) \
        if False else -2.0 * np.einsum("xia,jia->ij", dip_mo, x, optimize=True)
    return a, mf.e_tot


print("=" * 74)
print("PART 1 -- water polarizability, two independent routes")
print("=" * 74)
mol = make_mol(WATER)

t0 = time.time()
a_ff, e0 = alpha_finite_field(mol, "b3lyp", h=1e-3)
t_ff = time.time() - t0
print(f"  finite field ({t_ff:.1f}s, 19 SCFs)")
print(np.array2string(a_ff, precision=4, suppress_small=True))
print(f"  isotropic = {np.trace(a_ff) / 3:.4f} a.u. "
      f"= {np.trace(a_ff) / 3 * BOHR3_TO_A3:.4f} A^3")

try:
    t0 = time.time()
    a_cp, e1 = alpha_cphf(mol, "b3lyp")
    t_cp = time.time() - t0
    print(f"\n  CPHF ({t_cp:.1f}s, 1 SCF + 1 solve)")
    print(np.array2string(a_cp, precision=4, suppress_small=True))
    print(f"  isotropic = {np.trace(a_cp) / 3:.4f} a.u. "
          f"= {np.trace(a_cp) / 3 * BOHR3_TO_A3:.4f} A^3")
    denom = np.abs(a_ff).max()
    print(f"\n  max |difference| / max |alpha| = "
          f"{np.abs(a_cp - a_ff).max() / denom:.3e}")
    print(f"  energy agreement: {abs(e0 - e1):.2e} hartree")
except Exception as exc:
    import traceback
    traceback.print_exc()
    print("  CPHF route failed:", type(exc).__name__, exc)

print()
print("=" * 74)
print("PART 2 -- what the finite-field step size does")
print("=" * 74)
for h in (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4):
    a, _ = alpha_finite_field(mol, "b3lyp", h=h)
    print(f"  h = {h:8.1e}   alpha_iso = {np.trace(a) / 3:12.8f} a.u.")

print()
print("=" * 74)
print("PART 3 -- the physics a Raman spectrum has to reproduce")
print("=" * 74)
print("  Water, B3LYP/6-31G*: the symmetric O-H stretch is the strongest")
print("  line in the Raman spectrum, while in the IR it is the WEAKEST of")
print("  the three (bend 79.6, sym stretch 1.8, asym stretch 20.1 km/mol).")
print("  A Raman implementation that reuses the IR intensities, or that")
print("  mis-projects the tensor onto the modes, gets this backwards.")
print()
print("  Depolarization ratios are the sharp test: for linearly polarised")
print("  incident light rho = 3*gamma'^2/(45*abar'^2 + 4*gamma'^2) and can")
print("  never exceed 0.75.  A totally symmetric mode is polarised")
print("  (rho well below 0.75); a degenerate/antisymmetric one sits at 0.75.")
