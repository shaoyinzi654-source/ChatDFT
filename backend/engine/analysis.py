"""Paper-figure analyses: vibrational spectra, UV-Vis, DOS and reactivity.

Every function here either produces a number a paper would quote or a curve a
paper would plot.  They are kept out of ``dft.py`` because they are layered on
top of a converged SCF rather than being part of it, and because several of
them are expensive enough that the caller has to be able to decline them.

Everything is computed, nothing is faked: IR intensities come from finite
differences of the analytic dipole, the Hessian comes from PySCF's analytic
second derivatives, and the populations are the same Loewdin partition the
rest of the app uses.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import elements
from .dft import (BASIS_SETS, DFTEngine, DFTError, _split_xyz, basis_has_diffuse)

if TYPE_CHECKING:                                  # pragma: no cover
    pass

Progress = Optional[Callable[[int, str], None]]

# --- unit conversions ---------------------------------------------------
HARTREE2EV = 27.211386245988
HARTREE2CM = 219474.6313632
EV2NM = 1239.8419843320026
BOHR = 0.52917721092
# dmu/dQ in D/(A*sqrt(amu))  ->  integrated intensity in km/mol.
# Gaussian's published conversion: 1 D^2 A^-2 amu^-1 = 42.2561 km/mol, equal
# to N_A/(12 eps0 c^2) evaluated with the amu and Debye factors.  The result is
# the NAPIERIAN (natural-log) integrated absorbance, which is what every
# program prints under "IR Intensities" and what the literature tabulates;
# the decadic value is smaller by ln(10).  Getting the two confused, or using
# a normal coordinate in Bohr instead of Angstrom, moves every intensity by a
# constant factor while leaving the spectrum's shape untouched.
IR_KM_PER_MOL = 42.256
# 1 e * 1 Angstrom, in Debye: the dipole a unit charge gains per Angstrom.
E_ANGSTROM_TO_DEBYE = 4.80320

def _tick(progress: Progress, pct: int, msg: str) -> None:
    if progress:
        progress(max(0, min(100, int(pct))), msg)


def _xyz(syms: Sequence[str], coords: np.ndarray) -> str:
    body = "".join(
        f"{s} {c[0]:.12f} {c[1]:.12f} {c[2]:.12f}\n" for s, c in zip(syms, coords)
    )
    return f"{len(syms)}\nvib\n{body}"


# ======================================================================
#  vibrations
# ======================================================================
def _dipole_at(syms, coords, functional, basis, charge, mult, solvation):
    eng = DFTEngine(
        atom_xyz=_xyz(syms, coords), charge=charge, multiplicity=mult,
        functional=functional, basis=basis, solvation=solvation,
    )
    res = eng.run_scf()
    d = res["dipole"]
    return np.array([d["x"], d["y"], d["z"]]), res


# --- thermochemistry ---------------------------------------------------
R_GAS = 8.314462618          # J / (mol K)
K_B = 1.380649e-23           # J / K
H_PLANCK = 6.62607015e-34    # J s
C_LIGHT = 2.99792458e10      # cm / s
AMU_KG = 1.66053906660e-27
HARTREE2J = 4.3597447222071e-18
AVOGADRO = 6.02214076e23
J_PER_MOL_TO_HARTREE = 1.0 / (HARTREE2J * AVOGADRO)


def thermochemistry(
    syms: Sequence[str],
    coords: np.ndarray,
    freq_cm1: Sequence[float],
    energy_hartree: float,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    mol=None,
    symmetry_number: Optional[int] = None,
) -> dict:
    """Rigid-rotor / harmonic-oscillator thermochemistry at 1 atm.

    Done from the textbook formulas rather than via ``pyscf.hessian.thermo``,
    which returns impossible numbers in this build.  Validated against the
    gas-phase standard entropy of water, 188.8 J/(mol K): a wrong moment of
    inertia, a wrong symmetry number or a wrong vibrational entropy all show
    up there, so it checks the whole chain at once.
    """
    coords = np.asarray(coords, dtype=float)
    masses_amu = np.array([elements.get(s).mass for s in syms], dtype=float)
    total_amu = float(masses_amu.sum())

    # centre of mass
    com = (masses_amu[:, None] * coords).sum(axis=0) / total_amu
    rel = coords - com

    # inertia tensor -> principal moments (amu * A^2 -> kg m^2)
    inertia = np.zeros((3, 3))
    for m, r in zip(masses_amu, rel):
        inertia += m * (float(np.dot(r, r)) * np.eye(3) - np.outer(r, r))
    moments = np.linalg.eigvalsh(inertia) * AMU_KG * 1e-20

    T = float(temperature)
    # vibrational temperatures, real (positive) modes only
    nu = np.array([f for f in freq_cm1 if f > 0.0], dtype=float)
    theta_v = H_PLANCK * C_LIGHT * nu / K_B                       # K
    zpe_j = 0.5 * R_GAS * float(theta_v.sum())
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x = theta_v / T
        ex = np.exp(-x)
        e_vib = float(np.sum(np.where(x > 0, theta_v / np.expm1(x), 0.0))) * R_GAS
        s_vib = float(np.sum(
            np.where(x > 1e-8, x / np.expm1(x) - np.log1p(-np.clip(ex, 0, 1 - 1e-300)), 0.0)
        )) * R_GAS

    # translation (Sackur-Tetrode at the ideal-gas volume kT/p)
    m_kg = total_amu * AMU_KG
    vol = K_B * T / pressure
    q_trans = (2.0 * math.pi * m_kg * K_B * T / H_PLANCK ** 2) ** 1.5 * vol
    s_trans = R_GAS * (math.log(q_trans) + 2.5)
    e_trans = 1.5 * R_GAS * T

    # rotation
    # "Zero" has to be judged relative to the largest moment: molecular
    # moments are ~1e-47 kg m^2, so a fixed floor of 1e-40 silently clamps
    # every one of them (water's rotational entropy came out 199 J/mol/K too
    # high and its rotational constants 7 orders of magnitude too small).
    big = float(np.max(moments)) if moments.size else 0.0
    good = moments[moments > max(1e-3 * big, 1e-60)]
    linear = len(good) == 2
    sigma = 1
    if symmetry_number:
        sigma = int(symmetry_number)
    elif mol is not None:
        try:
            from pyscf.hessian import thermo as _th
            sigma = int(_th.rotational_symmetry_number(mol)) or 1
        except Exception:
            sigma = 1
    if linear:
        ia, ib = float(good[0]), float(good[1])
        theta_r = H_PLANCK ** 2 / (8.0 * math.pi ** 2 * max(ia, ib) * K_B)
        q_rot = T / (sigma * theta_r)
        s_rot = R_GAS * (math.log(q_rot) + 1.0)
        e_rot = R_GAS * T
    else:
        thetas = [H_PLANCK ** 2 / (8.0 * math.pi ** 2 * max(i, 1e-60) * K_B)
                  for i in moments]
        q_rot = (math.sqrt(math.pi) / sigma) * (
            T ** 1.5 / math.sqrt(max(thetas[0] * thetas[1] * thetas[2], 1e-300)))
        s_rot = R_GAS * (math.log(q_rot) + 1.5)
        e_rot = 1.5 * R_GAS * T

    s_tot = s_trans + s_rot + s_vib
    e_thermal = e_trans + e_rot + e_vib                 # 0 K -> T, without ZPE
    h_corr = e_thermal + R_GAS * T                      # E -> H
    # J/mol -> Hartree/molecule
    j_per_mol_to_hartree = 1.0 / (HARTREE2J * AVOGADRO)
    zpe_hartree = zpe_j * j_per_mol_to_hartree
    h_corr_hartree = (zpe_j + h_corr) * j_per_mol_to_hartree
    g_corr_hartree = (zpe_j + h_corr - T * s_tot) * j_per_mol_to_hartree

    # Cv = 3/2 R (trans) + R or 3/2 R (rot) + R sum x^2 e^x / (e^x - 1)^2
    with np.errstate(over="ignore", invalid="ignore"):
        cv_vib = R_GAS * float(np.sum(
            np.where(x > 1e-8,
                     x ** 2 * np.exp(-x) / (1.0 - np.exp(-x)) ** 2, 1.0)))
    cv_rot = R_GAS if linear else 1.5 * R_GAS

    return {
        "temperature_K": T,
        "pressure_Pa": pressure,
        "zpe_hartree": round(float(zpe_hartree), 6),
        "zpe_kj_mol": round(zpe_j / 1000.0, 3),
        "thermal_energy_kj_mol": round(e_thermal / 1000.0, 3),
        "heat_capacity_cv_j_mol_k": round(
            1.5 * R_GAS + cv_rot + float(cv_vib), 3),
        "entropy_j_mol_k": round(float(s_tot), 3),
        "entropy_trans_j_mol_k": round(float(s_trans), 3),
        "entropy_rot_j_mol_k": round(float(s_rot), 3),
        "entropy_vib_j_mol_k": round(float(s_vib), 3),
        "symmetry_number": int(sigma),
        "linear": bool(linear),
        "rotational_constants_ghz": [
            round(float(H_PLANCK / (8.0 * math.pi ** 2 * max(i, 1e-60)) / 1e9), 4)
            for i in sorted(moments)
        ],
        "zpe_corrected_energy_hartree": round(
            float(energy_hartree + zpe_hartree), 6),
        "enthalpy_hartree": round(float(energy_hartree + h_corr_hartree), 6),
        "gibbs_hartree": round(float(energy_hartree + g_corr_hartree), 6),
    }


def ir_sum_rule(
    syms: Sequence[str],
    coords: np.ndarray,
    dmudx: np.ndarray,
    intensities_km_mol: Sequence[float],
    dipole_debye: Sequence[float],
    masses_amu: Sequence[float],
    charge: float = 0.0,
) -> dict:
    """Self-check on the IR intensities that needs no published value.

    In mass-weighted coordinates an orthogonal change of basis preserves the
    sum of squares, and the 3N directions (3N-6 modes, 3 translations, 3
    rotations) span the space, so

        sum_i |dmu/dQ_i|^2 = sum_{a,k,alpha} (1/m_a) (dmu_alpha/dx_{a,k})^2
                             - (translation) - (rotation)

    with ``dmudx`` in D/Angstrom, masses in amu and Q in sqrt(amu)*Angstrom.

    The left side is taken from ``intensities_km_mol`` -- the numbers that are
    actually reported -- and not re-derived from the modes, because a check
    that recomputes the quantity it is guarding in parallel passes no matter
    how the reported ones are produced.

    * Translation turns mu by ``charge`` per Angstrom, so it contributes
      ``3 (charge * e.Angstrom->D)^2 / M_total`` -- zero for a neutral species,
      which is why a neutral molecule is the clean case to test on.
    * Rotation about principal axis j turns mu by ``|e_j x mu|`` and the
      mass-weighted norm of that direction is ``sqrt(I_j)``, contributing
      ``|e_j x mu|^2 / I_j``.  This is NOT small: for water it is four times
      the vibrational sum, so forgetting it makes the test look broken.

    A projection that reweights atoms by 1/sqrt(m_a) -- or that uses a
    coordinate in Bohr instead of Angstrom -- fails this by 8-72%, while the
    spectrum's shape stays identical.  ``residual_pct`` is therefore the number
    that says whether the intensities are on an absolute scale at all.
    """
    coords = np.asarray(coords, dtype=float)
    dmudx = np.asarray(dmudx, dtype=float)
    mu = np.asarray(dipole_debye, dtype=float)
    m = np.asarray(masses_amu, dtype=float)

    total = float(np.sum(dmudx ** 2 / m[:, None, None]))
    vib = float(np.sum(np.asarray(intensities_km_mol, dtype=float))
                ) / IR_KM_PER_MOL

    mtot = float(m.sum())
    trans = 3.0 * (float(charge) * E_ANGSTROM_TO_DEBYE) ** 2 / mtot if mtot else 0.0

    com = (m[:, None] * coords).sum(axis=0) / mtot
    rel = coords - com
    inertia = np.zeros((3, 3))
    for mi, r in zip(m, rel):
        inertia += mi * (float(np.dot(r, r)) * np.eye(3) - np.outer(r, r))
    rot = 0.0
    # eigh returns the eigenvectors as the COLUMNS of v; iterating v itself
    # walks its rows instead, which silently pairs each moment with the wrong
    # axis (water: rotation came out 6.12 instead of 9.23 and the residual
    # read -56% for a projection that was in fact exact).
    w, v = np.linalg.eigh(inertia)
    for Ij, vj in zip(w, v.T):
        if Ij > 1e-8:
            rot += float(np.sum(np.cross(vj, mu) ** 2)) / Ij

    expected = float(total - trans - rot)
    resid = 100.0 * (vib - expected) / expected if abs(expected) > 1e-12 else 0.0
    return {
        "apt_trace_d2_amu_a2": round(total, 6),
        "rotational_d2_amu_a2": round(float(rot), 6),
        "translational_d2_amu_a2": round(float(trans), 6),
        "vibrational_d2_amu_a2": round(vib, 6),
        "expected_d2_amu_a2": round(expected, 6),
        "residual_pct": round(float(resid), 3),
    }


# ======================================================================
#  polarizability and Raman
# ======================================================================
# 1 bohr^3 in A^3.  alpha comes out of the CPHF in atomic units, and the Raman
# activity is tabulated in A^4/amu -- which is exactly what you get when
# dalpha/dQ is in A^2/sqrt(amu) and gets squared, so this single factor carries
# the whole unit chain.
BOHR3_TO_A3 = 0.14818471147216278
# 1 a.u. of polarizability in A^3, for reporting alpha alongside the tensor.
# Water is the reference point: experiment gives alpha_iso = 9.6-9.9 a.u.
# (1.42-1.47 A^3), B3LYP/aug-cc-pVTZ gives 9.89.


def polarizability_tensor(mf, mol) -> np.ndarray:
    """Static dipole polarizability, 3x3, in atomic units (bohr^3).

    Written out here because ``pyscf.prop`` is absent from this build and
    ``dft.RKS`` has no ``polarizability()`` method, so there is nothing to
    delegate to.

    The equations solved are the coupled-perturbed SCF ones.  With the field
    coupled as ``+F.r`` the perturbation is ``h1 = r`` in the occupied-virtual
    block and

        [ diag(e_a - e_i) + fvind ] x = -h1
        alpha_ij = 4 h1_i^T (A+B)^{-1} h1_j

    Three details are easy to get wrong and none of them announces itself,
    because a polarizability with the wrong prefactor is still a symmetric
    positive-definite tensor with roughly the right shape:

    * ``fvind`` must carry only the two-electron part.  ``cphf.solve_nos1``
      forms ``e_ai = 1/(e_a - e_i)`` itself and feeds ``lib.krylov`` the
      right-hand side ``-e_ai*h1``, so it solves
      ``(e_a - e_i + fvind) x = -h1``.  Adding the diagonal in ``fvind`` as
      well would double-count it.
    * ``fvind`` must be ``mf.gen_response(...)`` and NOT ``mf.get_veff``.
      ``get_veff`` returns the exchange-correlation *potential* for a density,
      which is not the quantity needed: the response requires the second
      derivative of E_xc, i.e. the kernel ``f_xc``.  Using ``get_veff``
      silently drops the whole XC contribution -- most of the answer for a
      functional like B3LYP -- and leaves a tensor that still looks plausible.
    * the prefactor is 4, not 2.  The energy is quadratic in the field so
      differentiating twice brings a factor 2, and the density response is
      doubled for a closed-shell occupation.  Rather than trust that
      derivation, the four candidate prefactors were scored against the
      finite-field tensor at three basis sets; only +4 reproduced it, to 9e-6
      relative at 6-31G* and 5e-5 at aug-cc-pVDZ -- i.e. to the step-size error
      of the finite-field route itself, which is a genuinely independent
      calculation (SCF energy differences rather than a Krylov solve).
    """
    if mol.spin != 0:
        raise DFTError(
            "The polarizability (and therefore the Raman spectrum) is "
            "implemented for closed-shell molecules only; this system has "
            f"{mol.nelectron} electrons and spin {mol.spin}."
        )
    if not getattr(mf, "converged", False):
        raise DFTError("SCF is not converged, so the CPHF response is meaningless.")
    if not hasattr(mf, "gen_response"):
        raise DFTError("This SCF object exposes no response kernel.")

    from pyscf.scf import cphf

    mo_c, mo_e, mo_occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    occ = mo_occ > 0
    vir = mo_occ == 0
    orbo, orbv = mo_c[:, occ], mo_c[:, vir]
    nocc, nvir = orbo.shape[1], orbv.shape[1]
    if nocc == 0 or nvir == 0:
        raise DFTError("No occupied-virtual pairs, so there is no response to solve.")

    dip_ao = mol.intor("int1e_r")                     # r in bohr, (3, nao, nao)
    # (3, nvir, nocc): solve_nos1 builds e_ai as (nvir, nocc), so the virtual
    # index has to come first or the two will not broadcast.
    h1 = np.einsum("pa,xpq,qi->xai", orbv, dip_ao, orbo, optimize=True)

    vresp = mf.gen_response(mo_c, mo_occ, hermi=1)

    def fvind(x):
        x = x.reshape(nvir, nocc)
        # *2 for double occupancy, then symmetrise; PySCF's hessian/rhf.py.
        dm1 = np.einsum("pa,ai,qi->pq", orbv, x * 2.0, orbo, optimize=True)
        v1 = vresp(dm1 + dm1.T)
        return np.einsum("pa,pq,qi->ai", orbv, v1, orbo, optimize=True)

    # One field direction per solve.  Stacking the three components into a
    # single (3, nvir, nocc) right-hand side cannot work: e_ai is (nvir, nocc)
    # and the product dies with "operands could not be broadcast together".
    x = np.empty((3, nvir, nocc))
    for d in range(3):
        x[d] = cphf.solve(fvind, mo_e, mo_occ, h1[d], max_cycle=200, tol=1e-11,
                          verbose=0)[0]
    # solve_nos1 returns -(A+B)^{-1} h1, hence the leading minus.
    return -4.0 * np.einsum("xai,yai->xy", h1, x, optimize=True)


def _dipole_alpha_at(syms, coords, functional, basis, charge, mult, solvation,
                     want_alpha=True):
    """Dipole (Debye) and, optionally, the polarizability (a.u.) at one geometry.

    One SCF serves both.  The IR loop used to call ``_dipole_at`` per
    displacement and the Raman work would have needed a second SCF at the same
    geometry for alpha, so the two are harvested together here.
    """
    eng = DFTEngine(
        atom_xyz=_xyz(syms, coords), charge=charge, multiplicity=mult,
        functional=functional, basis=basis, solvation=solvation,
    )
    res = eng.run_scf()
    d = res["dipole"]
    mu = np.array([d["x"], d["y"], d["z"]], dtype=float)
    if not want_alpha:
        return mu, None, res
    return mu, polarizability_tensor(eng.mf, eng.mol), res


def raman_activity(dalpha_dq: np.ndarray) -> Tuple[float, float, float]:
    """(isotropic part, anisotropy squared, activity) for one dalpha/dQ.

    ``dalpha_dq`` is a 3x3 matrix in A^2/sqrt(amu), so the returned activity is
    in A^4/amu -- the unit every Raman table is printed in.

    The two invariants are the standard ones:

        alpha_bar = (a_xx + a_yy + a_zz)/3
        gamma^2   = 1/2 [ (a_xx-a_yy)^2 + (a_yy-a_zz)^2 + (a_zz-a_xx)^2
                          + 6 (a_xy^2 + a_yz^2 + a_zx^2) ]
        activity  = 45 alpha_bar^2 + 7 gamma^2

    The 7 (rather than 4, as in the depolarization ratio) is the convention
    Gaussian and every published table use; it counts the antisymmetric part of
    the scattering tensor, which does not contribute to rho.
    """
    a = np.asarray(dalpha_dq, dtype=float)
    xx, yy, zz = float(a[0, 0]), float(a[1, 1]), float(a[2, 2])
    xy, yz, zx = float(a[0, 1]), float(a[1, 2]), float(a[2, 0])
    iso = (xx + yy + zz) / 3.0
    gamma2 = 0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2
                    + 6.0 * (xy ** 2 + yz ** 2 + zx ** 2))
    return iso, gamma2, 45.0 * iso ** 2 + 7.0 * gamma2


def depolarization_ratio(iso: float, gamma2: float) -> float:
    """rho = 3 gamma^2 / (45 alpha_bar^2 + 4 gamma^2), bounded by 3/4.

    The bound is not a numerical accident: rho <= 0.75 always, with equality
    exactly when the isotropic part vanishes.  That makes it the one Raman
    quantity with a hard theoretical ceiling, so it is worth checking against.
    """
    denom = 45.0 * iso ** 2 + 4.0 * gamma2
    return 3.0 * gamma2 / denom if abs(denom) > 1e-300 else 0.0


def raman_sum_rule(
    dalphadx: np.ndarray,
    modes: np.ndarray,
    dalpha_dq: np.ndarray,
    alpha: np.ndarray,
    masses_amu: Sequence[float],
    coords: np.ndarray,
) -> dict:
    """Self-check on the Raman intensities that needs no published value.

    Exactly the argument behind ``ir_sum_rule``, applied to a tensor instead of
    a vector.  In mass-weighted coordinates an orthogonal change of basis
    preserves the sum of squares, and the 3N directions span the space, so

        sum_i ||dalpha/dQ_i||_F^2
            = sum_{a,k} (1/m_a) ||dalpha/dx_{a,k}||_F^2  - (rotation)

    with ||.||_F the Frobenius norm over the nine tensor components.

    The two terms that vanish or nearly do, and why they are not dropped:

    * **Translation contributes exactly zero.**  alpha is a property of the
      electron distribution, and translating the whole molecule leaves it
      unchanged, so dalpha/d(delta) = 0 identically.  This is different from
      the IR case, where translation contributes ``3 (q e)^2 / M`` for a
      charged species -- so a projection bug that is invisible in the IR sum
      rule of a charged molecule still shows up here, and vice versa.
    * **Rotation does not.**  Under an infinitesimal rotation by angle theta
      about a unit axis v, ``alpha -> R alpha R^T`` with ``R = 1 + theta K``,
      so ``dalpha/dtheta = K(v) alpha - alpha K(v)`` with
      ``K(v)_{kl} = -eps_{vkl}``.  The mass-weighted norm of that rotation
      direction is ``sqrt(I_v)``, giving a contribution
      ``||K alpha - alpha K||_F^2 / I_v``.  For water this is comparable to the
      vibrational sum, so leaving it out makes a correct projection look
      broken.

    ``residual_pct`` is the number that says whether the reported Raman
    activities are on an absolute scale at all.  A projection that reweights
    atoms by 1/sqrt(m_a), or a dalpha/dQ taken with Q in Bohr, moves every
    activity by a constant factor while leaving the *shape* of the spectrum and
    every depolarization ratio untouched -- the same class of defect the IR
    intensities had, and one that no normalised plot can reveal.
    """
    dalphadx = np.asarray(dalphadx, dtype=float)          # (natm, 3, 3, 3)
    dalpha_dq = np.asarray(dalpha_dq, dtype=float)        # (nmodes, 3, 3)
    m = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords, dtype=float)

    total = float(np.sum(np.sum(dalphadx ** 2, axis=(2, 3)) / m[:, None]))
    vib = float(np.sum(np.sum(dalpha_dq ** 2, axis=(1, 2))))

    mtot = float(m.sum())
    com = (m[:, None] * coords).sum(axis=0) / mtot
    rel = coords - com
    inertia = np.zeros((3, 3))
    for mi, r in zip(m, rel):
        inertia += mi * (float(np.dot(r, r)) * np.eye(3) - np.outer(r, r))

    a = np.asarray(alpha, dtype=float)
    rot = 0.0
    w, v = np.linalg.eigh(inertia)
    for Ij, vj in zip(w, v.T):
        if Ij <= 1e-8:
            continue
        K = np.array([[0.0, -vj[2], vj[1]],
                      [vj[2], 0.0, -vj[0]],
                      [-vj[1], vj[0], 0.0]])
        d = K @ a - a @ K
        rot += float(np.sum(d * d)) / Ij

    expected = total - rot
    resid = 100.0 * (vib - expected) / expected if abs(expected) > 1e-12 else 0.0
    return {
        "polarizability_frob2_per_amu": round(total, 6),
        "rotational_frob2_per_amu": round(float(rot), 6),
        "translational_frob2_per_amu": 0.0,
        "vibrational_frob2_per_amu": round(vib, 6),
        "expected_frob2_per_amu": round(expected, 6),
        "residual_pct": round(float(resid), 3),
        "translation_note": ("exactly zero: alpha is invariant under "
                             "translation, unlike the dipole"),
    }


def raman_spectrum(
    frequencies: Sequence[float],
    activities: Sequence[float],
    depolarization: Sequence[float] = (),
    fwhm: float = 20.0,
    xmin: float = 100.0,
    xmax: float = 4000.0,
    npts: int = 900,
    scale: float = 0.961,
    scale_source: str = "",
) -> dict:
    """Lorentzian-broadened Raman spectrum, in the same conventions as the IR one.

    Frequencies are scaled by the factor published for the level of theory
    before being drawn, so the Raman and IR figures of the same molecule line
    up band for band; the table keeps both values.

    The ordinate is the Raman activity normalised to its own maximum, which is
    what a computed Raman figure shows.  ``activities`` themselves stay in the
    payload on an absolute scale, so a reader can quote them.
    """
    if not frequencies:
        return {"x": [], "y": [], "peaks": []}
    f = np.asarray(frequencies, dtype=float)
    act = np.asarray(activities, dtype=float)
    dep = (np.asarray(depolarization, dtype=float)
           if len(depolarization) else np.zeros_like(f))
    keep = f > 0.0
    f, act = f[keep], act[keep]
    dep = dep[keep] if len(dep) else dep
    fs = f * scale
    lo = max(xmin, min(fs) - 150.0)
    hi = min(xmax, max(fs) + 150.0)
    if hi <= lo:
        hi = lo + 1.0
    x = np.linspace(lo, hi, npts)
    gamma = fwhm / 2.0
    y = np.zeros_like(x)
    for fi, ai in zip(fs, act):
        y += ai * (gamma ** 2) / ((x - fi) ** 2 + gamma ** 2)
    if y.max() > 0:
        y = y / y.max() * 100.0
    x, y = x[::-1], y[::-1]
    peaks = [
        {
            "frequency_cm1": round(float(a), 1),
            "scaled_cm1": round(float(b), 1),
            "activity_a4_amu": round(float(c), 4),
            "depolarization": round(float(d), 4),
        }
        for a, b, c, d in sorted(zip(f, fs, act, dep), key=lambda t: -t[2])
    ]
    return {
        "x": [round(float(v), 1) for v in x],
        "y": [round(float(v), 3) for v in y],
        "peaks": peaks,
        "fwhm_cm1": fwhm,
        "scale_factor": scale,
        "scale_source": scale_source,
        "activity_units": "A^4/amu",
        "xlabel": "Wavenumber (cm-1)",
        "ylabel": "Relative Raman intensity (%)",
    }


def attach_vibrational_spectra(result: dict) -> dict:
    """Add the IR and Raman curves to a finished vibrational analysis.

    Both figures are built here rather than in the server, for the same reason
    the analysis functions live here at all: the curve is part of the result,
    and a result assembled in two places drifts.  The practical consequence is
    that the API, the browser gate and the mutation harness all exercise these
    exact lines -- when the harness ran ``vibrations`` on its own it found
    ``raman_spectrum`` missing, which is how this function came to exist.

    The scale factor is applied to both curves, so the IR and Raman figures of
    one molecule line up band for band.
    """
    scale = float(result.get("freq_scale") or 1.0)
    src = str(result.get("freq_scale_source") or "")
    result["ir_spectrum"] = ir_curve(
        result.get("frequencies_cm1") or [],
        result.get("ir_intensities_km_mol") or [],
        scale=scale,
        scale_source=src,
    )
    raman = result.get("raman") or {}
    if raman.get("activities_a4_amu"):
        result["raman_spectrum"] = raman_spectrum(
            result.get("frequencies_cm1") or [],
            raman["activities_a4_amu"],
            depolarization=raman.get("depolarization") or (),
            scale=scale,
            scale_source=src,
        )
    return result


def vibrations(
    engine: DFTEngine,
    optimize_first: bool = True,
    progress: Progress = None,
    max_atoms: int = 24,
    with_raman: bool = True,
    raman_max_atoms: int = 16,
) -> dict:
    """Harmonic frequencies, IR and Raman intensities, and thermochemistry.

    Frequencies come from PySCF's analytic Hessian.  IR intensities need
    dmu/dQ and Raman needs dalpha/dQ, and this build has no
    ``pyscf.prop.infrared`` and no ``pyscf.prop.raman``, so both derivatives
    are taken by central differences -- 6N extra SCF+dipole evaluations, plus
    a coupled-perturbed solve per displacement for the polarizability.  That
    is what ``max_atoms`` and ``raman_max_atoms`` are for.

    The two spectra share everything expensive: the Hessian, the normal modes
    and the displaced-geometry SCFs.  Only the response solve is extra, which
    is why Raman is a flag here rather than a separate job kind.
    """
    t0 = time.time()

    # Frequencies are only meaningful at a stationary point, so optimise
    # unless the caller says the input geometry already is one.
    opt = None
    # The harvested single point at whatever geometry the Hessian is finally
    # put on.  A vibrational analysis is a figure in a paper and a figure has
    # to say what produced it; this result used to carry frequencies and
    # nothing else -- no functional, no basis, no energy, no solvent -- so the
    # IR spectrum could not be captioned from its own payload and the
    # Properties pane went blank beside it.
    base: dict = {}
    # Warnings do not survive the engine swap below.  The pre-optimisation
    # complains on the engine handed in, and the re-run SCF complains on the
    # fresh one that replaces it, so both have to be collected here or a
    # frequency calculation on an unrelaxed structure reports no warning at
    # all -- three 3N-6 numbers arriving with a minimum's authority.
    warns: list = []
    syms, coords = _split_xyz(engine.atom_xyz)
    coords = np.asarray(coords)
    if optimize_first:
        _tick(progress, 5, "Optimising geometry before the Hessian")
        opt = engine.optimize(
            progress=lambda p: _tick(progress, 5 + int(p.percent * 0.35),
                                     p.message)
        )
        warns += list(opt.get("warnings") or [])
        syms, coords = _split_xyz(opt["optimized_xyz"])
        coords = np.asarray(coords)
        eng = DFTEngine(
            atom_xyz=opt["optimized_xyz"], charge=engine.charge,
            multiplicity=engine.mult, functional=engine.functional,
            basis=engine.basis, solvation=engine.solvation,
        )
        _tick(progress, 45, "SCF at the optimised geometry")
        base = dict(eng.run_scf())
        warns += list(getattr(eng, "warnings", None) or [])
        engine = eng

    natm = len(syms)
    if natm > max_atoms:
        raise DFTError(
            f"IR intensities need 6N single points ({6 * natm} for this "
            f"molecule), which is too slow past {max_atoms} atoms in an "
            f"interactive session. Ask for the frequencies alone, or use a "
            f"smaller basis."
        )

    mf, mol = engine.mf, engine.mol
    if mf is None:
        base = dict(engine.run_scf())
        mf, mol = engine.mf, engine.mol
    elif getattr(engine, "last_result", None):
        # The caller converged the SCF itself; take the harvest it left
        # rather than paying for a second one.
        base = dict(engine.last_result)

    # --- analytic Hessian ---------------------------------------------
    # Must be mf.Hessian(): hessian.RHF(mf) on a DFT object silently omits the
    # exchange-correlation second derivative and gives water 2151/2952/3271
    # cm-1 instead of 1711/3721/3845.
    _tick(progress, 55, "Analytic Hessian")
    H = np.asarray(mf.Hessian().kernel())

    from pyscf.hessian import thermo

    info = thermo.harmonic_analysis(mol, H)
    freq = np.asarray(info["freq_wavenumber"], dtype=float)
    modes = np.asarray(info["norm_mode"], dtype=float)     # (nmodes, natm, 3)
    nmodes = len(freq)

    masses = np.array([elements.get(s).mass for s in syms], dtype=float)

    # --- dmu/dx and dalpha/dx by central differences --------------------
    # A full SCF is converged at each of the 6N displaced geometries, so the
    # derivative includes orbital relaxation (the CPHF term).  It used to be
    # handed the reference density matrix, but nothing ever read it -- a name
    # that promised a frozen-density derivative while the code silently did
    # the expensive and correct thing, which is the worst of both worlds.
    #
    # The polarizability is harvested from the same SCF as the dipole.  It
    # costs one extra CPHF solve per displacement (1-2 s at 6-31G*, against
    # 10-20 s for a finite-field alpha), so Raman roughly doubles the cost of
    # the vibrational analysis rather than multiplying it -- which is why it
    # is gated on its own atom cap below.
    want_alpha = bool(with_raman)
    if want_alpha and natm > raman_max_atoms:
        want_alpha = False
        raman_skip = (
            f"Raman needs a coupled-perturbed response at each of the "
            f"{6 * natm} displaced geometries, which is too slow past "
            f"{raman_max_atoms} atoms in an interactive session; the IR "
            f"intensities above are unaffected."
        )
    else:
        raman_skip = None
    if want_alpha and getattr(mol, "spin", 0) != 0:
        want_alpha = False
        raman_skip = (
            "Raman is implemented for closed-shell molecules only; this "
            "system has an unpaired electron."
        )

    delta = 0.01                                           # Angstrom
    dmudx = np.zeros((natm, 3, 3))                         # [atom, xyz, mu]
    dalphadx = (np.zeros((natm, 3, 3, 3)) if want_alpha else None)
    for i in range(natm):
        for k in range(3):
            cp = coords.copy(); cp[i, k] += delta
            cm = coords.copy(); cm[i, k] -= delta
            dp, ap, _ = _dipole_alpha_at(syms, cp, engine.functional,
                                         engine.basis, engine.charge,
                                         engine.mult, engine.solvation,
                                         want_alpha=want_alpha)
            dmm, am, _ = _dipole_alpha_at(syms, cm, engine.functional,
                                          engine.basis, engine.charge,
                                          engine.mult, engine.solvation,
                                          want_alpha=want_alpha)
            dmudx[i, k] = (dp - dmm) / (2.0 * delta)       # D / A
            if want_alpha:
                dalphadx[i, k] = (ap - am) / (2.0 * delta)  # A^3 / A
        _tick(progress, 55 + int(40 * (i + 1) / natm),
              f"Dipole derivatives: atom {i + 1}/{natm}")

    # --- project onto the normal modes ---------------------------------
    # PySCF's ``norm_mode`` is ALREADY dx/dQ in the mass-weighted sense:
    # hessian/thermo.py builds it as ``einsum('z,zri->izr', mass**-.5, mode)``
    # from orthonormal eigh eigenvectors, so
    #     sum_a m_a * norm_mode[k,a]^2 == sum_a mode[k,a]^2 == 1   exactly.
    # The 0.92-0.96 that a previous version saw was ``sum_a norm_mode^2``,
    # which is a different quantity (it omits the m_a weight); "renormalising"
    # by it, on top of a second division by sqrt(m_a), reweighted every atom's
    # contribution by 1/sqrt(m_a) -- oxygen by 1/4 next to hydrogen.  That is
    # mass-dependent, so no single scale factor can repair it: water's three
    # bands came out -6%/-48%/-21% off and ammonia's N-H stretches 4x too
    # intense.  The corrected projection was fixed with a sum rule that needs
    # no published intensity at all (see backend/contract.py, "IR sum rule").
    #
    # Units: norm_mode is dx/dQ with Q in sqrt(amu)*Bohr, but the Bohr factor
    # cancels when Q is rescaled to sqrt(amu)*Angstrom, so the SAME numbers
    # are dx/dQ in Angstrom per sqrt(amu) -- which is what the 42.256 constant
    # expects (Gaussian: 1 D^2 A^-2 amu^-1 = 42.2561 km/mol, napierian).
    dmudQ = np.einsum("kax,axc->kc", modes, dmudx)          # D/(A sqrt(amu))
    intensities = IR_KM_PER_MOL * np.sum(dmudQ ** 2, axis=1)

    # --- Raman: dalpha/dQ, activities and depolarization ratios ---------
    # Same projection, one tensor rank higher: alpha is a 3x3 so the
    # derivative carries two extra indices and the "intensity" is no longer a
    # single number but the pair (isotropic, anisotropic) that the activity and
    # rho are built from.
    #
    # Note what the sum rule below checks and what it does NOT.  It conserves
    # the Frobenius norm, sum_ij (dalpha_ij/dQ)^2, because that is the quantity
    # an orthogonal change of basis preserves.  The Raman *activity*,
    # 45 a_bar^2 + 7 gamma^2, is a different combination of the same nine
    # numbers, so the sum rule cannot be written in terms of the activities --
    # tempting, and wrong by exactly the anisotropy weighting.  The identity
    # sum_ij a_ij^2 = 3 a_bar^2 + (2/3) gamma^2 for a symmetric tensor is
    # asserted separately instead, which is what ties the two together.
    raman = None
    if want_alpha:
        a3 = BOHR3_TO_A3
        # The equilibrium tensor.  Reported on its own because alpha_iso is the
        # one polarizability number with a measured value to compare against
        # (water: 9.6-9.9 a.u.), so it is the anchor that says whether the
        # basis can support a polarizability at all -- 6-31G* gives 5.13 and is
        # simply not a basis for this property.
        try:
            alpha0 = polarizability_tensor(mf, mol)
        except Exception as exc:                             # noqa: BLE001
            want_alpha = False
            raman = {"available": False,
                     "reason": f"the polarizability could not be computed: {exc}"}
    if want_alpha:
        dalpha_dq = np.einsum("kax,axij->kij", modes, dalphadx) * a3
        iso = np.zeros(nmodes)
        gamma2 = np.zeros(nmodes)
        act = np.zeros(nmodes)
        for k in range(nmodes):
            iso[k], gamma2[k], act[k] = raman_activity(dalpha_dq[k])
        rho = np.array([depolarization_ratio(iso[k], gamma2[k])
                        for k in range(nmodes)])
        try:
            # alpha0 must be in A^3 here: the rotation term is
            # ||K alpha - alpha K||^2 / I with I in amu*A^2, which lands in
            # A^4/amu and has to match the Frobenius sums.
            rsum = raman_sum_rule(dalphadx * a3, modes, dalpha_dq,
                                  alpha0 * a3, masses, coords)
        except Exception as exc:                             # noqa: BLE001
            rsum = {"error": str(exc)}
        # The invariant that links the two conventions: for a symmetric 3x3 the
        # Frobenius norm and the two Placzek invariants are the same number in
        # different clothes.  If the anisotropy formula were wrong -- a missing
        # factor of 6, a dropped off-diagonal term -- the activity would be
        # wrong while this identity still held, but if the *projection* were
        # wrong both would move together and this would catch it.
        frob = np.sum(dalpha_dq ** 2, axis=(1, 2))
        inv = 3.0 * iso ** 2 + (2.0 / 3.0) * gamma2
        worst = float(np.max(np.abs(frob - inv) / np.maximum(frob, 1e-30)))
        raman = {
            "activities_a4_amu": [round(float(v), 4) for v in act],
            "depolarization": [round(float(v), 4) for v in rho],
            "isotropic_deriv_a2_sqrt_amu": [round(float(v), 4) for v in iso],
            "anisotropy_deriv_a2_sqrt_amu": [round(float(np.sqrt(max(g, 0.0))), 4)
                                             for g in gamma2],
            "alpha_iso_au": round(float(np.trace(alpha0) / 3.0), 4),
            "alpha_iso_a3": round(float(np.trace(alpha0) / 3.0 * a3), 4),
            "alpha_tensor_au": [[round(float(v), 4) for v in row]
                                for row in alpha0],
            "sum_rule": rsum,
            "frobenius_vs_invariants_pct": round(100.0 * worst, 6),
            "activity_units": "A^4/amu",
            "depolarization_bound": 0.75,
            "depolarization_note": (
                "rho = 3 gamma^2 / (45 a_bar^2 + 4 gamma^2) <= 0.75 always; "
                "a mode with a_bar = 0 by symmetry sits exactly at 0.75"),
        }
        # A Raman figure has to say whether its basis can support the property.
        # The number is produced either way -- it is just wrong by tens of per
        # cent without diffuse functions, and nothing in the curve's shape
        # reveals that, because the curve is normalised to its own maximum.
        basis_used = getattr(engine, "basis", "") or ""
        if not basis_has_diffuse(basis_used):
            raman["basis_warning"] = (
                f"{BASIS_SETS.get(basis_used, {}).get('label', basis_used)} has "
                f"no diffuse functions, and the polarizability is dominated by "
                f"the outer tail of the density. Measured on water, alpha_iso "
                f"is 5.13 a.u. at 6-31G* against 9.89 at aug-cc-pVTZ "
                f"(experiment 9.6-9.9), and the anisotropy is 3.25 against "
                f"0.66 -- so the activities and every depolarization ratio "
                f"here are only qualitative. Use a diffuse basis "
                f"(6-31+G*, aug-cc-pVDZ) for a Raman figure you intend to "
                f"publish."
            )
            raman["alpha_reliable"] = False
        else:
            raman["alpha_reliable"] = True
    elif raman_skip:
        raman = {"available": False, "reason": raman_skip}

    # --- thermochemistry -----------------------------------------------
    # PySCF's thermo.thermo() returns nonsense in this build (water: ZPE
    # 108.6 Eh instead of 0.021, E_tot +32.2 instead of -76.39), so this is
    # computed from the standard rigid-rotor / harmonic-oscillator formulas
    # instead.  Water's gas-phase S0 of 188.8 J/mol/K is the check.
    therm = thermochemistry(syms, coords, freq, float(engine.mf.e_tot),
                            temperature=298.15, mol=mol)
    zpe_hartree = therm["zpe_hartree"]

    imaginary = [float(f) for f in freq if f < 0.0]

    sc, sc_src = freq_scale(getattr(engine, "functional", ""),
                            getattr(engine, "basis", ""))

    # The intensities are only on an absolute scale if the projection onto the
    # normal modes conserves the sum of squares; this is the number that says
    # so, and it is zero-knowledge -- no measured intensity is involved.
    mu_vec = np.array([(base.get("dipole") or {}).get(k, 0.0) or 0.0
                       for k in ("x", "y", "z")], dtype=float)
    try:
        sumrule = ir_sum_rule(syms, coords, dmudx, intensities, mu_vec,
                              masses, float(engine.charge or 0))
    except Exception as exc:                                 # noqa: BLE001
        sumrule = {"error": str(exc)}

    out = dict(base)
    out.update({
        "frequencies_cm1": [round(float(f), 1) for f in freq],
        "ir_intensities_km_mol": [round(float(i), 2) for i in intensities],
        "n_imaginary": len(imaginary),
        "imaginary_cm1": [round(abs(f), 1) for f in imaginary],
        "zpe_hartree": round(float(zpe_hartree), 6),
        "zpe_kj_mol": round(float(zpe_hartree * 2625.4996), 2),
        "thermochemistry": therm,
        "normal_modes": [[[round(float(v), 5) for v in row] for row in m]
                         for m in modes],
        "optimized_first": bool(optimize_first),
        "optimized_xyz": opt["optimized_xyz"] if opt else engine.atom_xyz,
        # Frequencies are only frequencies at a stationary point, so whether
        # the pre-optimisation got there has to travel with them.  Left behind
        # in the engine's log, a vibrational analysis on an unrelaxed
        # structure looked exactly like one on a minimum.
        "opt_converged": (opt.get("opt_converged") if opt else None),
        "opt_grms": (opt.get("opt_grms") if opt else None),
        "opt_criteria": (opt.get("opt_criteria") if opt else None),
        "vib_seconds": round(time.time() - t0, 3),
        # What the harmonic frequencies have to be multiplied by to compare
        # with a measured band, and where that number came from.  Both belong
        # in the payload: the whole IR curve is shifted by this factor.
        "freq_scale": sc,
        "freq_scale_source": sc_src,
        "ir_sum_rule": sumrule,
        "ir_intensity_units": (
            "km/mol, napierian integrated absorbance; "
            "A = 42.256 |dmu/dQ|^2 with mu in D and Q in sqrt(amu)*Angstrom"),
    })
    if raman is not None:
        out["raman"] = raman
    if opt:
        out["opt_energy_hartree"] = opt.get("energy_hartree")
        out["n_steps"] = opt.get("n_steps")
    if warns:
        out["warnings"] = warns
    _tick(progress, 100, "Vibrational analysis complete")
    return out


# Harmonic frequencies overshoot experiment because the real potential is
# anharmonic, and the correction is a property of the *method*, not of the
# molecule: it is fitted per functional/basis against measured bands.  These
# are the NIST CCCBDB precomputed values (cccbdb.nist.gov/vibscalejustx.asp).
#
# One number used to be hardcoded, 0.961, which is B3LYP/6-31G* specifically,
# and it was applied to every level the app offers.  At HF the published
# factor is 0.899, so every IR band came out 7% too high -- about 220 cm-1 on
# an O-H stretch -- while the narration described it as "the 0.961 factor
# standard for this level of theory".  Where no factor is published for the
# level in use, the honest answer is to apply none and say so, not to borrow
# one from a different method.
FREQ_SCALE_FACTORS = {
    "hf": {"sto-3g": 0.817, "3-21g": 0.906, "6-31g": 0.903,
           "6-31g*": 0.899, "6-31g**": 0.903},
    "b3lyp": {"sto-3g": 0.892, "3-21g": 0.965, "6-31g": 0.962,
              "6-31g*": 0.960, "6-31g**": 0.961},
    "blyp": {"sto-3g": 0.925, "3-21g": 0.995, "6-31g": 0.992,
             "6-31g*": 0.992, "6-31g**": 0.992},
    "pbe": {"sto-3g": 0.914, "3-21g": 0.991, "6-31g": 0.986,
            "6-31g*": 0.986, "6-31g**": 0.986},
    "lda": {"sto-3g": 0.896, "3-21g": 0.984, "6-31g": 0.980,
            "6-31g*": 0.981, "6-31g**": 0.981},
    "pbe0": {"sto-3g": 0.882, "3-21g": 0.960, "6-31g": 0.956,
             "6-31g*": 0.950, "6-31g**": 0.953},
}

FREQ_SCALE_SOURCE = "NIST CCCBDB precomputed vibrational scaling factors"


def freq_scale(functional: str, basis: str) -> Tuple[float, str]:
    """The frequency scale factor published for this level of theory.

    Returns ``(scale, source)``.  ``scale`` is 1.0 -- apply nothing -- with a
    source string that says why, whenever the combination is not in the table.
    Inventing a factor by analogy is worse than quoting an unscaled harmonic
    frequency and labelling it as such.
    """
    table = FREQ_SCALE_FACTORS.get(str(functional or "").lower()) or {}
    value = table.get(str(basis or "").lower())
    if value is not None:
        return float(value), f"{FREQ_SCALE_SOURCE} ({functional}/{basis})"
    return 1.0, (
        f"No frequency scale factor is published for {functional}/{basis} in "
        f"the {FREQ_SCALE_SOURCE} set, so none was applied: these are unscaled "
        f"harmonic frequencies.")


def ir_curve(
    frequencies: Sequence[float],
    intensities: Sequence[float],
    fwhm: float = 20.0,
    xmin: float = 400.0,
    xmax: float = 4000.0,
    npts: int = 900,
    scale: float = 0.961,
    scale_source: str = "",
) -> dict:
    """Lorentzian-broadened IR spectrum.

    Papers quote scaled harmonic frequencies, so the curve is built on the
    scaled values while the table keeps both.  ``scale`` must be the factor
    published for the level the frequencies came from -- it is method-specific,
    and a single hardcoded value silently shifts every band of every other
    method (see ``freq_scale``).

    The x axis is returned in descending order (high wavenumber on the left),
    which is the convention every IR figure follows.  The curve is computed on
    an ascending grid and then flipped, so the payload is paper-ready on its
    own and consumers that export it do not have to know the convention.
    """
    if not frequencies:
        return {"x": [], "y": [], "peaks": []}
    f = np.asarray(frequencies, dtype=float)
    it = np.asarray(intensities, dtype=float)
    keep = f > 0.0                       # imaginary modes are not IR bands
    f, it = f[keep], it[keep]
    fs = f * scale
    lo = max(xmin, min(fs) - 150.0)
    hi = min(xmax, max(fs) + 150.0)
    if hi <= lo:
        hi = lo + 1.0
    x = np.linspace(lo, hi, npts)
    gamma = fwhm / 2.0
    y = np.zeros_like(x)
    for fi, ii in zip(fs, it):
        y += ii * (gamma ** 2) / ((x - fi) ** 2 + gamma ** 2)
    if y.max() > 0:
        y = y / y.max() * 100.0          # % transmittance-style normalisation
    x, y = x[::-1], y[::-1]              # IR convention: 4000 cm-1 on the left
    peaks = [
        {
            "frequency_cm1": round(float(a), 1),
            "scaled_cm1": round(float(b), 1),
            "intensity_km_mol": round(float(c), 2),
        }
        for a, b, c in sorted(zip(f, fs, it), key=lambda t: -t[2])
    ]
    return {
        "x": [round(float(v), 1) for v in x],
        "y": [round(float(v), 3) for v in y],
        "peaks": peaks,
        "fwhm_cm1": fwhm,
        "scale_factor": scale,
        "scale_source": scale_source,
        "xlabel": "Wavenumber (cm-1)",
        "ylabel": "Relative intensity (%)",
    }


# ======================================================================
#  UV-Vis
# ======================================================================
# The integrated absorption follows from the same transition moment as the
# oscillator strength:
#     integral(sigma dnu) = e^2 / (4 eps0 me c) * f        (SI, nu in Hz)
# and the molar decadic coefficient is eps = N_A sigma / (1000 ln 10).  Chaining
# the two with nu_tilde in cm^-1,
#     integral(eps dnu_tilde) = 2.3154e8 * f               (L mol^-1 cm^-2)
# which is the textbook f = 4.319e-9 integral(eps dnu_tilde).  The constant is
# derived here rather than quoted because a stray 2*pi or ln(10) in it is
# *invisible* in a curve normalised to its own maximum -- exactly the trap the
# IR intensity constant had, where the factor turned out to be ln(10).
# Cross-check: the same route reproduces the Einstein A coefficient of hydrogen
# Lyman-alpha (6.2617e8 /s against the published 6.2649e8) to 0.05%.
EPS_INTEGRAL_PER_F = 2.3154e8
# 1 eV in cm^-1.  Note EV2NM * EV2CM = 1.0000e7, which is why nu_tilde in cm^-1
# is just 1e7 / lambda in nm.
EV2CM = HARTREE2CM / HARTREE2EV
FWHM2SIGMA = 2.3548200450309493
# An auto-ranged window is clamped here so the figure stays a UV-Vis figure
# instead of running off into the vacuum ultraviolet.
UV_MIN_NM = 100.0
UV_MAX_NM = 1100.0


def _norm_cdf(z: float) -> float:
    """Standard normal CDF, for how much of a band lies inside the window."""
    return 0.5 * math.erfc(-float(z) / math.sqrt(2.0))


def uv_curve(
    states: Sequence[dict],
    fwhm_ev: float = 0.40,
    wmin: Optional[float] = None,
    wmax: Optional[float] = None,
    npts: int = 500,
    n_electrons: Optional[int] = None,
) -> dict:
    """Gaussian-broadened electronic absorption spectrum from TD-DFT states.

    Three things a UV-Vis figure has to get right, and why:

    *Broadening happens in energy.*  A vertical transition's band width is set
    by vibrational and lifetime broadening, which is a fixed width in energy,
    not in wavelength.  The curve is a sum of Gaussians in eV, evaluated at
    E = hc/lambda.  Broadening on the wavelength grid directly would stretch
    the blue side.

    *The ordinate carries no Jacobian.*  Beer-Lambert's eps(lambda) is a
    spectral density evaluated at lambda, so a band's height is fixed by f and
    by its width in energy and by nothing else.  Multiplying by |dE/dlambda|
    would conserve the *area* of the plot -- a different quantity -- and it
    tilts the spectrum by (lambda_max/lambda_min)^2, which is 19.75 across a
    180-800 nm window.  Two bands with equal f and equal width would then be
    drawn at 8.9:1 instead of 1:1.  The area-conserving quantity is reported
    separately, as `sum_rule`, integrated over wavenumber.

    *The window comes from the states.*  A fixed 180-800 nm window silently
    excludes every band of a small molecule -- ethylene's first six roots all
    sit below 146 nm -- and the figure then shows nothing but Gaussian tails
    normalised to 100%.  Unless the caller names a window, one is taken from
    the states themselves and the choice is reported in `window_source`.

    The curve is returned normalised to its own maximum (what the plot uses)
    and as an absolute molar decadic absorptivity in L mol^-1 cm^-1, which is
    the number a paper's table quotes.
    """
    usable = [
        s for s in states
        if s.get("energy_ev") and s.get("oscillator_strength") is not None
        and s["energy_ev"] > 1e-6
    ]
    if not usable:
        return {"x": [], "y": [], "epsilon": [], "peaks": [],
                "xlabel": "Wavelength (nm)"}
    e = np.array([s["energy_ev"] for s in usable], dtype=float)
    f = np.array([s["oscillator_strength"] for s in usable], dtype=float)

    sigma = fwhm_ev / FWHM2SIGMA
    source = "explicit"
    if wmin is None or wmax is None:
        # three sigma of the broadening, but never less than half an eV, so a
        # set of nearly degenerate roots still gets a window wider than itself
        pad = max(3.0 * sigma, 0.5)
        auto_hi = min(float(e.max()) + pad, EV2NM / UV_MIN_NM)
        auto_lo = max(float(e.min()) - pad, EV2NM / UV_MAX_NM)
        if not (auto_hi > auto_lo > 0.0):
            auto_hi, auto_lo = EV2NM / UV_MIN_NM, EV2NM / UV_MAX_NM
        if wmin is None:
            wmin = EV2NM / auto_hi
        if wmax is None:
            wmax = EV2NM / auto_lo
        source = "auto"
    wmin, wmax = float(wmin), float(wmax)
    if not (wmax > wmin > 0.0):
        wmin, wmax = EV2NM / UV_MIN_NM, EV2NM / UV_MAX_NM
        source = "fallback"

    wl = np.linspace(wmin, wmax, npts)
    ev = EV2NM / wl                                     # energy at each lambda
    nu = EV2NM / wl * EV2CM                             # cm^-1 at each lambda
    # Unit-area Gaussians in energy.  Without the 1/(sigma*sqrt(2pi)) the curve
    # is still a Gaussian of the right shape, so the omission is invisible in a
    # plot normalised to its own maximum -- but the area over wavenumber would
    # then be sum(f) times an arbitrary band width instead of sum(f) times the
    # eps<->f conversion, and the sum rule below could not be stated at all.
    g = np.zeros_like(wl)
    for ei, fi in zip(e, f):
        g += fi * np.exp(-0.5 * ((ev - ei) / sigma) ** 2)
    g /= sigma * math.sqrt(2.0 * math.pi)
    eps = g * (EPS_INTEGRAL_PER_F / EV2CM)              # L mol^-1 cm^-1
    y = eps / eps.max() * 100.0 if eps.max() > 0 else np.zeros_like(eps)

    peaks = []
    for i, s in enumerate(usable):
        lam = float(s.get("wavelength_nm") or EV2NM / s["energy_ev"])
        fi = float(s["oscillator_strength"])
        peaks.append({
            "state": s.get("label") or f"S{i + 1}",
            "energy_ev": round(float(s["energy_ev"]), 4),
            "wavelength_nm": round(lam, 1),
            "oscillator_strength": round(fi, 4),
            # the analytic maximum of an isolated band: independent of the
            # grid and of where the window was cut, so it is the number to
            # put in a table
            "epsilon_max_l_mol_cm": round(
                EPS_INTEGRAL_PER_F * fi
                / (sigma * EV2CM * math.sqrt(2.0 * math.pi)), 1),
            "in_window": bool(wmin <= lam <= wmax),
        })

    # The exact invariant.  With unit-area Gaussians in energy and the
    # conversion above, the area under the curve over wavenumber is that
    # conversion times the oscillator strength -- but only the part of each
    # band that lies inside the plotted window can contribute, so the expected
    # value carries an explicit coverage factor rather than a fudge.  The
    # measured value is the trapezoid of the *reported* epsilon, so a wrong
    # constant, a missing normalisation or a stray Jacobian all show up here.
    nu_lo, nu_hi = float(nu.min()), float(nu.max())
    sig_cm = sigma * EV2CM
    covered = 0.0
    for ei, fi in zip(e, f):
        ni = ei * EV2CM
        span = _norm_cdf((nu_hi - ni) / sig_cm) - _norm_cdf((nu_lo - ni) / sig_cm)
        covered += float(fi) * max(0.0, span)
    expected = EPS_INTEGRAL_PER_F * covered
    order = np.argsort(nu)                              # nu runs backwards in lambda
    measured = float(np.trapezoid(eps[order], nu[order]))
    sum_f = float(np.sum(f))
    sum_rule = {
        "measured_l_mol_cm2": measured,
        "expected_l_mol_cm2": expected,
        "residual_pct": round(100.0 * (measured - expected) / expected, 4)
        if expected else 0.0,
        "sum_f": round(sum_f, 6),
        "covered_fraction": round(covered / sum_f, 6) if sum_f else 0.0,
        "n_states": len(states),
        "n_used": len(usable),
        "n_electrons": int(n_electrons) if n_electrons else None,
        # The Thomas-Reiche-Kuhn sum rule says sum(f) over *all* transitions
        # equals the electron count.  A handful of roots captures a few per
        # cent of that, and a reader has to know it before treating a stick
        # list as a complete spectrum.
        "sum_f_over_n_electrons": (round(sum_f / float(n_electrons), 5)
                                   if n_electrons else None),
        "constant_l_mol_cm2_per_f": EPS_INTEGRAL_PER_F,
        "unit": "L mol^-1 cm^-2",
        "note": ("integral(eps dnu) = 2.3154e8 * sum(f), over the transitions "
                 "whose band lies inside the plotted window; the residual is "
                 "what the broadening and the grid actually deliver"),
    }

    strongest = max(usable, key=lambda s: abs(s.get("oscillator_strength") or 0.0))
    s_lam = float(strongest.get("wavelength_nm")
                  or EV2NM / strongest["energy_ev"])
    s_f = float(strongest["oscillator_strength"])
    notes: List[str] = []
    outside = [p for p in peaks if not p["in_window"]]
    if outside:
        notes.append(
            f"{len(outside)} of {len(peaks)} transitions lie outside the "
            f"plotted window ({wmin:.0f}-{wmax:.0f} nm), so their bands are cut "
            f"off; raise the window or lower the broadening to see them whole."
        )
    if not (wmin <= s_lam <= wmax):
        notes.append(
            f"The strongest transition (S{strongest.get('state')}, "
            f"f = {s_f:.4f}) is at {s_lam:.1f} nm, outside the plotted window "
            f"({wmin:.0f}-{wmax:.0f} nm), so the figure does not show it."
        )

    return {
        "x": [round(float(v), 2) for v in wl],
        "y": [round(float(v), 3) for v in y],
        "epsilon": [round(float(v), 4) for v in eps],
        "epsilon_max": round(float(eps.max()), 4),
        "epsilon_units": "L mol^-1 cm^-1",
        "peaks": peaks,
        "fwhm_ev": fwhm_ev,
        "fwhm_cm1": round(fwhm_ev * EV2CM, 2),
        "window_nm": [round(wmin, 1), round(wmax, 1)],
        "window_source": source,
        "broadened_in": "energy (eV), evaluated at E = hc/lambda",
        "ordinate": ("molar decadic absorptivity, evaluated at each "
                     "wavelength; no |dE/dlambda| factor, which would tilt the "
                     "spectrum by (lambda_max/lambda_min)^2"),
        "sum_rule": sum_rule,
        "strongest": {
            "state": strongest.get("state"),
            "wavelength_nm": round(s_lam, 1),
            "oscillator_strength": round(s_f, 4),
            "in_window": bool(wmin <= s_lam <= wmax),
        },
        "notes": notes,
        "xlabel": "Wavelength (nm)",
        "ylabel": "Relative absorbance (%)",
        "epsilon_ylabel": "Molar absorptivity (L mol-1 cm-1)",
    }


# ======================================================================
#  density of states
# ======================================================================
def dos(engine: DFTEngine, fwhm_ev: float = 0.5, npts: int = 500,
        progress: Progress = None) -> dict:
    """Total and element-projected density of states.

    The projection weights each molecular orbital by its Loewdin population
    on each atom, so the projected curves sum to the total by construction --
    the same partition used everywhere else, rather than a second scheme that
    could disagree with it.
    """
    _tick(progress, 10, "Projecting orbitals")
    mf, mol = engine.mf, engine.mol
    if mf is None:
        engine.run_scf()
        mf, mol = engine.mf, engine.mol

    import numpy as np

    en = np.asarray(mf.mo_energy, dtype=float)
    C = np.asarray(mf.mo_coeff)
    if en.ndim == 2:                       # open shell: average the channels
        en = en.mean(axis=0)
        C = C.mean(axis=0)
    en = en * HARTREE2EV
    nmo = len(en)

    shalf = DFTEngine._s_half(mf)
    slices = [mol.aoslice_by_atom()[i][2:] for i in range(mol.natm)]
    syms = [mol.atom_symbol(i) for i in range(mol.natm)]

    pop = np.zeros((nmo, mol.natm))
    for j in range(nmo):
        v = shalf @ C[:, j]
        for i, (p0, p1) in enumerate(slices):
            pop[j, i] = float(np.sum(v[p0:p1] ** 2))

    occ = np.asarray(mf.get_occ())
    if occ.ndim == 2:
        occ = occ.sum(axis=0)
    nocc = int(np.count_nonzero(occ > 1e-8))
    homo = float(en[nocc - 1]) if nocc else float(en[0])

    elements_present = []
    for s in syms:
        if s not in elements_present:
            elements_present.append(s)

    lo = float(en.min()) - 3.0
    hi = float(en.max()) + 3.0
    # The grid has to resolve the broadening.  A fixed 500 points spans
    # whatever range the basis happens to give -- 50 eV for water, several
    # hundred for benzene -- so the step can easily exceed the FWHM, and a
    # Gaussian sampled coarser than its own width is aliased: its peak comes
    # out low and its area wrong (water at FWHM 0.3 eV on a 0.1 eV grid
    # integrated to 16.9 states instead of 18).  Eight points per FWHM is
    # exact to machine precision for a Gaussian.
    step = fwhm_ev / 8.0
    npts = int(min(3000, max(npts, math.ceil((hi - lo) / step) + 1)))
    grid = np.linspace(lo, hi, npts)
    sigma = fwhm_ev / 2.3548200450309493

    # Each orbital contributes a Gaussian of unit *area*, not unit height, so
    # the curve is a density in states/eV and its area means something: the
    # total integrates to the number of orbitals and the part below the HOMO
    # integrates to the number of occupied ones.  Without the normalisation
    # the area came out as nmo * sigma * sqrt(2*pi) -- i.e. proportional to a
    # smoothing parameter anyone can change -- so the same molecule plotted at
    # FWHM 0.3 and 1.0 eV gave curves 3.3x apart in height, and a reader who
    # integrated the occupied region of water got 2.5 states instead of 5.
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

    def broaden(weights: np.ndarray) -> List[float]:
        w = np.zeros_like(grid)
        for ei, wi in zip(en, weights):
            if wi:
                w += wi * norm * np.exp(-0.5 * ((grid - ei) / sigma) ** 2)
        return w

    total = broaden(np.ones(nmo))
    projected = {}
    for s in elements_present:
        cols = [i for i, sym in enumerate(syms) if sym == s]
        projected[s] = broaden(pop[:, cols].sum(axis=1))

    _tick(progress, 100, "Density of states complete")
    return {
        "x_energy_ev": [round(float(v), 3) for v in grid],
        "x_relative_ev": [round(float(v - homo), 3) for v in grid],
        "total": [round(float(v), 4) for v in total],
        "projected": {
            s: [round(float(v), 4) for v in curve]
            for s, curve in projected.items()
        },
        "homo_ev": round(homo, 4),
        "lumo_ev": round(float(en[nocc]), 4) if nocc < nmo else None,
        "n_occupied": nocc,
        "n_orbitals": int(nmo),
        "fwhm_ev": fwhm_ev,
        "xlabel": "Energy (eV)",
        # states/eV: the area under the curve is a number of orbitals, so the
        # axis label has to say what is being counted.
        "ylabel": "Density of states (states/eV)",
        "note": (f"Gaussian broadening, FWHM {fwhm_ev} eV; the curve is "
                 f"normalised, so its area is a number of orbitals "
                 f"({nmo} in total, {nocc} below the HOMO)."),
    }


# ======================================================================
#  reactivity descriptors
# ======================================================================
def _mult_for(nelec: int) -> int:
    """Lowest multiplicity consistent with an electron count.

    (N electrons, multiplicity M) requires N + M odd, so an odd electron
    count needs an even multiplicity.  Adding or removing one electron from a
    closed-shell molecule therefore flips it to a doublet -- keeping the
    multiplicity fixed would be silently invalid.
    """
    return 2 if nelec % 2 else 1


def reactivity(engine: DFTEngine, with_fukui: bool = True,
               progress: Progress = None) -> dict:
    """Conceptual-DFT descriptors: global indices plus condensed Fukui.

    Global indices use Koopmans' theorem (I ~ -E_HOMO, A ~ -E_LUMO), which is
    an approximation but the standard one for a qualitative reactivity
    discussion.  Fukui functions use the real N-1 / N+1 densities.
    """
    res = getattr(engine, "last_result", None)
    if res is None:
        res = engine.run_scf()
        engine.last_result = res          # cache: the caller may only have mf
    homo = float(res["homo_ev"])
    lumo = float(res["lumo_ev"])

    ip = -homo               # Koopmans ionisation potential, eV
    ea = -lumo               # Koopmans electron affinity, eV
    mu = -0.5 * (ip + ea)    # chemical potential
    eta = 0.5 * (ip - ea)    # chemical hardness
    soft = 1.0 / (2.0 * eta) if abs(eta) > 1e-9 else float("nan")
    omega = (mu ** 2) / (2.0 * eta) if abs(eta) > 1e-9 else float("nan")
    chi = -mu                # Mulliken electronegativity

    out = {
        "homo_ev": round(homo, 4),
        "lumo_ev": round(lumo, 4),
        "gap_ev": round(lumo - homo, 4),
        "ionization_potential_ev": round(ip, 4),
        "electron_affinity_ev": round(ea, 4),
        "chemical_potential_ev": round(mu, 4),
        "electronegativity_ev": round(chi, 4),
        "hardness_ev": round(eta, 4),
        "softness_ev_inv": round(soft, 4) if soft == soft else None,
        "electrophilicity_ev": round(omega, 4) if omega == omega else None,
        "fukui": None,
    }

    if not with_fukui:
        return out

    mol = engine.mol
    natm = mol.natm
    syms = [mol.atom_symbol(i) for i in range(natm)]
    nelec = int(mol.nelectron)

    def charges_for(delta_charge: int, label: str, pct: int):
        q = nelec - delta_charge          # charge lowers the electron count
        m = _mult_for(q)
        _tick(progress, pct, f"{label} system (charge {delta_charge:+d})")
        eng = DFTEngine(
            atom_xyz=engine.atom_xyz, charge=delta_charge, multiplicity=m,
            functional=engine.functional, basis=engine.basis,
            solvation=engine.solvation,
        )
        r = eng.run_scf()
        lo = {c["atom"]: c["charge"] for c in r.get("lowdin_charges", [])}
        return [float(lo.get(i + 1, 0.0)) for i in range(natm)]

    q0 = [float(c["charge"]) for c in res.get("lowdin_charges", [])]
    q_plus = charges_for(+1, "Cation", 40)     # N-1 electrons
    q_minus = charges_for(-1, "Anion", 70)     # N+1 electrons

    # Fukui functions are differences of *population*, not of charge.
    # charge = Z - population, so gaining an electron lowers the charge:
    # working from charges flips the sign of every f and puts the reactive
    # site on the wrong atom (formaldehyde came out "most electrophilic: H3").
    z = [elements.get(s).z for s in syms]
    p0 = [z[i] - q0[i] for i in range(natm)]
    p_plus = [z[i] - q_plus[i] for i in range(natm)]     # N-1
    p_minus = [z[i] - q_minus[i] for i in range(natm)]   # N+1

    rows = []
    for i in range(natm):
        fp = p_minus[i] - p0[i]        # nucleophilic attack: f+ = p(N+1) - p(N)
        fm = p0[i] - p_plus[i]         # electrophilic attack: f- = p(N) - p(N-1)
        f0 = 0.5 * (p_minus[i] - p_plus[i])
        rows.append({
            "atom": i + 1,
            "symbol": syms[i],
            "q_N": round(q0[i], 4),
            "f_plus": round(fp, 4),
            "f_minus": round(fm, 4),
            "f_zero": round(f0, 4),
        })
    top_plus = max(rows, key=lambda r: r["f_plus"]) if rows else None
    top_minus = max(rows, key=lambda r: r["f_minus"]) if rows else None
    out["fukui"] = {
        "rows": rows,
        "scheme": "Loewdin charges, condensed Fukui from N-1 / N / N+1",
        "most_electrophilic_site": (
            f"{top_minus['symbol']}{top_minus['atom']}" if top_minus else None),
        "most_nucleophilic_site": (
            f"{top_plus['symbol']}{top_plus['atom']}" if top_plus else None),
    }
    _tick(progress, 100, "Reactivity analysis complete")
    return out


# ======================================================================
#  cube grids (MEP map, orbital and density isosurfaces)
# ======================================================================
# The Gaussian cube convention writes positions in bohr and flags that with a
# positive atom count in the header; 3Dmol reads exactly that (it applies the
# 0.529177 bohr->angstrom factor when the count is positive), so the files
# PySCF writes can be handed to the viewer unchanged.
ORBITAL_ISVAL = 0.032          # e/bohr^3, the usual orbital-plot contour
# An NTO pair below this occupation is not worth a cube file: it contributes
# less than a tenth of a percent and would only clutter the menu.
NTO_MIN_OCCUPATION = 1e-4
DENSITY_ISVAL = 0.002          # au, Bader's molecular surface
HARTREE_PER_E_TO_KCAL = 627.5094740631

# NCI (non-covalent interaction) index, Johnson et al. JACS 132, 6498 (2010).
# RDG = (1 / 2(3 pi^2)^(1/3)) |grad rho| / rho^(4/3); the second Hessian
# eigenvalue lambda2 tells bonding (negative) from non-bonding (positive).
RDG_PREFACTOR = 1.0 / (2.0 * (3.0 * np.pi ** 2) ** (1.0 / 3.0))
NCI_RDG_ISOVAL = 0.5           # au, the contour the NCI pictures are drawn at
NCI_RHO_CUT = 0.05             # au, above this the interaction is covalent
NCI_VDW_BAND = 0.005           # au, |sign(l2)rho| below this is van der Waals


def cube_mep(engine: DFTEngine, path: str, n: int = 50) -> str:
    """Write the molecular electrostatic potential to a cube file."""
    from pyscf import tools

    if engine.mf is None:
        engine.run_scf()
    mf = engine.mf
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    tools.cubegen.mep(engine.mol, path, dm, nx=n, ny=n, nz=n)
    return path


def cube_orbital(engine: DFTEngine, path: str, index: int, n: int = 50) -> str:
    """Write molecular orbital ``index`` (0-based) to a cube file."""
    from pyscf import tools

    if engine.mf is None:
        engine.run_scf()
    mf = engine.mf
    C = np.asarray(mf.mo_coeff)
    if C.ndim == 3:
        # Unrestricted: (2, nao, nmo).  Averaging alpha and beta would
        # produce an orbital that is neither, so take the alpha set and
        # say so in the label.
        C = C[0]
    if not (0 <= index < C.shape[1]):
        raise DFTError(
            f"Orbital {index} is out of range (0..{C.shape[1] - 1})."
        )
    tools.cubegen.orbital(engine.mol, path, C[:, index], nx=n, ny=n, nz=n)
    return path


def cube_coeff(mol, path: str, coeff: np.ndarray, n: int = 40) -> str:
    """Write an arbitrary orbital coefficient vector to a cube file."""
    from pyscf import tools

    tools.cubegen.orbital(mol, path, np.asarray(coeff, dtype=float),
                          nx=n, ny=n, nz=n)
    return path


# ----------------------------------------------------------------------
#  Natural transition orbitals
# ----------------------------------------------------------------------
def natural_transition_orbitals(engine: DFTEngine, state: int = 1,
                                npairs: int = 2, outdir: Optional[str] = None,
                                n: int = 40,
                                progress: Progress = None) -> dict:
    """Natural transition orbitals for one excited state.

    A TD-DFT excitation is a sum over occupied-virtual pairs, and that sum
    has no unique orbital picture: twenty pairs contributing 5% each is not
    a figure anybody can read.  The NTOs (Martin, J. Chem. Phys. 118, 4775
    (2003)) are the rotation of the occupied and virtual spaces that
    diagonalises the transition, so a transition is described by as few
    orbital pairs as possible -- usually one.

    With the amplitudes X (and Y for full TD-DFT) as a matrix T = X + Y,
    the singular value decomposition T = U s V^T gives

        hole/donor   NTOs:  occupied space rotated by U
        electron/acceptor:  virtual space rotated by V
        occupation numbers: s_j^2, normalised to sum to 1

    PySCF's ``tddft`` objects expose ``get_nto`` only for some solvers, so
    this is done directly from the amplitudes -- which also means the
    normalisation is ours to state explicitly rather than assume.
    """
    import os

    from pyscf import tdscf

    res = getattr(engine, "last_result", None)
    if engine.mf is None:
        res = engine.run_scf()
        engine.last_result = res
    mf = engine.mf
    if not mf.converged:
        raise DFTError("NTOs need a converged ground state.")
    if engine.mult != 1:
        raise DFTError("NTOs are implemented for closed-shell systems only.")

    state = max(1, int(state))
    _tick(progress, 5, f"TD-DFT for state {state}")
    td = tdscf.TDA(mf)
    td.nstates = state
    td.kernel()
    # ``td.converged`` is an array with one flag per root, not a bool, so a
    # plain truth test raises "the truth value of an array is ambiguous" and
    # takes the whole NTO job down with it.
    conv = getattr(td, "converged", True)
    if conv is not None:
        flags = np.asarray(conv).ravel()
        if flags.size and not bool(flags.all()):
            failed = int((~flags.astype(bool)).sum())
            raise DFTError(
                f"{failed} of {flags.size} TD-DFT roots did not converge; the "
                f"transition amplitudes would not describe a real excitation.")
    if len(td.e) < state:
        raise DFTError(f"TD-DFT returned {len(td.e)} states, not {state}.")

    X, Y = td.xy[state - 1]
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float) if Y is not None else np.zeros_like(X)
    T = X + Y
    if not np.isfinite(T).all():
        raise DFTError("the transition amplitudes are not finite.")

    U, s, Vt = np.linalg.svd(T)
    occ = s ** 2
    total = float(occ.sum())
    if total <= 0:
        raise DFTError("this state has zero transition amplitude.")
    occ = occ / total

    C = np.asarray(mf.mo_coeff)
    if C.ndim == 3:
        C = C[0]
    nocc = int(np.asarray(mf.mo_occ).sum() // 2)
    Cocc = C[:, :nocc]
    Cvir = C[:, nocc:]

    # ---- what the transition looks like in canonical orbitals ---------
    # Papers caption an NTO figure with "HOMO->LUMO (94%)", and that number
    # is a property of the canonical MO pair, not of the NTO, so both are
    # reported.  They agree only when the transition is already clean.
    canon = []
    flat = np.argsort(-(X ** 2).ravel())[:6]
    for f in flat:
        i, a = divmod(int(f), X.shape[1])
        w = float(X[i, a] ** 2) / float((X ** 2).sum())
        canon.append({
            "from_index": int(i), "to_index": int(nocc + a),
            "from_label": _mo_label(i, nocc), "to_label": _mo_label(nocc + a, nocc),
            "weight": round(w, 4),
        })

    _tick(progress, 40, "Building NTO cubes")
    os.makedirs(outdir, exist_ok=True)
    pairs = []
    surfaces = []
    # How many pairs carry real weight, and what the whole set sums to.  The
    # ``pairs`` list below is truncated at a threshold, so it cannot answer
    # "does this add up to one" -- that question needs every singular value.
    significant = int((occ > NTO_MIN_OCCUPATION).sum())
    for j in range(min(int(npairs), len(s))):
        hole = Cocc @ U[:, j]
        elec = Cvir @ Vt[j]
        w = float(occ[j])
        if w < 1e-4:
            break
        for role, vec in (("donor", hole), ("acceptor", elec)):
            stem = f"nto{state}_{role}{j + 1}"
            cube_coeff(engine.mol, os.path.join(outdir, f"{stem}.cube"), vec, n=n)
        pairs.append({
            "pair": j + 1, "occupation": round(w, 4),
            "donor_file": f"nto{state}_donor{j + 1}.cube",
            "acceptor_file": f"nto{state}_acceptor{j + 1}.cube",
        })
        for idx, (role, label) in enumerate((
                ("donor", f"NTO{state} hole (donor)"),
                ("acceptor", f"NTO{state} electron (acceptor)"))):
            surfaces.append({
                "key": f"nto{state}-{role}-{j + 1}",
                "file": (pairs[-1]["donor_file"] if role == "donor"
                         else pairs[-1]["acceptor_file"]),
                "label": f"{label} {w * 100:.0f}%",
                "kind": "orbital",
                "isoval": ORBITAL_ISVAL,
                "units": "e/bohr^3",
                "bipolar": True,
                "default": (j == 0 and role == "donor"),
                "note": (f"occupation number {w:.3f}; the pair carries "
                         f"{w * 100:.1f}% of the transition"),
            })

    _tick(progress, 100, "NTO complete")
    dominant = float(occ[0])
    if dominant > 0.9:
        character = ("a single orbital-pair excitation: the NTO picture is "
                     "essentially exact")
    elif dominant > 0.6:
        character = "mainly one pair, with a secondary contribution"
    else:
        character = ("a strongly mixed transition -- no single orbital pair "
                     "describes it, which the occupation numbers make explicit")
    return {
        "state": state,
        "energy_ev": round(float(td.e[state - 1] * HARTREE2EV), 4),
        "wavelength_nm": (round(1239.84198 / (float(td.e[state - 1]) * HARTREE2EV), 1)
                          if td.e[state - 1] > 0 else None),
        "pairs": pairs,
        "canonical": canon,
        "occupation_sum": round(float(occ.sum()), 10),
        "n_significant_pairs": significant,
        "dominant_occupation": round(dominant, 4),
        "character": character,
        "method": "TDA",
        "note": ("Natural transition orbitals from the Tamm-Dancoff "
                 "amplitudes (T = X). The occupation numbers are the squared "
                 "singular values, normalised to sum to 1; a value near 1 "
                 "means one orbital pair is the whole transition."),
        "surfaces": surfaces,
        "grid_n": n,
    }


def _mo_label(index: int, nocc: int) -> str:
    """Canonical-orbital label relative to the frontier orbitals."""
    rel = index - (nocc - 1)
    if rel == 0:
        return "HOMO"
    if rel < 0:
        return f"HOMO{rel}"
    if rel == 1:
        return "LUMO"
    return f"LUMO+{rel - 1}"


def density_derivatives(mol, dm: np.ndarray, coords: np.ndarray,
                        chunk: int = 8192, progress: Progress = None,
                        pct_from: int = 0, pct_to: int = 100):
    """Electron density, its gradient and its Hessian on a set of points.

    Everything is analytic: the AO values and their first and second
    derivatives come from libcint, and

        rho    = sum_mn D_mn phi_m phi_n
        d_a    = 2 sum_mn D_mn (d_a phi_m) phi_n
        H_ab   = 2 sum_mn D_mn [ (d_a d_b phi_m) phi_n
                                 + (d_a phi_m)(d_b phi_n) ]

    so the second term of the Hessian must not be dropped -- it is the same
    order as the first.  Verified against central differences to 3e-9
    (gradient) and 2e-8 (Hessian).
    """
    n = len(coords)
    rho = np.empty(n)
    grad = np.empty((n, 3))
    hess = np.empty((n, 3, 3))
    # libcint returns the six unique second derivatives in this order.
    tri = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for s in range(0, n, chunk):
        c = coords[s:s + chunk]
        A = np.asarray(mol.eval_gto("GTOval_sph_deriv2", c))
        if A.shape[1] != len(c) and A.shape[2] == len(c):
            A = np.swapaxes(A, 1, 2)        # -> (10, npts, nao)
        ao = A[0]
        a1 = A[1:4]
        a2 = A[4:10]
        k = len(c)
        Dphi = dm @ ao.T                    # (nao, k)
        rho[s:s + k] = np.einsum("im,mi->i", ao, Dphi)
        grad[s:s + k] = 2.0 * np.einsum("aim,mi->ia", a1, Dphi)
        for t, (a, b) in enumerate(tri):
            hess[s:s + k, a, b] = 2.0 * (
                np.einsum("im,mi->i", a2[t], Dphi)
                + np.einsum("im,mn,in->i", a1[a], dm, a1[b]))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            hess[s:s + k, b, a] = hess[s:s + k, a, b]
        if progress is not None:
            pct = pct_from + int((pct_to - pct_from) * (s + k) / max(n, 1))
            progress(pct, f"Density derivatives {s + k}/{n}")
    return rho, grad, hess


def read_cube(path: str):
    """Parse a Gaussian cube file -> (natoms, origin, axes, data).

    ``axes`` holds the three voxel vectors in bohr, so grid point (i, j, k)
    sits at ``origin + i*axes[0] + j*axes[1] + k*axes[2]``.
    """
    with open(path, "r") as fh:
        lines = fh.read().splitlines()
    if len(lines) < 6 + abs(int(lines[2].split()[0])):
        raise DFTError(f"{path}: truncated cube file")
    natoms = int(lines[2].split()[0])
    origin = np.array([float(v) for v in lines[2].split()[1:4]])
    axes = np.array([[float(v) for v in lines[3 + a].split()[1:4]]
                     for a in range(3)])
    nx, ny, nz = (abs(int(lines[3 + a].split()[0])) for a in range(3))
    body = " ".join(lines[6 + abs(natoms):]).split()
    data = np.asarray(body, dtype=float)
    if data.size != nx * ny * nz:
        raise DFTError(
            f"{path}: header claims {nx}x{ny}x{nz} = {nx * ny * nz} values "
            f"but {data.size} were read")
    return natoms, origin, axes, data.reshape(nx, ny, nz)


def mep_stats(mol, origin: np.ndarray, axes: np.ndarray, data: np.ndarray) -> dict:
    """Colour range and the extreme sites of an electrostatic-potential grid.

    Sampling matters more than it looks.  The raw grid diverges at the nuclei
    (water's cube reaches +25.6 Ha next to a hydrogen), so the values have to
    be read on the molecular surface, not over the whole box.

    The first attempt kept every point more than 2.4 bohr from any nucleus.
    That is a fixed-radius shell, so it is not the surface of anything: it
    fills the box with far-field points whose potential is near zero, and the
    percentiles it produced were unrelated to the extremes it reported --
    naphthalene came back with a colour range of +/-15 kcal/mol while the
    quoted extreme was 89.  It also sampled *inside* oxygen's van der Waals
    radius (2.87 bohr) while sitting well outside hydrogen's (2.27 bohr),
    so it weighed the atoms unevenly.

    Now the points are taken from the van der Waals sheet itself: keeping
    points whose distance to the nearest nucleus is 1.0-1.3 x that atom's van
    der Waals radius.  That is the surface a reagent actually sees, the
    nuclei are excluded by construction, and the colour range can be taken
    from the real extremes so the legend and the text always agree.
    """
    from pyscf.data import radii as _radii

    nx, ny, nz = data.shape
    idx = np.indices((nx, ny, nz)).reshape(3, -1).T.astype(float)
    pts = origin[None, :] + idx @ axes                  # (N, 3), bohr
    nuc = np.asarray(mol.atom_coords())                 # bohr
    flat = data.reshape(-1)
    if pts.shape[0] > 200000:                           # keep it cheap
        pick = np.linspace(0, pts.shape[0] - 1, 200000).astype(int)
        pts, flat = pts[pick], flat[pick]

    vdw = np.array([_radii.VDW[int(mol.atom_charge(i))] for i in range(mol.natm)])
    dist = np.linalg.norm(pts[:, None, :] - nuc[None, :, :], axis=2)
    # ratio < 1 means the point is inside some atom's van der Waals sphere
    ratio = dist / vdw[None, :]
    m = ratio.min(axis=1)
    near = ratio.argmin(axis=1)

    keep = (m >= 0.98) & (m <= 1.35)
    shell, shell_near = flat[keep], near[keep]
    if shell.size < 64:                                 # coarse grid fallback
        keep = m >= 0.98
        shell, shell_near = flat[keep], near[keep]
    if shell.size < 64:
        shell, shell_near = flat, near

    def site(i: int) -> dict:
        a = int(shell_near[i])
        v = float(shell[i])
        return {
            "site": f"{mol.atom_symbol(a)}{a + 1}",
            "value_hartree": round(v, 4),
            "value_kcal_mol": round(v * HARTREE_PER_E_TO_KCAL, 2),
        }

    lo = site(int(np.argmin(shell)))
    hi = site(int(np.argmax(shell)))
    vlo, vhi = float(shell.min()), float(shell.max())

    # Per-atom extremes on the same shell.  The global argmax is a knife edge:
    # for formaldehyde the carbon's most positive shell point (+0.0495 Ha) and
    # the hydrogens' (+0.0507 Ha) are 0.7 kcal/mol apart, so which atom the
    # "most positive site" lands on is decided by where a 40^3 cube happens to
    # put its points, not by the chemistry.  The per-atom figures are not
    # marginal -- the carbonyl oxygen's shell reaches -0.055 Ha while nothing
    # near the carbon goes below -0.012 -- and they are what a reader actually
    # wants: where the positive and negative regions sit.
    atom_extremes = []
    for a in range(mol.natm):
        sel = near == a
        sel = sel & keep
        atom_extremes.append({
            "site": f"{mol.atom_symbol(a)}{a + 1}",
            "element": mol.atom_symbol(a),
            "n_points": int(sel.sum()),
            "min_hartree": round(float(flat[sel].min()), 4) if sel.any() else None,
            "max_hartree": round(float(flat[sel].max()), 4) if sel.any() else None,
        })

    # A neutral molecule straddles zero, and then the range is kept symmetric
    # so that white sits at zero -- the convention every MEP figure follows.
    # An ion does not: hydroxide's potential is negative everywhere in the
    # shell, and forcing a symmetric range there throws away most of the
    # colour scale for the sake of a colour that never appears.
    straddles = vlo < 0.0 < vhi
    if straddles:
        r = min(max(max(abs(vlo), abs(vhi)), 0.02), 0.25)
        rng = [-r, r]
    else:
        rng = [vlo, vhi]
        if abs(vhi - vlo) > 0.4:                        # refuse to stretch
            rng = [-0.2, 0.2] if vhi < 0 else [0.0, 0.4]
        elif abs(vhi - vlo) < 0.01:
            rng = [vlo - 0.01, vhi + 0.01]
    lo_h, hi_h = round(rng[0], 4), round(rng[1], 4)

    return {
        "range": [lo_h, hi_h],
        "range_kcal_mol": [round(lo_h * HARTREE_PER_E_TO_KCAL, 2),
                           round(hi_h * HARTREE_PER_E_TO_KCAL, 2)],
        # "negative/positive", not "rich/poor": for an ion the potential has
        # one sign everywhere, and calling the least-negative point "electron
        # poor" is simply false.
        "most_negative": lo,
        "most_positive": hi,
        "atom_extremes": atom_extremes,
        "potential_changes_sign": bool(straddles),
        "net_charge": int(mol.charge),
        "sampling": "van der Waals sheet (1.0-1.3 x r_vdw)",
    }


def surfaces(engine: DFTEngine, outdir: str, n: int = 40,
             progress: Progress = None) -> dict:
    """Write the cube grids behind the two 3D figures a paper shows.

    Produces the electrostatic-potential map and isocontours of the electron
    density and of the frontier orbitals.  A 40^3 cube is about 1.2 MB, so
    the grids stay on disk and the caller serves them by URL rather than
    stuffing them into the JSON payload.
    """
    import os
    from pyscf import tools

    res = getattr(engine, "last_result", None)
    if res is None or engine.mf is None:
        res = engine.run_scf()
        engine.last_result = res

    os.makedirs(outdir, exist_ok=True)
    mol = engine.mol
    mf = engine.mf

    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        dm = dm.sum(axis=0)

    mo_e = np.asarray(mf.mo_energy)
    if mo_e.ndim == 2:
        mo_e = mo_e[0]
    nocc = int(res.get("nocc") or 0)
    nmo = int(mo_e.shape[0])
    unrestricted = np.asarray(mf.mo_coeff).ndim == 3

    items: List[Dict[str, object]] = []

    # --- electron density: Bader's molecular surface --------------------
    _tick(progress, 10, "Electron density grid")
    tools.cubegen.density(mol, os.path.join(outdir, "density.cube"), dm,
                          nx=n, ny=n, nz=n)
    items.append({
        "key": "density",
        "file": "density.cube",
        "label": "Electron density",
        "kind": "density",
        "isoval": DENSITY_ISVAL,
        "units": "e/bohr^3",
        "bipolar": False,
        "note": "0.002 e/bohr^3 -- Bader's molecular surface",
    })

    # --- electrostatic potential ---------------------------------------
    _tick(progress, 35, "Electrostatic potential grid")
    mp = os.path.join(outdir, "mep.cube")
    tools.cubegen.mep(mol, mp, dm, nx=n, ny=n, nz=n)
    _nat, origin, axes, vals = read_cube(mp)
    stats = mep_stats(mol, origin, axes, vals)
    items.append({
        "key": "mep",
        "file": "mep.cube",
        "label": "Electrostatic potential (MEP)",
        "kind": "mep",
        "default": True,
        "range": stats["range"],
        "range_kcal_mol": stats["range_kcal_mol"],
        "most_negative": stats["most_negative"],
        "most_positive": stats["most_positive"],
        # Per-atom extremes on the same sheet.  The global argmax is a knife
        # edge -- for formaldehyde the carbon's and the hydrogens' most
        # positive shell points are 0.7 kcal/mol apart, so which atom it names
        # depends on where the 40^3 cube happens to fall -- while the contrast
        # between the two elements is tens of kcal/mol.  Anything that wants
        # to say "the positive region sits over the carbon" needs these.
        "atom_extremes": stats["atom_extremes"],
        "potential_changes_sign": stats["potential_changes_sign"],
        "net_charge": stats["net_charge"],
        "units": "hartree/e",
        "note": ("red = negative, blue = positive; both the range and the "
                 "extremes are measured on the van der Waals sheet, where a "
                 "reagent actually meets the molecule"),
    })

    # --- frontier orbitals ---------------------------------------------
    # The file name cannot be the key: "lumo+1.cube" trips the server's
    # filename whitelist and 404s, so the grid shows in the menu and then
    # fails to load.  Spell the sign out instead.
    wanted = [
        ("homo-1", nocc - 2, "HOMO-1", "homo_minus1"),
        ("homo", nocc - 1, "HOMO", "homo"),
        ("lumo", nocc, "LUMO", "lumo"),
        ("lumo+1", nocc + 1, "LUMO+1", "lumo_plus1"),
    ]
    made = 0
    for i, (key, idx, name, stem) in enumerate(wanted):
        if not (0 <= idx < nmo):
            continue
        _tick(progress, 50 + 12 * i, f"{name} grid")
        cube_orbital(engine, os.path.join(outdir, f"{stem}.cube"), idx, n=n)
        occ = 2.0 if idx < nocc else 0.0
        items.append({
            "key": key,
            "file": f"{stem}.cube",
            "label": name,
            "kind": "orbital",
            "index": int(idx),
            "energy_ev": round(float(mo_e[idx] * HARTREE2EV), 4),
            "occupation": occ,
            "isoval": ORBITAL_ISVAL,
            "units": "e/bohr^3",
            "bipolar": True,
            "default": (key == "homo"),
            "note": ("both phases shown; the sign of an orbital is arbitrary, "
                     "only the nodal structure is meaningful"
                     + (", alpha set" if unrestricted else "")),
        })
        made += 1

    _tick(progress, 100, "Surfaces complete")
    return {
        "surfaces": items,
        "grid_n": n,
        "n_orbital_surfaces": made,
        "homo_index": nocc - 1,
        "lumo_index": nocc if nocc < nmo else None,
    }


# ----------------------------------------------------------------------
#  NCI / RDG: the weak-interaction figure
# ----------------------------------------------------------------------
def nci(engine: DFTEngine, outdir: Optional[str] = None, n: Optional[int] = None,
        margin: float = 3.5, rho_cut: float = NCI_RHO_CUT,
        npoints: int = 3600, progress: Progress = None) -> dict:
    """Reduced density gradient analysis of the non-covalent interactions.

    This produces the figure every adsorption and supramolecular paper
    carries: RDG plotted against ``sign(lambda2) * rho``.  Covalent bonds sit
    at high density and are deliberately excluded; what survives is the
    low-density, low-gradient region where hydrogen bonds, dispersion
    contacts and steric clashes live, and ``lambda2`` sorts them into
    attractive, van der Waals and repulsive.

    Three cube grids are written as well, one per class, each holding the RDG
    masked to that class.  A single grid cannot carry the three-way colouring
    -- the isosurface needs a value to contour and a *different* value to
    colour by -- so the classes are split into separate files instead.
    """
    import os

    from pyscf import tools

    res = getattr(engine, "last_result", None)
    if res is None or engine.mf is None:
        res = engine.run_scf()
        engine.last_result = res

    mol = engine.mol
    dm = np.asarray(engine.mf.make_rdm1())
    if dm.ndim == 3:
        dm = dm.sum(axis=0)

    natm = mol.natm
    if n is None:
        # A finer grid is needed for small systems, but the cost is cubic in
        # n and the number of AOs grows too, so larger molecules step down.
        n = 64 if natm <= 12 else (48 if natm <= 26 else 40)

    _tick(progress, 5, f"Building a {n}^3 grid")
    box = tools.cubegen.Cube(mol, n, n, n, margin=margin)
    coords = box.get_coords()

    _tick(progress, 10, "Evaluating the density and its derivatives")
    rho, grad, hess = density_derivatives(
        mol, dm, coords, progress=progress, pct_from=10, pct_to=75)

    lam = np.linalg.eigvalsh(hess)               # ascending: l1 <= l2 <= l3
    lam2 = lam[:, 1]
    del hess, lam
    sgn = np.sign(lam2) * rho
    safe = np.maximum(rho, 1e-30)
    rdg = RDG_PREFACTOR * np.linalg.norm(grad, axis=1) / safe ** (4.0 / 3.0)
    del grad, safe

    finite = np.isfinite(rdg) & np.isfinite(sgn)
    # The cores and the covalent bonds are not what this analysis is for.
    weak = finite & (rho <= rho_cut)
    shown = weak & (rdg <= 2.0)

    att = shown & (sgn < -NCI_VDW_BAND)
    vdw = shown & (np.abs(sgn) <= NCI_VDW_BAND)
    rep = shown & (sgn > NCI_VDW_BAND)

    # --- the spike ------------------------------------------------------
    # A single grid point dipping low proves nothing; a hydrogen bond puts a
    # handful of points there, so count them.
    band = weak & (rho >= 5e-3) & (sgn < -NCI_VDW_BAND)
    spike = band & (rdg < 0.35)
    n_spike = int(spike.sum())
    if n_spike:
        at_spike = float(np.median(sgn[spike]))
        min_rdg = float(rdg[band].min())
    else:
        at_spike = None
        min_rdg = float(rdg[band].min()) if band.any() else None

    # A van der Waals contact is one where the density is small *and* lambda2
    # nearly vanishes, so this band has its own, much lower, density window --
    # the 5e-3 floor used for the attractive band would leave it empty.
    vdw_band = weak & (rho >= 1e-3) & (rho <= 1e-2) & (np.abs(sgn) <= NCI_VDW_BAND)
    vdw_rdg = float(rdg[vdw_band].min()) if vdw_band.any() else None

    # --- scatter --------------------------------------------------------
    # Half the budget is spent on the low-RDG tail (that is where the
    # interaction is, and a uniform sample can miss a spike that is a dozen
    # points out of ten thousand); the other half is uniform, so the cloud
    # still shows where the bulk of the grid sits.
    rng = np.random.default_rng(20240517)
    per = max(1, npoints // 3)
    series = []
    for key, mask, label, colour in (
        ("attractive", att, "Attractive (H-bond, halogen, ...)", "#2f6fd0"),
        ("vdw", vdw, "van der Waals", "#3aa76d"),
        ("repulsive", rep, "Repulsive (steric)", "#d1493f"),
    ):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            series.append({"key": key, "label": label, "colour": colour,
                           "points": [], "count": 0})
            continue
        tail_n = min(idx.size, max(1, per // 2))
        order = np.argsort(rdg[idx], kind="stable")
        step = max(1, idx.size // tail_n)
        tail = idx[order[::step][:tail_n]]
        rest_n = min(idx.size, per - len(tail))
        rest = idx[rng.choice(idx.size, rest_n, replace=False)] \
            if rest_n > 0 else np.empty(0, dtype=int)
        sel = np.unique(np.concatenate([tail, rest]))
        pts = [[round(float(sgn[i]), 5), round(float(rdg[i]), 4)]
               for i in sel]
        series.append({"key": key, "label": label, "colour": colour,
                       "points": pts, "count": int(idx.size)})

    surfaces: List[Dict[str, object]] = []
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        for key, mask, label, colour in (
            ("nci_attractive", att, "Attractive", "#2f6fd0"),
            ("nci_vdw", vdw, "van der Waals", "#3aa76d"),
            ("nci_repulsive", rep, "Repulsive", "#d1493f"),
        ):
            _tick(progress, 80, f"{label} isosurface grid")
            field = np.where(mask, rdg, 0.0).reshape(n, n, n)
            path = os.path.join(outdir, f"{key}.cube")
            box.write(field, path, f"RDG masked to the {key} region")
            surfaces.append({
                "key": key,
                "file": f"{key}.cube",
                "label": label,
                "kind": "nci",
                "isoval": NCI_RDG_ISOVAL,
                "colour": colour,
                "opacity": 0.85,
                "bipolar": False,
                "units": "a.u.",
                "note": ("RDG = %.2f a.u. inside the low-density region "
                         "(rho <= %.3f); cores and covalent bonds are excluded"
                         % (NCI_RDG_ISOVAL, rho_cut)),
            })

    _tick(progress, 100, "NCI analysis complete")
    return {
        "scatter": {
            "x_label": "sign(lambda2) rho  (a.u.)",
            "y_label": "Reduced density gradient (a.u.)",
            "x_range": [-0.05, 0.05],
            "y_range": [0.0, 2.0],
            "series": series,
        },
        "surfaces": surfaces,
        "grid": {"n": n, "margin_bohr": margin, "points": int(coords.shape[0])},
        "rho_cut": rho_cut,
        "min_rdg": min_rdg,
        "rdg_at_spike": min_rdg,
        "sign_l2_rho_at_spike": at_spike,
        "n_spike_points": n_spike,
        "vdw_min_rdg": vdw_rdg,
        "has_weak_interaction": bool(n_spike >= 3),
        "n_points_shown": int(shown.sum()),
    }
