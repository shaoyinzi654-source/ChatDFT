"""Round-14 probe: is the Jacobian in uv_curve() correct?

Two questions, both answerable without any literature constant:

1. SHAPE.  For an absorbance spectrum plotted against wavelength, the
   ordinate at each lambda is the spectral density evaluated there --
   eps(lambda) = eps(nu(lambda)).  A band's peak height is set by f and by
   the band width in energy, not by where it sits on the wavelength axis.
   So two bands with equal f and equal FWHM_ev must come out with equal
   peak heights.  Any factor that depends on lambda breaks that.  The
   code multiplies by EV2NM/lambda**2, which varies by a factor of 20
   across the default 180-800 nm window.

2. SUM RULE.  Broadening in energy with unit-area Gaussians must give
   integral(eps dnu) = sum(f) * (the eps<->f conversion).  Without the
   conversion we still have an exact invariant: the area of the raw
   (un-normalised) curve over a wavelength grid, after the Jacobian, must
   equal sigma*sqrt(2*pi)*sum(f) -- because the Jacobian is exactly what
   makes integral over lambda equal integral over E.

Run:  python probes/probe_uv.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

EV2NM = 1239.8419843320026
SQRT2PI = math.sqrt(2.0 * math.pi)
FWHM2SIGMA = 2.3548200450309493

# The literature / textbook conversion, derived here from first principles
# rather than quoted:  integral(eps dnu_tilde) = 2.3154e8 * f
#   f            = (2/3) dE |mu|^2                      (atomic units)
#   integral(sigma dnu) = e^2/(4 eps0 me c) * f          (SI, nu in Hz)
#   eps          = N_A sigma / (1000 ln10)               (L/mol/cm)
#   => integral(eps dnu_tilde) = 2.3154e8 * f
EPS_INTEGRAL_PER_F = 2.3154e8          # L mol^-1 cm^-2 per unit f


def curve(states, fwhm_ev=0.40, wmin=180.0, wmax=800.0, npts=2000,
          jacobian=True, gauss_norm=False):
    """Standalone re-implementation of uv_curve with the two switches."""
    e = np.array([s[0] for s in states], dtype=float)
    f = np.array([s[1] for s in states], dtype=float)
    wl = np.linspace(wmin, wmax, npts)
    sigma = fwhm_ev / FWHM2SIGMA
    ev = EV2NM / wl
    y = np.zeros_like(wl)
    amp = 1.0 / (sigma * SQRT2PI) if gauss_norm else 1.0
    for ei, fi in zip(e, f):
        y += fi * amp * np.exp(-0.5 * ((ev - ei) / sigma) ** 2)
    if jacobian:
        y = y * (EV2NM / (wl ** 2))
    return wl, y


def peak_near(wl, y, target_nm, halfwin=40.0):
    m = np.abs(wl - target_nm) <= halfwin
    if not m.any():
        return float("nan"), float("nan")
    i = int(np.argmax(np.where(m, y, -np.inf)))
    return float(wl[i]), float(y[i])


print("=" * 74)
print("PART 1 -- SHAPE: two bands, equal f, equal FWHM, 200 nm vs 600 nm")
print("=" * 74)
E200, E600 = EV2NM / 200.0, EV2NM / 600.0
states = [(E200, 1.0), (E600, 1.0)]
print(f"  band energies: {E200:.4f} eV (200.0 nm), {E600:.4f} eV (600.0 nm)")

for label, jac in (("WITH Jacobian (current code)", True),
                   ("WITHOUT Jacobian (plain eps)", False)):
    wl, y = curve(states, jacobian=jac)
    y = y / y.max() * 100.0
    l1, h1 = peak_near(wl, y, 200.0)
    l2, h2 = peak_near(wl, y, 600.0)
    print(f"  {label:32s} 200nm peak {h1:7.2f}   600nm peak {h2:7.2f}"
          f"   ratio {h1 / h2:6.2f}")
    print(f"  {'':32s} (found at {l1:.1f} nm / {l2:.1f} nm)")

print()
print("  Expectation: equal f and equal FWHM_ev -> equal eps_max -> ratio 1.00.")
print(f"  The Jacobian itself varies by (800/180)^2 = {(800/180) ** 2:.2f} "
      "across the window,")
print("  so it cannot be a constant offset -- it is a shape distortion.")

print()
print("=" * 74)
print("PART 2 -- SUM RULE: area under the raw curve must be sigma*sqrt(2pi)*sum(f)")
print("=" * 74)
# Put the band well inside the window so truncation is negligible.
E = EV2NM / 400.0
wl, y_raw = curve([(E, 1.0)], jacobian=True)
area_lam = float(np.trapezoid(y_raw, wl))
sigma = 0.40 / FWHM2SIGMA
print(f"  integral(y_raw dlambda)          = {area_lam:.6e}")
print(f"  sigma*sqrt(2pi)*sum(f)           = {sigma * SQRT2PI * 1.0:.6e}")
print(f"  ratio                            = {area_lam / (sigma * SQRT2PI):.6f}")

# What the same thing looks like in energy space, i.e. with the Jacobian
# undone -- this is the invariant the sum rule is really about.
#
# NB the eV -> cm^-1 factor.  A density per eV is 8065.5439x a density per
# cm^-1, so multiplying by EPS_INTEGRAL_PER_F alone overshoots by exactly that
# factor.  (The first version of this probe forgot it and reported an integral
# of 1.87e12 against an expected 2.32e8 -- a factor of 8065.5, which is the
# giveaway.  probe_uv2.py Part D does the same check with the factor in.)
EV2CM = 8065.54393734935
_, y_abs = curve([(E, 1.0)], jacobian=False, gauss_norm=True)
wl2 = np.linspace(180.0, 800.0, 2000)
nu = 1e7 / wl2                                        # cm^-1
eps = y_abs * EPS_INTEGRAL_PER_F / EV2CM              # L/mol/cm
o = np.argsort(nu)                                    # nu falls as lambda rises
area_nu = float(np.trapezoid(eps[o], nu[o]))
print()
print(f"  integral(eps dnu_tilde) with unit-area Gaussians = {area_nu:.6e}")
print(f"  2.3154e8 * sum(f)                                = "
      f"{EPS_INTEGRAL_PER_F * 1.0:.6e}")
print(f"  ratio                                            = "
      f"{area_nu / EPS_INTEGRAL_PER_F:.6f}")
print("  (one band well inside the window, so this also checks that the")
print("   eps<->f constant is self-consistent with the broadening)")

print()
print("=" * 74)
print("PART 3 -- what the missing 1/(sigma*sqrt(2pi)) does to the plot")
print("=" * 74)
for fw in (0.20, 0.40, 0.80):
    wl3, a = curve([(E, 1.0)], fwhm_ev=fw, jacobian=False, gauss_norm=False)
    wl3, b = curve([(E, 1.0)], fwhm_ev=fw, jacobian=False, gauss_norm=True)
    na, nb = a / a.max() * 100.0, b / b.max() * 100.0
    print(f"  FWHM {fw:.2f} eV: raw max {a.max():.4f} vs normalised max "
          f"{b.max():.4f}; after the /max*100 both are identical "
          f"(max diff {np.max(np.abs(na - nb)):.2e})")
print("  -> the missing normalisation is invisible *because* of the /max*100.")
print("     It only matters once the ordinate is meant to be absolute.")

print()
print("=" * 74)
print("PART 4 -- truncation: is the plotted maximum the true maximum?")
print("=" * 74)
# Strong band just outside the default window, weak band inside it.
states = [(EV2NM / 150.0, 1.0), (EV2NM / 300.0, 0.01)]
wl4, y4 = curve(states, wmin=180.0, wmax=800.0)
y4n = y4 / y4.max() * 100.0
print(f"  strong band f=1.00 at 150.0 nm (OUTSIDE the 180-800 window)")
print(f"  weak   band f=0.01 at 300.0 nm (inside)")
print(f"  plotted curve maximum is {y4n.max():.1f}% at "
      f"{wl4[int(np.argmax(y4n))]:.1f} nm")
print(f"  true strongest transition is at 150.0 nm, which is not on the plot.")
print("  The normalised curve shows a peak at the window edge at 100% and")
print("  gives no hint that the real maximum is off-screen.")

print()
print("=" * 74)
print("PART 5 -- peaks list ordering")
print("=" * 74)
print("  IR spectrum (analysis.ir_spectrum) sorts peaks by descending")
print("  intensity:  sorted(zip(f, fs, it), key=lambda t: -t[2])")
print("  UV-Vis builds peaks in `usable` order, i.e. ascending energy, with")
print("  no sort at all -- and the frontend/planner both read `states`.")
