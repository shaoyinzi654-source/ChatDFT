"""Round-14 probe, part 2: real TD-DFT numbers + the eps<->f constant.

Run:  python -m probes.probe_uv2
"""
from __future__ import annotations

import math

import numpy as np

import backend.bootstrap as bootstrap

bootstrap.setup()

from backend.engine import analysis
from backend.engine.dft import DFTEngine

HARTREE2EV = 27.211386245988
EV2NM = 1239.8419843320026
EV2CM = 8065.544004611701          # 1 eV in cm^-1
FWHM2SIGMA = 2.3548200450309493
SQRT2PI = math.sqrt(2.0 * math.pi)
EPS_PER_F = 2.3154e8               # integral(eps dnu) / sum(f), L mol^-1 cm^-2

FORMALDEHYDE = """4
formaldehyde
C 0.000000 0.000000 0.000000
O 0.000000 0.000000 1.205000
H 0.000000 0.942900 -0.587600
H 0.000000 -0.942900 -0.587600
"""

ETHYLENE = """6
ethylene
C 0.000000 0.000000 0.667000
C 0.000000 0.000000 -0.667000
H 0.000000 0.928000 1.234000
H 0.000000 -0.928000 1.234000
H 0.000000 0.928000 -1.234000
H 0.000000 -0.928000 -1.234000
"""

print("=" * 74)
print("PART A -- the eps<->f constant, from first principles")
print("=" * 74)
e = 1.602176634e-19
eps0 = 8.8541878128e-12
me = 9.1093837015e-31
c = 2.99792458e8
NA = 6.02214076e23
# integral(sigma dnu) = e^2/(4 eps0 me c) * f   (SI, sigma in m^2, nu in Hz)
pref = e ** 2 / (4 * eps0 * me * c)
print(f"  e^2/(4 eps0 me c)                      = {pref:.6e} m^2 Hz")
# eps = NA*sigma/(1000*ln10); nu[Hz] = c*100*nu_tilde[cm^-1]
conv = (pref * 1e4) / (c * 100) * NA / (1000 * math.log(10))
print(f"  integral(eps dnu_tilde) per unit f     = {conv:.6e} L mol^-1 cm^-2")
print(f"  1/conv (the textbook 4.319e-9)         = {1.0 / conv:.6e}")
print(f"  value used here                        = {EPS_PER_F:.6e}")
print(f"  agreement                              = "
      f"{100 * (conv / EPS_PER_F - 1):+.3f} %")

# Cross-check the same constant through the Einstein A coefficient of
# hydrogen Lyman-alpha, where A_21 and f_12 are both published.
nu_ly = 2.4660675e15
A_ly = 6.2649e8
f_ly = 0.4162
A_pred = (2 * math.pi * e ** 2 * nu_ly ** 2 / (eps0 * me * c ** 3)) * (1 / 3) * f_ly
print(f"\n  Lyman-alpha check: A_21 predicted {A_pred:.4e} vs published "
      f"{A_ly:.4e}  ({100 * (A_pred / A_ly - 1):+.3f} %)")

print()
print("=" * 74)
print("PART B -- real TD-DFT: how much oscillator strength is in N roots?")
print("=" * 74)
for name, xyz, nelec in (("formaldehyde", FORMALDEHYDE, 16),
                         ("ethylene", ETHYLENE, 16)):
    eng = DFTEngine(xyz, functional="b3lyp", basis="6-31g*")
    res = eng.excited_states(nstates=10)
    st = res["excited_states"]
    tot = sum(s.get("oscillator_strength") or 0.0 for s in st)
    print(f"\n  {name}  (b3lyp/6-31g*, {len(st)} roots, {nelec} electrons)")
    for s in st:
        print(f"    S{s['state']:<3d} {s['energy_ev']:8.4f} eV  "
              f"{s['wavelength_nm']:7.1f} nm   f = "
              f"{s.get('oscillator_strength', float('nan')):9.6f}")
    print(f"    sum(f) = {tot:.6f}   sum(f)/N_elec = {tot / nelec:.5f}")

print()
print("=" * 74)
print("PART C -- shape: equal-f bands must come out equal height")
print("=" * 74)


def build(states, fwhm_ev, wmin, wmax, npts, jacobian, absolute):
    e = np.array([s[0] for s in states], float)
    f = np.array([s[1] for s in states], float)
    wl = np.linspace(wmin, wmax, npts)
    sig = fwhm_ev / FWHM2SIGMA
    ev = EV2NM / wl
    y = np.zeros_like(wl)
    amp = 1.0 / (sig * SQRT2PI) if absolute else 1.0
    for ei, fi in zip(e, f):
        y += fi * amp * np.exp(-0.5 * ((ev - ei) / sig) ** 2)
    if jacobian:
        y *= EV2NM / (wl ** 2)
    if absolute:
        y *= EPS_PER_F / EV2CM        # eps in L mol^-1 cm^-1 vs lambda
    return wl, y


two = [(EV2NM / 250.0, 1.0), (EV2NM / 500.0, 1.0)]
for label, jac, ab in (("current (Jacobian, relative)", True, False),
                       ("fixed  (no Jacobian, relative)", False, False),
                       ("fixed  (no Jacobian, absolute eps)", False, True)):
    wl, y = build(two, 0.40, 180, 800, 4000, jac, ab)
    yn = y / y.max() * 100.0
    i1 = int(np.argmin(np.abs(wl - 250.0)))
    i2 = int(np.argmin(np.abs(wl - 500.0)))
    print(f"  {label:34s} 250nm {yn[i1]:7.2f}   500nm {yn[i2]:7.2f}"
          f"   ratio {yn[i1] / yn[i2]:6.3f}")
wl, y = build(two, 0.40, 180, 800, 4000, False, True)
i1 = int(np.argmin(np.abs(wl - 250.0)))
i2 = int(np.argmin(np.abs(wl - 500.0)))
print(f"  absolute eps_max, f=1, FWHM 0.40 eV: {y[i1]:.1f} / {y[i2]:.1f} "
      f"L mol^-1 cm^-1")
print(f"  analytic 2.3154e8/(1.0645*0.40*8065.5) = "
      f"{EPS_PER_F / (1.0645 * 0.40 * EV2CM):.1f} L mol^-1 cm^-1")

print()
print("=" * 74)
print("PART D -- sum rule on the absolute curve: integral(eps dnu) = 2.3154e8*sum(f)")
print("=" * 74)
for lam in (250.0, 400.0, 600.0):
    wl, y = build([(EV2NM / lam, 1.0)], 0.40, 100, 2000, 20000, False, True)
    nu = 1e7 / wl
    o = np.argsort(nu)
    area = float(np.trapezoid(y[o], nu[o]))
    print(f"  band at {lam:5.1f} nm: integral = {area:.6e}   expected "
          f"{EPS_PER_F:.6e}   ratio {area / EPS_PER_F:.6f}")

print()
print("=" * 74)
print("PART E -- does the default 180-800 nm window contain the bands?")
print("=" * 74)
eng = DFTEngine(FORMALDEHYDE, functional="b3lyp", basis="6-31g*")
st = eng.excited_states(nstates=6)["excited_states"]
lams = [s["wavelength_nm"] for s in st]
print(f"  formaldehyde S1..S6 at {min(lams):.1f} .. {max(lams):.1f} nm")
inside = [l for l in lams if 180.0 <= l <= 800.0]
print(f"  inside 180-800 nm: {len(inside)} of {len(lams)}")
eng = DFTEngine(ETHYLENE, functional="b3lyp", basis="6-31g*")
st = eng.excited_states(nstates=6)["excited_states"]
lams = [s["wavelength_nm"] for s in st]
print(f"  ethylene    S1..S6 at {min(lams):.1f} .. {max(lams):.1f} nm")
print(f"  inside 180-800 nm: {sum(1 for l in lams if 180.0 <= l <= 800.0)} "
      f"of {len(lams)}")
