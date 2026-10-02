"""probe_nmr26.py -- what the product actually returns for a molecule with both
13C and 1H, and whether the single spectrum it returns is a spectrum.

`nmr.compute` groups every nucleus of the molecule together and hands the whole
list to one call of `nmr_spectrum`, which places one Lorentzian per group on a
single ppm axis using a single linewidth in Hz and a single spectrometer
frequency.  For methanol that axis has to hold 13C near 30 ppm and 1H near 3 ppm.

Two things follow, and both are checkable without any reference value:

  * a real NMR spectrum is acquired for ONE nucleus.  A 13C and a 1H window are
    a factor of four apart in width, so a figure that holds both is not a
    spectrum anybody can compare with;
  * the ppm width of a line depends on the Larmor frequency of the nucleus
    being observed.  ``spectrometer_mhz`` is the 1H frequency, so applying it to
    a 13C line makes the 13C linewidth wrong by the gyromagnetic ratio
    (100.6 MHz / 400 MHz = 0.2515 for 13C at 9.4 T).

Run:  python -u probes/probe_nmr26.py
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from backend.engine import nmr

METHANOL = """6
methanol
C 0.000000 0.000000 0.000000
O 0.000000 0.000000 1.430000
H 0.895000 0.000000 -0.440000
H -0.448000 0.775000 -0.440000
H -0.448000 -0.775000 -0.440000
H 0.885000 0.000000 1.850000
"""


def main():
    res = nmr.compute(METHANOL, basis="6-31g*", use_cache=True)
    print("=" * 88)
    print(" the spectrum the product returns for methanol (13C and 1H together)")
    print("=" * 88)
    print(f"  natoms {res['natoms']}  nao {res['nao']}  "
          f"active elements {res['active_elements']}")
    print()
    print(f"  {'atom':>5s} {'isotope':>8s} {'sigma':>10s} {'delta':>10s} "
          f"{'group?':>7s}")
    for n in res["nuclei"]:
        if not n["nmr_active"]:
            continue
        print(f"  {n['symbol']}{n['index']:>2d} {str(n['isotope']):>8s} "
              f"{n['sigma_iso_ppm']:10.3f} "
              f"{(n['delta_ppm'] if n['delta_ppm'] is not None else float('nan')):10.3f}")
    print()
    g = res["groups"]
    print(f"  groups: {[(x['isotope'], round(x['delta_ppm'], 3), x['count']) for x in g]}")
    print()

    sp = res["spectrum"]
    xs = np.array(sp["shift_ppm"])
    ys = np.array(sp["intensity"])
    step = xs[1] - xs[0]
    print(f"  ONE spectrum for all of it:")
    print(f"    window   {xs[0]:.3f} .. {xs[-1]:.3f} ppm   ({len(xs)} points)")
    print(f"    step     {step:.5f} ppm")
    print(f"    linewidth {sp['linewidth_hz']} Hz at {sp['spectrometer_mhz']} MHz"
          f"  ->  {sp['linewidth_hz'] / sp['spectrometer_mhz']:.5f} ppm HWHM")
    print(f"    step / HWHM = {step / (sp['linewidth_hz'] / sp['spectrometer_mhz']):.2f}")
    print()

    # How many grid points does each signal get?
    print(f"  {'signal':>16s} {'delta':>9s} {'height at its own shift':>24s}")
    for grp in g:
        pos = grp["delta_ppm"]
        i = int(np.argmin(np.abs(xs - pos)))
        print(f"  {str(grp['isotope']):>16s} {pos:9.3f} {ys[i]:24.6f}")
    print()
    # local maxima
    peaks = [(float(xs[i]), float(ys[i])) for i in range(1, len(ys) - 1)
             if ys[i] > ys[i - 1] and ys[i] > ys[i + 1]]
    print(f"  the figure has {len(peaks)} local maximum (maxima) for "
          f"{len(g)} signals:")
    for px, py in peaks:
        print(f"      {px:9.3f} ppm   height {py:.6f}")
    print()

    # the physical size of a 13C window vs a 1H window
    print("  a real acquisition:")
    for iso, mhz, width in (("1H", 400.0, 12.0), ("13C", 100.6, 220.0)):
        hw = 1.0 / mhz
        print(f"    {iso:>4s} at {mhz:6.1f} MHz: 1 Hz = {hw:.5f} ppm, "
              f"typical window {width:.0f} ppm -> "
              f"{width / hw:9.0f} points needed to resolve it")


if __name__ == "__main__":
    main()
