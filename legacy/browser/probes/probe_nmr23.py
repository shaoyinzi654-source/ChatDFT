"""probe_nmr23.py -- the decisive test for the ppm grid.

probe_nmr22 section 2 could not see anything, and the reason is instructive:
``nmr_spectrum`` builds its grid as ``linspace(min-pad, max+pad, points)``, so
translating every shift translates the grid with it and every peak keeps the
same offset from its nearest grid point.  A rigid translation is therefore
*invisible* -- which is exactly why the bug survived.

The window is set by the SPAN, though, and the span is a property of the
molecule.  A 1H spectrum that runs over 12 ppm has a grid step 6x coarser than
one that runs over 2 ppm, and the step is what decides whether a 1 Hz line
(0.0025 ppm HWHM at 400 MHz) is sampled at its top or half way down its flank.

The test: two lines that are equal by construction -- one nucleus each, same
linewidth -- must be drawn at the same height.  Nothing about them differs
except where they land relative to the grid.

Run:  python -u probes/probe_nmr23.py
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from backend.engine import nmr

np.set_printoptions(precision=6, suppress=True, linewidth=200)


def groups_of(shifts, symbol="H", tol=0.001):
    entries = [{"index": i, "symbol": symbol, "isotope": "1H",
                "delta_ppm": s, "sigma_iso_ppm": s}
               for i, s in enumerate(shifts)]
    return nmr.group_equivalent(entries, tol_ppm=tol)


def height_at(spec, shift):
    x = np.array(spec["shift_ppm"])
    y = np.array(spec["intensity"])
    i = int(np.argmin(np.abs(x - shift)))
    return float(y[i]), float(x[i] - shift), (x[1] - x[0])


def scan_offsets():
    print("=" * 86)
    print(" how far down its own flank can a peak be sampled?")
    print("=" * 86)
    print("  the worst offset is half a grid step; the drawn height there is")
    print("  hwhm^2 / (d^2 + hwhm^2) with d = step/2, relative to the true top")
    print()
    print(f"  {'span ppm':>9s} {'points':>7s} {'step ppm':>10s} "
          f"{'step/HWHM':>10s} {'worst height':>13s}")
    for span, points in ((2.0, 2000), (6.0, 2000), (12.0, 2000), (12.0, 5000)):
        # emulate the window: lo = 0-pad, hi = span+pad, pad = 0.08*span
        pad = max(0.5, 0.08 * span)
        width = span + 2 * pad
        step = width / (points - 1)
        hwhm = 1.0 / 400.0
        d = step / 2.0
        worst = hwhm ** 2 / (d ** 2 + hwhm ** 2)
        print(f"  {span:9.1f} {points:7d} {step:10.5f} {step / hwhm:10.2f} "
              f"{worst:13.4f}")


def two_equal_lines(span):
    print()
    print("=" * 86)
    print(f" two nuclei that MUST be drawn equal, in a {span:.0f} ppm window")
    print("=" * 86)
    # two single-1H lines at the two ends of the window; both count = 1
    a, b = 0.0, span
    groups = groups_of([a, b])
    spec = nmr.nmr_spectrum(groups, linewidth_hz=1.0, spectrometer_mhz=400.0)
    ha, da, step = height_at(spec, a)
    hb, db, _ = height_at(spec, b)
    print(f"  grid step {step:.5f} ppm   HWHM {1 / 400:.5f} ppm   "
          f"step/HWHM {step * 400:.2f}")
    print(f"  line at {a:.3f}: sampled height {ha:.6f}  "
          f"(offset from grid point {da:+.5f} ppm)")
    print(f"  line at {b:.3f}: sampled height {hb:.6f}  "
          f"(offset from grid point {db:+.5f} ppm)")
    print(f"  ratio of the two 'equal' lines: {max(ha, hb) / max(1e-12, min(ha, hb)):.3f}")
    # now the same molecule with the lines closer together: nothing physical
    # changed about either line, only the window
    return max(ha, hb) / max(1e-12, min(ha, hb))


def closing_lines():
    print()
    print("=" * 86)
    print(" the same two lines, brought closer together")
    print("=" * 86)
    for span in (12.0, 8.0, 4.0, 2.0, 1.0, 0.4):
        groups = groups_of([0.0, span])
        spec = nmr.nmr_spectrum(groups, linewidth_hz=1.0, spectrometer_mhz=400.0)
        ha, _, step = height_at(spec, 0.0)
        hb, _, _ = height_at(spec, span)
        print(f"  span {span:5.1f} ppm  step {step:.5f}  "
              f"heights {ha:.5f} / {hb:.5f}  ratio "
              f"{max(ha, hb) / max(1e-12, min(ha, hb)):.4f}")


def line_shape():
    print()
    print("=" * 86)
    print(" how many grid points does one line occupy?")
    print("=" * 86)
    for lw in (0.5, 1.0, 2.0, 5.0):
        groups = groups_of([0.0, 12.0])
        spec = nmr.nmr_spectrum(groups, linewidth_hz=lw,
                                spectrometer_mhz=400.0)
        x = np.array(spec["shift_ppm"])
        y = np.array(spec["intensity"])
        step = x[1] - x[0]
        hwhm = lw / 400.0
        above_half = int(np.sum(y > 0.5 * y.max()))
        print(f"  lw {lw:4.1f} Hz  HWHM {hwhm:.5f} ppm  step {step:.5f} ppm  "
              f"points above half height: {above_half}  "
              f"(a resolved line needs >= 3)")


if __name__ == "__main__":
    scan_offsets()
    for span in (12.0, 6.0, 2.0):
        two_equal_lines(span)
    closing_lines()
    line_shape()
