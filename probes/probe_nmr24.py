"""probe_nmr24.py -- the decisive grid test, done right.

probe_nmr23 tried to catch the under-sampling with two lines at the two ends of
the window and saw nothing, for two reasons worth writing down:

  * the window is built as linspace(min-pad, max+pad, points), so translating
    every shift translates the grid with it -- a rigid translation cannot
    expose the sampling offset at all;
  * putting the two lines at the two ENDS makes their offsets mirror images of
    each other, so they are sampled at the same height even when that height is
    wrong.

This probe removes both loopholes: the window is pinned with wmin/wmax (which
``nmr_spectrum`` already accepts), so the grid is fixed, and then

  1. one line is slid across a single grid step, with nothing else changing;
  2. two lines that are equal by construction are placed so that one sits on a
     grid point and the other sits between two.

Both are pure functions of the numbers in the figure, so any variation is the
figure's own doing.

Run:  python -u probes/probe_nmr24.py
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from backend.engine import nmr

np.set_printoptions(precision=6, suppress=True, linewidth=200)


def groups_of(shifts, symbol="H"):
    entries = [{"index": i, "symbol": symbol, "isotope": "1H",
                "delta_ppm": s, "sigma_iso_ppm": s}
               for i, s in enumerate(shifts)]
    return nmr.group_equivalent(entries, tol_ppm=0.001)


def grid_of(spec):
    x = np.array(spec["shift_ppm"])
    return x, np.array(spec["intensity"]), x[1] - x[0]


def height_at(spec, shift):
    x, y, step = grid_of(spec)
    i = int(np.argmin(np.abs(x - shift)))
    return float(y[i]), float(x[i] - shift)


def slide_one_line():
    print("=" * 88)
    print(" 1. one line slid across a single grid step, window pinned 0..12 ppm")
    print("=" * 88)
    wmin, wmax = 0.0, 12.0
    probe = nmr.nmr_spectrum(groups_of([5.0]), linewidth_hz=1.0,
                             spectrometer_mhz=400.0, wmin=wmin, wmax=wmax)
    _, _, step = grid_of(probe)
    hwhm = 1.0 / 400.0
    print(f"  grid step {step:.6f} ppm   HWHM {hwhm:.6f} ppm   "
          f"step/HWHM {step / hwhm:.2f}")
    print()
    print(f"  {'shift':>9s} {'offset':>9s} {'drawn':>8s} {'true':>8s} "
          f"{'drawn/true':>11s}")
    lo = 5.0
    rows = []
    for k in range(9):
        shift = lo + step * k / 8.0
        spec = nmr.nmr_spectrum(groups_of([shift]), linewidth_hz=1.0,
                                spectrometer_mhz=400.0, wmin=wmin, wmax=wmax)
        drawn, off = height_at(spec, shift)
        true = hwhm ** 2 / (off ** 2 + hwhm ** 2)
        rows.append(drawn / true)
        print(f"  {shift:9.5f} {off:+9.5f} {drawn:8.5f} {true:8.5f} "
              f"{drawn / true:11.4f}")
    print()
    print(f"  spread of drawn/true over one grid step: "
          f"{min(rows):.4f} .. {max(rows):.4f}   "
          f"(a correct figure is flat at 1.0000)")


def two_equal_lines():
    print()
    print("=" * 88)
    print(" 2. two nuclei that MUST be drawn equal, window pinned 0..12 ppm")
    print("=" * 88)
    wmin, wmax = 0.0, 12.0
    probe = nmr.nmr_spectrum(groups_of([5.0]), linewidth_hz=1.0,
                             spectrometer_mhz=400.0, wmin=wmin, wmax=wmax)
    x, _, step = grid_of(probe)
    hwhm = 1.0 / 400.0
    # a shift that lands exactly on a grid point, and one that lands half a
    # step away from the nearest one
    on = float(x[int(np.argmin(np.abs(x - 5.0)))])
    off = on + step / 2.0
    groups = groups_of([on, off])
    spec = nmr.nmr_spectrum(groups, linewidth_hz=1.0, spectrometer_mhz=400.0,
                            wmin=wmin, wmax=wmax)
    h_on, d_on = height_at(spec, on)
    h_off, d_off = height_at(spec, off)
    print(f"  both lines: one 1H each, 1.0 Hz, 400 MHz -- identical by "
          f"construction")
    print(f"  line at {on:.5f} ppm (offset {d_on:+.6f}): drawn height "
          f"{h_on:.5f}")
    print(f"  line at {off:.5f} ppm (offset {d_off:+.6f}): drawn height "
          f"{h_off:.5f}")
    print(f"  ratio: {max(h_on, h_off) / max(1e-12, min(h_on, h_off)):.3f}"
          f"   (must be 1.000)")


def resolution():
    print()
    print("=" * 88)
    print(" 3. how many grid points does a line get, as a function of span?")
    print("=" * 88)
    print(f"  {'span ppm':>9s} {'step ppm':>10s} {'step/HWHM':>10s} "
          f"{'pts > half height':>18s}  resolved?")
    for span in (2.0, 4.0, 8.0, 12.0, 20.0):
        spec = nmr.nmr_spectrum(groups_of([0.0, span]), linewidth_hz=1.0,
                                spectrometer_mhz=400.0)
        _, y, step = grid_of(spec)
        n = int(np.sum(y > 0.5 * y.max()))
        print(f"  {span:9.1f} {step:10.6f} {step * 400:10.2f} {n:18d}  "
              f"{'yes' if n >= 3 else 'NO'}")


if __name__ == "__main__":
    slide_one_line()
    two_equal_lines()
    resolution()
