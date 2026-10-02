"""probe_nmr25.py -- what a reader sees when two real signals share a window.

probe_nmr24 established that the ppm grid is coarser than the line: at a 12 ppm
window with 2000 points the step is 2.79 HWHM, so each line is sampled
somewhere on its own flank and the figure is normalised by the highest sample.

The first version of this probe used two lines 0.0035 ppm apart and found the
figure shows ONE maximum -- but that case cannot reach the product, because
group_equivalent() merges anything within 0.05 ppm.  The version that matters
is two signals that ARE reported separately: 0.5 ppm apart, which is a normal
1H pair (two different environments), in a window wide enough to hold them.

The window is pinned so the grid cannot move with the lines, and the pair is
slid across one grid step so every sampling offset is visited.

Run:  python -u probes/probe_nmr25.py
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from backend.engine import nmr


def groups_of(shifts, symbol="H"):
    entries = [{"index": i, "symbol": symbol, "isotope": "1H",
                "delta_ppm": s, "sigma_iso_ppm": s}
               for i, s in enumerate(shifts)]
    return nmr.group_equivalent(entries, tol_ppm=0.05)


def local_maxima(x, y):
    return [(float(x[i]), float(y[i])) for i in range(1, len(y) - 1)
            if y[i] > y[i - 1] and y[i] > y[i + 1]]


def peak_near(peaks, pos):
    if not peaks:
        return None
    return min(peaks, key=lambda p: abs(p[0] - pos))


def main():
    wmin, wmax = 0.0, 12.0
    sep = 0.5
    probe = nmr.nmr_spectrum(groups_of([5.0]), linewidth_hz=1.0,
                             spectrometer_mhz=400.0, wmin=wmin, wmax=wmax)
    x = np.array(probe["shift_ppm"])
    step = x[1] - x[0]
    hwhm = 1.0 / 400.0

    print("=" * 90)
    print(f" two 1H signals {sep} ppm apart, 1.0 Hz line, 400 MHz, "
          f"window pinned {wmin:.0f}..{wmax:.0f} ppm")
    print("=" * 90)
    print(f"  grid step {step:.6f} ppm = {step / hwhm:.2f} x HWHM "
          f"({hwhm:.6f} ppm);  the signals are {sep / step:.1f} grid steps "
          f"apart")
    print(f"  grouping tolerance 0.05 ppm, so {sep} ppm IS reported as two "
          f"signals")
    print()
    print(f"  {'pair at':>9s} {'maxima':>7s} {'left h':>9s} {'right h':>9s} "
          f"{'ratio':>8s}   {'A offset':>9s} {'B offset':>9s}")
    ratios = []
    nmax = {}
    for k in range(11):
        a = 5.0 + step * k / 10.0
        b = a + sep
        spec = nmr.nmr_spectrum(groups_of([a, b]), linewidth_hz=1.0,
                                spectrometer_mhz=400.0, wmin=wmin, wmax=wmax)
        xs = np.array(spec["shift_ppm"])
        ys = np.array(spec["intensity"])
        peaks = local_maxima(xs, ys)
        nmax[len(peaks)] = nmax.get(len(peaks), 0) + 1
        pa, pb = peak_near(peaks, a), peak_near(peaks, b)
        if pa and pb and pa[0] != pb[0]:
            r = max(pa[1], pb[1]) / max(1e-12, min(pa[1], pb[1]))
            ratios.append(r)
            oa = xs[int(np.argmin(np.abs(xs - a)))] - a
            ob = xs[int(np.argmin(np.abs(xs - b)))] - b
            print(f"  {a:9.5f} {len(peaks):7d} {pa[1]:9.5f} {pb[1]:9.5f} "
                  f"{r:8.4f}   {oa:+9.5f} {ob:+9.5f}")
        else:
            print(f"  {a:9.5f} {len(peaks):7d}  -- the two signals do not "
                  f"produce two maxima --")
    print()
    if ratios:
        print(f"  the two signals are equal by construction; the figure draws")
        print(f"  them with a height ratio between {min(ratios):.4f} and "
              f"{max(ratios):.4f}")
        print(f"  maxima count histogram: {nmax}")
    else:
        print(f"  maxima count histogram: {nmax}")


if __name__ == "__main__":
    main()
