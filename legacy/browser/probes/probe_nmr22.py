"""probe_nmr22.py -- three questions about the NMR spectrum and grouping.

1. Does the ppm grid resolve the linewidth?  A 1 Hz line at 400 MHz is
   0.0025 ppm HWHM; the default 2000-point grid spans the shift window, so the
   step is of the same order.  If the sampled peak does not reach the true
   peak, the normalisation divides by a number that depends on where the
   shift happens to fall -- i.e. the relative intensities in the figure depend
   on the grid, not on the molecule.

2. Does moving every shift by half a grid step change the figure?  This is the
   "sliding grid" test used for the rigid scan in round 10.  A figure whose
   peak heights move when nothing physical moves is not reproducible.

3. Is group_equivalent() independent of the order the atoms are listed in?
   It merges sequentially against a running mean, which is the classic way to
   get an order-dependent clustering.

Run:  python -u probes/probe_nmr22.py
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from backend.engine import nmr

np.set_printoptions(precision=6, suppress=True, linewidth=200)


def entry(index, symbol, delta, sigma=None):
    return {"index": index, "symbol": symbol, "isotope": "1H",
            "delta_ppm": delta, "sigma_iso_ppm": sigma if sigma else delta}


def groups_of(shifts, symbol="H"):
    return nmr.group_equivalent(
        [entry(i, symbol, s) for i, s in enumerate(shifts)], tol_ppm=0.001)


def section_1_grid():
    print("=" * 84)
    print(" 1. does the grid resolve the linewidth?")
    print("=" * 84)
    print(f"  {'lw Hz':>6s} {'points':>7s} {'HWHM ppm':>9s} {'step ppm':>9s} "
          f"{'step/HWHM':>10s} {'sampled peak':>13s} {'error':>9s}")
    for lw in (0.5, 1.0, 2.0, 5.0):
        for points in (2000,):
            groups = groups_of([3.0, 7.0])
            spec = nmr.nmr_spectrum(groups, linewidth_hz=lw,
                                    spectrometer_mhz=400.0, points=points)
            x = np.array(spec["shift_ppm"])
            y = np.array(spec["intensity"])
            hwhm = lw / 400.0
            step = x[1] - x[0]
            # true height of the envelope at the 3.0 ppm line, before any
            # normalisation: its own Lorentzian at d = 0 plus the tail of the
            # 7.0 ppm line
            d_other = 3.0 - 7.0
            true_at_3 = 1.0 * 1.0 + 1.0 * hwhm ** 2 / (d_other ** 2 + hwhm ** 2)
            d_other = 7.0 - 3.0
            true_at_7 = 1.0 * 1.0 + 1.0 * hwhm ** 2 / (d_other ** 2 + hwhm ** 2)
            true_max = max(true_at_3, true_at_7)
            # sampled maximum over both peaks
            sampled = float(y.max())
            print(f"  {lw:6.1f} {points:7d} {hwhm:9.5f} {step:9.5f} "
                  f"{step / hwhm:10.2f} {sampled:13.6f} "
                  f"{sampled - 1.0:9.5f}")


def section_2_slide():
    print()
    print("=" * 84)
    print(" 2. sliding every shift by half a grid step")
    print("=" * 84)
    # three lines, deliberately unequal so that a normalisation error shows up
    # as a change in RELATIVE heights
    base = groups_of([1.0, 3.0, 7.0])
    for lw in (1.0, 2.0, 5.0):
        a = nmr.nmr_spectrum(base, linewidth_hz=lw, spectrometer_mhz=400.0)
        x = np.array(a["shift_ppm"])
        step = x[1] - x[0]
        half = step / 2.0
        moved = [dict(g, delta_ppm=g["delta_ppm"] + half) for g in base]
        b = nmr.nmr_spectrum(moved, linewidth_hz=lw, spectrometer_mhz=400.0)
        ya = np.array(a["intensity"])
        yb = np.array(b["intensity"])
        # height of each line in each figure, taken at the nearest grid point
        ha = [float(ya[np.argmin(np.abs(x - g["delta_ppm"]))]) for g in base]
        hb = [float(yb[np.argmin(np.abs(np.array(b["shift_ppm"])
                                        - (g["delta_ppm"] + half)))])
              for g in base]
        print(f"  lw {lw:4.1f} Hz  step {step:.5f} ppm")
        print(f"    heights before: {np.array(ha)}")
        print(f"    heights after : {np.array(hb)}")
        print(f"    max change    : {np.abs(np.array(ha) - np.array(hb)).max():.6f}")
        # relative heights, which is what the reader compares
        ra = np.array(ha) / np.array(ha).max()
        rb = np.array(hb) / np.array(hb).max()
        print(f"    relative before {ra}, after {rb}")
        print(f"    max relative change: {np.abs(ra - rb).max():.6f}")


def section_3_grouping():
    print()
    print("=" * 84)
    print(" 3. is group_equivalent() independent of atom order?")
    print("=" * 84)
    # three carbons in a chain: 0.00, 0.04, 0.08 ppm.  A-B are within the
    # 0.05 tolerance, B-C are within it, A-C are not.
    shifts = [0.00, 0.04, 0.08]
    import itertools
    print(f"  shifts {shifts}, tol 0.05 ppm")
    seen = {}
    for perm in itertools.permutations(range(3)):
        es = [entry(i, "C", shifts[i]) for i in perm]
        g = nmr.group_equivalent(es, tol_ppm=0.05)
        sig = tuple(sorted(x["count"] for x in g))
        seen.setdefault(sig, []).append(perm)
    for sig, perms in sorted(seen.items()):
        print(f"    counts {sig}:  {len(perms)} of 6 orders")
        for p in perms:
            print(f"        order {p}")

    print()
    print("  a second case: 0.00, 0.03, 0.06, 0.09")
    shifts = [0.00, 0.03, 0.06, 0.09]
    seen = {}
    for perm in itertools.permutations(range(4)):
        es = [entry(i, "C", shifts[i]) for i in perm]
        g = nmr.group_equivalent(es, tol_ppm=0.05)
        sig = tuple(sorted(x["count"] for x in g))
        seen.setdefault(sig, []).append(perm)
    for sig, perms in sorted(seen.items()):
        print(f"    counts {sig}:  {len(perms)} of 24 orders")


if __name__ == "__main__":
    section_1_grid()
    section_2_slide()
    section_3_grouping()
