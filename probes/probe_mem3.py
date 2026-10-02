"""Probe -- fit the measured NMR peak memory against nao^4.

probe_mem1 sampled the peak RSS of shielding_tensor alone (baseline taken
after build_operators) for four molecules.  If the nao^4 model is right, those
four points must lie on a straight line through the origin when plotted
against nao^4, and the slope is the true constant.  Four points spanning
nao = 18..54 is a 55x range in nao^4, so a fixed overhead and a nao^3 term
would show up as curvature or a non-zero intercept.

Changes nothing.

Run:  python probes/probe_mem3.py
"""
import numpy as np

D = [("water", 18, 6.4), ("ammonia", 20, 8.6),
     ("formaldehyde", 32, 72.2), ("ethanol", 54, 584.3)]


def main() -> int:
    print("=== NMR peak memory: measured vs nao^4 ===\n")
    x = np.array([n ** 4 for _, n, _ in D], float)
    y = np.array([m for _, _, m in D], float)
    A = np.vstack([x, np.ones_like(x)]).T
    (k, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    print("slope     = %.1f bytes per nao^4" % (k * 1e6))
    print("intercept = %+.2f MB   (a fixed overhead would show here)" % c)
    pred = k * x + c
    for (nm, n, m), p in zip(D, pred):
        print("  %-12s nao=%3d  measured %7.1f MB  fit %7.1f MB  resid %+6.1f"
              % (nm, n, m, p, m - p))
    ss = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print("  R^2 = %.6f" % ss)
    print()
    for nao, who in ((82, "thiophene"), (98, "TMS")):
        print("  %-10s nao=%3d   shipped estimate (56) %.2f GB   "
              "at the measured slope %.2f GB"
              % (who, nao, 56 * nao ** 4 / 1e6 / 1024, k * nao ** 4 / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
