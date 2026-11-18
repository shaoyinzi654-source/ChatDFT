"""Precompute the NMR reference shieldings and write them to data/nmr_cache.

Why this is a script and not just a lazy cache
----------------------------------------------
Every chemical shift needs a reference compound computed with the same code at
the same level, and the 1H/13C standard is TMS: 17 atoms, nao = 98 at 6-31G*.
That costs about five minutes, which is fine once and unacceptable on every
first job a user runs.  So the cache is generated here and shipped as data.

The cache key carries the basis, the convergence tolerance and the field step,
because all three move the answer -- conv_tol in particular changes the 17O
shielding of water by 15 ppm between 1e-11 and 1e-13.  A reference computed at
the wrong tolerance would put a constant offset on every shift the product
prints and there would be no way to see it from the output.

That argument is why the key also carries a fingerprint of the *code* now (see
``nmr._method_signature``).  Everything in this file ships as data, so a user
who upgrades keeps the numbers the previous version computed -- and a reference
computed by different code is the same failure as one computed at the wrong
tolerance, with the same absence of evidence in the output.  Editing any of the
four functions that produce the number changes every key, so the old entries
stop being read and these are regenerated.

The consequence for anyone editing this project: **after changing the shielding
code, re-run this script.**  Nothing breaks if you do not -- the engine simply
recomputes each reference on first use and rewrites it -- but the first job
after such a change pays five minutes instead of the cache paying it once here.

Usage:

    python -m backend.gen_nmr_cache                  # the defaults
    python -m backend.gen_nmr_cache 6-31g* def2-svp  # several bases
    python -m backend.gen_nmr_cache --force          # ignore existing entries
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import backend.bootstrap

backend.bootstrap.setup()

from backend.engine import nmr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bases", nargs="*", default=["6-31g*"],
                    help="basis sets to generate (default 6-31g*)")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if a cache entry exists")
    ap.add_argument("--keys", nargs="*", default=None,
                    help="reference keys to generate (default: all)")
    args = ap.parse_args(argv)

    keys = args.keys or list(nmr.REFERENCES)
    out_dir = nmr._cache_dir()
    print(f"cache directory: {out_dir}")
    print(f"tolerance {nmr.CONV_TOL:.0e}, field step {nmr.FIELD_STEP:.0e} au")
    print()

    total = 0
    failures = 0
    for basis in args.bases:
        for key in keys:
            # The key carries a fingerprint of the geometry, so a reference
            # whose geometry changed lands on a new path instead of reusing
            # the old numbers.  The path comes from the engine rather than
            # being rebuilt here: this script writes the file the engine reads,
            # and two copies of the expression is one more place to forget a
            # term of the key.
            path = nmr.reference_cache_path(key, basis)
            if os.path.exists(path) and not args.force:
                print(f"  {key:14s} {basis:10s} already cached, skipping")
                continue
            t0 = time.time()
            try:
                res = nmr.reference_shieldings(key, basis, use_cache=False)
            except Exception as exc:                          # noqa: BLE001
                print(f"  {key:14s} {basis:10s} FAILED: {exc}")
                failures += 1
                continue
            # reference_shieldings(use_cache=False) does not write, so do it
            # here -- that is the whole point of this script.  Through the
            # same atomic writer the engine uses, so the two cannot drift
            # apart, and with the failure reported rather than swallowed:
            # this is the script whose whole job is to produce the cache.
            res["cached"] = False
            err = nmr._write_cache_atomic(path, res)
            if err is not None:
                print(f"  {key:14s} {basis:10s} CACHE WRITE FAILED: {err}")
                failures += 1
                continue
            els = ", ".join(
                f"{el} {v['sigma_iso_ppm']:.3f}"
                for el, v in res["element_shielding_ppm"].items())
            print(f"  {key:14s} {basis:10s} nao={res['nao']:3d}  "
                  f"{time.time() - t0:6.1f}s  {els}")
            total += 1

    print()
    print(f"wrote {total} entries, {failures} failures")

    # What the engine can actually read, and what it cannot.  A stale file is
    # inert -- the key moved, so nothing opens it -- but it is not invisible:
    # two ammonia entries sat in here, one live and one from a superseded
    # geometry, and a probe that paired them by the payload's ``key`` field
    # compared the orphan and reported a 1.15 ppm error no run had produced.
    inv = nmr.cache_inventory(bases=args.bases)
    print()
    print(f"cache inventory: {len(inv['live'])} reachable from the current "
          f"key, {len(inv['stale'])} not")
    for p in inv["stale"]:
        print(f"  unreachable: {os.path.basename(p)}")
    if inv["stale"]:
        print("  (these are left in place on purpose: deleting a file is not "
              "this script's")
        print("   job, and 'unreachable' is a fact about the current key, not "
              "about the file)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
