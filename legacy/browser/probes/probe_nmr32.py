"""Probe 32 -- where did the non-MMFF94 library geometries come from?

probe 30 settled the negative: 29 of the 62 library entries are not MMFF94
stationary points, and for 7 more MMFF94 has no parameters at all.  It did not
say what they are.  The water_dimer entry claims in its own description to be
"relaxed at B3LYP/6-31G* by this program", and the water entry quotes O-H
0.9686 A / 104.00 deg -- a pair of numbers that sit 0.0005 A from the MMFF94
minimum but are also what B3LYP/6-31G* gives for water.

So: take each suspect geometry, re-optimise it **from its own coordinates** at
B3LYP/6-31G* with this program's optimiser, and compare the result with the
input on internal distances.  If the optimiser returns the geometry it was
given, that geometry is a B3LYP/6-31G* stationary point and its parentage is
settled.  If it moves, the geometry belongs to neither candidate source and
the only honest label is the one that admits that.

This costs a real DFT optimisation per molecule, so it is limited to entries
small enough to finish quickly.  Changes nothing.

Run:  python probes/probe_nmr32.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backend.engine.dft import DFTEngine
from backend.engine.molecule import _load_library, parse_xyz

# Small entries that probe 30 flagged as non-MMFF94 (gradient > 1 kcal/mol/A)
# or that MMFF94 cannot parametrise at all.
SUSPECTS = [
    "water",
    "ammonia",
    "methane",
    "ethylene",
    "acetylene",
    "formaldehyde",
    "methanol",
    "co2",
    "nitrogen",
    "hydrogen",
    "hydroxide",
    "ammonium",
    "nitrate",
    "water_dimer",
]

MAX_ATOMS = 12


def fingerprint(atoms):
    """Internal-geometry fingerprint: multiset of (element pair, distance)."""
    pos = np.array([[a.x, a.y, a.z] for a in atoms])
    syms = [a.symbol for a in atoms]
    out = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            out.append((tuple(sorted((syms[i], syms[j]))), round(d, 6)))
    return sorted(out)


def deviation(a, b) -> float:
    fa, fb = fingerprint(a), fingerprint(b)
    if len(fa) != len(fb):
        return float("inf")
    return max(abs(da - db) for (_, da), (_, db) in zip(fa, fb))


def main() -> int:
    lib = _load_library()
    print("=== probe 32: are the odd library geometries B3LYP/6-31G*? ===\n")
    print("re-optimise each entry from its own coordinates at b3lyp/6-31g*")
    print("and measure how far the optimiser moved it.\n")
    print(f"{'entry':14s} {'natm':>5s} {'moved / A':>12s} {'seconds':>9s}   verdict")

    for key in SUSPECTS:
        e = lib.get(key)
        if e is None:
            print(f"{key:14s}      --      --          --   not in library")
            continue
        m = parse_xyz(e["xyz"], name=key)
        if len(m.atoms) > MAX_ATOMS:
            print(f"{key:14s} {len(m.atoms):5d}   (too big for this probe)")
            continue
        t0 = time.time()
        try:
            eng = DFTEngine(
                atom_xyz=e["xyz"],
                charge=int(e.get("charge", 0)),
                multiplicity=int(e.get("multiplicity", 1)),
                functional="b3lyp",
                basis="6-31g*",
            )
            res = eng.optimize(max_steps=40)
            opt = parse_xyz(res["optimized_xyz"], name=key)
            dev = deviation(m.atoms, opt.atoms)
            verdict = "B3LYP/6-31G* stationary point" if dev < 1e-3 else "moved - unknown source"
            print(f"{key:14s} {len(m.atoms):5d} {dev:12.6f} {time.time() - t0:9.1f}   {verdict}")
        except Exception as exc:
            print(f"{key:14s} {len(m.atoms):5d} {'--':>12s} {time.time() - t0:9.1f}   "
                  f"FAILED {type(exc).__name__}: {str(exc)[:60]}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
