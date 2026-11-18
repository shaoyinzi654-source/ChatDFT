"""Probe 33 -- which bond is wrong in the geometries that are not MMFF94?

probe 32 re-optimised 14 suspect entries at B3LYP/6-31G*.  water_dimer came
back unchanged (0.0001 A) and is therefore a genuine B3LYP/6-31G* minimum,
but methanol moved 0.68 A, and anything that moves that far was never a
stationary point of that level either.

A maximum internal-distance deviation hides what is actually different, so
this probe breaks it down per element pair: for every (O,H), (C,H), (C,O) ...
pair it compares the sorted list of library distances with the sorted list
from a fresh ETKDG+MMFF94 embedding.  No atom mapping is needed, because the
multiset of distances for a given element pair is invariant to atom order.

A big deviation concentrated in one pair says "a different bond length was
used", which is a provenance question.  A deviation spread over every pair,
or a distance that is chemically impossible, says the stored coordinates are
simply wrong -- which is a worse problem and a different fix.

Changes nothing.

Run:  python probes/probe_nmr33.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backend.engine.molecule import _load_library, from_smiles, parse_xyz

# Entries probe 30 found were not MMFF94 stationary points, or that MMFF94
# cannot parametrise.  Ordered by how far probe 32 found them to be from a
# B3LYP/6-31G* minimum where that was measured.
SUSPECTS = [
    "water", "ammonia", "methane", "ethylene", "acetylene", "formaldehyde",
    "methanol", "co2", "nitrogen", "hydrogen", "hydroxide", "ammonium",
    "nitrate", "water_dimer", "benzene", "methanethiol", "glycine", "urea",
    "tfa", "toluene", "styrene", "aniline", "phenol", "pyridine", "ozone",
    "oxygen", "hydrogen_fluoride", "hydrogen_chloride", "carbon_monoxide",
    "borane", "nitric_oxide", "methyl_radical", "hydroxyl_radical",
    "sulfur_dioxide", "acetic_acid",
]


def pairs(atoms):
    """{element pair: sorted distances} for a geometry."""
    pos = np.array([[a.x, a.y, a.z] for a in atoms])
    syms = [a.symbol for a in atoms]
    out: dict = {}
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            key = tuple(sorted((syms[i], syms[j])))
            out.setdefault(key, []).append(float(np.linalg.norm(pos[i] - pos[j])))
    return {k: sorted(v) for k, v in out.items()}


def main() -> int:
    lib = _load_library()
    print("=== probe 33: per-bond breakdown of the non-MMFF94 geometries ===\n")
    for key in SUSPECTS:
        e = lib.get(key)
        if e is None:
            continue
        try:
            lib_atoms = parse_xyz(e["xyz"], name=key).atoms
            fresh = from_smiles(e["smiles"]).atoms
        except Exception as exc:
            print(f"{key}: skipped ({type(exc).__name__}: {exc})")
            continue
        if sorted(a.symbol for a in lib_atoms) != sorted(a.symbol for a in fresh):
            print(f"{key}: atom composition differs - skipped")
            continue
        pl, pf = pairs(lib_atoms), pairs(fresh)
        worst = []
        for k in sorted(set(pl) | set(pf)):
            a, b = pl.get(k, []), pf.get(k, [])
            if len(a) != len(b):
                worst.append((float("inf"), k, None, None))
                continue
            for da, db in zip(a, b):
                worst.append((abs(da - db), k, da, db))
        worst.sort(key=lambda t: -t[0])
        top = [w for w in worst[:3] if w[0] > 1e-6]
        print(f"{key} ({len(lib_atoms)} atoms)")
        if not top:
            print("   every distance agrees with a fresh embedding")
        for d, k, da, db in top:
            print(f"   {k[0]}-{k[1]:2s}  library {da:7.4f}   fresh {db:7.4f}   "
                  f"delta {d:7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
