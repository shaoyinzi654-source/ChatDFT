"""Probe 29 -- is the library's claimed geometry provenance actually true?

``from_name`` stamps every library molecule with ``geometry_source =
MMFF_GEOMETRY``, and the payload writes that out as the ``A//B`` provenance
string a reader needs: a single point at A on a geometry optimised at B.  It
is a claim about every one of the 62 entries.

Two things made me doubt it:

* the water entry's geometry is O-H 0.96857 A / H-O-H 104.00 deg, while
  RDKit's current ETKDG+MMFF94 embedding of "O" gives 0.96900 / 103.978 --
  close, but not the same numbers;
* the water_dimer entry's *description* says "relaxed at B3LYP/6-31G* by this
  program", which is a different provenance from the one the code stamps on it.

So the library may be a mixture, with one label on all of it.  This probe
re-embeds every entry from its SMILES with the pipeline the codebase says
produced it and compares coordinates.  It changes nothing.

Run:  python probes/probe_nmr29.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backend.engine.molecule import _load_library, from_smiles, parse_xyz

# How far two INTERNAL distance matrices may differ and still count as the
# same geometry.  ETKDG+MMFF94 is deterministic given the seed, so an entry
# built by that pipeline should reproduce its internal distances to ~1e-6 A;
# anything looser is a different geometry that happens to be near.
#
# The comparison is on distances, never on coordinates: the first version of
# this probe compared coordinates and claimed 33 of 62 entries were wrong,
# which was the probe's own error -- ETKDG places a molecule wherever it
# likes, and "hydrogen" was reported as differing by 0.74 A, its own bond
# length.
TOL = 1e-4


def coords(xyz: str):
    m = parse_xyz(xyz, name="x")
    return np.array([[a.x, a.y, a.z] for a in m.atoms]), [a.symbol for a in m.atoms]


def main() -> int:
    lib = _load_library()
    print("=== probe 29: is every library geometry really MMFF94? ===\n")
    print(f"tolerance {TOL:g} A per atom\n")

    rows = []
    for key, e in sorted(lib.items()):
        smi = (e.get("smiles") or "").strip()
        if not smi:
            rows.append((key, "no SMILES", None))
            continue
        try:
            fresh = from_smiles(smi, name=key)
        except Exception as exc:                         # noqa: BLE001
            rows.append((key, f"embed failed: {exc}", None))
            continue

        have, have_syms = coords(e["xyz"])
        got = np.array([[a.x, a.y, a.z] for a in fresh.atoms])
        got_syms = [a.symbol for a in fresh.atoms]

        if sorted(have_syms) != sorted(got_syms) or have.shape != got.shape:
            rows.append((key, "composition differs", None))
            continue
        # Compare the INTERNAL geometry, as a sorted distance matrix, not the
        # raw coordinates.  ETKDG puts a molecule wherever it likes, so two
        # embeddings of the same conformer differ by a rigid-body motion and
        # by nothing else: comparing coordinates directly reported hydrogen as
        # differing by 0.74 A, which is just its own bond length.
        # A canonical internal-geometry fingerprint: for every pair of atoms,
        # (sorted element pair, distance), sorted as a whole.  Invariant under
        # translation, rotation, reflection AND atom reordering -- RDKit lists
        # atoms in a different order from the library's XYZ for eleven entries,
        # and a comparison that needed a matching order could not see them.
        def fingerprint(c, syms):
            n = len(syms)
            out = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = float(np.linalg.norm(c[i] - c[j]))
                    out.append((tuple(sorted((syms[i], syms[j]))),
                                round(d, 6)))
            return sorted(out)

        f_have = fingerprint(have, have_syms)
        f_got = fingerprint(got, got_syms)
        if len(f_have) != len(f_got) or [p[0] for p in f_have] != [
                p[0] for p in f_got]:
            rows.append((key, "bond pattern differs", None))
            continue
        d = max(abs(a[1] - b[1]) for a, b in zip(f_have, f_got))
        rows.append((key, "match" if d <= TOL else "DIFFERS", d))

    n_match = sum(1 for r in rows if r[1] == "match")
    n_diff = sum(1 for r in rows if r[1] == "DIFFERS")
    n_other = len(rows) - n_match - n_diff

    print(f"{'entry':22s} {'verdict':20s} max |d| / A")
    print("-" * 58)
    for key, verdict, d in rows:
        if d is None:
            print(f"{key:22s} {verdict:20s}")
        else:
            flag = "   <<<" if verdict == "DIFFERS" else ""
            print(f"{key:22s} {verdict:20s} {d:11.6f}{flag}")

    print()
    print(f"reproduced by ETKDG+MMFF94 : {n_match}")
    print(f"DIFFERENT geometry         : {n_diff}")
    print(f"could not be compared      : {n_other}")
    print()
    print("The code stamps geometry_source = MMFF94 on all of them.  Every")
    print("entry above marked DIFFERS carries a provenance string that is not")
    print("where its coordinates came from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
