"""Probe -- how marginal is the "most positive site" of formaldehyde?

The contract asserts that the most positive point on formaldehyde's van der
Waals shell sits on the carbonyl carbon, which is the chemistry (a carbonyl is
attacked at carbon).  After the library geometry was relaxed at B3LYP/6-31G*
the same assertion reports H instead, even though the geometry moved by only
about 0.003 A.

That is the signature of a quantity measured on a knife edge rather than a
claim being disproved: if the carbon's shell potential and the hydrogen's are
within a few percent, then which one wins is decided by which grid points the
40^3 cube happens to place in the 0.98-1.35 r_vdw shell, and the assertion is
testing the grid rather than the molecule.

This probe evaluates the shell potential of *every* atom separately, for the
geometry the library used to carry and for the relaxed one, so the margin can
be seen.  It changes nothing.

Run:  python probes/probe_esp1.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from backend.engine.analysis import mep_stats
from backend.engine.dft import DFTEngine
from backend.engine.molecule import _load_library

OLD = "data/molecules/library.round18.bak"


def shell_by_atom(xyz: str, label: str) -> None:
    eng = DFTEngine(atom_xyz=xyz, charge=0, multiplicity=1,
                    functional="b3lyp", basis="6-31g*")
    eng.run_scf()
    from pyscf import tools

    import tempfile

    dm = np.asarray(eng.mf.make_rdm1())
    path = os.path.join(tempfile.gettempdir(), f"probe_esp_{label}.cube")
    # the same grid the surfaces job writes: 40^3 over its own box
    tools.cubegen.mep(eng.mol, path, dm, nx=40, ny=40, nz=40)
    origin, axes, data, _ = _read_cube(path)
    stats = mep_stats(eng.mol, origin, axes, data)
    print(f"\n{label}:  most negative {stats['most_negative']}, "
          f"most positive {stats['most_positive']}")

    # per-atom extremes over the same shell the stats use
    nuc = np.asarray(eng.mol.atom_coords())
    idx = np.indices(data.shape).reshape(3, -1).T.astype(float)
    pts = origin[None, :] + idx @ axes
    flat = data.reshape(-1)
    from backend.engine.elements import get as element_get

    vdw = np.array([_vdw(eng.mol.atom_symbol(i)) for i in range(eng.mol.natm)])
    dist = np.linalg.norm(pts[:, None, :] - nuc[None, :, :], axis=2)
    ratio = dist / vdw[None, :]
    m = ratio.min(axis=1)
    near = ratio.argmin(axis=1)
    keep = (m >= 0.98) & (m <= 1.35)
    print(f"   {keep.sum()} shell points")
    rows = []
    for a in range(eng.mol.natm):
        sel = keep & (near == a)
        if sel.sum() < 3:
            rows.append((eng.mol.atom_symbol(a) + str(a + 1), sel.sum(),
                         float("nan"), float("nan")))
            continue
        rows.append((eng.mol.atom_symbol(a) + str(a + 1), int(sel.sum()),
                     float(flat[sel].max()), float(flat[sel].min())))
    for name, n, hi, lo in rows:
        print(f"   {name:4s} n={n:6d}  max {hi:+.4f} Ha  ({hi * 627.5095:+7.1f} "
              f"kcal/mol)   min {lo:+.4f}")
    os.remove(path)


def _vdw(sym: str) -> float:
    from backend.engine.elements import get as element_get

    # cube coordinates are in bohr, the vdW table is in angstrom
    return float(element_get(sym).vdw_radius or 1.7) * 1.88972612546


def _read_cube(path: str):
    """Minimal cube reader: origin, axes, data."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    natm = int(lines[2].split()[0])
    nx, ny, nz = (int(lines[3 + i].split()[0]) for i in range(3))
    axes = np.array([[float(x) for x in lines[3 + i].split()[1:4]]
                     for i in range(3)])
    origin = np.array([float(x) for x in lines[2].split()[1:]])
    data = np.array([float(x) for ln in lines[6 + natm:] for x in ln.split()]).reshape(nx, ny, nz)
    return origin, axes, data, natm


def main() -> int:
    lib = _load_library()
    with open(OLD, encoding="utf-8") as fh:
        was = json.load(fh)
    print("=== shell potential per atom: the geometry the library carried vs "
          "the relaxed one ===")
    for key in ("formaldehyde", "water"):
        print(f"\n--- {key} ---")
        shell_by_atom(was[key]["xyz"], "old")
        shell_by_atom(lib[key]["xyz"], "relaxed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
