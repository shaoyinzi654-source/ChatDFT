"""Bond-level analysis: Mayer bond orders and Hirshfeld charges.

Both are ordinary tables in a computational paper and neither existed here.
PySCF's ``prop`` subpackage is absent from this build, so the Hirshfeld
partition is written out from its definition rather than called.

Two conventions are worth stating because getting either wrong is silent:

*Mayer bond order.*  Two definitions circulate and differ by a factor of two
for a closed-shell molecule -- the spin-resolved one,
``sum_{mu,nu} [(Pa S)(Pa S) + (Pb S)(Pb S)]``, and the total-density one,
``sum_{mu,nu} (P S)(P S)`` with ``P = Pa + Pb``.  Measured here at
B3LYP/6-31G* against the published values, only the total-density form is
right: N2 2.829 (lit. ~2.83), CO 2.419 (~2.4), O-H in water 0.807 (~0.8),
C-H in methane 0.964 (~0.97), C-C in benzene 1.456 (~1.45).  The spin-resolved
form returns exactly half of each.

*Hirshfeld charge.*  ``q_A = Z_A - sum_g w_A(r_g) rho_mol(r_g) w_g`` with
``w_A = rho_A^free / sum_B rho_B^free``.  The free-atom density must be the
**spherically averaged** ground-state density: every first-row atom is
open-shell (C is 3P, N is 4S, O is 3P), so an unaveraged atom density is
orientation-dependent and the charges would change when the molecule is
rotated.  The averaging is done here on a radial grid crossed with a Lebedev
angular grid.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import elements

# Radial grid for the spherically averaged free-atom density, in Bohr.  The
# spacing is ~7.7%, fine for a function this smooth; the upper end is where an
# atom's density has long since vanished, and beyond it the interpolation
# returns 0 rather than extrapolating a tail that is not there.
RADIAL_MIN = 1e-5
RADIAL_MAX = 25.0
RADIAL_POINTS = 200
LEBEDEV_POINTS = 110

# Building and converging a free atom costs a few seconds; the same element
# recurs across a batch, and the profile depends only on (symbol, basis, xc).
_FREE_ATOM_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}


def _rho(mol, dm: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Electron density on a set of points.

    Delegated to ``fields.density_and_gradient`` rather than to
    ``numint.eval_rho``: this PySCF build's ``eval_ao`` documents and returns
    ``(N, nao)`` while ``eval_rho``'s internals assume ``(nao, N)``, so
    ``eval_rho`` raises an AssertionError on any grid.  Going through the same
    helper the field plots use also means the two can never disagree about what
    the density is.
    """
    from .fields import density_and_gradient

    return density_and_gradient(mol, dm, coords)[0]


def _angular_grid(points: int = LEBEDEV_POINTS) -> np.ndarray:
    from pyscf.dft import gen_grid

    # MakeAngularGrid returns (x, y, z, weight); only the directions are used,
    # because the radial integration here is a plain average over the sphere.
    return np.asarray(gen_grid.MakeAngularGrid(points), dtype=float)[:, :3]


def free_atom_profile(
    symbol: str,
    basis: str,
    xc: Optional[str],
    radial_points: int = RADIAL_POINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spherically averaged ground-state density of a free atom.

    Returns ``(radii_bohr, rho_avg)`` with ``rho_avg`` in electrons/Bohr^3.
    Cached: the result depends only on the element, the basis and the
    functional, never on the molecule the atom ends up in.
    """
    key = (symbol, str(basis).lower(), xc, radial_points)
    hit = _FREE_ATOM_CACHE.get(key)
    if hit is not None:
        return hit

    from pyscf import dft, gto, scf

    mult = elements.ground_state_multiplicity(symbol)
    spin = max(int(mult) - 1, 0)
    mol = gto.M(atom=f"{symbol} 0.0 0.0 0.0", basis=basis, spin=spin,
                charge=0, verbose=0)
    if xc is None:
        mf = scf.ROHF(mol) if spin else scf.RHF(mol)
    else:
        mf = dft.ROKS(mol) if spin else dft.RKS(mol)
        mf.xc = xc
    mf.verbose = 0
    # No checkpointing: these are throwaway atoms, and writing one makes the
    # free-atom step depend on where the process may write files (it raised
    # PermissionError on a temp path before this was turned off).
    mol.chkfile = None
    mf.chkfile = None
    mf.run()

    dm = np.asarray(mf.make_rdm1(), dtype=float)
    if dm.ndim == 3:                      # open shell: sum the two channels
        dm = dm.sum(axis=0)

    radii = np.logspace(math.log10(RADIAL_MIN), math.log10(RADIAL_MAX),
                        int(radial_points))
    ang = _angular_grid()
    coords = (radii[:, None, None] * ang[None, :, :]).reshape(-1, 3)
    rho = _rho(mol, dm, coords)
    rho_avg = rho.reshape(len(radii), -1).mean(axis=1)
    rho_avg = np.maximum(rho_avg, 0.0)

    _FREE_ATOM_CACHE[key] = (radii, rho_avg)
    return radii, rho_avg


def hirshfeld_charges(
    mol,
    mf,
    xc: Optional[str],
    progress=None,
) -> Tuple[List[dict], dict]:
    """Hirshfeld charges, and the numbers that say they are trustworthy.

    The second return value carries the invariant a reader can check: the
    charges must sum to the molecular charge, and the promolecular weights must
    sum to one at every grid point that carries density.  Both are exact for a
    correct implementation and are what the contract gate asserts.
    """
    # An HF calculation has no `grids` attribute -- that belongs to the DFT
    # object -- so the integration grid is built here when it is missing.
    # Hirshfeld charges are a property of the density, not of the functional,
    # and they are just as quotable at HF.
    grids = getattr(mf, "grids", None)
    if grids is None:
        from pyscf.dft.gen_grid import Grids

        grids = Grids(mol)
        grids.level = 3
    if getattr(grids, "coords", None) is None:
        grids.build()
    coords = np.asarray(grids.coords, dtype=float)
    weights = np.asarray(grids.weights, dtype=float)

    dm = np.asarray(mf.make_rdm1(), dtype=float)
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    rho_mol = _rho(mol, dm, coords)

    atom_coords = np.asarray(mol.atom_coords(), dtype=float)
    natm = int(mol.natm)
    profiles = np.zeros((natm, coords.shape[0]))
    for a in range(natm):
        sym = mol.atom_symbol(a)
        radii, rho_avg = free_atom_profile(sym, mol.basis, xc)
        d = np.linalg.norm(coords - atom_coords[a], axis=1)
        profiles[a] = np.interp(d, radii, rho_avg, left=rho_avg[0], right=0.0)
        if progress:
            progress(int(100.0 * (a + 1) / natm), f"Hirshfeld: atom {a+1}/{natm}")

    pro = profiles.sum(axis=0)
    # Where the promolecular density is zero there is no molecule either, so
    # the weights are undefined but the product is zero.  Setting them to zero
    # keeps the sum of weights at one wherever it matters.
    live = pro > 1e-12
    safe = np.where(live, pro, 1.0)

    out: List[dict] = []
    pop_total = 0.0
    for a in range(natm):
        w = np.where(live, profiles[a] / safe, 0.0)
        pop = float(np.sum(w * rho_mol * weights))
        pop_total += pop
        z = float(mol.atom_charge(a))
        out.append({
            "atom": a + 1,
            "symbol": mol.atom_symbol(a),
            "charge": round(z - pop, 5),
            "population": round(pop, 5),
        })

    # The weights must sum to exactly 1 at every point that carries density --
    # that is what makes them a partition.  Their *total* grid weight is a
    # different number and a much smaller one: most of the grid sits where the
    # molecule has no density at all (ammonium: 14% of the weight is live), so
    # asserting on the total would fail a perfectly correct implementation.
    wsum_pt = np.where(live, pro / safe, 0.0)
    wdev = float(np.max(np.abs(wsum_pt[live] - 1.0))) if live.any() else 0.0
    wsum_all = float(np.sum(weights))
    live_frac = (float(np.sum(np.where(live, weights, 0.0)) / wsum_all)
                 if wsum_all else 0.0)
    nelec = float(mol.nelectron)
    # Integrating the molecular density on this grid must give back the number
    # of electrons.  If it does not, the density was evaluated wrongly and
    # every charge below is meaningless -- a much better thing to catch here
    # than to discover from a table that looks plausible.
    grid_nelec = float(np.sum(rho_mol * weights))
    diag = {
        "charge_sum": round(float(sum(c["charge"] for c in out)), 6),
        "expected_charge_sum": round(
            float(sum(float(mol.atom_charge(a)) for a in range(natm))) - nelec, 6),
        "electron_sum": round(float(pop_total), 6),
        "n_electrons": round(nelec, 6),
        "grid_integral_electrons": round(grid_nelec, 5),
        "grid_integral_error_pct": round(100.0 * (grid_nelec - nelec) / nelec, 4),
        "weight_sum_max_deviation": wdev,
        "live_grid_weight_fraction": round(live_frac, 6),
        "spherically_averaged_free_atoms": True,
        "lebedev_angular_points": LEBEDEV_POINTS,
    }
    return out, diag


def mayer_bond_orders(mol, mf, threshold: float = 0.05) -> dict:
    """Mayer bond orders from the total density matrix.

    ``B_AB = sum_{mu in A, nu in B} (P S)_{mu nu} (P S)_{nu mu}`` with
    ``P = P_alpha + P_beta``.  See the module docstring for why this form and
    not the spin-resolved one.
    """
    S = np.asarray(mf.get_ovlp(), dtype=float)
    dm = np.asarray(mf.make_rdm1(), dtype=float)
    P = dm.sum(axis=0) if dm.ndim == 3 else dm
    M = P @ S

    slices = mol.aoslice_by_atom()
    natm = int(mol.natm)
    syms = [mol.atom_symbol(a) for a in range(natm)]

    B = np.zeros((natm, natm))
    for a in range(natm):
        ra = slice(slices[a][2], slices[a][3])
        for b in range(a + 1, natm):
            rb = slice(slices[b][2], slices[b][3])
            # sum_{mu in A, nu in B} (PS)_{mu nu} (PS)_{nu mu}.  The second
            # factor is the (B, A) block transposed, NOT the transpose of the
            # (A, B) block: those two agree only when A and B have the same
            # basis functions in the same order.  Using blk * blk.T gives
            # N2 2.041 instead of 2.829 -- plausible enough to be missed.
            B[a, b] = B[b, a] = float(np.sum(M[ra, rb] * M[rb, ra].T))

    bonds = []
    for a in range(natm):
        for b in range(a + 1, natm):
            if B[a, b] >= threshold:
                bonds.append({
                    "i": a + 1, "j": b + 1,
                    "from": syms[a], "to": syms[b],
                    "order": round(B[a, b], 4),
                })
    bonds.sort(key=lambda x: -x["order"])

    # Mayer's valence: the bond-order sum over the *other* atoms.  Including
    # the diagonal would add the self term and make nitrogen in N2 look
    # heptavalent.
    off = B.sum(axis=1) - np.diag(B)
    valence = [{"atom": a + 1, "symbol": syms[a], "valence": round(float(off[a]), 4)}
               for a in range(natm)]

    return {
        "bonds": bonds,
        "valence": valence,
        "total_bond_order": round(float(off.sum() / 2.0), 4),
        "convention": ("total density matrix P = P_alpha + P_beta; the "
                       "spin-resolved form returns exactly half of this"),
        "threshold": threshold,
    }


def bonding_analysis(mol, mf, xc: Optional[str], progress=None) -> dict:
    """Both analyses, with the failure of either reported rather than hidden.

    A missing table is worse than a stated error: the UI would render an empty
    bond list and a reader would take it for a molecule with no bonds.
    """
    out: dict = {}
    try:
        out["mayer"] = mayer_bond_orders(mol, mf)
    except Exception as exc:                                 # noqa: BLE001
        out["mayer_error"] = f"{type(exc).__name__}: {exc}"
    try:
        charges, diag = hirshfeld_charges(mol, mf, xc, progress=progress)
        out["hirshfeld"] = {"charges": charges, "diagnostics": diag}
    except Exception as exc:                                 # noqa: BLE001
        out["hirshfeld_error"] = f"{type(exc).__name__}: {exc}"
    return out
