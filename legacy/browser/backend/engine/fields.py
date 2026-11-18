"""Real-space fields a paper plots, computed rather than drawn.

Everything here produces a number from the wavefunction on a grid:

  * **ELF** -- the electron localisation function of Becke and Edgecombe
    (J. Chem. Phys. 92, 5397, 1990), the figure that shows where electron
    pairs live: bonding regions, lone pairs, shell structure.
  * **Laplacian of the density** -- Bader's ∇²ρ, negative where charge is
    concentrated (a covalent bond, a lone pair) and positive where it is
    depleted.  This is the quantity QTAIM uses to classify interactions.
  * **Spin density** -- ρα − ρβ, zero everywhere for a closed shell and the
    map of where the unpaired electron sits for a radical.
  * **Density difference** -- ρ(AB) − ρ(A) − ρ(B) with the fragments
    calculated in the *full* basis set (ghost atoms), which is what separates
    genuine charge transfer from the basis-set artefact.
  * **2D slices** of any of the above, because the figure most papers print
    is a contour map in a plane, not an isosurface.

Two rules are enforced throughout.  First, no field is interpolated or
smoothed into looking like something it is not: every value is evaluated from
the AO basis at that point.  Second, the ELF reference term uses the
spin-resolved Thomas-Fermi constant, so a uniform electron gas comes out at
exactly 0.5 -- the check that the normalisation is right.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dft import DFTEngine, DFTError

Progress = Optional[Callable[[int, str], None]]

BOHR = 0.52917721092

# (3/10) (6 pi^2)^(2/3): the Thomas-Fermi constant for a *single* spin
# channel.  For a closed shell the two channels sum back to
# (3/10)(3 pi^2)^(2/3) = 2.8712, which is why the uniform electron gas gives
# ELF = 0.5 exactly.
TF_SIGMA = 0.3 * (6.0 * math.pi ** 2) ** (2.0 / 3.0)

# Below this density the ELF and the Laplacian are numerically meaningless:
# the reference term goes to zero faster than the numerator.
RHO_FLOOR = 1e-30


def _tick(progress: Progress, pct: int, msg: str) -> None:
    if progress:
        progress(max(0, min(100, int(pct))), msg)


# ======================================================================
#  grids
# ======================================================================
def box_grid(mol, spacing: float = 0.20, margin: float = 3.0
             ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """A regular grid in a box around the molecule, in bohr.

    ``margin`` is in angstrom so it reads like a chemical distance; the
    returned coordinates are in bohr because that is what libcint wants.
    """
    pos = np.asarray(mol.atom_coords(), dtype=float)          # bohr
    vdw = np.array([mol.atom_radius(i, "vdw") if hasattr(mol, "atom_radius")
                    else 1.7 for i in range(mol.natm)])
    if vdw.size == 0:
        vdw = np.array([1.7])
    lo = pos.min(axis=0) - (vdw.max() + margin) / BOHR
    hi = pos.max(axis=0) + (vdw.max() + margin) / BOHR
    n = np.maximum(((hi - lo) / spacing).astype(int) + 1, 2)
    ax = [np.linspace(lo[a], hi[a], int(n[a])) for a in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.ascontiguousarray(g), (int(n[0]), int(n[1]), int(n[2]))


def _plane_basis(kind: str, atoms: np.ndarray,
                 indices: Optional[Sequence[int]] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two orthonormal in-plane vectors and the plane's normal.

    ``kind`` is 'xy', 'xz', 'yz', or 'atoms' (three atom indices define the
    plane -- the one a paper picks when it wants to cut through a ring or a
    bond).
    """
    k = (kind or "xy").lower()
    if k == "atoms" and indices and len(indices) >= 3:
        i, j, l = (int(x) for x in indices[:3])
        p0, p1, p2 = atoms[i], atoms[j], atoms[l]
        u = p1 - p0
        v = p2 - p0
        nrm = np.cross(u, v)
        nlen = np.linalg.norm(nrm)
        if nlen < 1e-8:
            raise DFTError("those three atoms are collinear; they do not "
                           "define a plane")
        nrm = nrm / nlen
        e1 = u / np.linalg.norm(u)
        e2 = np.cross(nrm, e1)
        e2 = e2 / np.linalg.norm(e2)
        return p0, e1, e2
    axis = {"xy": 2, "xz": 1, "yz": 0}.get(k, 2)
    nrm = np.zeros(3)
    nrm[axis] = 1.0
    e1 = np.zeros(3)
    e1[(axis + 1) % 3] = 1.0
    e2 = np.zeros(3)
    e2[(axis + 2) % 3] = 1.0
    origin = atoms.mean(axis=0).copy()
    origin[axis] = 0.0
    return origin, e1, e2


def plane_grid(mol, kind: str = "xy", spacing: float = 0.10,
               margin: float = 2.0, n: int = 80,
               indices: Optional[Sequence[int]] = None
               ) -> Dict[str, object]:
    """A 2D grid in a plane, in bohr, plus the vectors to plot it in angstrom.

    The plane passes through the molecular centroid (or through the three
    named atoms) and extends far enough to cover the whole molecule, so the
    contour map is never clipped at the edge of the picture.
    """
    atoms = np.asarray(mol.atom_coords(), dtype=float)        # bohr
    p0, e1, e2 = _plane_basis(kind, atoms, indices)
    # project every atom into the plane to find how far the map must reach
    d = atoms - p0
    us = d @ e1
    vs = d @ e2
    pad = (2.0 + margin) / BOHR
    u0, u1 = us.min() - pad, us.max() + pad
    v0, v1 = vs.min() - pad, vs.max() + pad
    # keep it square: a stretched contour map distorts every feature
    span = max(u1 - u0, v1 - v0)
    uc = 0.5 * (u0 + u1)
    vc = 0.5 * (v0 + v1)
    u0, u1 = uc - 0.5 * span, uc + 0.5 * span
    v0, v1 = vc - 0.5 * span, vc + 0.5 * span
    u = np.linspace(u0, u1, int(n))
    v = np.linspace(v0, v1, int(n))
    U, V = np.meshgrid(u, v, indexing="ij")
    coords = p0 + U.reshape(-1, 1) * e1 + V.reshape(-1, 1) * e2
    return {
        "coords": np.ascontiguousarray(coords),
        "u": u, "v": v, "n": int(n),
        "origin": p0, "e1": e1, "e2": e2,
        "u_range": (float(u0 * BOHR), float(u1 * BOHR)),
        "v_range": (float(v0 * BOHR), float(v1 * BOHR)),
    }


# ======================================================================
#  spin channels
# ======================================================================
def spin_channels(mf) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One (density matrix, occupied MO coefficients, occupations) per spin.

    An unrestricted calculation gives two real channels.  A restricted one
    gives a single channel holding *half* the density with occupation 1,
    because the ELF is built from same-spin quantities: ρσ = ρ/2, τσ = τ/2,
    ∇ρσ = ∇ρ/2.  Folding that in here means every downstream formula sees
    exactly one case.
    """
    dm = np.asarray(mf.make_rdm1())
    C = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    if dm.ndim == 3:                                   # UKS
        out = []
        for s in range(dm.shape[0]):
            Cs = np.asarray(C[s])
            os_ = np.asarray(occ[s])
            sel = os_ > 1e-8
            out.append((np.asarray(dm[s]), Cs[:, sel], np.ones(int(sel.sum()))))
        return out
    sel = occ > 1e-8
    return [(0.5 * dm, C[:, sel], np.ones(int(sel.sum())))]


# ======================================================================
#  the fields
# ======================================================================
def _ao_and_grad(mol, coords: np.ndarray):
    """AO values and first derivatives, shaped (nderiv, npts, nao)."""
    A = np.asarray(mol.eval_gto("GTOval_sph_deriv1", coords))
    if A.ndim == 3 and A.shape[1] != len(coords) and A.shape[2] == len(coords):
        A = np.swapaxes(A, 1, 2)
    return A[0], A[1:4]


def density_and_gradient(mol, dm: np.ndarray, coords: np.ndarray,
                         chunk: int = 8192,
                         progress: Progress = None) -> Tuple[np.ndarray, np.ndarray]:
    """ρ and ∇ρ for one density matrix."""
    n = len(coords)
    rho = np.empty(n)
    grad = np.empty((n, 3))
    for s in range(0, n, chunk):
        c = coords[s:s + chunk]
        ao, a1 = _ao_and_grad(mol, c)
        k = len(c)
        Dphi = dm @ ao.T
        rho[s:s + k] = np.einsum("im,mi->i", ao, Dphi)
        grad[s:s + k] = 2.0 * np.einsum("aim,mi->ia", a1, Dphi)
        if progress is not None:
            progress(int(100 * (s + k) / n), f"density {s + k}/{n}")
    return rho, grad


def kinetic_density(mol, Cocc: np.ndarray, occ: np.ndarray, coords: np.ndarray,
                    chunk: int = 4096,
                    progress: Progress = None) -> np.ndarray:
    """τ = ½ Σ_i n_i |∇ψ_i|² -- the positive kinetic energy density.

    The ½ is not decoration: it is what makes τ equal the Thomas-Fermi
    density C_F ρ^(5/3) for a uniform electron gas, which is the condition
    the ELF reference term is normalised against.
    """
    n = len(coords)
    tau = np.empty(n)
    w = np.asarray(occ, dtype=float)
    for s in range(0, n, chunk):
        c = coords[s:s + chunk]
        ao, a1 = _ao_and_grad(mol, c)
        k = len(c)
        # (3, k, nocc)
        g = np.einsum("apm,mi->api", a1, Cocc)
        tau[s:s + k] = 0.5 * np.einsum("api,i,api->p", g, w, g)
        if progress is not None:
            progress(int(100 * (s + k) / n), f"kinetic density {s + k}/{n}")
    return tau


def elf(mol, channels: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        coords: np.ndarray, chunk: int = 4096,
        progress: Progress = None) -> np.ndarray:
    """Becke-Edgecombe electron localisation function on a set of points.

        D_σ  = τ_σ − |∇ρ_σ|² / (8 ρ_σ)
        D_σ⁰ = (3/10)(6π²)^(2/3) ρ_σ^(5/3)
        ELF  = 1 / (1 + (Σσ D_σ / Σσ D_σ⁰)²)

    The 1/8 is the von Weizsäcker term for a singly occupied orbital, and it
    is the number that makes D vanish wherever one orbital dominates -- which
    is exactly why ELF approaches 1 in a lone pair and in a bond.
    """
    n = len(coords)
    D = np.zeros(n)
    D0 = np.zeros(n)
    for dm_s, C_s, occ_s in channels:
        for s in range(0, n, chunk):
            c = coords[s:s + chunk]
            k = len(c)
            rho_s, grad_s = density_and_gradient(mol, dm_s, c)
            tau_s = kinetic_density(mol, C_s, occ_s, c)
            safe = np.maximum(rho_s, RHO_FLOOR)
            D[s:s + k] += tau_s - (grad_s ** 2).sum(axis=1) / (8.0 * safe)
            D0[s:s + k] += TF_SIGMA * np.power(safe, 5.0 / 3.0)
            if progress is not None:
                progress(int(100 * (s + k) / n), f"ELF {s + k}/{n}")
    safe0 = np.where(D0 > RHO_FLOOR, D0, 1.0)
    ratio = D / safe0
    out = 1.0 / (1.0 + ratio ** 2)
    # where there is no density at all the function is undefined, not 1
    out = np.where(D0 > RHO_FLOOR, out, 0.0)
    return np.clip(out, 0.0, 1.0)


def laplacian(mol, dm: np.ndarray, coords: np.ndarray, chunk: int = 4096,
              progress: Progress = None) -> np.ndarray:
    """∇²ρ = trace of the density Hessian.

    Negative where electronic charge is locally concentrated, positive where
    it is depleted: the sign convention Bader's QTAIM is built on.
    """
    from .analysis import density_derivatives

    _, _, hess = density_derivatives(mol, dm, coords, chunk=chunk,
                                     progress=progress)
    return hess[:, 0, 0] + hess[:, 1, 1] + hess[:, 2, 2]


def spin_density(mol, dms: Sequence[np.ndarray], coords: np.ndarray,
                 chunk: int = 8192,
                 progress: Progress = None) -> Tuple[np.ndarray, np.ndarray]:
    """(ρα − ρβ, ρα + ρβ) on the grid.

    The total comes along for free and is what the spin density has to be
    read against: a spin density of 0.01 means something completely
    different at a bond midpoint and in empty space.
    """
    n = len(coords)
    rho = np.zeros(n)
    for dm_s in dms:
        r, _ = density_and_gradient(mol, dm_s, coords, chunk=chunk,
                                    progress=progress)
        rho += r
    if len(dms) == 2:
        ra, _ = density_and_gradient(mol, dms[0], coords, chunk=chunk)
        rb, _ = density_and_gradient(mol, dms[1], coords, chunk=chunk)
        return ra - rb, rho
    return np.zeros(n), rho


# ======================================================================
#  density difference
# ======================================================================
def _ghost_mol(mol, keep: Sequence[int]):
    """The same basis set, with every atom outside ``keep`` turned ghost.

    A fragment calculated in the *full* basis is the only density-difference
    convention that is free of basis-set superposition artefact: without the
    ghost functions each fragment is artificially compact, and the difference
    map then mostly shows the missing basis rather than the interaction.
    """
    from pyscf import gto

    keep = set(int(i) for i in keep)
    atom = []
    nelec = 0
    for i in range(mol.natm):
        sym = mol.atom_symbol(i)
        # atom_coords() is in bohr and gto.M defaults to angstrom; passing
        # bohr numbers straight through silently stretches every bond by
        # 1.89x, which converges happily and produces a fragment density
        # that looks nothing like the fragment.
        xyz = tuple(float(x) * BOHR for x in mol.atom_coords()[i])
        atom.append((sym if i in keep else "ghost-" + sym, xyz))
        if i in keep:
            nelec += mol.atom_charge(i)
    nelec -= int(mol.charge)
    # a fragment with an odd number of electrons is a doublet; building it
    # as a closed shell would not converge and would be the wrong state
    spin = max(0, int(nelec) % 2)
    return gto.M(atom=atom, basis=mol.basis, charge=0, spin=spin,
                 verbose=0)


def fragment_density(mol, keep: Sequence[int], dm_full: np.ndarray,
                     coords: np.ndarray, functional: str,
                     solvation: Optional[str] = None,
                     progress: Progress = None) -> np.ndarray:
    """Density of one fragment, relaxed in the field of the ghost basis."""
    from pyscf import dft as pyscf_dft

    gh = _ghost_mol(mol, keep)
    spin = int(gh.spin)
    mf = pyscf_dft.RKS(gh) if spin == 0 else pyscf_dft.UKS(gh)
    mf.xc = functional
    mf.chkfile = None
    if solvation:
        try:
            from pyscf import solvent as pyscf_solvent
            mf = pyscf_solvent.ddCOSMO(mf)
        except Exception:                 # noqa: BLE001
            pass
    mf.kernel(dm0=None)
    if not mf.converged:
        raise DFTError(
            f"the fragment SCF did not converge ({functional}); the "
            f"difference map would compare a converged complex against an "
            f"unconverged fragment")
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    rho, _ = density_and_gradient(gh, dm, coords, progress=progress)
    return rho


def density_difference(engine: DFTEngine, fragments: Sequence[Sequence[int]],
                       coords: np.ndarray,
                       progress: Progress = None) -> Dict[str, object]:
    """Δρ = ρ(AB… ) − Σ ρ(fragment), each fragment in the full basis.

    Returns the grid and, because the figure is useless without them, the
    magnitudes: how much charge moves and where the largest accumulation and
    depletion sit.
    """
    mol = engine.mol
    mf = engine.mf
    if mf is None:
        raise DFTError("run an SCF before asking for a density difference")
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        dm = dm.sum(axis=0)

    _tick(progress, 5, "Density of the complex")
    rho_tot, _ = density_and_gradient(mol, dm, coords)
    parts = [rho_tot]
    notes: List[str] = []
    for k, frag in enumerate(fragments):
        _tick(progress, 10 + int(80 * k / max(1, len(fragments))),
              f"Fragment {k + 1}/{len(fragments)}")
        try:
            parts.append(fragment_density(
                mol, frag, dm, coords, engine.functional,
                getattr(engine, "solvation", None)))
        except Exception as exc:                     # noqa: BLE001
            notes.append(f"fragment {k + 1}: {exc}")
    if len(parts) != len(fragments) + 1:
        raise DFTError("; ".join(notes) or "a fragment could not be calculated")

    delta = parts[0].copy()
    for p in parts[1:]:
        delta -= p
    _tick(progress, 100, "Density difference complete")

    # integrate the positive and negative parts: the amount of charge that
    # moved is the number a paper quotes alongside the map
    vol = None
    return {
        "values": delta,
        "total_density": rho_tot,
        "n_fragments": len(fragments),
        "max": float(delta.max()),
        "min": float(delta.min()),
        "notes": notes,
    }


# ======================================================================
#  one entry point
# ======================================================================
FIELD_KINDS = {
    "elf": ("Electron localisation function", "dimensionless", 0.0, 1.0),
    "laplacian": ("Laplacian of the density", "e/bohr^5", None, None),
    "spin": ("Spin density", "e/bohr^3", None, None),
    "density": ("Electron density", "e/bohr^3", 0.0, None),
    "gradient": ("|grad rho|", "e/bohr^4", 0.0, None),
    "difference": ("Density difference", "e/bohr^3", None, None),
}


def compute_field(engine: DFTEngine, kind: str, plane: str = "xy",
                  n: int = 80, spacing: float = 0.10,
                  indices: Optional[Sequence[int]] = None,
                  fragments: Optional[Sequence[Sequence[int]]] = None,
                  progress: Progress = None) -> Dict[str, object]:
    """Evaluate one field on a 2D plane and return it ready to plot."""
    kind = (kind or "elf").lower()
    if kind not in FIELD_KINDS:
        raise DFTError(
            f"unknown field '{kind}'; available: "
            + ", ".join(sorted(FIELD_KINDS)))

    mf = engine.mf
    if mf is None:
        raise DFTError("run an SCF before asking for a field")

    mol = engine.mol
    grid = plane_grid(mol, plane, spacing=spacing, n=n, indices=indices)
    coords = grid["coords"]

    label, units, lo, hi = FIELD_KINDS[kind]
    rdm = np.asarray(mf.make_rdm1())
    dms = [np.asarray(x) for x in rdm] if rdm.ndim == 3 else [rdm]
    dm_t = rdm if rdm.ndim == 2 else rdm.sum(axis=0)

    if kind == "elf":
        _tick(progress, 10, "ELF on the plane")
        vals = elf(mol, spin_channels(mf), coords, progress=progress)
        note = ("Becke-Edgecombe ELF; 1 = a localised electron pair "
                "(bond or lone pair), 0.5 = a uniform electron gas, "
                "0 = no localisation")
    elif kind == "laplacian":
        _tick(progress, 10, "Laplacian on the plane")
        vals = laplacian(mol, dm_t, coords, progress=progress)
        note = ("Laplacian of the electron density; negative (blue) where "
                "charge is locally concentrated, positive (red) where it is "
                "depleted")
    elif kind == "spin":
        _tick(progress, 10, "Spin density on the plane")
        vals, total = spin_density(mol, dms, coords, progress=progress)
        if len(dms) < 2:
            note = ("This is a closed-shell calculation, so the spin density "
                    "is zero everywhere by definition; the map is blank "
                    "because that is the correct answer, not because it "
                    "failed. Ask for a radical or a triplet to see structure.")
        else:
            note = ("Spin density rho(alpha) - rho(beta); where the unpaired "
                    "electron actually is")
    elif kind == "density":
        _tick(progress, 10, "Density on the plane")
        vals, _ = density_and_gradient(mol, dm_t, coords, progress=progress)
        note = "Total electron density"
    elif kind == "gradient":
        _tick(progress, 10, "|grad rho| on the plane")
        _, g = density_and_gradient(mol, dm_t, coords, progress=progress)
        vals = np.linalg.norm(g, axis=1)
        note = "Magnitude of the density gradient"
    else:                                              # difference
        if not fragments or len(fragments) < 2:
            raise DFTError(
                "a density difference needs at least two fragments; give the "
                "atom index groups that make up each part")
        _tick(progress, 10, "Density difference on the plane")
        res = density_difference(engine, fragments, coords, progress=progress)
        vals = res["values"]
        note = ("rho(complex) - sum rho(fragment), every fragment relaxed in "
                "the full basis set (ghost atoms) so the map shows charge "
                "rearrangement, not the missing basis")

    # where the atoms sit inside the plane, in angstrom.  Without this the
    # map is an abstract picture: the reader cannot tell which lobe belongs
    # to which atom.
    rel = np.asarray(mol.atom_coords(), dtype=float) - grid["origin"]
    us = (rel @ grid["e1"]) * BOHR
    vs = (rel @ grid["e2"]) * BOHR
    atoms = [{"symbol": mol.atom_symbol(i),
              "u": round(float(us[i]), 4), "v": round(float(vs[i]), 4)}
             for i in range(mol.natm)]

    _tick(progress, 100, "Field complete")
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    stats = {
        "min": float(finite.min()) if finite.size else 0.0,
        "max": float(finite.max()) if finite.size else 0.0,
        "mean": float(finite.mean()) if finite.size else 0.0,
    }
    return {
        "field": kind,
        "label": label,
        "units": units,
        "plane": plane,
        "n": int(n),
        "u_range": grid["u_range"],
        "v_range": grid["v_range"],
        "u_axis": {"x": float(grid["e1"][0]), "y": float(grid["e1"][1]),
                   "z": float(grid["e1"][2])},
        "v_axis": {"x": float(grid["e2"][0]), "y": float(grid["e2"][1]),
                   "z": float(grid["e2"][2])},
        "values": vals.reshape(int(n), int(n)).T.tolist(),
        "atoms": atoms,
        "stats": stats,
        "levels": contour_levels(vals, kind),
        "note": note,
    }


def contour_levels(vals: np.ndarray, kind: str, count: int = 12) -> List[float]:
    """Contour values that actually describe the field.

    ELF has a natural, fixed scale (0 to 1) and papers draw it at the same
    values every time, so those are given literally.  The other fields are
    auto-ranged on a symmetric colour scale, because a Laplacian map drawn
    from its own min to its own max is dominated by the core and shows
    nothing in the bonding region.
    """
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return []
    if kind == "elf":
        return [round(x, 2) for x in
                np.linspace(0.0, 1.0, 11)]
    if kind == "spin" or kind == "difference":
        m = float(np.abs(v).max()) or 1.0
        # clip at the 99.5th percentile: the core singularity would otherwise
        # flatten the whole map
        m = float(np.percentile(np.abs(v), 99.5)) or m
        return [round(float(x), 6) for x in np.linspace(-m, m, 2 * count + 1)]
    if kind == "laplacian":
        lo = float(np.percentile(v, 1.0))
        hi = float(np.percentile(v, 99.0))
        m = max(abs(lo), abs(hi)) or 1.0
        return [round(float(x), 6) for x in np.linspace(-m, m, 2 * count + 1)]
    lo = float(np.percentile(v, 0.5))
    hi = float(np.percentile(v, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return [round(float(x), 6) for x in np.linspace(lo, hi, count)]
