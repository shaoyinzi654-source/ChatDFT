"""Reaction paths, barriers and transition states.

The figure this produces is the one every mechanism paper opens with: energy
on the vertical axis, reaction coordinate on the horizontal, a reactant well,
a barrier, a product well, and a number in kcal/mol on each step.

How it is computed, and what that costs in rigour:

  * The path is a **relaxed scan**.  At each value of the reaction coordinate
    every other degree of freedom is optimised on analytic gradients, with a
    harmonic restraint holding the coordinate at its target value.  That is a
    genuine minimum-energy path along that coordinate -- not a rigid scan,
    which would show a barrier inflated by the strain of frozen angles.
  * The **transition state** is the maximum of that path, refined by a
    parabolic fit and a further restrained optimisation, then *verified*:
    the analytic Hessian must show exactly one imaginary frequency, and the
    eigenvector of that mode must point along the reaction coordinate.  A
    stationary point with two imaginary frequencies is a saddle of the wrong
    order and is reported as a failure, not as a transition state.
  * The **endpoints** are fully optimised, so the reported reaction energy is
    between two real minima rather than between two points on a grid.

Nothing here is extrapolated or smoothed: every point on the curve is a
converged SCF at a relaxed geometry.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dft import DFTEngine, DFTError, _mol_to_xyz, _split_xyz

Progress = Optional[Callable[[int, str], None]]

HARTREE2KCAL = 627.5094740631
BOHR = 0.52917721092
AMU = 1822.888486209          # m_e, for mass weighting


def _tick(progress: Progress, pct: int, msg: str) -> None:
    if progress:
        progress(max(0, min(100, int(pct))), msg)


# ======================================================================
#  reaction coordinates
# ======================================================================
@dataclass
class Coordinate:
    """A bond length or a bond angle, in angstrom / degrees for reporting."""

    kind: str                       # "bond" | "angle"
    atoms: Tuple[int, ...]
    label: str = ""

    def __post_init__(self) -> None:
        self.atoms = tuple(int(a) for a in self.atoms)
        if self.kind == "bond" and len(self.atoms) != 2:
            raise DFTError("a bond coordinate needs exactly two atoms")
        if self.kind == "angle" and len(self.atoms) != 3:
            raise DFTError("an angle coordinate needs exactly three atoms")
        if self.kind == "torsion" and len(self.atoms) != 4:
            raise DFTError("a torsion coordinate needs exactly four atoms")
        if self.kind not in ("bond", "angle", "torsion"):
            raise DFTError(f"unknown coordinate kind '{self.kind}'")
        if not self.label:
            self.label = ("-" + self.kind).join(
                [f"{self.kind} {self.atoms}"])

    def display(self) -> str:
        return (f"{self.kind} {self.atoms}")

    def unit(self) -> str:
        return "angstrom" if self.kind == "bond" else "degree"

    def value(self, coords_bohr: np.ndarray) -> float:
        c = np.asarray(coords_bohr, dtype=float)
        if self.kind == "bond":
            i, j = self.atoms
            return float(np.linalg.norm(c[i] - c[j]) * BOHR)
        if self.kind == "angle":
            i, j, k = self.atoms
            u = c[i] - c[j]
            v = c[k] - c[j]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < 1e-12 or nv < 1e-12:
                raise DFTError("an angle coordinate has two atoms at one point")
            cosang = float(np.dot(u, v) / (nu * nv))
            return float(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
        # torsion (dihedral), the coordinate a conformational barrier
        # follows -- ethane, butadiene, H2O2 and every ring flip
        i, j, k, l = self.atoms
        b0 = c[i] - c[j]
        b1 = c[k] - c[j]
        b2 = c[l] - c[k]
        n1 = np.linalg.norm(b1)
        if n1 < 1e-12:
            raise DFTError("a torsion coordinate has three collinear atoms")
        b1 = b1 / n1
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = float(np.dot(v, w))
        y = float(np.dot(np.cross(b1, v), w))
        return float(math.degrees(math.atan2(y, x)))

    def gradient(self, coords_bohr: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """dq/dR, by central differences of the coordinate itself.

        Differentiating the geometry rather than the energy is cheap -- no
        SCF is involved -- and it removes the sign errors that the analytic
        angle derivative invites.
        """
        c = np.asarray(coords_bohr, dtype=float).copy()
        g = np.zeros_like(c)
        for a in range(c.shape[0]):
            for x in range(3):
                plus = c.copy()
                plus[a, x] += h
                minus = c.copy()
                minus[a, x] -= h
                g[a, x] = (self.value(plus) - self.value(minus)) / (2.0 * h)
        return g


def _bonds_of(symbols: Sequence[str], coords_bohr: np.ndarray,
              limit: float = 2.2) -> List[Tuple[int, int, float]]:
    """Candidate bonds by distance, longest first.

    Longest first because the bond a reaction breaks is usually the one that
    is already stretched.
    """
    out = []
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            if symbols[i] == "H" and symbols[j] == "H":
                continue
            d = float(np.linalg.norm(coords_bohr[i] - coords_bohr[j]) * BOHR)
            heavy = symbols[i] != "H" and symbols[j] != "H"
            if d < (limit if heavy else 1.6):
                out.append((i, j, d))
    out.sort(key=lambda t: -t[2])
    return out


def guess_coordinate(symbols: Sequence[str], coords_bohr: np.ndarray,
                     xyz: str = "") -> Coordinate:
    """The bond a reaction request most likely means."""
    import re

    # an explicit "between C1 and O2" or "scan 1 2"
    m = re.search(r"(?:between|scan)\D*(\d+)\D+(\d+)", xyz, re.IGNORECASE)
    if m:
        i, j = int(m.group(1)) - 1, int(m.group(2)) - 1
        if 0 <= i < len(symbols) and 0 <= j < len(symbols) and i != j:
            return Coordinate("bond", (i, j))
    bonds = _bonds_of(symbols, coords_bohr)
    if not bonds:
        raise DFTError("no bond could be chosen as the reaction coordinate")
    i, j, _ = bonds[0]
    return Coordinate("bond", (i, j))


# ======================================================================
#  restrained relaxation
# ======================================================================
def _gradient(engine: DFTEngine) -> Tuple[float, np.ndarray]:
    """Converged energy and analytic nuclear gradient, in hartree/bohr."""
    mf = engine.mf
    if mf is None or not mf.converged:
        res = engine.run_scf()
        mf = engine.mf
        if not mf.converged:
            raise DFTError("the SCF did not converge; no gradient is available")
        return float(res["energy_hartree"]), np.asarray(
            mf.nuc_grad_method().kernel(), dtype=float)
    return float(mf.e_tot), np.asarray(mf.nuc_grad_method().kernel(),
                                       dtype=float)


def constrained_relax(symbols: Sequence[str], coords_bohr: np.ndarray,
                      coord: Coordinate, target: float, charge: int,
                      mult: int, functional: str, basis: str,
                      max_steps: int = 40,
                      solvation: Optional[str] = None,
                      progress: Progress = None,
                      pct_from: int = 0, pct_to: int = 100
                      ) -> Dict[str, object]:
    """Relax every coordinate except ``coord``, which is held at ``target``.

    geomeTRIC takes the constraint as a file in its own format and holds the
    coordinate *exactly*, which a harmonic restraint does not: a restraint
    with a finite force constant leaves the coordinate short of the target by
    whatever the molecular forces can push, and the whole curve is then
    compressed towards the equilibrium geometry.  A restrained fallback is
    kept for the case where geomeTRIC cannot be used, and the caller is told
    which one ran.
    """
    import os
    import tempfile

    key = {"bond": "distance", "angle": "angle",
           "torsion": "dihedral"}[coord.kind]
    idx = " ".join(str(a + 1) for a in coord.atoms)      # geomeTRIC is 1-based
    # geomeTRIC's constraint file takes angles and dihedrals in DEGREES.  It
    # stores radians internally, so converting here produced a target of 3.14
    # *degrees* and folded every molecule onto itself.
    value = float(target)

    xyz = _xyz_block(symbols, coords_bohr)
    engine = DFTEngine(atom_xyz=xyz, charge=charge, multiplicity=mult,
                       functional=functional, basis=basis, solvation=solvation)

    def _finish(mol_eq, backend: str) -> Dict[str, object]:
        xyz_eq = _mol_to_xyz(mol_eq)
        sym2, ang = _split_xyz(xyz_eq)
        cb = np.asarray(ang, dtype=float) / BOHR
        sp = DFTEngine(atom_xyz=xyz_eq, charge=charge, multiplicity=mult,
                       functional=functional, basis=basis,
                       solvation=solvation)
        res = sp.run_scf(progress=None)
        return {
            "coords_bohr": cb,
            "energy_hartree": float(res["energy_hartree"]),
            "coord_value": coord.value(cb),
            "steps": int(getattr(mol_eq, "nsteps", 0) or 0),
            "xyz": xyz_eq,
            "backend": backend,
        }

    try:
        from pyscf.geomopt.geometric_solver import optimize as _geo_opt

        engine._build_mol()
        mf = engine._build_mf()
        path = os.path.join(tempfile.mkdtemp(prefix="chatdft_scan_"),
                            "constraints.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"$set\n{key} {idx} {value:.6f}\n")

        counter = [0]

        def _cb(envs):                                   # geomeTRIC callback
            counter[0] += 1
            if progress is not None and counter[0] % 2 == 0:
                pct = pct_from + int((pct_to - pct_from) * min(
                    1.0, counter[0] / max(1.0, max_steps)))
                progress(pct, f"relax step {counter[0]} holding "
                              f"{coord.display()} = {target:.4f} "
                              f"{coord.unit()}")

        mol_eq = _geo_opt(
            mf, constraints=path, maxsteps=max_steps, callback=_cb,
            convergence_energy=2e-5, convergence_grms=8e-4,
            convergence_gmax=1.2e-3, convergence_drms=2.4e-3,
            convergence_dmax=3.6e-3,
        )
        return _finish(mol_eq, "geometric (exact constraint)")
    except Exception as exc:                             # noqa: BLE001
        if progress is not None:
            progress(pct_from, f"constrained optimiser unavailable ({exc}); "
                               f"falling back to a harmonic restraint")
    return _restrained_relax(symbols, coords_bohr, coord, target, charge,
                             mult, functional, basis, max_steps=max_steps,
                             solvation=solvation, progress=progress,
                             pct_from=pct_from, pct_to=pct_to)


def _restrained_relax(symbols: Sequence[str], coords_bohr: np.ndarray,
                      coord: Coordinate, target: float, charge: int,
                      mult: int, functional: str, basis: str,
                      kappa: float = 4.0, max_steps: int = 40,
                      grms_tol: float = 6e-4,
                      solvation: Optional[str] = None,
                      progress: Progress = None,
                      pct_from: int = 0, pct_to: int = 100
                      ) -> Dict[str, object]:
    """Minimise E + kappa/2 (q - target)^2 with a damped BFGS step.

    The fallback path.  ``kappa`` is in hartree per (angstrom)^2 for a bond
    and per (radian)^2 for an angle.  It is deliberately stiff: the point of
    the scan is that the coordinate moves, not that the molecule relaxes
    around it.
    """
    from .elements import get as _element  # noqa: F401  (validates symbols)

    n = len(symbols)
    x = np.asarray(coords_bohr, dtype=float).copy()
    hess_inv = np.eye(3 * n) * 0.4

    def energy_and_grad(xx: np.ndarray) -> Tuple[float, np.ndarray]:
        xyz = _xyz_block(symbols, xx)
        eng = DFTEngine(atom_xyz=xyz, charge=charge, multiplicity=mult,
                        functional=functional, basis=basis,
                        solvation=solvation)
        e, g = _gradient(eng)
        q = coord.value(xx)
        dq = coord.gradient(xx)
        d = q - target
        e_eff = e + 0.5 * kappa * d * d
        g_eff = g + kappa * d * dq
        return e_eff, g_eff

    prev_e = None
    steps = 0
    for step in range(int(max_steps)):
        steps = step + 1
        e, g = energy_and_grad(x)
        gnorm = float(np.sqrt((g ** 2).mean()))
        if progress is not None:
            pct = pct_from + int((pct_to - pct_from) * (step + 1) / max_steps)
            progress(pct, f"relax step {step + 1}: |g| = {gnorm:.4f}, "
                          f"{coord.display()} = {coord.value(x):.4f}")
        if gnorm < grms_tol:
            break
        if prev_e is not None and abs(prev_e - e) < 1e-7 and step > 2:
            break
        prev_e = e
        # damped BFGS on the full 3N space; the restraint is folded into the
        # gradient, so no constraint machinery is needed
        step_vec = -hess_inv @ g.ravel()
        max_move = 0.25 / BOHR          # 0.25 A per step
        norm = float(np.linalg.norm(step_vec))
        if norm > max_move:
            step_vec *= max_move / norm
        x_new = x + step_vec.reshape(-1, 3)
        e_new, g_new = energy_and_grad(x_new)
        s = (x_new - x).ravel()
        y = (g_new - g).ravel()
        sy = float(s @ y)
        if sy > 1e-10:
            rho = 1.0 / sy
            A = np.eye(3 * n) - rho * np.outer(s, y)
            hess_inv = A @ hess_inv @ A.T + rho * np.outer(s, s)
        else:
            hess_inv = np.eye(3 * n) * 0.4
        if e_new > e and norm > 1e-4:
            # reject the step and shrink: a barrier point in a scan is not a
            # minimum of the restrained function, and the line search must not
            # walk over the ridge
            hess_inv *= 0.5
            x = x + 0.25 * step_vec.reshape(-1, 3)
        else:
            x = x_new

    e_final, _ = energy_and_grad(x)
    return {
        "coords_bohr": x,
        "energy_hartree": e_final,
        "coord_value": coord.value(x),
        "steps": steps,
        "xyz": _xyz_block(symbols, x),
        "backend": "harmonic restraint (fallback)",
    }


def _xyz_block(symbols: Sequence[str], coords_bohr: np.ndarray) -> str:
    lines = [str(len(symbols)), "reaction"]
    for s, c in zip(symbols, coords_bohr):
        lines.append(f"{s} {c[0] * BOHR:.10f} {c[1] * BOHR:.10f} "
                     f"{c[2] * BOHR:.10f}")
    return "\n".join(lines) + "\n"


def full_relax(symbols: Sequence[str], coords_bohr: np.ndarray,
               charge: int, mult: int, functional: str, basis: str,
               solvation: Optional[str] = None, max_steps: int = 60,
               progress: Progress = None, pct_from: int = 0,
               pct_to: int = 100) -> Dict[str, object]:
    """Unrestrained minimisation, used to confirm the two endpoints."""
    n = len(symbols)
    x = np.asarray(coords_bohr, dtype=float).copy()
    hess_inv = np.eye(3 * n) * 0.4

    def eg(xx):
        eng = DFTEngine(atom_xyz=_xyz_block(symbols, xx), charge=charge,
                        multiplicity=mult, functional=functional,
                        basis=basis, solvation=solvation)
        return _gradient(eng)

    prev_e = None
    for step in range(int(max_steps)):
        e, g = eg(x)
        gnorm = float(np.sqrt((g ** 2).mean()))
        if progress is not None:
            progress(pct_from + int((pct_to - pct_from) * (step + 1)
                                    / max_steps),
                     f"relaxing the endpoint: |g| = {gnorm:.4f}")
        if gnorm < 3e-4:
            break
        if prev_e is not None and abs(prev_e - e) < 1e-8 and step > 2:
            break
        prev_e = e
        sv = -hess_inv @ g.ravel()
        norm = float(np.linalg.norm(sv))
        if norm > 0.3 / BOHR:
            sv *= (0.3 / BOHR) / norm
        x_new = x + sv.reshape(-1, 3)
        e_new, g_new = eg(x_new)
        s = (x_new - x).ravel()
        y = (g_new - g).ravel()
        sy = float(s @ y)
        if sy > 1e-10:
            rho = 1.0 / sy
            A = np.eye(3 * n) - rho * np.outer(s, y)
            hess_inv = A @ hess_inv @ A.T + rho * np.outer(s, s)
        else:
            hess_inv = np.eye(3 * n) * 0.4
        x = x_new if e_new <= e else x + 0.3 * sv.reshape(-1, 3)
    e_final, g_final = eg(x)
    return {
        "coords_bohr": x,
        "energy_hartree": e_final,
        "grad_rms": float(np.sqrt((g_final ** 2).mean())),
        "xyz": _xyz_block(symbols, x),
    }


# ======================================================================
#  frequencies, to verify a stationary point
# ======================================================================
# An eigenvalue of the mass-weighted Hessian below this (in hartree per
# bohr^2 per amu) counts as imaginary.  It is about 50 cm^-1: below that the
# residual is quadrature noise, not a genuine unstable direction.
IMAG_TOL = -1.0e-4
# A frequency this large or larger is a real molecular vibration, so an
# imaginary mode at this magnitude is a broken geometry rather than a
# saddle.  Used by the gates, not by the sign test.
ABSURD_IMAG_CM = 1500.0


def _zero_mode_vectors(symbols: Sequence[str], coords_bohr: np.ndarray,
                       masses: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Mass-weighted translations and rotations, orthonormalised.

    Why this is needed.  The XC contribution to the Hessian is a *numerical*
    quadrature over a grid built for the molecule, so the computed Hessian is
    not exactly invariant under a rigid translation or rotation: those six
    directions come back with a curvature of the order of the grid noise
    instead of zero.  Left in place they are counted as imaginary modes --
    which is how a perfectly ordinary minimum (staggered ethane, gradient
    9e-5) was reported as a twelve-fold saddle.

    A linear molecule has only five: the rotation about its own axis produces
    a zero-length vector, which the orthonormalisation below discards.
    """
    n = len(symbols)
    sm = np.sqrt(np.repeat(masses, 3)).reshape(n, 3)
    com = (masses[:, None] * np.asarray(coords_bohr, dtype=float)).sum(0)
    com = com / masses.sum()
    rel = np.asarray(coords_bohr, dtype=float) - com
    raw = []
    for a in range(3):
        e = np.zeros(3)
        e[a] = 1.0
        raw.append((sm * e).ravel())                   # translation
        raw.append((sm * np.cross(e, rel)).ravel())    # rotation
    out: List[np.ndarray] = []
    for v in raw:
        w = v.copy()
        for u in out:
            w = w - float(w @ u) * u
        nv = float(np.linalg.norm(w))
        if nv > tol:
            out.append(w / nv)
    return np.array(out) if out else np.zeros((0, 3 * n))


def mode_analysis(symbols: Sequence[str], coords_bohr: np.ndarray, charge: int,
                  mult: int, functional: str, basis: str,
                  solvation: Optional[str] = None,
                  grid_level: int = 5) -> Dict[str, object]:
    """Analytic Hessian -> harmonic frequencies and the imaginary modes.

    Only the sign pattern matters here: a true transition state has exactly
    one imaginary frequency, a minimum none, and anything else is a saddle of
    the wrong order.

    Two details are easy to get wrong and both were.

      * ``mf.Hessian().kernel()`` returns shape ``(natm, natm, 3, 3)`` -- the
        two *atom* indices come first.  Reshaping that straight to ``(3N, 3N)``
        interleaves atom and Cartesian indices and produces a matrix that is
        not even symmetric; it is not the Hessian and its eigenvalues are
        meaningless.  The atom indices have to be brought together first:
        ``transpose(0, 2, 1, 3)``.  Checked against a Hessian built by finite
        differences of the analytic gradient, which agrees to 5e-7.
      * The translations and rotations are projected out before diagonalising,
        so quadrature noise cannot masquerade as an imaginary mode.
    """
    from . import elements

    xyz = _xyz_block(symbols, coords_bohr)
    eng = DFTEngine(atom_xyz=xyz, charge=charge, multiplicity=mult,
                    functional=functional, basis=basis, solvation=solvation,
                    grid_level=grid_level)
    eng.run_scf()
    hess = np.asarray(eng.mf.Hessian().kernel(), dtype=float)
    n3 = 3 * len(symbols)
    if hess.shape != (n3, n3):
        hess = hess.transpose(0, 2, 1, 3).reshape(n3, n3)
    hess = 0.5 * (hess + hess.T)

    masses = np.array([elements.get(s).mass for s in symbols])
    m3 = np.repeat(masses, 3)
    mw = hess / np.sqrt(np.outer(m3, m3))
    mw = 0.5 * (mw + mw.T)

    zero = _zero_mode_vectors(symbols, coords_bohr, masses)
    n_zero = int(zero.shape[0])
    if n_zero:
        proj = np.eye(n3) - zero.T @ zero
        mw = proj @ mw @ proj
        mw = 0.5 * (mw + mw.T)

    evals, evecs = np.linalg.eigh(mw)
    # hartree/(bohr^2 amu) -> cm^-1
    conv = math.sqrt(4.3597447222071e-18 / (5.29177210903e-11 ** 2
                                            * 1.66053906660e-27))
    conv /= 2.0 * math.pi * 2.99792458e10
    freqs = np.sign(evals) * np.sqrt(np.abs(evals)) * conv

    # Which eigenvectors are the projected-out rigid-body directions?  They
    # are the ones that live in the span of ``zero``; identifying them by
    # overlap rather than by "the six smallest eigenvalues" keeps a genuine
    # low-frequency vibration from being discarded when a mode is imaginary.
    if n_zero:
        ov = ((zero @ evecs) ** 2).sum(0)              # (3N,)
        rigid = set(np.argsort(ov)[-n_zero:].tolist())
    else:
        rigid = set()

    order = [int(i) for i in np.argsort(freqs) if int(i) not in rigid]
    vib_freqs = [float(freqs[i]) for i in order]
    imag = [(float(freqs[i]), evecs[:, i]) for i in order
            if evals[i] < IMAG_TOL]
    # The rotational symmetry number belongs to the thermochemistry, not to
    # the mode analysis, but it has to be read off the geometry we just put a
    # Hessian on -- and a barrier is a *difference* of two of these, so a
    # symmetry number of 1 on one side and 6 on the other would put
    # RT ln 6 = 1.06 kcal/mol of pure artefact into delta-G.
    try:
        from pyscf.hessian import thermo as _th
        sigma = int(_th.rotational_symmetry_number(eng.mol)) or 1
    except Exception:                                      # noqa: BLE001
        sigma = 1
    return {
        "symmetry_number": sigma,
        "frequencies": vib_freqs,
        "n_imaginary": len(imag),
        "imaginary": [f for f, _ in imag],
        "imaginary_vectors": [v for _, v in imag],
        "masses": masses,
        "n_rigid_modes": n_zero,
        "n_vibrations": len(vib_freqs),
        "imag_threshold_cm": round(
            math.copysign(math.sqrt(abs(IMAG_TOL)) * conv, IMAG_TOL), 1),
        "grid_level": int(grid_level),
        "hessian_shape": [int(x) for x in hess.shape],
        # Raw eigen-decomposition, in mass-weighted coordinates.  Kept for the
        # IRC, which uses it as a preconditioner; not part of the JSON payload.
        "eigenvalues": evals,
        "eigenvectors": evecs,
        "rigid_indices": sorted(rigid),
    }


def irc_preconditioner(ma: Dict[str, object],
                       floor: float = 1.0e-3) -> np.ndarray:
    """A metric in which the soft reaction mode dominates the descent step.

    Steepest descent in mass-weighted Cartesian coordinates is governed by the
    *stiffest* mode, not the reaction mode: the direction ``-g/|g|`` is set by
    whichever perpendicular vibration is furthest from its minimum, and the
    torsion then advances by a fraction of a degree per step.  Measured on
    ethane, a 0.10 amu^1/2 bohr step had to be cut to 0.0125 by the line search
    before it would descend at all, and ten steps moved the torsion 1.1 degrees
    out of the 60 needed -- an "IRC" that would need five hundred steps.

    Weighting each mode by the inverse of its curvature fixes that: the
    torsion (lambda ~ 3.5e-3) gets a weight a hundred times the C-H stretch
    (lambda ~ 0.34), so the step follows the valley instead of crossing it.
    This is the local quadratic approximation of Page and McIver with the
    Hessian held fixed at the saddle, which is what makes it cheap.
    """
    evals = np.asarray(ma["eigenvalues"], dtype=float)
    evecs = np.asarray(ma["eigenvectors"], dtype=float)
    rigid = set(int(i) for i in ma.get("rigid_indices", []))
    n = evals.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        if i in rigid:
            continue
        w = 1.0 / max(abs(float(evals[i])), float(floor))
        u = evecs[:, i]
        P += w * np.outer(u, u)
    return P


def _mode_alignment(vec: np.ndarray, coord: Coordinate,
                    coords_bohr: np.ndarray, masses: np.ndarray) -> float:
    """How much of the imaginary mode lies along the reaction coordinate.

    The mode is mass-weighted; the coordinate gradient is not, so the
    gradient is mass-weighted the same way before the projection.  Without
    this the overlap is meaningless and a mode that is really a methyl
    rotation can look like it points at the breaking bond.
    """
    m3 = np.repeat(masses, 3)
    dq = coord.gradient(coords_bohr).ravel() / np.sqrt(m3)
    nv = float(np.linalg.norm(vec))
    nd = float(np.linalg.norm(dq))
    if nv < 1e-12 or nd < 1e-12:
        return 0.0
    return abs(float(vec @ dq) / (nv * nd))


# ======================================================================
#  intrinsic reaction coordinate
# ======================================================================
def geometry_fingerprint(coords_bohr: np.ndarray) -> np.ndarray:
    """Sorted interatomic distances, in angstrom.

    This is the rotation-, translation- and atom-order-invariant way to ask
    "are these two structures the same conformer?".  It is needed because the
    obvious tests both fail on the ethane torsion:

      * a raw RMSD between the two relaxed endpoints is dominated by the rigid
        rotation that separates them, because they were relaxed independently;
      * the reaction coordinate itself does not settle it either.  Relaxing
        from 30 deg reaches the 60 deg minimum and relaxing from 150 deg
        reaches the 180 deg one, 120 deg away -- yet 60 and 180 are the *same*
        staggered conformer, related by the methyl's three-fold symmetry.

    Sorted distances are identical for 60 and 180 (same point group) and
    different for HCN and HNC, which is exactly the discrimination wanted.
    """
    c = np.asarray(coords_bohr, dtype=float)
    n = c.shape[0]
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            d.append(float(np.linalg.norm(c[i] - c[j])) * BOHR)
    return np.sort(np.array(d))


def irc_from_ts(symbols: Sequence[str], ts_coords_bohr: np.ndarray,
                imaginary_vec: np.ndarray, masses: np.ndarray,
                charge: int, mult: int, functional: str, basis: str,
                solvation: Optional[str] = None, steps: int = 8,
                ds: float = 0.10, start: float = 0.25,
                coord: Optional[Coordinate] = None,
                degenerate: bool = False,
                q_reactant: Optional[float] = None,
                q_product: Optional[float] = None,
                ts_energy: Optional[float] = None,
                preconditioner: Optional[np.ndarray] = None,
                grid_level: int = 5,
                progress: Progress = None, pct_from: int = 0,
                pct_to: int = 100) -> Dict[str, object]:
    """Mass-weighted steepest descent from the saddle, in both directions.

    Why this is not redundant with the Hessian check.  One imaginary
    frequency proves the point is a *saddle*; it does not prove the saddle
    belongs to the reaction being drawn.  Following the imaginary mode down
    both ways does: each branch must reach a minimum, and the two minima must
    be the reactant and the product.

    The integration is a unit-speed descent in mass-weighted Cartesian
    coordinates,
        dy/ds = -g_y / |g_y|,   y = sqrt(m) x,   g_y = g_x / sqrt(m),
    with the rigid-body directions projected out of ``g_y`` at every step.
    Without that projection the residual translation the numerical gradient
    carries is integrated along with the chemistry and the molecule simply
    walks off, which looks like a converging IRC until the energies are
    checked.

    Two details that this got wrong first time round, and that the numbers
    make visible:

      * the step is ``-g_y/|g_y|`` on *both* branches.  The sign belongs to
        the initial displacement off the saddle, not to the step: applying it
        to the step as well marched the backward branch straight back up the
        hill (it reported +29 kcal/mol at the end of six "descent" steps).
      * the first displacement is larger than the subsequent steps.  At the
        saddle the gradient is nearly zero -- the torsion is at its maximum,
        so dE/dq vanishes -- and the direction ``-g/|g|`` is then mostly
        quadrature noise.  Starting 0.25 amu^1/2 bohr down the mode puts the
        first real gradient an order of magnitude above that noise, which is
        why the frequencies are computed on a denser grid than the scan uses.
    """
    n = len(symbols)
    m3 = np.repeat(masses, 3)
    sm = np.sqrt(m3)
    step_ceiling = float(ds)
    x_ts = np.asarray(ts_coords_bohr, dtype=float).copy()
    y_ts = (x_ts.ravel() * sm)

    v = np.asarray(imaginary_vec, dtype=float).ravel()
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        raise DFTError("the imaginary mode has zero length; no IRC is possible")
    v = v / nv
    zero = _zero_mode_vectors(symbols, x_ts, masses)
    if zero.shape[0]:
        proj = np.eye(3 * n) - zero.T @ zero
        v = proj @ v
        v = v / float(np.linalg.norm(v))

    def energy_and_grad(xx: np.ndarray):
        eng = DFTEngine(atom_xyz=_xyz_block(symbols, xx), charge=charge,
                        multiplicity=mult, functional=functional,
                        basis=basis, solvation=solvation,
                        grid_level=grid_level)
        e, g = _gradient(eng)
        return e, np.asarray(g, dtype=float)

    def to_x(yy: np.ndarray) -> np.ndarray:
        return yy.reshape(-1, 3) / sm.reshape(-1, 3)

    branches: Dict[str, object] = {}
    total = max(1, 2 * steps)
    done = 0
    for side, name in ((+1.0, "forward"), (-1.0, "backward")):
        ds = float(step_ceiling)
        y = y_ts + side * float(start) * v
        x = to_x(y)
        e, g = energy_and_grad(x)
        pts = []
        used: List[float] = []
        for step in range(int(steps)):
            gy = g.ravel() / sm
            if zero.shape[0]:
                gy = gy - zero.T @ (zero @ gy)
            nrm = float(np.linalg.norm(gy))
            row = {
                "step": step,
                "energy_hartree": float(e),
                "grad_rms": float(np.sqrt((g ** 2).mean())),
                "coord": (round(float(coord.value(x)), 4) if coord else None),
                "step_size": round(used[-1], 4) if used else None,
                "xyz": _xyz_block(symbols, x),
            }
            pts.append(row)
            done += 1
            _tick(progress, pct_from + int((pct_to - pct_from) * done / total),
                  f"IRC {name}: step {step + 1}/{steps}, "
                  f"|g| = {row['grad_rms']:.4f}")
            if nrm < 1e-8:
                break
            # A descent path cannot go uphill, so the step is accepted only
            # if it lowers the energy; otherwise it is halved and retried.
            # Plain fixed-step descent on a soft torsion zig-zags across the
            # stiff C-H modes instead: the direction -g/|g| is dominated by
            # whichever perpendicular mode is furthest from its minimum, and
            # a 0.10 amu^1/2 bohr step along it costs ~0.8 kcal/mol of
            # stretch energy, so every second point came out *higher* than
            # the one before it and the "IRC" climbed its own well.
            #
            # The direction is preconditioned when the saddle's Hessian is
            # available, so the step follows the soft reaction mode instead of
            # being dominated by the stiff perpendicular ones.
            d = gy if preconditioner is None else (preconditioner @ gy)
            dn = float(np.linalg.norm(d))
            if dn < 1e-14:
                break
            ds_cur = float(ds)
            step_ok = False
            for _ in range(10):
                y_try = y - ds_cur * (d / dn)
                x_try = to_x(y_try)
                e_try, g_try = energy_and_grad(x_try)
                if e_try <= e:
                    step_ok = True
                    break
                ds_cur *= 0.5
            if not step_ok:
                break                       # no descent: stop honestly here
            y, x, e, g = y_try, x_try, e_try, g_try
            used.append(ds_cur)
            # let the step grow back after a success, or it stays pinned at
            # whatever the first hard step forced it down to
            ds = min(float(step_ceiling), ds_cur * 1.6)

        energies = [p["energy_hartree"] for p in pts]
        descends = all(energies[i + 1] <= energies[i] + 1e-9
                       for i in range(len(energies) - 1))
        end = pts[-1]
        # Which minimum did this branch reach?  Judged on the reaction
        # coordinate, which is rotation- and translation-invariant, rather
        # than on a geometry RMSD: the two endpoints are relaxed
        # independently and can end up in different orientations, which would
        # make a raw RMSD report whichever one happened to be aligned.
        toward = None
        if (not degenerate and end["coord"] is not None
                and q_reactant is not None and q_product is not None):
            toward = ("reactant"
                      if abs(end["coord"] - q_reactant)
                      < abs(end["coord"] - q_product) else "product")
        branches[name] = {
            "points": pts,
            "n_steps": len(pts),
            "descends_monotonically": bool(descends),
            "final_energy_kcal": round(
                (energies[-1] - energies[0]) * HARTREE2KCAL, 3),
            "total_drop_kcal": round(
                (energies[-1] - float(ts_energy)) * HARTREE2KCAL, 3)
            if ts_energy is not None else None,
            "final_coord": end["coord"],
            "mean_step_size": round(float(np.mean(used)), 4) if used else None,
            "toward": toward,
        }

    fwd, bwd = branches["forward"], branches["backward"]
    connects = (fwd.get("toward") is not None
                and bwd.get("toward") is not None
                and fwd.get("toward") != bwd.get("toward"))
    notes = []
    if not (fwd["descends_monotonically"] and bwd["descends_monotonically"]):
        notes.append("At least one branch does not descend at every step; the "
                     "step size is probably too large for this surface.")
    if connects:
        notes.append(f"The two branches reach different minima "
                     f"({bwd['toward']} and {fwd['toward']}), so the saddle "
                     f"connects the reactant to the product.")
    elif degenerate:
        notes.append("The two endpoints relaxed to the same conformer (same "
                     "sorted interatomic distances, same energy), so this "
                     "reaction is degenerate: the wells either side of the "
                     "maximum are equivalent, and the two branches are mirror "
                     "images of one another by symmetry. No direction is "
                     "assigned for that reason.")
    elif fwd.get("toward") is not None:
        notes.append(f"Both branches approach the same minimum "
                     f"({fwd.get('toward')}); the two wells are equivalent.")
    return {
        "steps": int(steps),
        "step_size": float(ds),
        "start_displacement": float(start),
        "grid_level": int(grid_level),
        "preconditioned": bool(preconditioner is not None),
        "degenerate_reaction": bool(degenerate),
        "branches": branches,
        "connects_reactant_to_product": bool(connects),
        "note": " ".join(notes),
    }


# ======================================================================
#  the whole job
# ======================================================================
def reaction_path(engine: DFTEngine, coord: Optional[Coordinate] = None,
                  npoints: int = 9, span: Optional[float] = None,
                  max_steps: int = 40, verify_ts: bool = True,
                  q_from: Optional[float] = None,
                  q_to: Optional[float] = None,
                  irc: bool = True, irc_steps: Optional[int] = None,
                  thermo: bool = True, thermo_max_atoms: int = 20,
                  temperature: float = 298.15,
                  progress: Progress = None) -> dict:
    """Relaxed scan along one coordinate, with the barrier and a verified TS.

    ``thermo`` puts the Hessian-derived corrections on the barrier.  A relaxed
    scan yields a difference of electronic energies at 0 K; the zero-point
    term is typically 1-3 kcal/mol and T*dS adds more, so an electronic
    difference is not an activation energy and must not be reported as one.
    It costs one analytic Hessian on top of the one the TS verification
    already pays for, so it is capped by ``thermo_max_atoms``.
    """
    symbols, coords_ang = _split_xyz(engine.atom_xyz)
    coords_bohr = np.asarray(coords_ang, dtype=float) / BOHR

    if coord is None:
        coord = guess_coordinate(symbols, coords_bohr, engine.atom_xyz)

    q0 = coord.value(coords_bohr)
    # The window runs *outward from the current geometry*, not symmetrically
    # around it.  A reaction path starts at the reactant and stretches; a
    # symmetric window spends half its points on the repulsive wall, which
    # costs as much as the interesting half and tells you nothing.
    if q_from is not None and q_to is not None:
        values = np.linspace(float(q_from), float(q_to), npoints)
    elif coord.kind == "bond":
        step = float(span) if span else 0.12
        lo = max(0.60, q0 - 0.15)
        hi = lo + step * (npoints - 1)
        values = np.linspace(lo, hi, npoints)
    elif coord.kind == "torsion":
        step = float(span) if span else 15.0
        lo = q0
        hi = q0 + step * (npoints - 1)
        values = np.linspace(lo, hi, npoints)
    else:
        step = float(span) if span else 12.0
        if q0 >= 150.0:
            # a near-linear angle is the reactant; the path bends it inwards
            hi = q0
            lo = max(20.0, q0 - step * (npoints - 1))
        else:
            lo = max(20.0, q0 - step * (npoints - 1) / 2.0)
            hi = min(178.0, lo + step * (npoints - 1))
        values = np.linspace(lo, hi, npoints)

    _tick(progress, 2, f"Relaxed scan along {coord.display()} from "
                       f"{values[0]:.3f} to {values[-1]:.3f} {coord.unit()}")
    t0 = time.time()
    points = []
    for k, q in enumerate(values):
        pct = 2 + int(76 * k / max(1, len(values)))
        r = constrained_relax(
            symbols, coords_bohr, coord, float(q), engine.charge,
            engine.mult, engine.functional, engine.basis, max_steps=max_steps,
            solvation=engine.solvation, progress=progress,
            pct_from=pct, pct_to=pct + int(76 / max(1, len(values))))
        points.append({
            "coord": round(float(r["coord_value"]), 5),
            "target": round(float(q), 5),
            "energy_hartree": float(r["energy_hartree"]),
            "steps": int(r["steps"]),
            "xyz": r["xyz"],
            "coords_bohr": r["coords_bohr"],
        })

    energies = np.array([p["energy_hartree"] for p in points])

    # ---- the barrier -----------------------------------------------------
    imax = int(np.argmax(energies))
    bracketed = 0 < imax < len(points) - 1
    if not bracketed:
        # A monotonic curve is a real result -- it says the coordinate is not
        # the one the reaction follows, or the window is too narrow -- so it
        # is reported with the profile rather than thrown away.  What must
        # not happen is calling the endpoint a transition state.
        _tick(progress, 80, "No barrier inside the window")
        ts = None
    else:
        _tick(progress, 80, "Refining the maximum")
        ts = _refine_maximum(symbols, coord, points, imax, engine,
                             max_steps=max_steps, progress=progress)

    # ---- the two endpoints, fully relaxed --------------------------------
    _tick(progress, 86, "Relaxing the reactant")
    left = full_relax(symbols, points[0]["coords_bohr"], engine.charge,
                      engine.mult, engine.functional, engine.basis,
                      engine.solvation, progress=progress, pct_from=86,
                      pct_to=90)
    _tick(progress, 90, "Relaxing the product")
    right = full_relax(symbols, points[-1]["coords_bohr"], engine.charge,
                       engine.mult, engine.functional, engine.basis,
                       engine.solvation, progress=progress, pct_from=90,
                       pct_to=94)

    e_react = float(left["energy_hartree"])
    e_prod = float(right["energy_hartree"])
    e_ts = float(ts["energy_hartree"]) if ts else None
    barrier = (e_ts - e_react) * HARTREE2KCAL if ts else None
    reaction = (e_prod - e_react) * HARTREE2KCAL

    # One zero for the whole figure.  The scan points and the two relaxed
    # endpoints were originally referenced to two different minima, which put
    # the curve and the wells on different scales; a plot of both would then
    # have been wrong by whatever the two references differ by, with no sign
    # that anything was off.  Everything below is relative to the lowest
    # energy anywhere on the path.
    e0 = float(min([float(energies.min()), e_react, e_prod]
                   + ([e_ts] if e_ts is not None else [])))
    ref = e0
    for p in points:
        p["relative_kcal"] = round(
            (p["energy_hartree"] - e0) * HARTREE2KCAL, 3)

    # ---- verify the transition state -------------------------------------
    verification: Dict[str, object] = {"checked": False}
    imaginary_vec = None
    ts_masses = None
    irc_precond = None
    ts_modes = None
    if ts is None:
        verification = {
            "checked": False,
            "reason": ("no maximum was found inside the scanned window, so "
                       "there is no stationary point to verify"),
        }
    elif verify_ts:
        _tick(progress, 95, "Verifying the transition state (Hessian)")
        try:
            ma = mode_analysis(symbols, ts["coords_bohr"], engine.charge,
                               engine.mult, engine.functional, engine.basis,
                               engine.solvation)
            align = 0.0
            if ma["n_imaginary"]:
                align = _mode_alignment(ma["imaginary_vectors"][0], coord,
                                        ts["coords_bohr"],
                                        np.asarray(ma["masses"]))
                imaginary_vec = np.asarray(ma["imaginary_vectors"][0])
                ts_masses = np.asarray(ma["masses"])
                irc_precond = irc_preconditioner(ma)
            verification = {
                "checked": True,
                "n_imaginary": ma["n_imaginary"],
                "imaginary_cm": [round(f, 1) for f in ma["imaginary"]],
                "mode_alignment": round(float(align), 3),
                "is_transition_state": bool(
                    ma["n_imaginary"] == 1 and align > 0.4),
                "frequencies_cm": [round(f, 1) for f in ma["frequencies"]],
                "symmetry_number": int(ma.get("symmetry_number", 1)),
                "n_real_modes": int(sum(1 for f in ma["frequencies"]
                                        if f > 0.0)),
            }
            ts_modes = ma
        except Exception as exc:                       # noqa: BLE001
            verification = {"checked": False, "error": str(exc)}
            ts_modes = None
    else:
        ts_modes = None

    # ---- the barrier is not one number -----------------------------------
    # What a relaxed scan measures is a difference of *electronic* energies at
    # 0 K, with the nuclei frozen.  Papers quote activation energies, and the
    # two are not the same quantity: the zero-point term alone moves a barrier
    # by 1-3 kcal/mol -- more than the whole barrier for a methyl torsion --
    # and the entropy term moves it again.  So the electronic difference is
    # reported as what it is, and the Hessian-derived corrections are put
    # beside it when they can be afforded.
    #
    # At the saddle the reaction coordinate is not a vibration, so it is left
    # out of the partition function -- the standard transition-state-theory
    # treatment, and the reason a TS has 3N-7 real modes where the minimum has
    # 3N-6.  That one missing mode is exactly the zero-point difference that
    # lowers the corrected barrier below the electronic one.
    barrier_thermo: Dict[str, object] = {"computed": False}
    if ts is None:
        barrier_thermo["reason"] = "no transition state, so no barrier to correct"
    elif not verification.get("is_transition_state"):
        barrier_thermo["reason"] = ("the maximum is not a verified saddle, so "
                                    "its partition function would be wrong")
    elif ts_modes is None:
        barrier_thermo["reason"] = "no Hessian at the transition state"
    elif not thermo:
        barrier_thermo["reason"] = "thermochemistry was switched off"
    elif len(symbols) > thermo_max_atoms:
        barrier_thermo["reason"] = (
            f"zero-point and Gibbs corrections need a Hessian "
            f"({len(symbols)} atoms is past the {thermo_max_atoms}-atom limit "
            f"for an interactive job)")
    else:
        try:
            _tick(progress, 96, "Zero-point and Gibbs corrections (Hessian)")
            from .analysis import thermochemistry as _thermo_fn

            def _stationary(freqs, coords_b, energy, sigma):
                return _thermo_fn(
                    list(symbols), np.asarray(coords_b, dtype=float) * BOHR,
                    list(freqs), float(energy), temperature=temperature,
                    symmetry_number=int(sigma or 1))

            th_ts = _stationary(ts_modes["frequencies"], ts["coords_bohr"],
                                e_ts, ts_modes.get("symmetry_number"))
            ma_r = mode_analysis(symbols, left["coords_bohr"], engine.charge,
                                 engine.mult, engine.functional, engine.basis,
                                 engine.solvation)
            if ma_r["n_imaginary"]:
                barrier_thermo["reason"] = (
                    f"the relaxed reactant still has "
                    f"{ma_r['n_imaginary']} imaginary mode(s), so it is not a "
                    f"minimum and its partition function is meaningless")
            else:
                th_r = _stationary(ma_r["frequencies"], left["coords_bohr"],
                                   e_react, ma_r.get("symmetry_number"))
                d = HARTREE2KCAL
                barrier_thermo = {
                    "computed": True,
                    "temperature_K": round(float(temperature), 2),
                    "electronic_kcal": round(float(barrier), 3),
                    "zpe_corrected_kcal": round(
                        (th_ts["zpe_corrected_energy_hartree"]
                         - th_r["zpe_corrected_energy_hartree"]) * d, 3),
                    "enthalpy_kcal": round(
                        (th_ts["enthalpy_hartree"]
                         - th_r["enthalpy_hartree"]) * d, 3),
                    "gibbs_kcal": round(
                        (th_ts["gibbs_hartree"]
                         - th_r["gibbs_hartree"]) * d, 3),
                    "reactant": {
                        "n_real_modes": int(sum(1 for f in ma_r["frequencies"]
                                                if f > 0.0)),
                        "zpe_kcal": round(
                            th_r["zpe_hartree"] * d, 3),
                        "symmetry_number": int(ma_r.get("symmetry_number", 1)),
                    },
                    "transition_state": {
                        "n_real_modes": int(
                            sum(1 for f in ts_modes["frequencies"]
                                if f > 0.0)),
                        "zpe_kcal": round(
                            th_ts["zpe_hartree"] * d, 3),
                        "symmetry_number": int(
                            ts_modes.get("symmetry_number", 1)),
                    },
                    "note": (
                        "Harmonic oscillator / rigid rotor at "
                        f"{temperature:g} K and 1 atm. The transition state's "
                        "imaginary mode is not a vibration and is left out of "
                        "its partition function, so it has one real mode "
                        "fewer than the reactant."),
                }
        except Exception as exc:                       # noqa: BLE001
            barrier_thermo = {"computed": False,
                              "reason": f"corrections failed: {exc}"}

    # ---- the intrinsic reaction coordinate -------------------------------
    # Only meaningful once the saddle is real: following a mode that is not
    # the reaction mode downhill produces a curve that looks like an IRC and
    # proves nothing.
    irc_result = None
    if irc and imaginary_vec is not None and verification.get(
            "is_transition_state"):
        # Each IRC point is a full SCF plus a gradient on a dense grid, so the
        # number of steps has to shrink as the molecule grows or a 20-atom
        # barrier would cost longer than the scan that found it.
        n_irc = int(irc_steps) if irc_steps else (
            12 if len(symbols) <= 12 else 8 if len(symbols) <= 24 else 5)
        # A torsion like ethane's is degenerate: both ends of the scan relax
        # to the same conformer, so "which minimum does this branch reach"
        # has no answer and must not be invented.  Two tests, both of which
        # have to hold, and both invariant under the rigid rotation that
        # separates two independently relaxed endpoints:
        #   * the energies agree, and
        #   * the sorted interatomic distances agree, which sees through the
        #     120 deg shift between the 60 and 180 deg staggered conformers.
        fp_react = geometry_fingerprint(np.asarray(left["coords_bohr"],
                                                   dtype=float))
        fp_prod = geometry_fingerprint(np.asarray(right["coords_bohr"],
                                                  dtype=float))
        same_shape = (fp_react.shape == fp_prod.shape
                      and float(np.abs(fp_react - fp_prod).max()) < 0.02)
        degenerate = (same_shape
                      and abs(e_react - e_prod) * HARTREE2KCAL < 0.15)
        q_react = float(coord.value(np.asarray(left["coords_bohr"],
                                               dtype=float)))
        q_prod = float(coord.value(np.asarray(right["coords_bohr"],
                                              dtype=float)))
        _tick(progress, 97, "Following the imaginary mode down both ways (IRC)")
        try:
            irc_result = irc_from_ts(
                symbols, ts["coords_bohr"], imaginary_vec,
                ts_masses if ts_masses is not None else np.ones(len(symbols)),
                engine.charge, engine.mult, engine.functional, engine.basis,
                solvation=engine.solvation, steps=n_irc,
                ds=0.10, start=0.25, coord=coord,
                degenerate=degenerate, q_reactant=q_react, q_product=q_prod,
                ts_energy=e_ts,
                preconditioner=irc_precond,
                progress=progress, pct_from=97, pct_to=99)
        except Exception as exc:                       # noqa: BLE001
            irc_result = {"error": str(exc)}
    elif irc and ts is not None:
        irc_result = {
            "skipped": ("the maximum is not a verified transition state, so "
                        "following its lowest mode would not follow the "
                        "reaction"),
        }

    _tick(progress, 100, "Reaction path complete")
    return {
        "job_type": "reaction",
        "coordinate": {"kind": coord.kind, "atoms": list(coord.atoms),
                       "label": coord.display(), "unit": coord.unit()},
        "points": [{k: v for k, v in p.items() if k != "coords_bohr"}
                   for p in points],
        "reference_hartree": ref,
        "barrier_bracketed": bool(bracketed),
        "barrier_kcal": round(barrier, 3) if barrier is not None else None,
        "barrier_thermo": barrier_thermo,
        "reaction_kcal": round(reaction, 3),
        "ts": ({
            "coord": ts["coord"],
            "energy_hartree": e_ts,
            "relative_kcal": round((e_ts - e0) * HARTREE2KCAL, 3),
            "xyz": ts["xyz"],
        } if ts else None),
        "reactant": {
            "energy_hartree": e_react,
            "relative_kcal": round((e_react - e0) * HARTREE2KCAL, 3),
            "xyz": left["xyz"],
            "grad_rms": left["grad_rms"],
        },
        "product": {
            "energy_hartree": e_prod,
            "relative_kcal": round((e_prod - e0) * HARTREE2KCAL, 3),
            "xyz": right["xyz"],
            "grad_rms": right["grad_rms"],
        },
        "verification": verification,
        "irc": irc_result,
        "n_points": len(points),
        "seconds": round(time.time() - t0, 2),
        "functional": engine.functional,
        "basis": engine.basis,
    }


def _refine_maximum(symbols, coord, points, imax, engine, max_steps=40,
                    progress=None) -> Dict[str, object]:
    """Parabolic refinement of the scan maximum, then a restrained relax."""
    lo = points[max(0, imax - 1)]
    mid = points[imax]
    hi = points[min(len(points) - 1, imax + 1)]
    x1, x2, x3 = lo["coord"], mid["coord"], hi["coord"]
    y1, y2, y3 = (lo["energy_hartree"], mid["energy_hartree"],
                  hi["energy_hartree"])
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    guess = x2
    if abs(denom) > 1e-12:
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1)
             + x1 * x1 * (y2 - y3)) / denom
        if abs(a) > 1e-12:
            cand = -b / (2.0 * a)
            # the vertex is only a refinement, not a relocation: keep it
            # inside the bracket or the fit has failed
            if x1 <= cand <= x3:
                guess = cand

    coords_bohr = mid["coords_bohr"]
    r = constrained_relax(symbols, coords_bohr, coord, float(guess),
                         engine.charge, engine.mult, engine.functional,
                         engine.basis, max_steps=max_steps,
                         solvation=engine.solvation, progress=progress,
                         pct_from=80, pct_to=86)
    return {"coord": round(float(r["coord_value"]), 5),
            "energy_hartree": float(r["energy_hartree"]),
            "xyz": r["xyz"], "coords_bohr": r["coords_bohr"]}
