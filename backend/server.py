"""
ChatDFT HTTP API.

Run with::

    python -m backend.server            # from the project root
    uvicorn backend.server:app --port 8000

Endpoints
---------
GET  /                       -> single-page application
GET  /api/health             -> engine status and capability report
GET  /api/methods            -> functionals and basis sets
GET  /api/library            -> built-in molecule library
POST /api/molecule/resolve   -> name / SMILES / formula -> 3D structure
POST /api/chat               -> natural-language job submission
POST /api/job                -> structured job submission (no language layer)
GET  /api/job/{id}           -> poll a submitted job
POST /api/analyze            -> immediate analysis of an XYZ block
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# allow "python backend/server.py" as well as "-m backend.server"
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.engine import BENCHMARKS, DFTEngine, FUNCTIONALS, BASIS_SETS, DFTError
    from backend.engine import molecule as molmod
    from backend.engine.dft import scan_bond
    from backend.engine import analysis
    from backend.engine import reaction
    from backend.engine import nmr as nmrmod
    from backend.agent import planner as agent
    from backend.agent import designer
    from backend.agent.llm import default_client as default_llm
else:
    from .engine import BENCHMARKS, DFTEngine, FUNCTIONALS, BASIS_SETS, DFTError
    from .engine import molecule as molmod
    from .engine.dft import scan_bond
    from .engine import analysis
    from .engine import reaction
    from .engine import nmr as nmrmod
    from .agent import planner as agent
    from .agent import designer
    from .agent.llm import default_client as default_llm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT, "frontend", "static")
# Cube grids are megabytes each, so they live on disk under a per-job token
# and are served by URL instead of travelling inside the JSON result.
# Overridable so a gate run can write them somewhere disposable.
CUBE_DIR = os.environ.get("CHATDFT_CUBE_DIR") or os.path.join(
    ROOT, "data", "cubes")
CUBE_SLOTS = 12                     # directories kept in the recycling pool
_CUBE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.cube$")
_CUBE_TOKEN = re.compile(r"^[0-9a-f]{8,64}$")


_CUBE_LOCK = threading.Lock()
_CUBE_BUSY: set = set()             # tokens owned by a job that is still alive


def _cube_slots() -> list:
    """Existing cube directories, oldest first."""
    try:
        names = os.listdir(CUBE_DIR)
    except OSError:
        return []
    out = []
    for d in names:
        if not _CUBE_TOKEN.match(d):
            continue
        p = os.path.join(CUBE_DIR, d)
        if not os.path.isdir(p):
            continue
        try:
            out.append((os.path.getmtime(p), d))
        except OSError:
            out.append((0.0, d))
    out.sort()
    return [d for _m, d in out]


def _cube_token() -> str:
    """Take a directory to write cube grids into.

    The cache is a fixed pool of directories that are **overwritten in
    place**, never deleted.  That is deliberate.  Deletion is the one
    filesystem operation that can be made to wait indefinitely -- a backup
    agent, an antivirus scan, or a bulk-delete guard can hold an unlink open
    with no error and no timeout -- and a previous version of this function
    called ``shutil.rmtree`` while a job was waiting for it.  The job wedged
    at 8% for 89 minutes with no stack trace, and the guard eventually killed
    the whole server.  Overwriting sidesteps the entire class of failure:
    the disk footprint is bounded by the pool, and no code path ever removes
    a file.

    Space is reclaimed only when the user asks for it, via
    ``POST /api/admin/clear-cubes``.
    """
    os.makedirs(CUBE_DIR, exist_ok=True)
    with _CUBE_LOCK:
        slots = _cube_slots()
        free = [d for d in slots if d not in _CUBE_BUSY]
        if len(slots) >= CUBE_SLOTS and free:
            token = free[0]
        else:
            token = uuid.uuid4().hex
            try:
                os.makedirs(os.path.join(CUBE_DIR, token), exist_ok=True)
            except OSError:
                pass
        _CUBE_BUSY.add(token)
        return token


def _release_cube_token(token: Optional[str]) -> None:
    """Hand a directory back to the pool once its job is done."""
    if not token:
        return
    with _CUBE_LOCK:
        _CUBE_BUSY.discard(token)


def _clear_cubes() -> int:
    """Delete every cached cube directory.  Only ever called on request."""
    removed = 0
    for d in _cube_slots():
        if d in _CUBE_BUSY:
            continue
        shutil.rmtree(os.path.join(CUBE_DIR, d), ignore_errors=True)
        removed += 1
    return removed


# ======================================================================
#  job registry
# ======================================================================
class Job:
    def __init__(self, job_id: str, kind: str, payload: dict):
        self.id = job_id
        self.kind = kind
        self.payload = payload
        self.status = "queued"
        self.percent = 0.0
        self.stage = "queued"
        self.message = "Waiting for a free worker"
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created = time.time()
        self.started: Optional[float] = None
        self.last_update: Optional[float] = None
        self.finished: Optional[float] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Mark the job as picked up by a worker.

        Without this a job only leaves "queued" the first time the engine
        reports progress.  A job that hangs before its first progress
        callback is therefore indistinguishable from one still waiting for a
        worker, and with only two workers in the pool a single hang silently
        starves every later job while the API cheerfully reports
        "Waiting for a free worker" forever.
        """
        with self._lock:
            now = time.time()
            self.status = "running"
            self.stage = "starting"
            self.percent = 0.0
            self.message = "Starting"
            self.started = now
            self.last_update = now

    def update(self, stage: str, percent: float, message: str) -> None:
        with self._lock:
            self.status = "running"
            self.stage = stage
            self.percent = percent
            self.message = message
            self.last_update = time.time()

    def succeed(self, result: dict) -> None:
        with self._lock:
            self.status = "completed"
            self.percent = 100.0
            self.stage = "done"
            self.message = "Completed"
            self.result = result
            self.finished = time.time()

    def fail(self, error: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = error
            self.message = error
            self.finished = time.time()

    def to_dict(self, include_result: bool = True) -> dict:
        d = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "percent": round(self.percent, 1),
            "message": self.message,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "elapsed": round((self.finished or time.time()) - self.created, 2),
            # how long this job has been executing without finishing.  With a
            # two-worker pool this is the only way to tell "slow" from
            # "wedged", because Python cannot kill a running thread.
            "running_seconds": (
                round(time.time() - self.started, 2) if self.started else None),
            "idle_seconds": (
                round(time.time() - self.last_update, 2)
                if self.last_update and not self.finished else None),
            "error": self.error,
        }
        if include_result:
            d["result"] = self.result
        return d


JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
MAX_JOBS_KEPT = 200

# DFT is CPU-bound and PySCF is not thread-safe across all code paths;
# a small serialised pool keeps the machine responsive.
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chatdft")


def _submit(job: Job, fn, *args, **kwargs) -> None:
    def runner():
        job.start()
        try:
            result = fn(job, *args, **kwargs)
            job.succeed(result)
        except DFTError as exc:
            job.fail(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            job.fail(f"Unexpected error: {exc}\n{traceback.format_exc()[-1200:]}")

    EXECUTOR.submit(runner)


def _prune_jobs() -> None:
    with JOBS_LOCK:
        if len(JOBS) <= MAX_JOBS_KEPT:
            return
        ordered = sorted(JOBS.items(), key=lambda kv: kv[1].created)
        for jid, _ in ordered[: len(JOBS) - MAX_JOBS_KEPT]:
            JOBS.pop(jid, None)


# ======================================================================
#  request models
# ======================================================================
class ChatRequest(BaseModel):
    message: str = Field(..., description="Natural language request")
    auto_run: bool = Field(True, description="Execute immediately or only plan")
    functional: Optional[str] = None
    basis: Optional[str] = None
    charge: Optional[int] = None
    multiplicity: Optional[int] = None


class JobRequest(BaseModel):
    molecule: str
    kind: str = Field(
        "single_point",
        description="single_point|geometry_optimization|excited_states|scan|"
                    "vibrations|nmr|dos|reactivity|surfaces|nci")
    functional: str = "b3lyp"
    basis: str = "6-31g*"
    # None means "use the molecule's own charge / ground-state spin", so a
    # triplet such as O2 is not silently downgraded to a singlet.
    charge: Optional[int] = None
    multiplicity: Optional[int] = None
    nstates: int = 6
    solvation: Optional[str] = None
    max_steps: int = 40
    # Raman comes out of the same Hessian and the same displaced geometries as
    # the IR spectrum, so it rides along with the vibrational analysis instead
    # of being a job kind of its own.  It costs one coupled-perturbed solve per
    # displacement, which is why it can be declined.
    with_raman: bool = True
    # UV-Vis figure controls.  None means "take the window from the states":
    # a fixed 180-800 nm default shows nothing but Gaussian tails for a
    # molecule whose bands are all in the deep UV.
    uv_fwhm_ev: float = 0.40
    uv_wmin: Optional[float] = None
    uv_wmax: Optional[float] = None
    # NMR figure controls.  The spectrometer frequency is what turns the ppm
    # axis into a Hz axis, and therefore what decides how wide a 1 Hz line
    # looks; 400 MHz is the usual 1H frequency for a routine instrument.
    nmr_linewidth_hz: float = 1.0
    nmr_spectrometer_mhz: float = 400.0
    name: str = ""


class ResolveRequest(BaseModel):
    spec: str
    kind: str = "auto"
    charge: int = 0
    multiplicity: int = 1


class AnalyzeRequest(BaseModel):
    xyz: str
    functional: str = "b3lyp"
    basis: str = "6-31g*"
    charge: Optional[int] = None
    multiplicity: Optional[int] = None
    optimize: bool = False
    name: str = ""


class SeriesRequest(BaseModel):
    molecules: List[str] = Field(..., description="Names, formulas or SMILES")
    functional: str = "b3lyp"
    basis: str = "6-31g*"
    solvation: Optional[str] = None


class FieldRequest(BaseModel):
    molecule: str
    kind: str = Field("elf", description="elf | laplacian | spin | density | "
                                         "gradient | difference")
    functional: str = "b3lyp"
    basis: str = "6-31g*"
    plane: str = Field("xy", description="xy | xz | yz | atoms")
    n: int = Field(80, description="points per side of the plane grid")
    indices: Optional[List[int]] = Field(
        None, description="three atom indices defining the plane")
    fragments: Optional[List[List[int]]] = Field(
        None, description="atom index groups for a density difference")
    solvation: Optional[str] = None


class NtoRequest(BaseModel):
    molecule: str
    state: int = Field(1, description="which excited state, 1 = the lowest")
    functional: str = "b3lyp"
    basis: str = "6-31g*"
    npairs: int = Field(2, description="how many NTO pairs to write")


class ReactionRequest(BaseModel):
    molecule: str
    coordinate: str = Field(
        "", description="'torsion 2 0 1 5' / 'bond 1 2' / 'angle 1 2 3' with "
                        "1-based indices; empty picks the stretched bond")
    npoints: int = Field(9, description="relaxed points along the path")
    functional: str = "b3lyp"
    basis: str = "6-31g*"
    solvation: Optional[str] = None


class DesignRequest(BaseModel):
    text: str = Field(..., description="What to design, in plain language")
    functional: str = "b3lyp"
    basis: str = "6-31g*"
    n_candidates: int = 12
    dft_top: int = Field(3, description="how many of the best to calculate")
    generations: int = Field(1, description="refinement rounds")
    use_model: bool = Field(True, description="let the language model propose")
    alert_filter: Optional[str] = Field(
        None, description="reject PAINS / BRENK / NIH structural alerts")
    optimize_leads: Optional[bool] = Field(
        None, description="relax the leads at the DFT level before measuring")


# ======================================================================
#  worker functions
# ======================================================================
def _enforce_spin_parity(mol: molmod.Molecule) -> None:
    """Keep the spin multiplicity consistent with the electron count.

    A system with N electrons and multiplicity M is physical only when N + M
    is odd: 10 electrons cannot form a doublet and 9 electrons cannot form a
    singlet.  Only inconsistent choices are corrected, so a genuine triplet
    such as O2 (16 electrons, M = 3) is left alone.
    """
    n = mol.total_electrons()
    if (n + mol.multiplicity) % 2 == 0:
        mol.multiplicity = 2 if n % 2 else 1


def _run_single_point(job: Job, mol: molmod.Molecule, functional: str, basis: str,
                      solvation: Optional[str]) -> dict:
    engine = DFTEngine(
        atom_xyz=mol.to_xyz(),
        charge=mol.charge,
        multiplicity=mol.multiplicity,
        functional=functional,
        basis=basis,
        solvation=solvation,
        # Only this path fills the Properties pane, which is where the bond
        # orders and Hirshfeld charges are read.  Every other job runs SCFs in
        # a loop (scans, dipole derivatives, optimisation steps) and must not
        # pay for a population analysis it will never show.
        bonding=True,
    )
    result = engine.run_scf(progress=lambda p: job.update(p.stage, p.percent, p.message))
    result["job_type"] = "single_point"
    result["molecule"] = mol.to_dict()
    return result


def _resolve_scan_bond(mol: "molmod.Molecule", pair: str) -> Tuple[int, int]:
    """Choose the bond a scan should drive.

    A named pair ("O-H") wins if the molecule actually has one; otherwise the
    longest bond is used, since that is normally the weakest and the one a
    chemist wants to probe.
    """
    bonds = mol.bonds or mol.perceive_bonds()
    if not bonds:
        raise DFTError(
            f"No bonds were perceived in {mol.name or mol.formula}, so there is "
            "nothing to scan."
        )

    if pair:
        want = {
            (s.strip().capitalize() if len(s.strip()) > 1 else s.strip().upper())
            for s in pair.split("-") if s.strip()
        }
        matches = [
            (i, j, L) for i, j, L in bonds
            if {mol.atoms[i].symbol, mol.atoms[j].symbol} == want
        ]
        if matches:
            matches.sort(key=lambda b: -b[2])
            return matches[0][0], matches[0][1]
        available = sorted(
            "-".join(sorted((mol.atoms[i].symbol, mol.atoms[j].symbol)))
            for i, j, _ in bonds
        )
        raise DFTError(
            f"{mol.name or mol.formula} has no {pair} bond. Available pairs: "
            + ", ".join(dict.fromkeys(available)) + "."
        )

    i, j, _ = max(bonds, key=lambda b: b[2])
    return i, j


def _run_scan(job: Job, mol: "molmod.Molecule", i: int, j: int,
              functional: str, basis: str, solvation: Optional[str]) -> dict:
    if mol.natoms() > 80:
        raise DFTError(
            f"This molecule has {mol.natoms()} atoms. ChatDFT caps jobs at 80 "
            "atoms to keep the interactive session responsive."
        )
    return scan_bond(
        mol.to_xyz(),
        i,
        j,
        functional=functional,
        basis=basis,
        charge=mol.charge,
        multiplicity=mol.multiplicity,
        solvation=solvation,
        progress=lambda p: job.update(p.stage, p.percent, p.message),
    )


def _run_job(job: Job, mol: molmod.Molecule, kind: str, functional: str, basis: str,
             nstates: int, solvation: Optional[str], max_steps: int,
             uv_fwhm_ev: float = 0.40, uv_wmin: Optional[float] = None,
             uv_wmax: Optional[float] = None,
             with_raman: bool = True,
             nmr_linewidth_hz: float = 1.0,
             nmr_spectrometer_mhz: float = 400.0) -> dict:
    if mol.natoms() > 80:
        raise DFTError(
            f"This molecule has {mol.natoms()} atoms. ChatDFT caps jobs at 80 atoms "
            "to keep the interactive session responsive - try a smaller system or "
            "a minimal basis set on your local machine."
        )
    if kind == "geometry_optimization":
        if mol.natoms() > 50:
            raise DFTError(
                f"Geometry optimisation of {mol.natoms()} atoms is too heavy for the "
                "interactive session (limit 50). Run a single point instead."
            )
        engine = DFTEngine(
            atom_xyz=mol.to_xyz(),
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            functional=functional,
            basis=basis,
            solvation=solvation,
            bonding=True,
        )
        result = engine.optimize(
            max_steps=max_steps,
            progress=lambda p: job.update(p.stage, p.percent, p.message),
        )
    elif kind == "scan":
        # the structured endpoint has no way to name a pair, so fall back to
        # the same default the chat path uses: the longest bond.
        i, j = _resolve_scan_bond(mol, "")
        return _run_scan(job, mol, i, j, functional, basis, solvation)
    elif kind == "excited_states":
        engine = DFTEngine(
            atom_xyz=mol.to_xyz(),
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            functional=functional,
            basis=basis,
            solvation=solvation,
        )
        result = engine.excited_states(
            nstates=nstates,
            progress=lambda p: job.update(p.stage, p.percent, p.message),
        )
        # The state list alone is not a figure.  Papers plot a broadened
        # spectrum, so build it here rather than making the front end guess.
        # The electron count is handed over so the payload can report what
        # fraction of the Thomas-Reiche-Kuhn sum rule these roots captured --
        # without it a stick list looks like a complete spectrum.
        result["uv_spectrum"] = analysis.uv_curve(
            result.get("excited_states") or [],
            fwhm_ev=uv_fwhm_ev,
            wmin=uv_wmin,
            wmax=uv_wmax,
            n_electrons=result.get("nelec"),
        )
    elif kind == "vibrations":
        if mol.natoms() > 24:
            raise DFTError(
                f"IR intensities need 6N single points ({6 * mol.natoms()} here); "
                f"that is too slow past 24 atoms in an interactive session."
            )
        engine = DFTEngine(
            atom_xyz=mol.to_xyz(),
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            functional=functional,
            basis=basis,
            solvation=solvation,
        )
        result = analysis.vibrations(
            engine,
            optimize_first=True,
            progress=lambda p, m: job.update("vibrations", p, m),
            with_raman=bool(with_raman),
        )
        # No warning merge here: analysis.vibrations() already carries the
        # pre-optimisation's and the re-run SCF's complaints across the engine
        # swap it does internally.  Doing it again would show each one twice.
        #
        # Both curves are built by the engine, not here: the Raman figure comes
        # out of the same Hessian and the same 6N displaced geometries as the
        # IR one, and they are routinely plotted as a pair.  The whole curve is
        # shifted by the scale factor, so it has to be the one published for
        # the level these frequencies were computed at -- 0.961 is B3LYP/6-31G*
        # and nothing else.
        analysis.attach_vibrational_spectra(result)
        result["job_type"] = "vibrations"
    elif kind == "nmr":
        # The shielding is computed at Hartree-Fock regardless of what
        # functional was asked for, and the payload says so.  This is not a
        # choice: pyscf.dft.numint cannot take the complex density a GIAO
        # response needs, and the real-arithmetic substitute neither converges
        # nor gives a non-zero response (see engine/nmr.py).  Passing the
        # functional through would only hide that.
        result = nmrmod.compute(
            mol.to_xyz(),
            basis=basis,
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            functional=functional,
            progress=lambda p, m: job.update("nmr", p, m),
            linewidth_hz=float(nmr_linewidth_hz),
            spectrometer_mhz=float(nmr_spectrometer_mhz),
        )
        result["job_type"] = "nmr"
    elif kind == "dos":
        engine = DFTEngine(
            atom_xyz=mol.to_xyz(),
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            functional=functional,
            basis=basis,
            solvation=solvation,
        )
        result = engine.run_scf()
        result["dos"] = analysis.dos(engine)
        result["job_type"] = "dos"
    elif kind == "reactivity":
        engine = DFTEngine(
            atom_xyz=mol.to_xyz(),
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            functional=functional,
            basis=basis,
            solvation=solvation,
        )
        result = analysis.reactivity(
            engine,
            with_fukui=True,
            progress=lambda p, m: job.update("reactivity", p, m),
        )
        base = engine.last_result or {}
        result.update({k: v for k, v in base.items()
                       if k not in result and k != "fukui"})
        result["job_type"] = "reactivity"
    elif kind == "surfaces":
        if mol.natoms() > 40:
            raise DFTError(
                f"Cube grids for {mol.natoms()} atoms are too large for the "
                "interactive session (limit 40)."
            )
        engine = DFTEngine(
            atom_xyz=mol.to_xyz(),
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            functional=functional,
            basis=basis,
            solvation=solvation,
        )
        # Every stage reports, so a wedged job says where it wedged.  The
        # cubes are the slow part and they give no feedback of their own.
        job.update("scf", 5, "Running the ground-state SCF")
        result = engine.run_scf()
        job.update("grids", 8, "Preparing the grid directory")
        token = _cube_token()
        try:
            grid_n = 40 if mol.natoms() <= 20 else 32
            sur = analysis.surfaces(
                engine,
                os.path.join(CUBE_DIR, token),
                n=grid_n,
                progress=lambda p, m: job.update("surfaces", p, m),
            )
        finally:
            # The grids are written and the URLs handed out, so the directory
            # is no longer owned by this job; on failure it must go back or
            # the pool would leak one slot per crashed run.
            _release_cube_token(token)
        sur["url_base"] = f"/api/cubes/{token}"
        result["surfaces"] = sur
        result["job_type"] = "surfaces"
    elif kind == "nci":
        if mol.natoms() > 40:
            raise DFTError(
                f"An NCI grid for {mol.natoms()} atoms is too large for the "
                "interactive session (limit 40)."
            )
        engine = DFTEngine(
            atom_xyz=mol.to_xyz(),
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            functional=functional,
            basis=basis,
            solvation=solvation,
        )
        job.update("scf", 5, "Running the ground-state SCF")
        result = engine.run_scf()
        job.update("grids", 8, "Preparing the grid directory")
        token = _cube_token()
        try:
            out = analysis.nci(
                engine,
                os.path.join(CUBE_DIR, token),
                progress=lambda p, m: job.update("nci", p, m),
            )
        finally:
            _release_cube_token(token)
        out["url_base"] = f"/api/cubes/{token}"
        result["nci"] = out
        result["job_type"] = "nci"
    elif kind == "single_point":
        result = _run_single_point(job, mol, functional, basis, solvation)
    else:
        # Refuse rather than degrade.  This branch used to run a single point
        # for any unrecognised kind, which is the origin of two bugs: a
        # "compare" that lost its second molecule and a "scan" that was never
        # implemented both reported themselves as complete jobs while actually
        # being single points.
        raise DFTError(
            f"Unsupported job type '{kind}'. Supported: single_point, "
            "geometry_optimization, excited_states, scan, vibrations, dos, "
            "reactivity, surfaces."
        )

    result["molecule"] = mol.to_dict()
    # A//B.  What the numbers were computed *on* matters as much as what they
    # were computed *with*: a HOMO-LUMO gap at B3LYP/6-31G* on a force-field
    # conformer and on a DFT minimum differ by tenths of an eV, and a dipole
    # by more.  The designer has always carried this; every other job left it
    # unsaid, so a figure could not be compared with anything or reproduced.
    # Any run that produced its own optimised geometry says so; the rest fall
    # back to wherever the coordinates came from.
    if result.get("optimized_xyz"):
        result["geometry_source"] = (
            f"DFT-optimised ({FUNCTIONALS[functional]['label']} / "
            f"{BASIS_SETS[basis]['label']})")
    else:
        result.setdefault("geometry_source", mol.geometry_source)
    return result


# ======================================================================
#  series: the same calculation on a set of molecules
# ======================================================================
SERIES_PROPERTIES = [
    ("gap_ev", "HOMO-LUMO gap", "eV", 3),
    ("homo_ev", "HOMO", "eV", 3),
    ("lumo_ev", "LUMO", "eV", 3),
    ("dipole", "Dipole", "D", 3),
    ("energy_ev", "Total energy", "eV", 4),
    ("relative_kcal", "Relative energy", "kcal/mol", 3),
]


def _dipole_magnitude(result: dict) -> Optional[float]:
    d = result.get("dipole") or {}
    try:
        return float(d.get("magnitude"))
    except (TypeError, ValueError):
        return None


def _run_series(job: Job, specs: List[str], functional: str, basis: str,
                solvation: Optional[str]) -> dict:
    """Run one single point per molecule, at one level of theory.

    A failure on one molecule must not lose the others: the point is kept
    with its error attached, because a series where furan failed but the
    other five succeeded is still a usable figure, and silently dropping
    it would make the trend look like a real gap in the data.
    """
    points: List[dict] = []
    total = max(1, len(specs))
    for k, spec in enumerate(specs):
        base = 6 + int(88 * k / total)
        span = int(88 / total)
        job.update("scf", base, f"({k + 1}/{total}) {spec}")
        row: Dict[str, Any] = {
            "spec": spec, "name": spec, "smiles": "", "formula": "",
            "natoms": None, "nelectrons": None,
            "energy_hartree": None, "energy_ev": None,
            "homo_ev": None, "lumo_ev": None, "gap_ev": None,
            "dipole": None, "converged": False, "error": None,
            "scf_seconds": None,
        }
        try:
            mol = molmod.resolve(spec)
        except Exception as exc:
            row["error"] = f"could not build a structure: {exc}"
            points.append(row)
            continue
        _enforce_spin_parity(mol)
        row.update({
            "name": mol.name or spec,
            "smiles": mol.smiles or "",
            "formula": mol.formula,
            "natoms": mol.natoms(),
            "nelectrons": mol.total_electrons(),
            # The series compares numbers across molecules, which is only a
            # fair comparison if every one of them was treated the same way --
            # and "the same way" includes where the coordinates came from.
            "geometry_source": mol.geometry_source,
        })
        try:
            engine = DFTEngine(
                atom_xyz=mol.to_xyz(), charge=mol.charge,
                multiplicity=mol.multiplicity, functional=functional,
                basis=basis, solvation=solvation)
            result = engine.run_scf(progress=lambda p: job.update(
                "scf", base + int(span * (p.percent or 0) / 100.0),
                f"{mol.name}: {p.message}"))
        except Exception as exc:
            row["error"] = str(exc)[:300]
            points.append(row)
            continue
        row.update({
            "energy_hartree": result.get("energy_hartree"),
            "energy_ev": result.get("energy_ev"),
            "homo_ev": result.get("homo_ev"),
            "lumo_ev": result.get("lumo_ev"),
            "gap_ev": result.get("gap_ev"),
            "dipole": _dipole_magnitude(result),
            "converged": bool(result.get("converged")),
            "scf_seconds": result.get("scf_seconds"),
        })
        if not result.get("converged"):
            row["error"] = "SCF did not converge"
        points.append(row)

    ok = [p for p in points if p["gap_ev"] is not None]
    # Relative energies are what a series diagram plots, but subtracting two
    # total energies is only a physical difference when both sides have the
    # same atoms: benzene minus furan is not an isomerisation energy, it is
    # the energy of two carbon atoms and a NH unit.  Reporting that as
    # "23915 kcal/mol" would be a number with no meaning attached, so the
    # column is produced only within one composition.
    formulas = {p.get("formula") for p in ok if p.get("formula")}
    comparable = len(formulas) == 1
    for p in points:
        p["relative_kcal"] = None
    if ok and comparable:
        ref = min(p["energy_hartree"] for p in ok
                  if p["energy_hartree"] is not None)
        for p in points:
            p["relative_kcal"] = (
                round((p["energy_hartree"] - ref) * 627.5094740631, 3)
                if (p["energy_hartree"] is not None and ref is not None)
                else None)

    return {
        "job_type": "series",
        "functional": functional,
        "basis": basis,
        "functional_label": FUNCTIONALS[functional]["label"],
        "basis_label": BASIS_SETS[basis]["label"],
        "solvation": solvation,
        "points": points,
        "n_ok": len(ok),
        "n_failed": len(points) - len(ok),
        # Only over the points that actually have a geometry.  A molecule that
        # could not be built has no coordinates at all, and folding its
        # "unknown" into the summary makes the whole set read as unqualified.
        "geometry_source": (
            "; ".join(sorted({p["geometry_source"] for p in points
                              if p.get("geometry_source")})) or "unknown"),
        "relative_meaningful": comparable,
        "relative_note": (
            "Relative energies are given because every species has the same "
            "formula, so the differences are isomerisation energies."
            if comparable else
            "No relative energies: the molecules do not share a formula, so "
            "the difference of their total energies is not a physical "
            "quantity."),
        "properties": [
            {"key": key, "label": label, "unit": unit, "decimals": dp}
            for key, label, unit, dp in SERIES_PROPERTIES
        ],
    }


# ======================================================================
#  real-space fields: ELF, Laplacian, spin density, density difference
# ======================================================================
FIELD_KINDS = {
    "elf": "Electron localisation function",
    "laplacian": "Laplacian of the density",
    "spin": "Spin density",
    "density": "Electron density",
    "gradient": "Density gradient magnitude",
    "difference": "Density difference",
}


def _connected_groups(mol) -> List[List[int]]:
    """Atom groups held together by bonds, for the density difference.

    A dimer arrives as two separate molecules in one XYZ block, and the
    meaningful comparison is complex-minus-monomers.  Guessing that from the
    bond graph is right far more often than asking the user for indices.
    """
    n = mol.natoms()
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for b in (mol.bonds or []):
        i, j = int(b[0]), int(b[1])
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g)]


def _run_field(job: Job, spec: str, kind: str, functional: str, basis: str,
               plane: str, n: int, indices: Optional[List[int]],
               fragments: Optional[List[List[int]]],
               solvation: Optional[str]) -> dict:
    """One SCF, then one real-space field on a 2D plane."""
    from backend.engine import fields as fieldmod

    job.update("scf", 3, f"Building {spec}")
    mol = molmod.resolve(spec)
    _enforce_spin_parity(mol)

    if kind == "difference":
        if not fragments:
            groups = _connected_groups(mol)
            if len(groups) < 2:
                raise DFTError(
                    "a density difference needs at least two separate "
                    "molecules in one structure, and this one is a single "
                    "connected fragment; give the atom groups explicitly")
            fragments = groups

    job.update("scf", 6, f"SCF on {mol.name}")
    engine = DFTEngine(
        atom_xyz=mol.to_xyz(), charge=mol.charge,
        multiplicity=mol.multiplicity, functional=functional,
        basis=basis, solvation=solvation)
    res = engine.run_scf(progress=lambda p: job.update(
        "scf", 6 + int(44 * (p.percent or 0) / 100.0),
        f"{mol.name}: {p.message}"))

    def prog(pct, msg):
        job.update("field", 50 + int(50 * pct / 100.0), msg)

    out = fieldmod.compute_field(
        engine, kind=kind, plane=plane, n=n, indices=indices,
        fragments=fragments, progress=prog)

    out.update({
        "job_type": "field",
        "kind": kind,
        "molecule": mol.to_dict(),
        "functional": functional,
        "basis": basis,
        "functional_label": FUNCTIONALS[functional]["label"],
        "basis_label": BASIS_SETS[basis]["label"],
        "solvation": solvation,
        "multiplicity": mol.multiplicity,
        "energy_hartree": res.get("energy_hartree"),
        "homo_ev": res.get("homo_ev"),
        "lumo_ev": res.get("lumo_ev"),
        "gap_ev": res.get("gap_ev"),
        "converged": bool(res.get("converged")),
        "fragments": fragments,
    })
    return out


def _nto_state(text: str) -> int:
    """Which excited state an NTO request means.

    'NTO of the second excited state' is a different figure from the first,
    and defaulting silently to state 1 would show the wrong orbitals with no
    indication that anything was guessed.
    """
    low = (text or "").lower()
    words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}
    for w, v in words.items():
        if re.search(rf"\b{w}\b", low):
            return v
    m = re.search(r"\bs(?:tate)?\s*([1-9])\b", low)
    if m:
        return int(m.group(1))
    m = re.search(r"([1-9])\s*(?:st|nd|rd|th)\s+excited", low)
    if m:
        return int(m.group(1))
    m = re.search(r"第\s*([1-9一二三四五])\s*(?:激发态|个)", text or "")
    if m:
        cn = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
        t = m.group(1)
        return cn.get(t, int(t) if t.isdigit() else 1)
    return 1


def _run_nto(job: Job, spec: str, state: int, functional: str, basis: str,
             npairs: int = 2) -> dict:
    """Natural transition orbitals for one excited state, as cube files."""
    mol = molmod.resolve(spec)
    _enforce_spin_parity(mol)
    engine = DFTEngine(
        atom_xyz=mol.to_xyz(), charge=mol.charge,
        multiplicity=mol.multiplicity, functional=functional, basis=basis)
    job.update("scf", 5, f"SCF on {mol.name}")
    job.update("scf", 8, "Running the ground-state SCF")
    res = engine.run_scf(progress=lambda p: job.update(
        "scf", 8 + int(12 * (p.percent or 0) / 100.0),
        f"{mol.name}: {p.message}"))
    if not res.get("converged"):
        raise DFTError("the ground state did not converge; NTOs need one")

    token = _cube_token()
    try:
        out = analysis.natural_transition_orbitals(
            engine, state=state, npairs=npairs,
            outdir=os.path.join(CUBE_DIR, token),
            n=40 if mol.natoms() <= 20 else 32,
            progress=lambda p, m: job.update("nto", 20 + int(80 * p / 100.0), m))
    finally:
        _release_cube_token(token)

    # The isosurface items go under "surfaces" so the existing 3D viewer
    # picks them up unchanged; the NTO numbers stay together under "nto" so
    # the panel can describe the transition without re-deriving it.
    items = out.pop("surfaces")
    grid_n = out.pop("grid_n", 40)
    return {
        "job_type": "nto",
        "molecule": mol.to_dict(),
        "functional": functional,
        "basis": basis,
        "functional_label": FUNCTIONALS[functional]["label"],
        "basis_label": BASIS_SETS[basis]["label"],
        "nto": out,
        "surfaces": {
            "surfaces": items,
            "grid_n": grid_n,
            "url_base": f"/api/cubes/{token}",
            "n_orbital_surfaces": len(items),
        },
    }


def _run_reaction(job: Job, spec: str, coord_spec: str, functional: str,
                  basis: str, npoints: int = 9,
                  solvation: Optional[str] = None) -> dict:
    """Relaxed scan along one internal coordinate, plus the barrier and TS.

    ``coord_spec`` is the planner's ``"kind i j [k l]"`` text with 1-based
    indices, or an empty string to let the engine choose the most stretched
    bond.  Indices are checked against the molecule *before* any SCF runs, so
    a typo costs a second rather than a minute.
    """
    mol = molmod.resolve(spec)
    _enforce_spin_parity(mol)

    coord = None
    parsed = agent.parse_coordinate(coord_spec)
    if parsed:
        kind, idx = parsed
        natm = mol.natoms()
        if max(idx) >= natm:
            raise DFTError(
                f"the reaction coordinate {coord_spec!r} names atom "
                f"{max(idx) + 1}, but {mol.name} has only {natm} atoms")
        coord = reaction.Coordinate(kind, idx)

    engine = DFTEngine(atom_xyz=mol.to_xyz(), charge=mol.charge,
                       multiplicity=mol.multiplicity, functional=functional,
                       basis=basis, solvation=solvation)
    job.update("reaction", 2,
               f"Setting up a relaxed scan on {mol.name}"
               + (f" along {coord.display()}" if coord else ""))
    out = reaction.reaction_path(
        engine, coord=coord, npoints=max(3, int(npoints)),
        progress=lambda p, m: job.update(
            "reaction", max(2, min(99, int(p))), m))
    out["molecule"] = mol.to_dict()
    out["functional_label"] = FUNCTIONALS[functional]["label"]
    out["basis_label"] = BASIS_SETS[basis]["label"]
    return out


# ======================================================================
#  molecular design
# ======================================================================
def _run_design(job: Job, brief_text: str, functional: str, basis: str,
                n_candidates: Optional[int], dft_top: Optional[int],
                generations: Optional[int], use_model: bool,
                alert_filter: Optional[str] = None,
                optimize_leads: Optional[bool] = None) -> dict:
    """Parse a design brief, generate, filter, rank and calculate."""
    client = default_llm() if use_model else None
    if client is not None and not client.available:
        client = None

    brief = designer.parse_brief(brief_text, client=client)
    if functional:
        brief.functional = functional
    if basis:
        brief.basis = basis
    if n_candidates:
        brief.n_candidates = max(1, min(100, int(n_candidates)))
    if dft_top is not None:
        brief.dft_top = max(0, min(5, int(dft_top)))
    if generations:
        brief.generations = max(1, min(4, int(generations)))
    # an explicit request overrides what the sentence was read as; None means
    # "not specified", so the brief keeps whatever the text implied
    if alert_filter:
        wanted = str(alert_filter).strip().upper()
        brief.alert_filter = wanted if wanted in ("PAINS", "BRENK", "NIH") \
            else "none"
    if optimize_leads is not None:
        brief.optimize_leads = bool(optimize_leads)

    result = designer.design(
        brief, client=client,
        progress=lambda stage, pct, msg: job.update(stage, pct, msg))
    return result.to_dict()


# ======================================================================
#  lifespan
# ======================================================================
ENGINE_STATUS: Dict[str, Any] = {"pyscf": False, "rdkit": False, "version": None}


def _probe_engines() -> None:
    # Make libcint / libxc / MKL discoverable before the first pyscf import.
    try:
        from backend import bootstrap

        ENGINE_STATUS["bootstrap"] = bootstrap.setup()
    except Exception as exc:  # pragma: no cover - defensive
        ENGINE_STATUS["bootstrap"] = {"applied": False, "reason": str(exc)}
    try:
        import pyscf

        ENGINE_STATUS["pyscf"] = True
        ENGINE_STATUS["version"] = pyscf.__version__
    except Exception as exc:
        ENGINE_STATUS["pyscf"] = False
        ENGINE_STATUS["pyscf_error"] = str(exc)
    try:
        import rdkit

        ENGINE_STATUS["rdkit"] = True
    except Exception:
        ENGINE_STATUS["rdkit"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _probe_engines()
    print(f"[ChatDFT] PySCF available: {ENGINE_STATUS['pyscf']} "
          f"({ENGINE_STATUS['version']}), RDKit: {ENGINE_STATUS['rdkit']}")
    yield
    EXECUTOR.shutdown(wait=False)


app = FastAPI(
    title="ChatDFT API",
    description="Conversational density functional theory",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
#  routes
# ======================================================================
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "engines": ENGINE_STATUS,
        "functionals": list(FUNCTIONALS.keys()),
        "basis_sets": list(BASIS_SETS.keys()),
        "library_size": len(molmod.library_names()),
        "active_jobs": sum(1 for j in JOBS.values() if j.status in ("queued", "running")),
    }


@app.get("/api/methods")
def methods():
    return {
        "functionals": [
            {"key": k, "label": v["label"], "type": v["type"]}
            for k, v in FUNCTIONALS.items()
        ],
        "basis_sets": [
            {"key": k, "label": v["label"], "quality": v["quality"]}
            for k, v in BASIS_SETS.items()
        ],
        "benchmarks": BENCHMARKS,
    }


@app.get("/api/library")
def library():
    return {"molecules": molmod.library_names()}


@app.post("/api/molecule/resolve")
def resolve_molecule(req: ResolveRequest):
    try:
        mol = molmod.resolve(req.spec, req.kind, req.charge, req.multiplicity)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if req.charge:
        mol.charge = req.charge
    if req.multiplicity:
        mol.multiplicity = req.multiplicity
    return mol.to_dict()


# ----------------------------------------------------------------------
@app.post("/api/chat")
def chat(req: ChatRequest):
    """Natural-language entry point: plan, then execute."""
    planner, llm_active = agent.create_planner()
    intent = planner.plan(req.message)

    # explicit overrides from the UI take priority over parsed values
    if req.functional:
        intent.functional = req.functional
    if req.basis:
        intent.basis = req.basis
    if req.charge is not None:
        intent.charge = req.charge
    if req.multiplicity is not None:
        intent.multiplicity = req.multiplicity

    response: Dict[str, Any] = {
        "intent": {
            "job_type": intent.job_type,
            "molecule": intent.molecule,
            "molecule2": intent.molecule2,
            "functional": intent.functional,
            "functional_label": FUNCTIONALS.get(intent.functional, {}).get("label", ""),
            "basis": intent.basis,
            "basis_label": BASIS_SETS.get(intent.basis, {}).get("label", ""),
            "charge": intent.charge,
            "multiplicity": intent.multiplicity,
            "nstates": intent.nstates,
            "molecules": list(intent.molecules),
            "field_kind": intent.field_kind,
            "reaction_coord": intent.reaction_coord,
            "solvation": intent.solvation,
            "confidence": intent.confidence,
            "notes": intent.notes,
        },
        # ``llm_active`` says the model was *configured*, not that it answered.
        # A call that fails mid-flight is planned by the keyword parser with
        # the reason in ``notes``, so the honest answer to "who planned this?"
        # is the planner's own record of whether it reached the model.
        "planner": "llm" if (llm_active and not getattr(planner, "last_error", ""))
                   else "local",
        "raw": req.message,
    }

    # ---- non-computational intents ---------------------------------
    if intent.job_type == "library":
        response["reply"] = (
            "The ChatDFT library ships with "
            f"{len(molmod.library_names())} molecules spanning basics, aromatics, "
            "heterocycles, carbonyls, alcohols, acids, biomolecules, ions and "
            "radicals. Open the Library panel to browse and load any of them, or "
            "just name one in a message - for example \"optimise caffeine\" or "
            "\"TD-DFT on benzene\"."
        )
        response["action"] = "library"
        return response

    # ---- automatic molecular design -----------------------------------
    # Handled before the concept-question branch on purpose: a design request
    # is full of property words ("a wide HOMO-LUMO gap"), so it looks exactly
    # like a "what is ..." question and would otherwise be answered with a
    # paragraph of theory instead of a molecule.
    if intent.job_type == "design":
        if not req.auto_run:
            response["reply"] = (
                f"Ready to design molecules for: \"{req.message.strip()}\". "
                "Confirm to start the search."
            )
            response["action"] = "confirm"
            return response

        job = Job(str(uuid.uuid4())[:8], "design", {
            "text": req.message,
            "functional": intent.functional,
            "basis": intent.basis,
            "n_candidates": 12,
            "dft_top": 3,
            "generations": 1,
            "use_model": True,
        })
        with JOBS_LOCK:
            JOBS[job.id] = job
        _prune_jobs()
        _submit(job, _run_design, req.message, intent.functional, intent.basis,
                None, None, None, True)
        response["job_id"] = job.id
        response["action"] = "design"
        response["molecule"] = None
        response["reply"] = (
            "Designing molecules for that brief. I am generating candidates, "
            "discarding the ones that break the hard limits, ranking the rest "
            "by desirability, and then calculating the best few at "
            f"{FUNCTIONALS[intent.functional]['label']} / "
            f"{BASIS_SETS[intent.basis]['label']}. The Design panel fills in "
            "as the results arrive."
        )
        return response

    # ---- natural transition orbitals ---------------------------------
    if intent.job_type == "nto":
        state = _nto_state(req.message)
        if not intent.molecule:
            response["reply"] = _fallback_help()
            response["action"] = "answer"
            return response
        try:
            molmod.resolve(intent.molecule)
        except Exception as exc:
            response["reply"] = (
                f"I could not build a structure for **{intent.molecule}**. "
                f"{exc}\n\nYou can give me a common name (formaldehyde), a "
                "formula (CH2O), a SMILES string, or a full XYZ block."
            )
            response["action"] = "error"
            return response
        if not req.auto_run:
            response["reply"] = (
                f"Ready to compute the natural transition orbitals of state "
                f"{state} of {intent.molecule or 'the molecule'} at "
                f"{FUNCTIONALS[intent.functional]['label']} / "
                f"{BASIS_SETS[intent.basis]['label']}. Confirm to start."
            )
            response["action"] = "confirm"
            response["state"] = state
            return response
        job = Job(str(uuid.uuid4())[:8], "nto", {
            "molecule": intent.molecule, "state": state,
            "functional": intent.functional, "basis": intent.basis,
        })
        with JOBS_LOCK:
            JOBS[job.id] = job
        _prune_jobs()
        _submit(job, _run_nto, intent.molecule, state, intent.functional,
                intent.basis, 2)
        response["job_id"] = job.id
        response["action"] = "nto"
        response["state"] = state
        response["reply"] = (
            f"Computing the **natural transition orbitals** of state {state} "
            f"of {intent.molecule} at "
            f"{FUNCTIONALS[intent.functional]['label']} / "
            f"{BASIS_SETS[intent.basis]['label']}. The donor and acceptor "
            "orbitals will appear in the Surfaces tab."
        )
        return response

    # ---- a reaction path --------------------------------------------
    if intent.job_type == "reaction":
        if not intent.molecule:
            response["reply"] = _fallback_help()
            response["action"] = "answer"
            return response
        try:
            molmod.resolve(intent.molecule)
        except Exception as exc:
            response["reply"] = (
                f"I could not build a structure for **{intent.molecule}**. "
                f"{exc}\n\nYou can give me a common name (ethane), a formula "
                "(C2H6), a SMILES string, or a full XYZ block."
            )
            response["action"] = "error"
            return response
        parsed = agent.parse_coordinate(intent.reaction_coord)
        if intent.reaction_coord and not parsed:
            response["reply"] = (
                f"I could not read the reaction coordinate "
                f"**{intent.reaction_coord}**. Use `bond 1 2`, "
                "`angle 1 2 3` or `torsion 2 0 1 5` with 1-based atom indices."
            )
            response["action"] = "error"
            return response
        coord_txt = (f"{parsed[0]} {' '.join(str(i + 1) for i in parsed[1])}"
                     if parsed else "the most stretched bond")
        if not req.auto_run:
            response["reply"] = (
                f"Ready to scan a reaction path for {intent.molecule} along "
                f"**{coord_txt}** at "
                f"{FUNCTIONALS[intent.functional]['label']} / "
                f"{BASIS_SETS[intent.basis]['label']}: a relaxed scan, the "
                "barrier, and a frequency check on the maximum. Confirm to "
                "start."
            )
            response["action"] = "confirm"
            response["coordinate"] = intent.reaction_coord
            return response
        job = Job(str(uuid.uuid4())[:8], "reaction", {
            "molecule": intent.molecule,
            "coordinate": intent.reaction_coord,
            "functional": intent.functional, "basis": intent.basis,
            "npoints": intent.reaction_points, "solvation": intent.solvation,
        })
        with JOBS_LOCK:
            JOBS[job.id] = job
        _prune_jobs()
        _submit(job, _run_reaction, intent.molecule, intent.reaction_coord,
                intent.functional, intent.basis, intent.reaction_points,
                intent.solvation)
        response["job_id"] = job.id
        response["action"] = "reaction"
        response["coordinate"] = intent.reaction_coord
        response["reply"] = (
            f"Running a **relaxed reaction path** for {intent.molecule} along "
            f"**{coord_txt}** at "
            f"{FUNCTIONALS[intent.functional]['label']} / "
            f"{BASIS_SETS[intent.basis]['label']}. Every point is a full "
            "geometry optimisation with that coordinate held fixed, and the "
            "maximum is checked with an analytic Hessian. The Reaction tab "
            "will draw the energy profile."
        )
        return response

    # ---- a real-space field -----------------------------------------
    if intent.job_type == "field":
        kind = intent.field_kind or "elf"
        if not intent.molecule:
            response["reply"] = _fallback_help()
            response["action"] = "answer"
            return response
        try:
            molmod.resolve(intent.molecule)
        except Exception as exc:
            response["reply"] = (
                f"I could not build a structure for **{intent.molecule}**. "
                f"{exc}\n\nYou can give me a common name (benzene), a formula "
                "(C6H6), a SMILES string, or a full XYZ block."
            )
            response["action"] = "error"
            return response
        if not req.auto_run:
            response["reply"] = (
                f"Ready to compute the {FIELD_KINDS.get(kind, kind)} of "
                f"{intent.molecule or 'the molecule'} at "
                f"{FUNCTIONALS[intent.functional]['label']} / "
                f"{BASIS_SETS[intent.basis]['label']}. Confirm to start."
            )
            response["action"] = "confirm"
            response["field"] = kind
            return response
        job = Job(str(uuid.uuid4())[:8], "field", {
            "molecule": intent.molecule, "kind": kind,
            "functional": intent.functional, "basis": intent.basis,
            "plane": "xy", "n": 80, "solvation": intent.solvation,
        })
        with JOBS_LOCK:
            JOBS[job.id] = job
        _prune_jobs()
        _submit(job, _run_field, intent.molecule, kind, intent.functional,
                intent.basis, "xy", 80, None, None, intent.solvation)
        response["job_id"] = job.id
        response["action"] = "field"
        response["field"] = kind
        response["reply"] = (
            f"Computing the **{FIELD_KINDS.get(kind, kind)}** of "
            f"{intent.molecule} at "
            f"{FUNCTIONALS[intent.functional]['label']} / "
            f"{BASIS_SETS[intent.basis]['label']}. The Fields tab will draw "
            "the contour map."
        )
        return response

    # ---- a series of three or more ---------------------------------
    # Before the compare branch: with three or more molecules named,
    # compare would take the first two and the rest would vanish with no
    # error and no mention in the reply.
    if intent.job_type == "series" and len(intent.molecules) >= 2:
        if not req.auto_run:
            response["reply"] = (
                f"Ready to run {len(intent.molecules)} molecules "
                f"({', '.join(intent.molecules)}) at "
                f"{FUNCTIONALS[intent.functional]['label']} / "
                f"{BASIS_SETS[intent.basis]['label']}. Confirm to start."
            )
            response["action"] = "confirm"
            return response

        job = Job(str(uuid.uuid4())[:8], "series", {
            "molecules": intent.molecules,
            "functional": intent.functional,
            "basis": intent.basis,
            "solvation": intent.solvation,
        })
        with JOBS_LOCK:
            JOBS[job.id] = job
        _prune_jobs()
        _submit(job, _run_series, list(intent.molecules), intent.functional,
                intent.basis, intent.solvation)
        response["job_id"] = job.id
        response["action"] = "series"
        response["molecules"] = list(intent.molecules)
        response["reply"] = (
            f"Running **{len(intent.molecules)} molecules** "
            f"({', '.join(intent.molecules)}) at "
            f"{FUNCTIONALS[intent.functional]['label']} / "
            f"{BASIS_SETS[intent.basis]['label']} so the numbers are directly "
            "comparable. The Series tab will show the trend as each one "
            "finishes."
        )
        return response

    if intent.job_type == "info" or (not intent.molecule and intent.job_type != "library"):
        concept = agent.find_concept(req.message)
        if concept:
            response["reply"] = concept
            response["action"] = "answer"
            return response
        if not intent.molecule:
            response["reply"] = _fallback_help()
            response["action"] = "answer"
            return response

    # ---- computational intents -------------------------------------
    if not intent.molecule:
        response["reply"] = _fallback_help()
        response["action"] = "answer"
        return response

    try:
        mol = molmod.resolve(intent.molecule)
    except Exception as exc:
        response["reply"] = (
            f"I could not build a structure for **{intent.molecule}**. {exc}\n\n"
            "You can give me a common name (benzene), a formula (C6H6), a SMILES "
            "string, or a full XYZ block."
        )
        response["action"] = "error"
        return response

    # apply parsed charge/multiplicity unless the molecule dictates them
    if intent.charge:
        mol.charge = intent.charge
    if intent.multiplicity > 1:
        mol.multiplicity = intent.multiplicity

    # electron parity guard
    _enforce_spin_parity(mol)

    if not req.auto_run:
        response["reply"] = (
            f"Ready to run a **{intent.job_type.replace('_', ' ')}** on "
            f"{mol.name} ({mol.formula}) with "
            f"{FUNCTIONALS[intent.functional]['label']} / "
            f"{BASIS_SETS[intent.basis]['label']}. Confirm to start."
        )
        response["action"] = "confirm"
        response["molecule"] = mol.to_dict()
        return response

    # ---- compare ---------------------------------------------------
    if intent.job_type == "compare" and intent.molecule2:
        try:
            mol2 = molmod.resolve(intent.molecule2)
        except Exception as exc:
            response["reply"] = f"I could not build the second structure: {exc}"
            response["action"] = "error"
            return response

        job = Job(str(uuid.uuid4())[:8], "compare", {
            "molecule": intent.molecule,
            "molecule2": intent.molecule2,
            "functional": intent.functional,
            "basis": intent.basis,
        })
        with JOBS_LOCK:
            JOBS[job.id] = job
        _prune_jobs()

        def compare_worker(j: Job) -> dict:
            j.update("scf", 5, f"Computing {mol.name}")
            ra = _run_single_point(j, mol, intent.functional, intent.basis, intent.solvation)
            j.update("scf", 55, f"Computing {mol2.name}")
            rb = _run_single_point(j, mol2, intent.functional, intent.basis, intent.solvation)
            return {"a": ra, "b": rb}

        _submit(job, compare_worker)
        response["job_id"] = job.id
        response["action"] = "compute"
        response["molecule"] = mol.to_dict()
        response["molecule2"] = mol2.to_dict()
        response["reply"] = (
            f"Running both molecules at {FUNCTIONALS[intent.functional]['label']} / "
            f"{BASIS_SETS[intent.basis]['label']} so the comparison is valid. "
            "Results will appear shortly."
        )
        return response

    # ---- bond scan -------------------------------------------------
    if intent.job_type == "scan":
        try:
            i, j = _resolve_scan_bond(mol, intent.bond)
        except DFTError as exc:
            response["reply"] = str(exc)
            response["action"] = "error"
            return response

        job = Job(str(uuid.uuid4())[:8], "scan", {
            "molecule": intent.molecule,
            "functional": intent.functional,
            "basis": intent.basis,
            "atoms": [i + 1, j + 1],
            "solvation": intent.solvation,
        })
        with JOBS_LOCK:
            JOBS[job.id] = job
        _prune_jobs()
        _submit(job, _run_scan, mol, i, j, intent.functional, intent.basis,
                intent.solvation)
        response["job_id"] = job.id
        response["action"] = "compute"
        response["molecule"] = mol.to_dict()
        response["reply"] = (
            f"Driving the **{mol.atoms[i].symbol}{i + 1}-"
            f"{mol.atoms[j].symbol}{j + 1}** bond of "
            f"**{mol.name or mol.formula}** at "
            f"{FUNCTIONALS[intent.functional]['label']} / "
            f"{BASIS_SETS[intent.basis]['label']}. The curve will appear shortly."
        )
        return response

    # A compare request that arrives here has no second molecule (the compare
    # branch above needs one).  Downgrade it rather than creating a job whose
    # kind says "compare" while _run_job actually performs a single point --
    # which is what used to happen, and the narration then reported a
    # comparison that never ran.
    if intent.job_type == "compare" and not intent.molecule2:
        intent.job_type = "single_point"

    # The payload records how the job was requested so that /api/job/{id} can
    # rebuild the written explanation later, without re-running anything.
    job = Job(str(uuid.uuid4())[:8], intent.job_type, {
        "molecule": intent.molecule,
        "molecule2": intent.molecule2,
        "functional": intent.functional,
        "basis": intent.basis,
        "nstates": intent.nstates,
        "solvation": intent.solvation,
    })
    with JOBS_LOCK:
        JOBS[job.id] = job
    _prune_jobs()

    _submit(
        job,
        _run_job,
        mol,
        intent.job_type,
        intent.functional,
        intent.basis,
        intent.nstates,
        intent.solvation,
        40,
    )

    response["job_id"] = job.id
    response["action"] = "compute"
    response["molecule"] = mol.to_dict()
    response["reply"] = (
        f"Running a **{intent.job_type.replace('_', ' ')}** on "
        f"**{mol.name or mol.formula}** ({mol.formula}, {mol.natoms()} atoms) with "
        f"{FUNCTIONALS[intent.functional]['label']} / {BASIS_SETS[intent.basis]['label']}"
        + (f" in implicit solvent (eps={intent.solvation})" if intent.solvation else "")
        + ". I will report the electronic structure as soon as it converges."
    )
    return response


def _fallback_help() -> str:
    return (
        "I am ChatDFT, a conversational front end to a real density functional "
        "theory engine. Ask me things like:\n\n"
        "* *\"Optimise the geometry of benzene with B3LYP/6-31G*\"*\n"
        "* *\"What is the HOMO-LUMO gap of pyridine?\"*\n"
        "* *\"Compute the first 5 excited states of formaldehyde\"*\n"
        "* *\"TD-DFT on naphthalene at PBE0/def2-TZVP in water\"*\n"
        "* *\"Compare phenol and aniline\"*\n"
        "* *\"Explain what a basis set is\"*\n\n"
        "You can name a molecule, give a formula, paste a SMILES string, or drop "
        "in an XYZ block."
    )


# ----------------------------------------------------------------------
@app.post("/api/job")
def submit_job(req: JobRequest):
    try:
        mol = molmod.resolve(req.molecule)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if req.charge is not None:
        mol.charge = req.charge
    if req.multiplicity is not None:
        mol.multiplicity = req.multiplicity
    _enforce_spin_parity(mol)
    if req.name:
        mol.name = req.name

    job = Job(str(uuid.uuid4())[:8], req.kind, req.model_dump())
    with JOBS_LOCK:
        JOBS[job.id] = job
    _prune_jobs()

    _submit(
        job,
        _run_job,
        mol,
        req.kind,
        req.functional,
        req.basis,
        req.nstates,
        req.solvation,
        req.max_steps,
        req.uv_fwhm_ev,
        req.uv_wmin,
        req.uv_wmax,
        req.with_raman,
        req.nmr_linewidth_hz,
        req.nmr_spectrometer_mhz,
    )
    return {"job_id": job.id, "molecule": mol.to_dict(), "status": "queued"}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    payload = job.to_dict()
    if job.status == "completed" and job.result:
        payload["explanation"] = _narrate(job)
    return payload


def _narrate(job: "Job") -> str:
    """The written analysis shown in the UI's Analysis tab.

    Rebuilt on demand from the stored payload and the finished result, so it
    costs nothing at calculation time and stays correct if the wording of the
    narration functions changes.
    """
    if job.kind == "compare":
        # ``explain_compare`` used to be unreachable: this branch returned a
        # one-line stub, so the comparison's Analysis tab never showed the
        # gap/dipole comparison the function was written to produce.
        res = job.result or {}
        res_a, res_b = res.get("a"), res.get("b")
        mol_a = mol_b = None
        try:
            mol_a = molmod.resolve(job.payload.get("molecule", ""))
            mol_b = molmod.resolve(job.payload.get("molecule2", ""))
        except Exception:
            pass
        if mol_a is None or mol_b is None or not res_a or not res_b:
            return (
                "Comparison complete. Both molecules were treated at identical "
                "level of theory so the orbital energies and gaps are directly "
                "comparable; see the side-by-side panel."
            )
        intent = agent.JobIntent(
            job_type="compare",
            functional=res_a.get("functional", "b3lyp"),
            basis=res_a.get("basis", "6-31g*"),
        )
        try:
            return agent.explain_compare(mol_a, mol_b, res_a, res_b, intent)
        except Exception as exc:  # pragma: no cover - narration must never 500
            return f"(Narration unavailable: {exc})"

    if job.kind == "series":
        if not job.result:
            return "Series complete."
        intent = agent.JobIntent(
            job_type="series",
            functional=job.payload.get("functional", "b3lyp"),
            basis=job.payload.get("basis", "6-31g*"),
        )
        try:
            return agent.explain_series(job.result, intent)
        except Exception as exc:                 # narration must never 500
            return f"(Narration unavailable: {exc})"

    if job.kind == "nto":
        if not job.result:
            return "NTO calculation complete."
        mol = None
        try:
            mol = molmod.resolve(job.payload.get("molecule", ""))
        except Exception:
            pass
        try:
            return agent.explain_nto(mol, job.result, agent.JobIntent(
                job_type="nto"))
        except Exception as exc:                 # narration must never 500
            return f"(Narration unavailable: {exc})"

    if job.kind == "field":
        if not job.result:
            return "Field calculation complete."
        intent = agent.JobIntent(
            job_type="field",
            functional=job.payload.get("functional", "b3lyp"),
            basis=job.payload.get("basis", "6-31g*"),
        )
        mol = None
        try:
            mol = molmod.resolve(job.payload.get("molecule", ""))
        except Exception:
            pass
        try:
            return agent.explain_field(mol, job.result, intent)
        except Exception as exc:                 # narration must never 500
            return f"(Narration unavailable: {exc})"

    if job.kind == "reaction":
        if not job.result:
            return "Reaction path complete."
        try:
            return agent.explain_reaction(job.result, agent.JobIntent(
                job_type="reaction"))
        except Exception as exc:                 # narration must never 500
            return f"(Narration unavailable: {exc})"

    if job.kind == "design":
        if not job.result:
            return "Design run complete."
        try:
            return designer.explain_design(job.result)
        except Exception as exc:                 # narration must never 500
            return f"(Narration unavailable: {exc})"

    mol_spec = job.payload.get("molecule", "")
    mol = None
    if mol_spec:
        try:
            mol = molmod.resolve(mol_spec)
        except Exception:
            mol = None
    if mol is None:
        return "Calculation complete. See the Properties, Orbitals and Geometry tabs for the numbers."

    intent = agent.JobIntent(
        job_type=job.kind,
        functional=job.result.get("functional", "b3lyp"),
        basis=job.result.get("basis", "6-31g*"),
    )
    try:
        return agent.summarise_result(job.kind, mol, job.result, intent)
    except Exception as exc:  # pragma: no cover - narration must never 500
        return f"(Narration unavailable: {exc})"


@app.get("/api/jobs")
def list_jobs():
    with JOBS_LOCK:
        items = [j.to_dict(include_result=False) for j in JOBS.values()]
    items.sort(key=lambda d: -d["created"])
    return {"jobs": items[:50]}


@app.delete("/api/job/{job_id}")
def delete_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.pop(job_id, None)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    return {"deleted": job_id}


# ----------------------------------------------------------------------
@app.post("/api/analyze")
def analyze_xyz(req: AnalyzeRequest):
    """Synchronous analysis of a user-supplied structure (fast path)."""
    try:
        mol = molmod.parse_xyz(req.xyz, name=req.name or "imported")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if req.charge is not None:
        mol.charge = req.charge
    if req.multiplicity is not None:
        mol.multiplicity = req.multiplicity
    _enforce_spin_parity(mol)

    job = Job(str(uuid.uuid4())[:8], "geometry_optimization" if req.optimize else "single_point", {})
    with JOBS_LOCK:
        JOBS[job.id] = job
    _prune_jobs()

    kind = "geometry_optimization" if req.optimize else "single_point"
    _submit(job, _run_job, mol, kind, req.functional, req.basis, 6, None, 40)
    return {"job_id": job.id, "molecule": mol.to_dict(), "status": "queued"}


# ----------------------------------------------------------------------
@app.get("/api/llm")
def llm_status():
    """Which model the design agent would talk to, and whether it answers."""
    client = default_llm()
    return {"config": client.describe(), "available": client.available}


@app.post("/api/llm/probe")
def llm_probe():
    """Actually ping every configured model.  Costs one request each."""
    client = default_llm()
    if not client.available:
        raise HTTPException(status_code=400, detail="No model configured")
    return client.probe()


@app.post("/api/nto")
def submit_nto(req: NtoRequest):
    """Natural transition orbitals for one excited state."""
    if req.state < 1 or req.state > 20:
        raise HTTPException(status_code=400,
                            detail="state must be between 1 and 20")
    job = Job(str(uuid.uuid4())[:8], "nto", req.model_dump())
    with JOBS_LOCK:
        JOBS[job.id] = job
    _prune_jobs()
    _submit(job, _run_nto, req.molecule, int(req.state), req.functional,
            req.basis, max(1, min(4, int(req.npairs))))
    return {"job_id": job.id, "molecule": req.molecule, "state": req.state,
            "status": "queued"}


@app.post("/api/reaction")
def submit_reaction(req: ReactionRequest):
    """Relaxed reaction path with the barrier and a verified transition state."""
    if req.coordinate and not agent.parse_coordinate(req.coordinate):
        raise HTTPException(
            status_code=400,
            detail="coordinate must look like 'bond 1 2', 'angle 1 2 3' or "
                   "'torsion 2 0 1 5', with 1-based atom indices")
    npoints = max(3, min(21, int(req.npoints or 9)))
    job = Job(str(uuid.uuid4())[:8], "reaction", req.model_dump())
    with JOBS_LOCK:
        JOBS[job.id] = job
    _prune_jobs()
    _submit(job, _run_reaction, req.molecule, req.coordinate, req.functional,
            req.basis, npoints, req.solvation)
    return {"job_id": job.id, "molecule": req.molecule,
            "coordinate": req.coordinate or "auto", "status": "queued"}


@app.post("/api/field")
def submit_field(req: FieldRequest):
    """Compute a real-space field (ELF, Laplacian, spin, ...) on a plane."""
    kind = (req.kind or "elf").lower()
    if kind not in FIELD_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown field '{kind}'; choose one of "
                   f"{', '.join(sorted(FIELD_KINDS))}")
    if req.plane == "atoms" and (not req.indices or len(req.indices) < 3):
        raise HTTPException(
            status_code=400,
            detail="plane='atoms' needs three atom indices")
    n = max(20, min(200, int(req.n or 80)))
    job = Job(str(uuid.uuid4())[:8], "field", req.model_dump())
    with JOBS_LOCK:
        JOBS[job.id] = job
    _prune_jobs()
    _submit(job, _run_field, req.molecule, kind, req.functional, req.basis,
            req.plane, n, req.indices, req.fragments, req.solvation)
    return {"job_id": job.id, "kind": kind, "molecule": req.molecule,
            "status": "queued"}


@app.get("/api/fields")
def list_fields():
    """The fields the engine can actually compute, for the UI menu."""
    return {"fields": [{"key": k, "label": v} for k, v in FIELD_KINDS.items()]}


@app.post("/api/series")
def submit_series(req: SeriesRequest):
    """Run the same calculation on a set of molecules and return the trend."""
    specs = [s.strip() for s in req.molecules if s and s.strip()]
    if len(specs) < 2:
        raise HTTPException(status_code=400,
                            detail="Give at least two molecules")
    if len(specs) > 24:
        raise HTTPException(status_code=400,
                            detail="At most 24 molecules in one series")
    job = Job(str(uuid.uuid4())[:8], "series", req.model_dump())
    with JOBS_LOCK:
        JOBS[job.id] = job
    _prune_jobs()
    _submit(job, _run_series, specs, req.functional, req.basis, req.solvation)
    return {"job_id": job.id, "molecules": specs, "status": "queued"}


@app.post("/api/design")
def submit_design(req: DesignRequest):
    """Start a design run.  Returns a job id to poll."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Describe what to design")
    job = Job(str(uuid.uuid4())[:8], "design", req.model_dump())
    with JOBS_LOCK:
        JOBS[job.id] = job
    _prune_jobs()
    _submit(job, _run_design, req.text, req.functional, req.basis,
            req.n_candidates, req.dft_top, req.generations, req.use_model,
            req.alert_filter, req.optimize_leads)
    return {"job_id": job.id, "status": "queued"}


@app.get("/api/designs")
def designs():
    """The saved design history, newest first."""
    return {"designs": designer.list_designs()}


@app.get("/api/design/{ident}")
def get_design(ident: str):
    data = designer.load_design(ident)
    if data is None:
        raise HTTPException(status_code=404, detail="No such design")
    data["explanation"] = designer.explain_design(data)
    return data


@app.get("/api/cubes/{token}/{name}")
def get_cube(token: str, name: str):
    """Serve one cube grid.

    Both path segments are matched against a whitelist rather than joined
    blindly: without that, ``?name=../../server.py`` would walk out of the
    cube directory.
    """
    if not _CUBE_TOKEN.match(token) or not _CUBE_NAME.match(name):
        raise HTTPException(404, "Not found")
    path = os.path.join(CUBE_DIR, token, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    # no-store, not max-age: directories are recycled, so the same URL serves
    # a different molecule's grid on the next run.  An hour of caching here
    # would paint yesterday's orbital onto today's molecule.
    return FileResponse(path, media_type="chemical/x-cube",
                        headers={"Cache-Control": "no-store"})


@app.post("/api/admin/clear-cubes")
def clear_cubes():
    """Drop the whole cube cache.

    Nothing in the server deletes cached grids on its own -- the pool is
    recycled by overwriting -- so this is the only way the directory can
    shrink, and it is only ever reached when something explicitly asks.
    """
    removed = _clear_cubes()
    return {"removed": removed}


# ----------------------------------------------------------------------
#  static front end
# ----------------------------------------------------------------------
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/favicon.ico")
    def favicon():
        path = os.path.join(STATIC_DIR, "favicon.svg")
        if os.path.exists(path):
            return FileResponse(path, media_type="image/svg+xml")
        return JSONResponse({}, status_code=204)


def main() -> None:
    import uvicorn

    host = os.environ.get("CHATDFT_HOST", "127.0.0.1")
    port = int(os.environ.get("CHATDFT_PORT", "8000"))
    print(f"[ChatDFT] Serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
