"""
ChatDFT computation core.

A thin, well-instrumented wrapper around PySCF that exposes the quantities
a chemist actually asks for:

    * Hartree-Fock and DFT (LDA/GGA/hybrid/meta-GGA) single points
    * geometry optimisation (BFGS on analytic gradients)
    * HOMO / LUMO / gap, Kohn-Sham orbital ladder
    * Mulliken and Löwdin population analysis
    * dipole moment and its magnitude
    * bond lengths / angles / dihedrals of the optimised structure
    * TD-DFT (TDA) vertical excitation energies, for the excited-state panel
    * optional solvation via a polarisable continuum model (ddCOSMO)

Design notes
------------
* Every heavy object is created inside the job and released afterwards, so
  long-running servers do not accumulate memory.
* All results are plain dicts (JSON serialisable) with explicit units, so
  the frontend never has to guess what a number means.
* Failures raise :class:`DFTError` with a chemist-readable message rather
  than letting a raw PySCF traceback escape to the UI.
"""

from __future__ import annotations

import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HARTREE2EV = 27.211386245988
HARTREE2KCAL = 627.5094740631
DEBYE = 2.541746473  # a.u. -> Debye

# The implicit-solvent model _build_mf actually attaches.  A solvated single
# point and a gas-phase one differ in every number a paper quotes -- energy,
# dipole, HOMO-LUMO gap -- and a figure can only be reproduced if the model
# and its dielectric travel with the result, not just the request.
SOLVENT_MODEL = "ddCOSMO"

# Curated functional / basis sets. Keys are the public names the UI uses.
FUNCTIONALS = {
    "hf": {"label": "Hartree-Fock", "xc": None, "type": "wavefunction"},
    "lda": {"label": "LDA (SVWN5)", "xc": "lda", "type": "dft"},
    "pbe": {"label": "PBE (GGA)", "xc": "pbe", "type": "dft"},
    "blyp": {"label": "BLYP (GGA)", "xc": "blyp", "type": "dft"},
    "bp86": {"label": "BP86 (GGA)", "xc": "bp86", "type": "dft"},
    "b3lyp": {"label": "B3LYP (hybrid)", "xc": "b3lyp", "type": "dft"},
    "pbe0": {"label": "PBE0 (hybrid)", "xc": "pbe0", "type": "dft"},
    "m06l": {"label": "M06-L (meta-GGA)", "xc": "m06l", "type": "dft"},
    "tpss": {"label": "TPSS (meta-GGA)", "xc": "tpss", "type": "dft"},
    "wb97x": {"label": "wB97X-D (range-sep.)", "xc": "wb97x", "type": "dft"},
}

BASIS_SETS = {
    "sto-3g": {"label": "STO-3G (minimal)", "quality": 1},
    "3-21g": {"label": "3-21G (split valence)", "quality": 2},
    "6-31g": {"label": "6-31G", "quality": 3},
    "6-31g*": {"label": "6-31G* (polarised)", "quality": 4},
    "6-31g**": {"label": "6-31G** (polarised)", "quality": 4},
    # Diffuse functions.  These are not a luxury: the polarizability of a
    # molecule is dominated by the outer tail of the density, and no basis
    # above this line can describe it.  Measured on water with B3LYP
    # (probe_raman3.py), alpha_iso in a.u.:
    #     6-31G*        5.13      (-48% against experiment)
    #     6-31+G*       6.88
    #     cc-pVTZ       7.18
    #     aug-cc-pVDZ   9.52      experiment 9.6-9.9
    #     aug-cc-pVTZ   9.89
    # The anisotropy is worse than the trace: 3.25 a.u. at 6-31G* against 0.66
    # at aug-cc-pVTZ, and the anisotropy is what the depolarization ratio is
    # made of.  So a Raman spectrum computed without them has roughly the right
    # band positions and thoroughly wrong intensities and rho values.
    "6-31+g*": {"label": "6-31+G* (diffuse)", "quality": 4, "diffuse": True},
    "6-31++g**": {"label": "6-31++G** (diffuse)", "quality": 4, "diffuse": True},
    "def2-svp": {"label": "def2-SVP", "quality": 4},
    "6-311g*": {"label": "6-311G* (triple-zeta)", "quality": 5},
    "cc-pvdz": {"label": "cc-pVDZ (correlation-consistent)", "quality": 5},
    "aug-cc-pvdz": {"label": "aug-cc-pVDZ (diffuse)", "quality": 5,
                    "diffuse": True},
    "def2-tzvp": {"label": "def2-TZVP (triple-zeta)", "quality": 6},
    "aug-cc-pvtz": {"label": "aug-cc-pVTZ (diffuse)", "quality": 6,
                    "diffuse": True},
}


def basis_has_diffuse(basis: str) -> bool:
    """Whether a basis carries diffuse functions, from the published name.

    Read off the name rather than the basis data because the naming is a hard
    convention: ``aug-`` prefixes and ``+`` signs mean diffuse, and nothing
    else does.  Used to warn when a polarizability is reported from a basis
    that cannot support one -- the number will still be produced, it will just
    be wrong by tens of per cent, and a figure has to say so.
    """
    key = (basis or "").strip().lower()
    if key.startswith("aug-"):
        return True
    if "+" in key:
        return True
    # Some published diffuse sets do not say so in the name.
    return key in {"ma-def2-svp", "ma-def2-tzvp", "def2-tzvppd", "def2-svpd"}


# Practical ceilings so a browser request cannot pin a CPU forever.
MAX_ATOMS_FAST = 60
MAX_ATOMS_OPT = 40

# Reference values for the built-in self-test / benchmark panel.
# Geometries are the experimental or high-level reference values; energies
# are literature HF/6-31G* and B3LYP/6-31G* numbers used purely as sanity
# checks, not as accuracy claims.
BENCHMARKS = [
    {
        "molecule": "water",
        "name": "Water",
        "property": "O-H bond length",
        "reference": 0.9572,
        "unit": "Angstrom",
        "tolerance": 0.03,
        "method": "geometry_optimization",
    },
    {
        "molecule": "water",
        "name": "Water",
        "property": "H-O-H angle",
        "reference": 104.52,
        "unit": "deg",
        "tolerance": 3.0,
        "method": "geometry_optimization",
    },
    {
        "molecule": "methane",
        "name": "Methane",
        "property": "C-H bond length",
        "reference": 1.087,
        "unit": "Angstrom",
        "tolerance": 0.03,
        "method": "geometry_optimization",
    },
    {
        "molecule": "benzene",
        "name": "Benzene",
        "property": "C-C bond length",
        "reference": 1.390,
        "unit": "Angstrom",
        "tolerance": 0.04,
        "method": "geometry_optimization",
    },
    {
        "molecule": "ammonia",
        "name": "Ammonia",
        "property": "H-N-H angle",
        "reference": 106.7,
        "unit": "deg",
        "tolerance": 3.0,
        "method": "geometry_optimization",
    },
]


class DFTError(RuntimeError):
    """Raised for any problem that should be reported to the user verbatim."""


@dataclass
class JobProgress:
    stage: str
    percent: float
    message: str


ProgressFn = Callable[[JobProgress], None]


# ======================================================================
#  the driver
# ======================================================================
class DFTEngine:
    """Stateful wrapper that owns one PySCF mean-field object."""

    def __init__(
        self,
        atom_xyz: str,
        charge: int = 0,
        multiplicity: int = 1,
        functional: str = "b3lyp",
        basis: str = "6-31g*",
        max_cycle: int = 100,
        conv_tol: float = 1e-9,
        verbose: int = 0,
        solvation: Optional[str] = None,
        grid_level: Optional[int] = None,
        bonding: bool = False,
    ):
        self.atom_xyz = atom_xyz
        self.charge = int(charge)
        self.mult = int(multiplicity)
        self.functional = functional.lower()
        self.basis = basis.lower()
        self.max_cycle = int(max_cycle)
        self.conv_tol = float(conv_tol)
        self.verbose = int(verbose)
        self.solvation = solvation
        # The XC quadrature grid is a *numerical* integral, so the Hessian it
        # produces is not exactly invariant under translation or rotation.
        # A denser grid shrinks that noise; frequency work asks for it.
        self.grid_level = int(grid_level) if grid_level else None
        # Mayer bond orders and Hirshfeld charges are wanted on the result a
        # person reads, not on the hundreds of displaced geometries a dipole
        # derivative, a bond scan or a geometry optimisation walks through.
        # Off by default so those inner loops stay as cheap as they were.
        self.analyze_bonding = bool(bonding)

        self.mol = None
        self.mf = None
        self.log: List[str] = []
        self.warnings: List[str] = []

        if self.functional not in FUNCTIONALS:
            raise DFTError(
                f"Unknown functional '{functional}'. Available: {', '.join(FUNCTIONALS)}"
            )
        if self.basis not in BASIS_SETS:
            raise DFTError(
                f"Unknown basis set '{basis}'. Available: {', '.join(BASIS_SETS)}"
            )

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"[{stamp}] {msg}")

    def _build_mol(self):
        from backend import bootstrap

        bootstrap.setup()  # make libcint/libxc/MKL discoverable before import
        from pyscf import gto

        try:
            mol = gto.M(
                atom=_xyz_to_atom_spec(self.atom_xyz),
                charge=self.charge,
                spin=self.mult - 1,
                basis=self.basis,
                verbose=self.verbose,
                unit="Angstrom",
            )
        except Exception as exc:
            raise DFTError(f"PySCF rejected the molecular geometry: {exc}")

        n_elec = mol.nelectron
        if n_elec <= 0:
            raise DFTError("The requested charge leaves the molecule with no electrons.")
        if (mol.spin + n_elec) % 2 != 0:
            raise DFTError(
                f"Multiplicity {self.mult} is incompatible with {n_elec} electrons "
                f"(charge {self.charge})."
            )
        self.mol = mol
        self._log(
            f"Built {mol.natm}-atom system, {n_elec} electrons, "
            f"spin={self.mult}, {mol.nao} basis functions"
        )
        return mol

    def _build_mf(self):
        from pyscf import dft, scf

        mol = self.mol
        info = FUNCTIONALS[self.functional]

        if info["type"] == "wavefunction":
            mf = scf.UHF(mol) if self.mult > 1 else scf.RHF(mol)
            mf.xc = ""
        else:
            mf = dft.UKS(mol) if self.mult > 1 else dft.RKS(mol)
            mf.xc = info["xc"]
            if self.functional == "wb97x":
                # wB97X-D is a range-separated functional with dispersion
                try:
                    mf = mf.density_fit(auxbasis="def2-universal-jfit")
                except Exception:
                    pass
                mf._numint = mf._numint  # keep default grid, just note it
                self.warnings.append(
                    "wB97X-D range separation is approximated; treat absolute "
                    "energies with care."
                )
            mf.grids.level = 3
            if self.grid_level:
                mf.grids.level = max(mf.grids.level, self.grid_level)

        if self.solvation:
            try:
                mf = mf.ddCOSMO()
                mf.with_solvent.eps = float(self.solvation)
                self._log(f"Applied implicit solvation (epsilon = {self.solvation})")
            except Exception as exc:
                self.warnings.append(f"Solvation model unavailable: {exc}")

        mf.max_cycle = self.max_cycle
        mf.conv_tol = self.conv_tol
        mf.conv_tol_grad = math.sqrt(self.conv_tol)
        # PySCF writes a temporary checkpoint file by default.  ChatDFT keeps
        # every result in memory and returns it to the caller, so the chkfile is
        # pure overhead -- and on Windows h5py's probe of the temp path can fail
        # outright.  Disable it.
        mf.chkfile = None
        self.mf = mf
        return mf

    # ------------------------------------------------------------------
    def run_scf(self, progress: Optional[ProgressFn] = None) -> dict:
        """Converge the SCF and harvest every scalar property."""
        if progress:
            progress(JobProgress("setup", 5, "Building basis and integrals"))

        self._build_mol()
        mf = self._build_mf()

        if progress:
            progress(
                JobProgress(
                    "scf",
                    20,
                    f"Running {FUNCTIONALS[self.functional]['label']} SCF in {self.basis}",
                )
            )

        t0 = time.time()
        # Use a Newton/second-order solver as backstop for stubborn cases.
        try:
            energy = mf.kernel()
        except Exception as exc:
            raise DFTError(f"SCF failed with an internal error: {exc}")
        elapsed = time.time() - t0

        if not mf.converged:
            self._log("First SCF pass did not converge; retrying with a second-order solver.")
            if progress:
                progress(JobProgress("scf", 55, "Refining SCF with Newton solver"))
            try:
                mf = mf.newton()
                mf.max_cycle = self.max_cycle
                mf.kernel()
            except Exception as exc:
                self._log(f"Newton fallback failed: {exc}")

        if not mf.converged:
            self.warnings.append(
                "SCF did not converge to the tight threshold. Numbers below are "
                "indicative only - consider a smaller basis or a different functional."
            )
        else:
            self._log(f"SCF converged in {elapsed:.2f}s, E = {energy:.8f} Ha")

        if progress:
            progress(JobProgress("analysis", 75, "Extracting orbitals and populations"))

        result = self._harvest(mf, energy, elapsed)

        # Geometry of the structure that was actually computed.  A single point
        # reports the input geometry; an optimisation overwrites this with the
        # relaxed one.
        result["geometry"] = _geometry_table(self.atom_xyz)
        result["job_type"] = "single_point"

        if progress:
            progress(JobProgress("done", 100, "Calculation complete"))
        return result

    # ------------------------------------------------------------------
    def _harvest(self, mf, energy: float, elapsed: float) -> dict:
        import numpy as np

        mol = self.mol
        is_open = self.mult > 1

        # ---- orbital energies ----------------------------------------
        if is_open:
            mo_energy = np.asarray(mf.mo_energy)
            # alpha channel highest occupied / lowest virtual
            occ_a = np.asarray(mf.mo_occ)[0]
            nocc = int(np.sum(occ_a > 1e-6))
            homo = float(mo_energy[0][nocc - 1]) if nocc > 0 else float("nan")
            lumo = float(mo_energy[0][nocc]) if nocc < len(mo_energy[0]) else float("nan")
        else:
            mo_energy = np.asarray(mf.mo_energy)
            occ = np.asarray(mf.mo_occ)
            nocc = int(np.sum(occ > 1e-6))
            homo = float(mo_energy[nocc - 1]) if nocc > 0 else float("nan")
            lumo = float(mo_energy[nocc]) if nocc < len(mo_energy) else float("nan")

        gap = lumo - homo if (homo == homo and lumo == lumo) else float("nan")

        # ---- full orbital ladder (for the energy-level diagram) ------
        ladder = []
        if not is_open:
            for k in range(min(len(mo_energy), 40)):
                ladder.append(
                    {
                        "index": k,
                        "energy_ha": float(mo_energy[k]),
                        "energy_ev": float(mo_energy[k] * HARTREE2EV),
                        "occupation": float(occ[k]),
                        "label": "HOMO" if k == nocc - 1 else ("LUMO" if k == nocc else ""),
                    }
                )
            nocc_slots = nocc
        else:
            # In an unrestricted calculation every alpha orbital holds one
            # electron, so occupation alone cannot identify a SOMO.  The beta
            # channel fills the lowest orbitals; alpha orbitals above that
            # count are the genuinely singly-occupied ones.
            occ_b = np.asarray(mf.mo_occ)[1]
            nocc_b = int(np.sum(occ_b > 1e-6))
            for k in range(min(len(mo_energy[0]), 40)):
                if k == nocc - 1:
                    label = "HOMO"          # highest occupied (alpha)
                elif k == nocc:
                    label = "LUMO"
                elif nocc_b <= k < nocc - 1:
                    label = "SOMO"
                else:
                    label = ""
                ladder.append(
                    {
                        "index": k,
                        "energy_ha": float(mo_energy[0][k]),
                        "energy_ev": float(mo_energy[0][k] * HARTREE2EV),
                        "occupation": float(occ_a[k]),
                        "label": label,
                    }
                )
            nocc_slots = nocc

        # ---- populations ---------------------------------------------
        charges_mulliken, charges_lowdin = self._populations(mf)

        # ---- bonding: Mayer bond orders and Hirshfeld charges ---------
        # Mayer bond orders and Hirshfeld charges are the two tables a paper
        # quotes when it says "the bond order increases" or "the charge
        # transfers".  Both are cheap next to the SCF that just finished.
        bonding_result: dict = {}
        if self.analyze_bonding:
            try:
                from . import bonding as _bonding

                bonding_result = _bonding.bonding_analysis(
                    self.mol, mf, FUNCTIONALS[self.functional].get("xc"))
            except Exception as exc:                         # noqa: BLE001
                self._log(f"bonding analysis failed: {exc}")
                bonding_result = {"error": f"{type(exc).__name__}: {exc}"}

        # ---- dipole --------------------------------------------------
        # PySCF returns the total (alpha+beta) dipole for open-shell UKS too,
        # so this is valid for radicals as well as closed-shell molecules.
        dipole = {"x": 0.0, "y": 0.0, "z": 0.0, "magnitude": 0.0, "unit": "Debye"}
        try:
            mu = mf.dip_moment(unit="Debye", verbose=0)
            dipole = {
                "x": float(mu[0]),
                "y": float(mu[1]),
                "z": float(mu[2]),
                "magnitude": float(np.linalg.norm(mu)),
                "unit": "Debye",
            }
        except Exception as exc:
            self._log(f"Dipole moment unavailable: {exc}")

        # ---- frontier-orbital composition ----------------------------
        frontier = self._frontier_composition(mf, nocc, is_open)

        # ---- spin contamination ---------------------------------------
        # A single unrestricted determinant is not an eigenfunction of S^2,
        # and the SCF can also just land on the wrong state.  <S^2> is how a
        # paper shows that an open-shell calculation *is* the spin state it
        # claims, and here there was no way to see it: a clean doublet and a
        # broken-symmetry mess produced byte-for-byte the same payload, so
        # energies, orbitals and charges were all read with the same
        # confidence.  Deviations past ~10% of S(S+1) are the usual point at
        # which unrestricted results stop being trustworthy.
        spin = None
        if is_open:
            s_exp = 0.5 * (self.mult - 1)
            want = s_exp * (s_exp + 1.0)
            try:
                ss = float(mf.spin_square()[0])
                dev = ss - want
                rel = (dev / want * 100.0) if want > 0.0 else 0.0
                spin = {
                    "s_squared": round(ss, 5),
                    "s_squared_expected": round(want, 5),
                    "contamination": round(dev, 5),
                    "contamination_pct": round(rel, 2),
                    "multiplicity": self.mult,
                }
                self._log(
                    f"<S^2> = {ss:.4f} against the exact {want:.4f} for a "
                    f"multiplicity-{self.mult} state (deviation {rel:+.2f}%)")
                if want > 0.0 and rel > 10.0:
                    self.warnings.append(
                        f"The unrestricted wavefunction is not the spin state "
                        f"it was asked for: <S^2> = {ss:.3f} against "
                        f"{want:.3f} for multiplicity {self.mult}, a {rel:.0f}% "
                        f"deviation. Energies and spin densities from this "
                        f"calculation should not be reported as those of a "
                        f"pure spin state.")
            except Exception as exc:                      # noqa: BLE001
                self._log(f"Spin analysis unavailable: {exc}")

        return {
            "energy_hartree": float(energy),
            "energy_ev": float(energy * HARTREE2EV),
            "energy_kcal": float(energy * HARTREE2KCAL),
            "homo_ha": homo,
            "homo_ev": homo * HARTREE2EV if homo == homo else float("nan"),
            "lumo_ha": lumo,
            "lumo_ev": lumo * HARTREE2EV if lumo == lumo else float("nan"),
            "gap_ha": gap,
            "gap_ev": gap * HARTREE2EV if gap == gap else float("nan"),
            "nocc": nocc_slots,
            "nbf": int(mol.nao),
            "nelec": int(mol.nelectron),
            "converged": bool(mf.converged),
            "scf_seconds": round(elapsed, 3),
            "functional": self.functional,
            "functional_label": FUNCTIONALS[self.functional]["label"],
            "basis": self.basis,
            "basis_label": BASIS_SETS[self.basis]["label"],
            "multiplicity": self.mult,
            "charge": self.charge,
            "solvation": self.solvation,
            "solvation_model": (SOLVENT_MODEL if self.solvation else None),
            "orbital_ladder": ladder,
            # An unrestricted calculation has two spin channels with *different*
            # orbital energies, and the ladder plots only one of them.  A
            # figure captioned "orbital energies" that silently shows alpha
            # alone is not reproducible, so the payload has to say which
            # channel it is.
            "ladder_channel": ("alpha (spin up)" if is_open
                               else "closed shell"),
            "mulliken_charges": charges_mulliken,
            "lowdin_charges": charges_lowdin,
            "bonding": bonding_result,
            "dipole": dipole,
            "frontier_orbitals": frontier,
            "spin": spin,
            "warnings": list(self.warnings),
            "log": list(self.log),
        }

    # ------------------------------------------------------------------
    def _populations(self, mf) -> Tuple[List[dict], List[dict]]:
        import numpy as np

        symbols = [self.mol.atom_symbol(i) for i in range(self.mol.natm)]
        mull, lowd = [], []
        try:
            pop, chg = mf.mulliken_pop(verbose=0)
            for i, c in enumerate(chg):
                mull.append(
                    {"atom": i + 1, "symbol": symbols[i], "charge": round(float(c), 5)}
                )
        except Exception as exc:
            self._log(f"Mulliken analysis failed: {exc}")
            mull = [
                {"atom": i + 1, "symbol": s, "charge": 0.0} for i, s in enumerate(symbols)
            ]

        try:
            lowd = self._lowdin_charges(mf)
        except Exception as exc:
            self._log(f"Loewdin analysis failed: {exc}")
            lowd = []
        return mull, lowd

    @staticmethod
    def _s_half(mf):
        """S^(1/2) for the symmetric (Loewdin) orthogonalisation.

        Shared by the Loewdin charges and the frontier-orbital composition so
        the two cannot drift apart.
        """
        import numpy as np

        S = np.asarray(mf.get_ovlp())
        w, v = np.linalg.eigh(S)
        w = np.where(w > 1e-10, w, 1e-10)
        return (v * np.sqrt(w)) @ v.T

    def _lowdin_charges(self, mf) -> List[dict]:
        """Loewdin charges: the Mulliken partition in the symmetrically
        orthogonalised (S^-1/2) basis.

        PySCF's ``mulliken_pop`` takes no ``xctype`` argument -- the call used
        to raise ``TypeError: got an unexpected keyword argument 'xctype'``,
        and because the surrounding handler swallowed every exception this
        list came back empty for *every* job while the UI still advertised
        Loewdin charges.  The transform is short enough to do explicitly.
        """
        import numpy as np

        from . import elements

        s_half = self._s_half(mf)
        dm = np.asarray(mf.make_rdm1())
        if dm.ndim == 3:                 # open shell: sum the two spin channels
            dm = dm.sum(axis=0)
        pop = np.real(np.diag(s_half @ dm @ s_half))

        out = []
        slices = self.mol.aoslice_by_atom()
        for i in range(self.mol.natm):
            p0, p1 = slices[i][2:]
            z = elements.get(self.mol.atom_symbol(i)).z
            out.append(
                {
                    "atom": i + 1,
                    "symbol": self.mol.atom_symbol(i),
                    "charge": round(float(z - pop[p0:p1].sum()), 5),
                }
            )
        return out

    # ------------------------------------------------------------------
    def _frontier_composition(self, mf, nocc: int, is_open: bool) -> dict:
        """Per-atom Loewdin population of the HOMO and LUMO, in percent.

        The population is ``sum_{mu in A} |(S^(1/2) c)_mu|^2`` -- a sum of
        squares in the symmetrically orthogonalised basis.

        This used to be the naive ``sum |c_mu|^2`` over each atom's AOs, which
        is not a population at all: MOs are normalised as ``c^T S c = 1``, not
        ``c^T c = 1``, so that expression silently drops every cross term.  It
        agreed well enough on HOMOs, but it inverted the LUMO of water --
        reporting O1 46%, H2 27%, H3 27% where the correct gross populations
        are H2 53%, H3 53% and O1 *negative*, because the LUMO is O-H
        antibonding and the cross terms are negative.

        The Mulliken gross population ``sum c_mu (S c)_mu`` is the textbook
        alternative and is what the naive version was trying to approximate,
        but it genuinely goes negative for antibonding orbitals, which cannot
        be drawn as a share of anything.  Loewdin is the right choice for a
        display: it is an exact partition (it sums to ``c^T S c = 1``), it is
        never negative, and it uses the same S^(1/2) as the Loewdin charges.
        """
        import numpy as np

        out = {"homo": [], "lumo": []}
        try:
            if is_open:
                C = np.asarray(mf.mo_coeff[0])
            else:
                C = np.asarray(mf.mo_coeff)
            # atom -> ao slice
            slices = []
            for i in range(self.mol.natm):
                p0, p1 = self.mol.aoslice_by_atom()[i][2:]
                slices.append((p0, p1))
            s_half = self._s_half(mf)

            def comp(vec):
                ortho = s_half @ vec
                dense = np.zeros(self.mol.natm)
                for i, (p0, p1) in enumerate(slices):
                    dense[i] = float(np.sum(ortho[p0:p1] ** 2))
                total = dense.sum()
                if total > 0:
                    dense = dense / total * 100.0
                return [
                    {
                        "atom": i + 1,
                        "symbol": self.mol.atom_symbol(i),
                        "percent": round(float(dense[i]), 2),
                    }
                    for i in range(self.mol.natm)
                    if dense[i] > 0.5
                ]

            if nocc > 0:
                out["homo"] = comp(C[:, nocc - 1])
            if nocc < C.shape[1]:
                out["lumo"] = comp(C[:, nocc])
        except Exception as exc:
            self._log(f"Frontier orbital composition unavailable: {exc}")
        return out

    # ------------------------------------------------------------------
    # The optimiser is given these, and the result is checked against the
    # same numbers -- one definition, so the two cannot drift apart.  PySCF's
    # own convergence flag is unusable: its ``optimize()`` returns
    # ``kernel(...)[1]``, throwing the flag away, and ``kernel`` catches
    # NotConvergedError and hands back the last geometry with conv=False.
    OPT_CONVERGENCE = {
        "convergence_energy": 1e-5,   # hartree
        "convergence_grms": 3e-4,     # hartree/bohr
        "convergence_gmax": 4.5e-4,   # hartree/bohr
        "convergence_drms": 1.2e-3,   # angstrom
        "convergence_dmax": 1.8e-3,   # angstrom
    }

    def optimize(self, max_steps: int = 40, progress: Optional[ProgressFn] = None) -> dict:
        """Geometry optimisation on analytic gradients.

        Tries the backends in order of preference and falls back if one is not
        available on this platform: ``geometric`` (default) then ``pyberny``.
        Both consume the same PySCF mean-field object and return an optimised
        ``gto.Mole``, so the caller does not care which one ran.
        """
        self._build_mol()
        mf = self._build_mf()

        if progress:
            progress(JobProgress("opt", 10, "Optimising geometry (analytic gradients)"))
        self._log(f"Starting geometry optimisation, max {max_steps} steps")

        t0 = time.time()
        mol_eq, backend, errors = None, None, []
        # geomeTRIC calls back once per energy+gradient evaluation, which is
        # the only way to learn how many cycles it actually took: the object
        # ``optimize`` returns is a bare Mole and carries no history.
        steps = [0]

        def _count_step(_locals=None) -> None:
            steps[0] += 1

        try:
            from pyscf.geomopt.geometric_solver import optimize as _geo_opt

            mol_eq = _geo_opt(
                mf,
                maxsteps=max_steps,
                callback=_count_step,
                **self.OPT_CONVERGENCE,
            )
            backend = "geometric"
        except Exception as exc:  # noqa: BLE001 - try the next backend
            errors.append(f"geometric: {exc}")
            self._log(f"geometric backend unavailable ({exc}); trying pyberny")

        if mol_eq is None:
            try:
                from pyscf.geomopt.berny_solver import optimize as _berny_opt

                mol_eq = _berny_opt(mf, maxsteps=max_steps)
                backend = "pyberny"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"pyberny: {exc}")

        if mol_eq is None:
            raise DFTError(
                "Geometry optimisation failed with every available backend "
                f"({'; '.join(errors)}). The initial structure may be too "
                "strained - try a cheaper functional/basis or supply a better "
                "starting geometry."
            )

        elapsed = time.time() - t0

        # re-run a single point on the optimised geometry for the properties
        if progress:
            progress(JobProgress("scf", 70, "Single point on optimised geometry"))

        opt_xyz = _mol_to_xyz(mol_eq)
        sp = DFTEngine(
            atom_xyz=opt_xyz,
            charge=self.charge,
            multiplicity=self.mult,
            functional=self.functional,
            basis=self.basis,
            max_cycle=self.max_cycle,
            conv_tol=self.conv_tol,
            verbose=self.verbose,
            solvation=self.solvation,
            # Inherited, not hardcoded: the final single point is what carries
            # the numbers a person reads, so it needs whatever the caller asked
            # for -- but only that.
            bonding=self.analyze_bonding,
        )
        result = sp.run_scf(progress=None)

        # Trust the physics, not the flag.  geomeTRIC reports convergence to
        # PySCF, PySCF discards it, and the structure it hands back is the
        # last one it visited whether or not it was a stationary point.  So
        # "converged" used to be logged unconditionally, and an unconverged
        # geometry went downstream as a minimum -- into the vibrational
        # frequencies, the surfaces and the reaction path, none of which are
        # meaningful away from one.  A stationary point is a point whose
        # gradient is below the thresholds actually used, and that is
        # checkable: one analytic gradient, which the optimisation has
        # already paid for several times over.
        import numpy as np

        conv_ok = None
        grms = gmax = None
        if sp.mf is not None and sp.mf.converged:
            g = np.asarray(sp.mf.nuc_grad_method().kernel(), dtype=float)
            grms = float(np.sqrt((g ** 2).mean()))
            gmax = float(np.abs(g).max())
            conv_ok = bool(
                grms <= self.OPT_CONVERGENCE["convergence_grms"]
                and gmax <= self.OPT_CONVERGENCE["convergence_gmax"])
        if conv_ok:
            self._log(
                f"Geometry optimisation converged with {backend}: "
                f"|g|rms {grms:.2e}, |g|max {gmax:.2e} hartree/bohr "
                f"(criteria {self.OPT_CONVERGENCE['convergence_grms']:.0e} "
                f"and {self.OPT_CONVERGENCE['convergence_gmax']:.0e})")
        elif conv_ok is None:
            self._log("Geometry optimisation finished, but the SCF at the "
                      "final geometry did not converge, so the gradient "
                      "criterion could not be checked")
            self.warnings.append(
                "The optimisation could not be checked for convergence "
                "because the SCF at the final geometry did not converge; "
                "treat the structure as unrelaxed.")
        else:
            self._log(
                f"Geometry optimisation with {backend} did NOT reach the "
                f"gradient criteria: |g|rms {grms:.2e} and |g|max {gmax:.2e} "
                f"hartree/bohr against "
                f"{self.OPT_CONVERGENCE['convergence_grms']:.0e} and "
                f"{self.OPT_CONVERGENCE['convergence_gmax']:.0e}, after "
                f"{steps[0]} of {max_steps} steps")
            self.warnings.append(
                f"The geometry optimisation did not converge: the residual "
                f"gradient is |g|rms {grms:.2e} hartree/bohr after "
                f"{steps[0]} steps. Frequencies, surfaces and barriers "
                f"computed here are not those of a stationary point.")

        result["optimized_xyz"] = opt_xyz
        result["opt_seconds"] = round(elapsed, 3)
        result["opt_backend"] = backend
        result["opt_converged"] = conv_ok
        result["opt_grms"] = None if grms is None else round(grms, 8)
        result["opt_gmax"] = None if gmax is None else round(gmax, 8)
        result["opt_criteria"] = dict(self.OPT_CONVERGENCE)
        result["n_steps"] = steps[0] or None
        result["geometry"] = _geometry_table(opt_xyz)
        result["log"] = self.log + result.get("log", [])
        result["warnings"] = self.warnings + result.get("warnings", [])
        result["job_type"] = "geometry_optimization"

        if progress:
            progress(JobProgress("done", 100, "Optimisation complete"))
        return result

    # ------------------------------------------------------------------
    def excited_states(self, nstates: int = 6, progress: Optional[ProgressFn] = None) -> dict:
        """TD-DFT (Tamm-Dancoff) vertical excitations on the ground state."""
        from pyscf import tdscf

        self.run_scf(progress=None)
        mf = self.mf
        if not mf.converged:
            raise DFTError("TD-DFT needs a converged ground state; SCF did not converge.")

        if self.mult > 1:
            raise DFTError("Excited-state analysis is implemented for closed-shell systems only.")

        if progress:
            progress(JobProgress("tddft", 60, f"Computing {nstates} excited states (TDA)"))

        self._log(f"Running TDA with {nstates} roots")
        t0 = time.time()
        try:
            td = tdscf.TDA(mf)
            td.nstates = int(nstates)
            td.kernel()
        except Exception as exc:
            raise DFTError(f"TD-DFT failed: {exc}")
        elapsed = time.time() - t0

        # In PySCF 2.6 oscillator_strength is a method; older releases exposed
        # it as an array attribute.  Accept either.
        osc = getattr(td, "oscillator_strength", None)
        if callable(osc):
            try:
                osc = osc()
            except Exception as exc:
                self._log(f"Oscillator strengths unavailable: {exc}")
                osc = None

        states = []
        for k, e in enumerate(td.e):
            row = {
                "state": k + 1,
                "energy_ha": float(e),
                "energy_ev": float(e * HARTREE2EV),
                "wavelength_nm": float(1239.84198 / (e * HARTREE2EV)) if e > 0 else None,
            }
            if osc is not None and k < len(osc):
                row["oscillator_strength"] = float(osc[k])
                row["active"] = bool(osc[k] > 0.01)
            states.append(row)

        base = self._harvest(mf, mf.e_tot, 0.0)
        base["excited_states"] = states
        base["tddft_seconds"] = round(elapsed, 3)
        base["job_type"] = "excited_states"
        # Tamm-Dancoff, not full TD-DFT -- and the two are not interchangeable:
        # on formaldehyde they differ by up to 0.42 eV at 6-31G*, so a figure
        # captioned "TD-DFT" cannot be reproduced from the number below.  The
        # payload says which one ran so the exported figure can too.
        base["excited_method"] = "TDA"
        base["excited_method_label"] = "TDA (Tamm-Dancoff)"
        # TD-DFT is a vertical excitation from the input geometry, so report
        # that geometry -- same shape as the single-point result.
        base["geometry"] = _geometry_table(self.atom_xyz)
        base["log"] = self.log + base.get("log", [])
        base["warnings"] = self.warnings + base.get("warnings", [])

        if progress:
            progress(JobProgress("done", 100, "Excited states complete"))
        return base


# ======================================================================
#  module-level helpers
# ======================================================================
def _is_atom_line(line: str) -> bool:
    """True if ``line`` reads "Sym x y z" rather than being a header line."""
    parts = line.split()
    if len(parts) < 4 or not parts[0].isalpha():
        return False
    try:
        [float(v) for v in parts[1:4]]
    except ValueError:
        return False
    return True


def _split_xyz(block: str) -> Tuple[List[str], List[List[float]]]:
    """Parse an XYZ block into (symbols, coords), tolerating a missing comment.

    An XYZ block is an atom count, a free-text comment, then the atoms.  The
    comment is optional in practice, and a *blank* one disappears when blank
    lines are filtered -- so any parser that assumes "skip exactly two lines"
    silently drops the first atom.  That is not hypothetical: it built water
    as H2 (2 atoms, E = -1.06 Ha) with no error anywhere.  Both the PySCF atom
    spec and the geometry table go through here so they cannot disagree.
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        raise DFTError("Empty molecular geometry.")

    start = 0
    declared = None
    if lines[0].lstrip("+-").isdigit():
        declared = int(lines[0])
        # Skip the comment only if there is one to skip.
        start = 1 if len(lines) > 1 and _is_atom_line(lines[1]) else min(2, len(lines))

    syms: List[str] = []
    coords: List[List[float]] = []
    for ln in lines[start:]:
        if not _is_atom_line(ln):
            break
        p = ln.split()
        syms.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
        if declared is not None and len(syms) == declared:
            break

    if not syms:
        raise DFTError("No atoms found in the molecular geometry.")
    if declared is not None and len(syms) != declared:
        raise DFTError(
            f"XYZ block declares {declared} atoms but {len(syms)} were parsed."
        )
    return syms, coords


def _xyz_to_atom_spec(block: str) -> str:
    """Turn an XYZ block into the bare atom list PySCF's ``gto.M`` expects.

    ``gto.M(atom=...)`` accepts "Symbol x y z" lines, a Z-matrix, or a *path*
    to a file -- but not an XYZ string with its two-line header.  Passing the
    header through makes PySCF read the atom count as a one-field Z-matrix
    line and fail.  Strip it here.
    """
    syms, coords = _split_xyz(block)
    return "\n".join(
        f"{s} {x} {y} {z}" for s, (x, y, z) in zip(syms, coords)
    )


def _mol_to_xyz(mol) -> str:
    lines = [str(mol.natm), "optimized"]
    coords = mol.atom_coords(unit="Angstrom")
    for i in range(mol.natm):
        s = mol.atom_symbol(i)
        x, y, z = coords[i]
        lines.append(f"{s:<3s} {x:16.8f} {y:16.8f} {z:16.8f}")
    return "\n".join(lines) + "\n"


def _geometry_table(xyz: str) -> dict:
    """Bond lengths, angles and dihedrals from an XYZ block."""
    import numpy as np

    # Same parser the PySCF atom spec uses -- see _split_xyz.
    syms, raw = _split_xyz(xyz)
    n = len(syms)
    coords = np.asarray(raw)

    from .elements import bond_cutoff

    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if 0.4 < d <= bond_cutoff(syms[i], syms[j]):
                bonds.append((i, j, d))

    bond_rows = [
        {
            "label": f"{syms[i]}{i + 1}-{syms[j]}{j + 1}",
            "value": round(d, 4),
            "unit": "Angstrom",
        }
        for i, j, d in sorted(bonds, key=lambda b: b[2])
    ]

    angle_rows = []
    adj: Dict[int, List[int]] = {}
    for i, j, _ in bonds:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    for c in range(n):
        nb = adj.get(c, [])
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                i, j = nb[a], nb[b]
                v1 = coords[i] - coords[c]
                v2 = coords[j] - coords[c]
                cosang = float(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                )
                ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
                angle_rows.append(
                    {
                        "label": f"{syms[i]}{i + 1}-{syms[c]}{c + 1}-{syms[j]}{j + 1}",
                        "value": round(ang, 2),
                        "unit": "deg",
                    }
                )

    dihedral_rows = []
    for i, j, _ in bonds:
        for k in adj.get(j, []):
            if k == i:
                continue
            for l in adj.get(k, []):
                if l in (i, j):
                    continue
                b0, b1, b2 = coords[i], coords[j], coords[k]
                b3 = coords[l]
                b0_ = b0 - b1
                b1_ = b2 - b1
                b2_ = b3 - b2
                n1 = np.cross(b0_, b1_)
                n2 = np.cross(b1_, b2_)
                m = np.cross(n1, b1_ / (np.linalg.norm(b1_) + 1e-12))
                x = float(np.dot(n1, n2))
                y = float(np.dot(m, n2))
                dih = abs(math.degrees(math.atan2(y, x)))
                dihedral_rows.append(
                    {
                        "label": f"{syms[i]}{i + 1}-{syms[j]}{j + 1}-"
                        f"{syms[k]}{k + 1}-{syms[l]}{l + 1}",
                        "value": round(dih, 2),
                        "unit": "deg",
                    }
                )
                if len(dihedral_rows) >= 30:
                    break
            if len(dihedral_rows) >= 30:
                break
        if len(dihedral_rows) >= 30:
            break

    return {
        "bonds": bond_rows,
        "angles": angle_rows[:40],
        "dihedrals": dihedral_rows[:25],
    }


# ======================================================================
#  rigid bond scan
# ======================================================================
def scan_bond(
    atom_xyz: str,
    i: int,
    j: int,
    functional: str = "b3lyp",
    basis: str = "6-31g*",
    charge: int = 0,
    multiplicity: int = 1,
    solvation: Optional[str] = None,
    steps: int = 12,
    lo_scale: float = 0.85,
    hi_scale: float = 1.45,
    progress: Optional[ProgressFn] = None,
) -> dict:
    """Rigid potential-energy scan along the bond between atoms ``i`` and ``j``.

    The two atoms are displaced symmetrically about their midpoint along the
    bond axis and every other atom is held fixed.  A rigid scan is the usual
    first look at a bond: it locates the well and shows its shape without the
    cost of relaxing the whole molecule at every point, which is why the
    minimum it finds sits slightly *above* the true equilibrium length.

    Returns the full single-point result for the sampled geometry nearest the
    input bond length (so the Properties / Orbitals panes stay populated), plus
    a ``scan`` block holding the curve.
    """
    import numpy as np

    from . import molecule as molmod

    mol = molmod.parse_xyz(atom_xyz)
    n = mol.natoms()
    if not (0 <= i < n and 0 <= j < n) or i == j:
        raise DFTError("A bond scan needs two distinct atom indices.")

    p = np.array([mol.atoms[i].x, mol.atoms[i].y, mol.atoms[i].z])
    q = np.array([mol.atoms[j].x, mol.atoms[j].y, mol.atoms[j].z])
    vec = q - p
    r0 = float(np.linalg.norm(vec))
    if r0 < 1e-6:
        raise DFTError("The two selected atoms coincide, so there is no bond to scan.")

    u = vec / r0
    mid = 0.5 * (p + q)

    distances = np.linspace(lo_scale * r0, hi_scale * r0, steps)
    # A field that reports how long the scan took has to actually be timed:
    # it was hardcoded to None, which is the same shape as the step count an
    # optimisation used to report.
    t_scan = time.time()
    fixed = [
        np.array([a.x, a.y, a.z]) for a in mol.atoms
    ]
    points: list[dict] = []
    results: list[dict] = []

    for k, r in enumerate(distances):
        if progress:
            progress(JobProgress("scan", 5.0 + 90.0 * k / steps,
                                 f"Scan point {k + 1}/{steps}: r = {r:.3f} A"))
        lines = []
        for idx, atom in enumerate(mol.atoms):
            if idx == i:
                xyz = mid - 0.5 * r * u
            elif idx == j:
                xyz = mid + 0.5 * r * u
            else:
                xyz = fixed[idx]
            lines.append(f"{atom.symbol} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}")
        block = f"{n}\nscan\n" + "\n".join(lines)

        engine = DFTEngine(
            atom_xyz=block,
            charge=charge,
            multiplicity=multiplicity,
            functional=functional,
            basis=basis,
            solvation=solvation,
        )
        res = engine.run_scf()
        results.append(res)
        points.append(
            {
                "index": k,
                "r": float(r),
                "energy_hartree": float(res["energy_hartree"]),
                "converged": bool(res["converged"]),
            }
        )

    energies = [pt["energy_hartree"] for pt in points]
    kmin = int(np.argmin(energies))
    emin = energies[kmin]
    for pt in points:
        pt["energy_rel_kcal"] = float((pt["energy_hartree"] - emin) * HARTREE2KCAL)

    # --- where the bottom actually is ----------------------------------
    # The grid minimum is one of the sampled points, so it is quantised, and
    # refining the grid does not steadily improve it: measured on water's O-H
    # well against a 101-point reference, the grid minimum is off by +0.0115 A
    # at 8 points, +0.0040 at 12, +0.0005 at 16, -0.0091 at 25 and -0.0054 at
    # 40 -- worse at 25 than at 16, because it is whichever point happened to
    # land nearest.  Quoting it to three decimals, as the narration did,
    # claims a precision the grid cannot support.
    #
    # A parabola through the three points centred on it recovers the true
    # bottom to 0.003 A at the default density and 0.0004 A at 25 points, at
    # no extra SCF cost.  Both are reported: the sampled point for what was
    # actually computed, the vertex for where the well bottoms out.
    grid_step = float(distances[1] - distances[0]) if steps > 1 else 0.0
    fitted = None
    if 0 < kmin < len(points) - 1:
        h = points[kmin]["r"] - points[kmin - 1]["r"]
        y1, y2, y3 = (points[kmin - 1]["energy_hartree"],
                      points[kmin]["energy_hartree"],
                      points[kmin + 1]["energy_hartree"])
        den = y1 - 2.0 * y2 + y3
        if abs((points[kmin + 1]["r"] - points[kmin]["r"]) - h) < 1e-9 \
                and abs(den) > 1e-14:
            fitted = {
                "r": round(float(points[kmin]["r"]
                                 + 0.5 * h * (y1 - y3) / den), 4),
                "energy_hartree": float(y2 - 0.125 * (y1 - y3) ** 2 / den),
                "method": "parabola through the three lowest points",
            }

    # the point nearest the input bond length becomes the reported structure
    kbase = int(np.argmin([abs(pt["r"] - r0) for pt in points]))
    base = results[kbase]
    base["job_type"] = "scan"
    # A well whose lowest point is the first or last one sampled has its
    # bottom outside the window, and whatever number is reported for it is an
    # endpoint, not a minimum.  Say so rather than letting it pass for one.
    if kmin == 0 or kmin == len(points) - 1:
        base["warnings"] = list(base.get("warnings") or []) + [
            f"The lowest point of this scan is at its {('inner' if kmin == 0 else 'outer')} "
            f"end (r = {distances[kmin]:.3f} A), so the bond was scanned over the "
            f"wrong range and no minimum was actually bracketed."]
    base["scan"] = {
        "atoms": [i + 1, j + 1],
        "labels": [f"{mol.atoms[i].symbol}{i + 1}", f"{mol.atoms[j].symbol}{j + 1}"],
        "element_pair": f"{mol.atoms[i].symbol}-{mol.atoms[j].symbol}",
        "r0": round(r0, 4),
        "steps": int(steps),
        "points": points,
        "minimum": {
            "r": round(float(distances[kmin]), 4),
            "energy_hartree": float(emin),
        },
        "minimum_fitted": fitted,
        # How far the sampled minimum can sit from the true one.  For a
        # unimodal well the grid minimum is one of the two points bracketing
        # the bottom, so one spacing is a safe bound and the payload can say
        # it instead of leaving the reader to guess.
        "grid_step": round(grid_step, 5),
        "minimum_index": kmin,
        "rigid": True,
    }
    base["scan_seconds"] = round(time.time() - t_scan, 3)
    base["log"] = list(base.get("log") or [])

    if progress:
        progress(JobProgress("done", 100, "Bond scan complete"))
    return base
