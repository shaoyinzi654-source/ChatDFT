"""
Molecular quality filters for the design agent.

Three independent, literature-standard checks that decide whether a proposed
structure is worth a DFT calculation at all:

* **Synthetic accessibility** (Ertl & Schuffenhauer, *J. Cheminform.* **1**,
  8 (2009)) -- 1 (easy) to 10 (very hard), from fragment contributions plus a
  ring-complexity and stereocentre penalty.  A molecule nobody can make is
  not a result.
* **QED** (Bickerton *et al.*, *Nat. Chem.* **4**, 90 (2012)) -- the
  geometric mean of eight desirability functions (MW, logP, HBD, HBA, PSA,
  rotatable bonds, aromatic rings, alerts).
* **Structural alerts** -- the PAINS (Baell & Holloway, *J. Med. Chem.*
  **53**, 2719 (2010)), BRENK (Brenk *et al.*, *ChemMedChem* **3**, 435
  (2008)) and NIH filter sets as shipped with RDKit.

The alerts are deliberately **not** rejections by default.  A real drug can
legitimately trip one, and silently deleting every flagged molecule would
narrow the design space for no stated reason.  They are reported instead, so
the report can never present an assay-interfering compound as if it were
clean.  Rejection is available, but it has to be asked for.

Every function returns ``None`` rather than a number when its data is
unavailable, so a missing optional dependency can never be mistaken for a
good score.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ALERT_SETS",
    "alerts",
    "alert_lines",
    "available",
    "qed_score",
    "sa_score",
]

# the three RDKit catalogues, in the order they are reported
ALERT_SETS: Tuple[str, ...] = ("PAINS", "BRENK", "NIH")

_SASCORER: Optional[object] = None
_SASCORER_TRIED = False
_CATALOGS: Dict[str, object] = {}
_CATALOGS_TRIED = False


# ----------------------------------------------------------------------
# synthetic accessibility
# ----------------------------------------------------------------------
def _sascorer():
    """The RDKit ``Contrib/SA_Score`` module, loaded by file path.

    It is a contribution, not an installed package, so it is loaded from its
    own directory rather than imported -- that also keeps the directory off
    ``sys.path``, where it would shadow nothing useful but is not ours to
    modify.
    """
    global _SASCORER, _SASCORER_TRIED
    if _SASCORER is not None or _SASCORER_TRIED:
        return _SASCORER
    _SASCORER_TRIED = True
    try:
        from rdkit.Chem import RDConfig

        path = os.path.join(RDConfig.RDContribDir, "SA_Score", "sascorer.py")
        if not os.path.isfile(path):
            return None
        spec = importlib.util.spec_from_file_location("chatdft_sascorer", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # the fragment-score table loads lazily on the first call; force it
        # now so a broken data file is reported as "unavailable" rather than
        # as an exception in the middle of a design run
        module.readFragmentScores()
        _SASCORER = module
    except Exception:
        _SASCORER = None
    return _SASCORER


def sa_score(mol) -> Optional[float]:
    """Ertl synthetic accessibility, 1 (easy) .. 10 (very hard)."""
    if mol is None:
        return None
    module = _sascorer()
    if module is None:
        return None
    try:
        value = float(module.calculateScore(mol))
    except Exception:
        return None
    return value if value == value else None


def qed_score(mol) -> Optional[float]:
    """QED drug-likeness, 0 (poor) .. 1 (ideal)."""
    if mol is None:
        return None
    try:
        from rdkit.Chem import QED

        value = float(QED.qed(mol))
    except Exception:
        return None
    return value if value == value else None


# ----------------------------------------------------------------------
# structural alerts
# ----------------------------------------------------------------------
def _catalogs() -> Dict[str, object]:
    global _CATALOGS, _CATALOGS_TRIED
    if _CATALOGS or _CATALOGS_TRIED:
        return _CATALOGS
    _CATALOGS_TRIED = True
    try:
        from rdkit.Chem import FilterCatalog
        from rdkit.Chem.FilterCatalog import FilterCatalogParams

        built: Dict[str, object] = {}
        for name in ALERT_SETS:
            enum = getattr(FilterCatalogParams.FilterCatalogs, name, None)
            if enum is None:
                continue
            built[name] = FilterCatalog.FilterCatalog(
                FilterCatalogParams(enum))
        _CATALOGS = built
    except Exception:
        _CATALOGS = {}
    return _CATALOGS


def alerts(mol, sets: Sequence[str] = ALERT_SETS) -> Dict[str, List[str]]:
    """``{filter set: [matched filter names]}``, empty sets omitted."""
    out: Dict[str, List[str]] = {}
    if mol is None:
        return out
    catalogs = _catalogs()
    if not catalogs:
        return out
    for name in sets:
        catalog = catalogs.get(name)
        if catalog is None:
            continue
        try:
            matches = catalog.GetMatches(mol)
        except Exception:
            continue
        names = []
        for match in matches:
            try:
                description = match.GetDescription()
            except Exception:
                continue
            if description and description not in names:
                names.append(description)
        if names:
            out[name] = names
    return out


def alert_lines(mol, sets: Sequence[str] = ALERT_SETS) -> List[str]:
    """The same information flattened to ``"PAINS: quinone_A(370)"``."""
    return [f"{name}: {hit}"
            for name, hits in alerts(mol, sets).items() for hit in hits]


def available() -> Dict[str, bool]:
    """Which checks can actually run here, so a report can say so."""
    return {
        "sa_score": _sascorer() is not None,
        "qed": qed_score(_dummy()) is not None,
        "alerts": bool(_catalogs()),
    }


def _dummy():
    try:
        from rdkit import Chem

        return Chem.MolFromSmiles("CCO")
    except Exception:
        return None
