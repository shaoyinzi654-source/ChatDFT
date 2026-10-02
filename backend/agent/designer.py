"""
Automatic molecular design.

The point of this module is that the design space is **not** a list of
molecules somebody typed in once.  Candidates come from four independent
sources that are unioned together:

1. **The language model** reads the brief and proposes SMILES with a
   rationale.  It is the only source that can invent something genuinely new
   for a goal it has never seen.
2. **The built-in library** seeds the run with molecules the app already
   knows, so a vague brief still produces sensible chemistry.
3. **Site substitution** walks every hydrogen-bearing position of every seed
   and attaches each of ~30 substituents, one and two at a time.
4. **Fragment assembly** recombines cores, linkers and substituents, which
   is where the combinatorial explosion lives: 40 cores x 30 substituents on
   one site plus 30 x 30 on two sites is already tens of thousands of
   distinct structures, and nothing bounds how many are drawn.

Every candidate is then validated by RDKit (single fragment, sane valence,
allowed elements), filtered against the hard constraints from the brief, and
ranked by a **desirability function** rather than a hand-made weighted sum.
The top few are finally handed to PySCF and ranked again on *measured*
HOMO / LUMO / gap / dipole, so the numbers in the report are real.

Two things this module deliberately does not do:

* It never invents a DFT number.  Electronic properties are absent until
  the calculation has actually run, and the report says which is which.
* It never fails just because the model is unreachable.  A rate-limited or
  unconfigured endpoint downgrades the run to rule-based mode and says so;
  sources 2-4 do not need the network at all.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Overridable so a test or a gate can point the design history somewhere
# disposable: the history is pruned by unlinking old files, and a gate that
# runs a few dozen designs should not be deleting anything the user owns.
DESIGN_DIR = os.environ.get("CHATDFT_DESIGN_DIR") or os.path.join(
    ROOT, "data", "designs")


# ======================================================================
#  RDKit access
# ======================================================================
_RDKIT = None
_QUALITY = None


def _rk():
    """Import RDKit once, with its chatty logger already switched off."""
    global _RDKIT
    if _RDKIT is None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

        RDLogger.DisableLog("rdApp.*")
        _RDKIT = {
            "Chem": Chem,
            "Crippen": Crippen,
            "Descriptors": Descriptors,
            "rdMolDescriptors": rdMolDescriptors,
        }
    return _RDKIT


def chem():
    return _rk()["Chem"]


def rdkit_available() -> bool:
    try:
        _rk()
        return True
    except Exception:
        return False


def _quality():
    """The quality module (SA score, QED, structural alerts), imported late.

    It is a pure add-on: if its optional data files are missing every one of
    its functions returns ``None`` and the design run carries on with the
    plain descriptors.
    """
    global _QUALITY
    if _QUALITY is None:
        from . import quality as _mod

        _QUALITY = _mod
    return _QUALITY


# ======================================================================
#  properties
# ======================================================================
@dataclass(frozen=True)
class Property:
    """One number a design can be judged on.

    ``source`` is either ``rdkit`` (a descriptor, available for every
    candidate, cheap) or ``dft`` (only available for the handful that were
    actually calculated).  Mixing the two in one score is fine as long as
    the report says which is which, which is why it is recorded here.
    """

    key: str
    label: str
    unit: str
    source: str
    low: float                      # desirability 0 / 1 boundary
    high: float                     # the other boundary
    decimals: int = 2
    better: str = "max"             # default aim if the user just names it
    path: Optional[str] = None      # dotted path into a DFT result dict
    getter: Optional[Callable] = None


def _d(mol, name):
    return float(getattr(_rk()["Descriptors"], name)(mol))


def _rd(mol, name):
    return float(getattr(_rk()["rdMolDescriptors"], name)(mol))


def _dft_value(result: dict, path: str) -> Optional[float]:
    node: Any = result
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    try:
        value = float(node)
    except (TypeError, ValueError):
        return None
    return value if value == value else None      # drop NaN


PROPERTIES: Dict[str, Property] = {}


def _register(*props: Property) -> None:
    for p in props:
        PROPERTIES[p.key] = p


_register(
    Property("molwt", "Molecular weight", "g/mol", "rdkit", 150.0, 500.0, 1,
             getter=lambda m: _d(m, "MolWt")),
    Property("logp", "logP (lipophilicity)", "", "rdkit", -1.0, 4.0,
             getter=lambda m: float(_rk()["Crippen"].MolLogP(m))),
    Property("tpsa", "Topological polar surface area", "A^2", "rdkit", 20.0, 120.0,
             getter=lambda m: _d(m, "TPSA")),
    Property("hbd", "Hydrogen-bond donors", "", "rdkit", 0.0, 4.0, 0,
             getter=lambda m: _rd(m, "CalcNumHBD")),
    Property("hba", "Hydrogen-bond acceptors", "", "rdkit", 1.0, 8.0, 0,
             getter=lambda m: _rd(m, "CalcNumHBA")),
    Property("rotb", "Rotatable bonds", "", "rdkit", 0.0, 8.0, 0,
             getter=lambda m: _rd(m, "CalcNumRotatableBonds")),
    Property("rings", "Rings", "", "rdkit", 1.0, 3.0, 0,
             getter=lambda m: _rd(m, "CalcNumRings")),
    Property("arom_rings", "Aromatic rings", "", "rdkit", 0.0, 2.0, 0,
             getter=lambda m: _rd(m, "CalcNumAromaticRings")),
    Property("fsp3", "Fraction C(sp3)", "", "rdkit", 0.2, 0.8,
             getter=lambda m: _rd(m, "CalcFractionCSP3")),
    Property("heavy", "Heavy atoms", "", "rdkit", 6.0, 30.0, 0,
             getter=lambda m: float(m.GetNumHeavyAtoms())),
    Property("mr", "Molar refractivity", "", "rdkit", 30.0, 100.0,
             getter=lambda m: float(_rk()["Crippen"].MolMR(m))),
    # ---- "is this molecule worth making at all" ------------------------
    # Ertl & Schuffenhauer, J. Cheminform. 1, 8 (2009).  1 = easy, 10 =
    # very hard, so the useful direction is minimisation.
    Property("sa", "Synthetic accessibility", "", "rdkit", 1.0, 6.0,
             better="min", getter=lambda m: _quality().sa_score(m)),
    # Bickerton et al., Nat. Chem. 4, 90 (2012).
    Property("qed", "QED drug-likeness", "", "rdkit", 0.2, 0.9, 3,
             better="max", getter=lambda m: _quality().qed_score(m)),
    # ---- measured, only present after a calculation -------------------
    Property("gap", "HOMO-LUMO gap", "eV", "dft", 2.0, 8.0, better="max",
             path="gap_ev"),
    Property("homo", "HOMO energy", "eV", "dft", -9.0, -4.0, better="max",
             path="homo_ev"),
    Property("lumo", "LUMO energy", "eV", "dft", -3.0, 2.0, better="min",
             path="lumo_ev"),
    Property("dipole", "Dipole moment", "D", "dft", 0.0, 6.0, better="max",
             path="dipole.magnitude"),
    Property("energy", "Total energy", "Ha", "dft", -2000.0, 0.0, 4,
             better="min", path="energy_hartree"),
)


def property_value(mol, key: str, dft: Optional[dict] = None) -> Optional[float]:
    """Value of one property, or None when it has not been measured."""
    prop = PROPERTIES.get(key)
    if prop is None:
        return None
    if prop.source == "dft":
        return _dft_value(dft, prop.path) if dft else None
    try:
        value = float(prop.getter(mol))
    except Exception:
        return None
    return value if value == value else None


# ======================================================================
#  desirability  (Derringer & Suich, J. Qual. Technol. 12, 214 (1980))
# ======================================================================
@dataclass
class Target:
    """What the user wants from one property.

    ``aim`` is ``max``, ``min`` or ``near``.  ``low``/``high`` are the
    boundaries of the desirability ramp and default to the property's own.
    """

    key: str
    aim: str = "max"
    weight: float = 1.0
    low: Optional[float] = None
    high: Optional[float] = None
    value: Optional[float] = None          # only meaningful for aim="near"

    def bounds(self) -> Tuple[float, float]:
        prop = PROPERTIES.get(self.key)
        base_low, base_high = (prop.low, prop.high) if prop else (0.0, 1.0)
        low = base_low if self.low is None else self.low
        high = base_high if self.high is None else self.high
        if high <= low:
            high = low + 1e-9
        return float(low), float(high)

    def to_dict(self) -> dict:
        return {"key": self.key, "aim": self.aim, "weight": self.weight,
                "low": self.low, "high": self.high, "value": self.value,
                "label": PROPERTIES[self.key].label if self.key in PROPERTIES
                else self.key}


def desirability(value: Optional[float], target: Target) -> Optional[float]:
    """One property mapped to 0..1.  None when the value is not measured.

    The ramp is linear on the property itself.  A value outside the window
    gives exactly 0, which is the useful part: because the overall score is
    a geometric mean, one impossible criterion zeroes the whole candidate
    instead of being averaged away by the others.
    """
    if value is None or value != value:
        return None
    low, high = target.bounds()
    if target.aim == "max":
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return (value - low) / (high - low)
    if target.aim == "min":
        if value <= low:
            return 1.0
        if value >= high:
            return 0.0
        return (high - value) / (high - low)
    # near a value: triangular, zero at both boundaries
    want = target.value
    if want is None:
        want = 0.5 * (low + high)
    if not (low <= want <= high):
        want = min(max(want, low), high)
    if value <= low or value >= high:
        return 0.0
    if value <= want:
        return (value - low) / (want - low) if want > low else 1.0
    return (high - value) / (high - want) if high > want else 1.0


def overall(candidates: List[Tuple[float, float]]) -> float:
    """Weighted geometric mean of (desirability, weight) pairs.

    Returns 0.0 as soon as any desirability is 0 -- the defining behaviour
    of desirability, and the reason one violated hard criterion cannot be
    compensated by good scores elsewhere.
    """
    if not candidates:
        return 0.0
    for d, _w in candidates:
        if d is not None and d <= 0.0:
            return 0.0
    total_w = sum(w for d, w in candidates if d is not None)
    if total_w <= 0:
        return 0.0
    acc = 0.0
    for d, w in candidates:
        if d is None or d <= 0.0:
            continue
        acc += w * math.log(d)
    return math.exp(acc / total_w)


# ======================================================================
#  the brief
# ======================================================================
@dataclass
class DesignBrief:
    """Everything the generators need, parsed out of one sentence."""

    text: str = ""
    goal: str = ""
    targets: List[Target] = field(default_factory=list)
    n_candidates: int = 12
    generations: int = 1
    dft_top: int = 3
    max_atoms: int = 30
    min_atoms: int = 3
    max_molwt: Optional[float] = None
    min_molwt: Optional[float] = None
    max_logp: Optional[float] = None
    min_logp: Optional[float] = None
    allowed_elements: Optional[List[str]] = None
    forbidden_elements: List[str] = field(default_factory=list)
    required_smarts: List[str] = field(default_factory=list)
    forbidden_smarts: List[str] = field(default_factory=list)
    seeds: List[str] = field(default_factory=list)
    # Molecules the brief *named* -- "similar to porphine", "类似阿司匹林".
    # These are not merely seeds.  A seed is one starting point among many,
    # thrown into a shuffled pool; a named molecule is what the request is
    # about, so it is mutated deliberately every round.
    seed_focus: List[str] = field(default_factory=list)
    # what those molecules are called, so the report can name them instead of
    # printing the SMILES the user never typed
    seed_names: List[str] = field(default_factory=list)
    functional: str = "b3lyp"
    basis: str = "6-31g*"
    notes: List[str] = field(default_factory=list)
    # Declared rather than attached in ``parse_brief``: the flag is read on
    # every run, and a brief built by hand (which the API and the self test
    # both allow) used to raise AttributeError from deep inside ``design``.
    has_numeric: bool = False
    # ``none`` reports structural alerts, ``pains``/``brenk``/``nih`` reject
    # them.  Reporting is the default because a real drug may legitimately
    # trip one; deleting such molecules unasked would shrink the design space
    # for no stated reason.
    alert_filter: str = "none"
    # Relax the lead candidates at the DFT level before the single point.
    # Off by default: it turns a 20 s screen into a several-minute job.
    optimize_leads: bool = False

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "targets"}
        d["targets"] = [t.to_dict() for t in self.targets]
        return d

    def summary(self) -> str:
        bits = []
        for t in self.targets:
            label = PROPERTIES[t.key].label if t.key in PROPERTIES else t.key
            word = {"max": "maximise", "min": "minimise", "near": "hit"}[t.aim]
            bits.append(f"{word} {label}")
        if self.max_molwt:
            bits.append(f"MW <= {self.max_molwt:g}")
        if self.forbidden_elements:
            bits.append("no " + "/".join(self.forbidden_elements))
        if self.required_smarts:
            bits.append(f"{len(self.required_smarts)} required substructure(s)")
        return "; ".join(bits) if bits else "no quantitative targets"


# Used only when the user gave no number at all.  A balanced, drug-like
# window rather than "more is better": maximising logP blindly ranks hexane
# above every real solvent, which is not what anybody means by "design me a
# molecule".
DEFAULT_TARGETS = [
    Target("logp", "near", 1.0, value=2.0, low=-2.0, high=6.0),
    Target("tpsa", "near", 1.0, value=60.0, low=0.0, high=140.0),
]


# ----------------------------------------------------------------------
#  rule-based parsing (no network, no model)
# ----------------------------------------------------------------------
_KEYWORD_TARGETS: List[Tuple[str, str, str]] = [
    # regex fragment, property key, default aim
    (r"分子量|摩尔质量|molecular\s*weight|\bmw\b|mol\s*wt", "molwt", "near"),
    (r"log\s?p|脂水分配|疏水参数|亲脂", "logp", "max"),
    (r"tpsa|极性表面积|拓扑极性", "tpsa", "max"),
    (r"氢键供体|给体|hbd|h-?bond\s*donor", "hbd", "min"),
    (r"氢键受体|受体|hba|h-?bond\s*acceptor", "hba", "max"),
    (r"能隙|带隙|禁带|\bgap\b|homo-?lumo", "gap", "max"),
    (r"\bhomo\b|最高占据", "homo", "min"),
    (r"\blumo\b|最低未占", "lumo", "min"),
    (r"偶极|dipole", "dipole", "max"),
    (r"可旋转键|rotatable", "rotb", "min"),
    (r"芳香环|苯环数|aromatic\s*ring", "arom_rings", "max"),
    (r"\bfsp3\b|sp3\s*比例|sp3\s*分数", "fsp3", "max"),
    (r"合成难度|合成可及|可合成|易合成|synthetic\s*access|sascore|"
     r"\bsa\s*score\b", "sa", "min"),
    (r"类药|成药|药性|drug-?like|\bqed\b", "qed", "max"),
]

# Hard filters that are about *quality* rather than a number.
_ALERT_RULES: List[Tuple[str, str]] = [
    (r"pains|泛筛选干扰|假阳性化合物", "PAINS"),
    (r"brenk|反应性基团|reactive\s*group", "BRENK"),
    (r"避免.*警报|no\s*structural\s*alerts|排除.*警示", "NIH"),
]

_OPTIMIZE_WORDS = re.compile(
    r"优化几何|几何优化|结构优化|弛豫|relax|optimis|optimiz|"
    r"geometry\s*optimi", re.I)

_MAX_WORDS = (r"(?:大|高|宽|max(?:imis|imiz)e?|highest|largest|widest|wide|"
              r"large|broad|提升|提高|尽量大|越大越好|越高越好)")
_MIN_WORDS = (r"(?:小|低|窄|min(?:imis|imiz)e?|lowest|smallest|narrow|tight|"
              r"降低|减少|尽量小|越小越好|越低越好)")

# orbital energies and logP are routinely negative, so the sign is part of
# the number -- "HOMO above -6 eV" is meaningless without it
_NUM = r"(-?[0-9]+(?:\.[0-9]+)?)"

_RANGE = re.compile(_NUM + r"\s*(?:-|~|到|至|—)\s*" + _NUM, re.I)
_RANGE_BETWEEN = re.compile(
    r"between\s*" + _NUM + r"\s*(?:and|to|和|到)\s*" + _NUM, re.I)
_NEAR = re.compile(
    r"(?:约|大约|接近|等于|=|around|about|near|target|close\s*to)\s*" + _NUM, re.I)
_LTE = re.compile(
    r"(?:<=|≤|小于|低于|不超过|最多|below|under|max(?:imum)?|at\s*most|"
    r"less\s*than|no\s*more\s*than)\s*" + _NUM, re.I)
_GTE = re.compile(
    r"(?:>=|≥|大于|高于|至少|最少|above|over|min(?:imum)?|at\s*least|"
    r"more\s*than|no\s*less\s*than)\s*" + _NUM, re.I)

# A 40-character window reaches into the next clause unless it is cut short:
# "HOMO above -6 eV and dipole below 2 D" read the *dipole* bound as the
# HOMO bound.  "and" ends a clause for this purpose; ranges are matched on
# the uncut window because "between 1 and 3" needs its "and".
#
# The full-width comma `，` and the enumeration comma `、` belong here too.
# They were missing, and because the *preceding* 16 characters were also
# searched for a relation word without being cut at a clause boundary,
# "logP 小于 2，能隙大于 4 eV" read the gap as "minimise, <= 2" -- logP's
# bound leaked forwards and the gap's own "4" was never seen.  Any Chinese
# brief written with normal punctuation was parsed this way.
_CLAUSE_BREAK = re.compile(
    r"[,;，、。；.]|\band\b|\bwith\b|\bbut\b|和|且|以及", re.I)
_HARD_BREAK = re.compile(r"[,;，、。；]")

_REQUIRED_SUBSTRUCTURES: List[Tuple[str, str]] = [
    (r"羧基|carbox(?:ylic|y)", "C(=O)[OX2H1]"),
    (r"羟基|hydroxyl|\boh\b", "[OX2H]"),
    (r"氨基|amine|amino", "[NX3;H2,H1,H0;!$(N=*)]"),
    (r"酰胺|amide", "C(=O)N"),
    (r"磺酸|sulfonic", "S(=O)(=O)[OX2H1]"),
    (r"磺酰胺|sulfonamide", "S(=O)(=O)N"),
    (r"腈|氰基|cyano|nitrile", "C#N"),
    (r"硝基|nitro", "[N+](=O)[O-]"),
    (r"苯环|芳香环|aromatic|phenyl", "c1ccccc1"),
    (r"杂环|heterocycle|heteroaromatic", "[a;!#6]"),
    (r"吡啶|pyridine", "n1ccccc1"),
    (r"咪唑|imidazole", "c1ncc[nH]1"),
    (r"胍|guanidine", "NC(=N)N"),
    (r"醚键|ether", "[OD2]([#6])[#6]"),
    (r"酯|ester", "C(=O)[OX2][#6]"),
    (r"酮|ketone", "[#6]C(=O)[#6]"),
    (r"醛|aldehyde", "[CX3H1](=O)[#6]"),
    (r"硫醇|thiol", "[SX2H]"),
    (r"脲|urea", "NC(=O)N"),
]

_FORBIDDEN_SUBSTRUCTURES: List[Tuple[str, str]] = [
    (r"过氧|peroxide", "[OX2][OX2]"),
    (r"酰氯|acid\s*chloride", "C(=O)[Cl]"),
    (r"叠氮|azide", "[N-]=[N+]=N"),
    (r"环氧|epoxide", "C1OC1"),
    (r"亚硝基|nitroso", "N=O"),
]

_ELEMENT_RULES: List[Tuple[str, List[str]]] = [
    (r"无卤素|不含卤|不含卤素|halogen[- ]?free|no\s*halogen", ["F", "Cl", "Br", "I"]),
    (r"无氟|不含氟|fluorine[- ]?free", ["F"]),
    (r"无氯|不含氯|chlorine[- ]?free", ["Cl"]),
    (r"无硫|不含硫|sulfur[- ]?free|sulphur[- ]?free", ["S"]),
    (r"无金属|不含金属|metal[- ]?free", ["Li", "Na", "K", "Mg", "Ca", "Fe",
                                          "Cu", "Zn", "Pd", "Pt", "Ru"]),
]

_ELEMENT_REQUIRED: List[Tuple[str, str]] = [
    (r"含氮|有氮|nitrogen[- ]?containing|含\s*N", "N"),
    (r"含氧|有氧|oxygen[- ]?containing|含\s*O", "O"),
    (r"含硫|有硫|sulfur[- ]?containing", "S"),
    (r"含氟|有氟|fluorinated", "F"),
    (r"含磷|有磷", "P"),
]


def _parse_number_targets(text: str) -> List[Target]:
    """Pick up "<keyword> <number>" style requests, in either language."""
    out: List[Target] = []
    low = text.lower()
    for pattern, key, default_aim in _KEYWORD_TARGETS:
        for match in re.finditer(pattern, low, re.I):
            # Look only in a short window around the keyword.  Reading the
            # whole sentence is what made "a *small* amine with a wide gap"
            # come out as "minimise the gap": the qualifier belongs to a
            # different noun entirely.
            context = low[max(0, match.start() - 24): match.end() + 40]
            window = low[match.end(): match.end() + 40]
            clause = _CLAUSE_BREAK.split(window)[0]
            aim = default_aim
            if re.search(_MAX_WORDS, context):
                aim = "max"
            elif re.search(_MIN_WORDS, context):
                aim = "min"

            # The relation word decides the direction, not the property's
            # usual preference: "HOMO above -6 eV" means maximise even
            # though a low HOMO is more often what is wanted.
            # ranges are read from the uncut window ("between 1 and 3"),
            # bounds only from the first clause (see _CLAUSE_BREAK)
            # ranges survive commas? no -- but they must survive "and", so
            # the cut is only on hard punctuation.  Without this the gap in
            # "gap above 5 eV, logP between 1 and 3" picks up logP's range.
            rng = (_RANGE.search(_HARD_BREAK.split(window)[0])
                   or _RANGE_BETWEEN.search(_HARD_BREAK.split(window)[0]))
            near = _NEAR.search(clause)
            # "at most 3 rotatable bonds" puts the relation *before* the
            # property name, so the short span ahead of it is searched too --
            # but only back to the previous clause boundary.  Reading past it
            # is how "logP below 2, gap above 4 eV" made the gap inherit
            # logP's bound and ignore its own.
            ahead_window = low[max(0, match.start() - 16): match.start()]
            ahead = _HARD_BREAK.split(ahead_window)[-1]
            lte = _LTE.search(clause) or _LTE.search(ahead)
            gte = _GTE.search(clause) or _GTE.search(ahead)
            if rng and not near:
                lo, hi = float(rng.group(1)), float(rng.group(2))
                target: Optional[Target] = Target(key, "near", 1.0,
                                                  value=0.5 * (lo + hi),
                                                  low=lo, high=hi)
            elif near:
                value = float(near.group(1))
                target = Target(key, "near", 1.0, value=value,
                                low=value * 0.85, high=value * 1.15)
            elif lte:
                # an upper bound is a request for low values
                target = Target(key, "min", 1.0, high=float(lte.group(1)))
            elif gte:
                target = Target(key, "max", 1.0, low=float(gte.group(1)))
            else:
                m = re.search(_NUM, clause)
                if m is None:
                    # The property was named but no number given -- "a wide
                    # HOMO-LUMO gap" is still a target, judged on the
                    # property's own window rather than being dropped.
                    target = Target(key, aim, 1.0)
                elif aim == "near":
                    value = float(m.group(1))
                    target = Target(key, "near", 1.0, value=value,
                                    low=value * 0.85, high=value * 1.15)
                else:
                    target = Target(key, aim, 1.0, low=float(m.group(1)))
            if target is not None and all(t.key != key for t in out):
                out.append(target)
            break
    return out


def _dedupe_targets(targets: Sequence[Target]) -> List[Target]:
    """One target per property, and no orphaned HOMO/LUMO alongside a gap.

    "a wide HOMO-LUMO gap" matches three patterns -- the gap, and the words
    HOMO and LUMO on their own.  Left alone that ranks the same molecule
    three times on overlapping numbers, which double-counts the electronics
    and buries whatever else was asked for.
    """
    out: List[Target] = []
    for target in targets:
        if any(t.key == target.key for t in out):
            continue
        out.append(target)
    if any(t.key == "gap" for t in out):
        out = [t for t in out if t.key not in ("homo", "lumo")]
    return out


def _drop_redundant_targets(targets: Sequence[Target],
                            brief: "DesignBrief") -> List[Target]:
    """Remove targets that a hard constraint already guarantees.

    "molecular weight below 250" is recorded as ``max_molwt`` and enforced
    as a filter.  Scoring on it as well is worse than useless: the property's
    window starts at 150, so every small molecule scores a perfect 1.0 and
    the ranking collapses into a tie.  The bound still bites -- it just
    bites earlier, as a filter.

    logP is deliberately *not* dropped.  A lipophilicity limit is a real
    design axis rather than a mere eligibility test, and unlike a mass
    window it still separates the survivors.
    """
    if brief.max_molwt is None and brief.min_molwt is None:
        return list(targets)
    return [t for t in targets
            if not (t.key == "molwt" and t.aim in ("min", "max"))]


def _parse_goal_words(text: str) -> List[Target]:
    """Goals that are really property statements in disguise."""
    low = text.lower()
    out: List[Target] = []
    pairs = [
        (r"水溶|亲水|hydrophilic|water[- ]soluble|溶于?水", "logp", "min"),
        (r"脂溶|疏水|亲脂|hydrophobic|lipophilic", "logp", "max"),
        (r"透膜|透脑|血脑|bbb|permeab|membrane", "tpsa", "min"),
        (r"极性大|高极性|polar", "tpsa", "max"),
        (r"导电|共轭|小能隙|narrow\s*gap|small\s*gap", "gap", "min"),
        (r"稳定|大能隙|wide\s*gap|large\s*gap|绝缘", "gap", "max"),
        (r"给电子|易氧化|electron[- ]?donat", "homo", "max"),
        (r"吸电子|易还原|electron[- ]?accept|electron[- ]?withdraw", "lumo", "min"),
        (r"刚性|rigid|刚性大", "rotb", "min"),
        (r"柔性|flexible", "rotb", "max"),
    ]
    for pattern, key, aim in pairs:
        if re.search(pattern, low):
            if all(t.key != key for t in out):
                out.append(Target(key, aim, 1.0))
    return out


def _match_substructures(text: str, table: Sequence[Tuple[str, str]]
                         ) -> List[str]:
    """SMARTS from a keyword table, longest match first, overlaps dropped.

    The keywords overlap: "磺酰胺" (sulfonamide) contains "酰胺" (amide) and
    "硫醇" contains nothing else, but "羧基" sits inside "羧基酸"-like words.
    Scanning the table entry by entry and taking every hit made "design a
    sulfonamide" also demand an amide, which then filled the pool with
    sulfonyl amides carrying an extra C(=O)N that nobody asked for.
    Taking the longest match at each position and discarding the shorter
    ones it covers is what a reader does.
    """
    low = text.lower()
    spans: List[Tuple[int, int, str]] = []
    for pattern, smarts in table:
        for match in re.finditer(pattern, low, re.I):
            if match.end() > match.start():
                spans.append((match.start(), match.end(), smarts))
    # longest first; ties go to the earlier position so the order is stable
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    taken: List[Tuple[int, int]] = []
    out: List[str] = []
    for start, end, smarts in spans:
        if any(start < t_end and end > t_start for t_start, t_end in taken):
            continue
        taken.append((start, end))
        if smarts not in out:
            out.append(smarts)
    return out


def _parse_seeds(text: str) -> List[str]:
    """SMILES-looking tokens, plus library names the user typed."""
    seeds: List[str] = []
    for token in re.findall(r"[A-Za-z0-9@+\-\[\]()=#/\\\.]{4,}", text):
        if not re.search(r"[A-Za-z]", token):
            continue
        if re.search(r"[=#\[\]\(\)]", token) or re.search(r"[cnospCNOSPF]",
                                                          token):
            mol = safe_mol(token)
            if mol is not None and mol.GetNumHeavyAtoms() >= 2:
                seeds.append(canonical(token))
    return list(dict.fromkeys(seeds))[:8]


# ----------------------------------------------------------------------
#  "a molecule like X": the named starting point
# ----------------------------------------------------------------------
# A brief can name the molecule it wants analogues of, and the two ways of
# saying so put the name on opposite sides of the cue: "similar to porphine"
# and "porphine analogues".  Both are read, because they are the same
# request.  "以X为骨架" is the third form, and it belongs to the second
# pattern -- the name still precedes the cue.
#
# Only a *cue* triggers this.  Naming a molecule without one is ambiguous:
# "containing benzene" is a substructure demand, not "start from benzene",
# and reading every library name out of the sentence turned "design 4 amines"
# into a hunt for histamine through a substring match.
# `like` is preceded by a hyphen in ordinary chemical prose -- "drug-like",
# "lead-like", "water-like" -- none of which names anything.  A lookbehind
# keeps those out; without it "design a drug-like amine" read "amine" as a
# compound name and, through a fuzzy library lookup, picked histamine.
_CUE_THEN_NAME = re.compile(
    r"(?:similar\s+to|close\s+to|(?<!-)like\b|based\s+on|derived\s+from|"
    r"inspired\s+by|analogues?\s+of|analogs?\s+of|derivatives?\s+of|"
    r"类似|相似|基于|参照|参考)", re.I)
_NAME_THEN_CUE = re.compile(
    r"(?:analogues?|analogs?|derivatives?|bioisosteres?|"
    r"类似物|衍生物|为骨架|为母体|为核心|为基础|为模板|为起点)", re.I)


def _resolve_library_words(window: str) -> Tuple[str, str]:
    """First library molecule named in ``window``: ``(smiles, display name)``.

    Chinese is not word-separated, so the aliases are searched as substrings
    of the window -- **longest first**, because 苯酚 (phenol) contains 苯
    (benzene) and the longer name is the one that was written.  A one-character
    alias only counts if it is the whole window: 水 means water in "类似水",
    but it is just a syllable in "溶于水".

    Latin names are matched **exactly**, never fuzzily.  A substring lookup
    found *something* for almost any chemical word -- "amine" resolved to
    histamine, "acid" to acetic acid -- which turned ordinary prose into a
    named starting point.  Every compound a user actually names is a library
    key, a display name or a formula, and all three are covered without it.
    """
    from ..engine import molecule as molmod

    trimmed = window.strip()
    for alias in sorted(molmod._ALIAS_TO_KEY, key=len, reverse=True):
        if alias not in window:
            continue
        if len(alias) == 1 and alias != trimmed:
            continue
        entry, _key = molmod._find_library_entry(alias, fuzzy=True)
        if entry and entry.get("smiles"):
            return entry["smiles"], (molmod._entry_name(entry) or alias)

    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", window)
    for word in sorted(set(words), key=len, reverse=True):
        entry, _key = molmod._find_library_entry(word, fuzzy=False)
        if entry and entry.get("smiles"):
            return entry["smiles"], (molmod._entry_name(entry) or word)
    return "", ""


def _parse_named_seeds(text: str) -> Tuple[List[str], List[str]]:
    """``(SMILES, display name)`` for every library molecule the brief names."""
    found: List[Tuple[str, str]] = []
    seen: set = set()
    for cue in _CUE_THEN_NAME.finditer(text):
        window = _HARD_BREAK.split(text[cue.end(): cue.end() + 48])[0]
        smi, name = _resolve_library_words(window)
        if smi and smi not in seen:
            seen.add(smi)
            found.append((smi, name))
    for cue in _NAME_THEN_CUE.finditer(text):
        back = _HARD_BREAK.split(text[max(0, cue.start() - 48): cue.start()])[-1]
        smi, name = _resolve_library_words(back)
        if smi and smi not in seen:
            seen.add(smi)
            found.append((smi, name))
    return [s for s, _n in found], [n for _s, n in found]


def parse_brief(text: str, client=None) -> DesignBrief:
    """Turn one sentence into a brief.

    The model is asked first because it reads intent far better than a
    keyword table, but its answer is validated field by field: an unknown
    property name is dropped, a nonsensical number is dropped, and anything
    the model omits is filled in from the rules.  A brief that the model got
    completely wrong is therefore still a usable brief.
    """
    brief = DesignBrief(text=text, goal=text.strip())
    low = text.lower()

    # ---- count of candidates ----------------------------------------
    m = re.search(r"([0-9]+)\s*(?:个|种|分子|个分子)", text)
    if m:
        brief.n_candidates = max(1, min(200, int(m.group(1))))
    else:
        m = re.search(r"(?:top|best)\s*([0-9]+)|([0-9]+)\s*candidates", low)
        if m:
            brief.n_candidates = max(1, min(200, int(m.group(1) or m.group(2))))
        else:
            # "design 4 amines", "give me 6 molecules", "propose 3 candidates"
            m = re.search(
                r"\b([0-9]+)\s+(?:amines?|molecules?|candidates?|compounds?"
                r"|structures?|drugs?)\b", low)
            if m:
                brief.n_candidates = max(1, min(200, int(m.group(1))))

    m = re.search(r"([0-9]+)\s*(?:代|轮|generation)", text)
    if m:
        brief.generations = max(1, min(5, int(m.group(1))))

    # ---- hard constraints --------------------------------------------
    m = re.search(r"(?:原子数|atoms?)\s*(?:<=|≤|小于|不超过|below|under)\s*" + _NUM, low)
    if m:
        brief.max_atoms = int(float(m.group(1)))
    else:
        m = re.search(_NUM + r"\s*(?:个原子|atoms)", low)
        if m:
            brief.max_atoms = int(float(m.group(1)))

    m = re.search(r"(?:分子量|mw|mol\s*wt)\s*(?:<=|≤|小于|不超过|below|under)\s*" + _NUM, low)
    if m:
        brief.max_molwt = float(m.group(1))
    m = re.search(r"(?:分子量|mw|mol\s*wt)\s*(?:>=|≥|大于|至少|above)\s*" + _NUM, low)
    if m:
        brief.min_molwt = float(m.group(1))
    m = re.search(r"log\s?p\s*(?:<=|≤|小于|不超过|below|under)\s*" + _NUM, low)
    if m:
        brief.max_logp = float(m.group(1))
    m = re.search(r"log\s?p\s*(?:>=|≥|大于|至少|above)\s*" + _NUM, low)
    if m:
        brief.min_logp = float(m.group(1))

    for pattern, elements in _ELEMENT_RULES:
        if re.search(pattern, low):
            for e in elements:
                if e not in brief.forbidden_elements:
                    brief.forbidden_elements.append(e)

    for pattern, element in _ELEMENT_REQUIRED:
        if re.search(pattern, low):
            m2 = re.search(pattern, low)
            if m2:
                brief.required_smarts.append(f"[#{_atomic_num(element)}]")

    for smarts in _match_substructures(text, _REQUIRED_SUBSTRUCTURES):
        if smarts not in brief.required_smarts:
            brief.required_smarts.append(smarts)
    for smarts in _match_substructures(text, _FORBIDDEN_SUBSTRUCTURES):
        if smarts not in brief.forbidden_smarts:
            brief.forbidden_smarts.append(smarts)

    # ---- targets ------------------------------------------------------
    brief.targets = _drop_redundant_targets(
        _dedupe_targets(_parse_number_targets(text) + _parse_goal_words(text)),
        brief)
    brief.has_numeric = bool(brief.targets)
    if not brief.targets:
        brief.targets = list(DEFAULT_TARGETS)
        brief.notes.append(
            "No numeric target was given, so candidates are judged on a "
            "balanced lipophilicity / polar-surface profile, and the "
            "molecules proposed for the stated goal are listed first. "
            "Say something like \"logP below 2 and TPSA above 60\" to steer "
            "the search by numbers instead.")

    # only ask for DFT when an electronic property is actually wanted
    if not any(PROPERTIES[t.key].source == "dft" for t in brief.targets
               if t.key in PROPERTIES):
        brief.dft_top = min(brief.dft_top, 1)

    brief.seeds = _parse_seeds(text)

    # A molecule the brief names is the starting point of the whole run, and
    # it used to be dropped entirely: "design 5 molecules similar to porphine"
    # ran a generic fragment draw and never touched porphine.  The named
    # molecules go to the front of the seed list *and* into ``seed_focus``,
    # which the generator mutates deliberately every round.
    named, named_names = _parse_named_seeds(text)
    if named:
        brief.seeds = [s for s in named if s not in brief.seeds] + brief.seeds
        brief.seed_focus = list(named)
        brief.seed_names = list(named_names)
        brief.notes.append(
            "The brief names " + ", ".join(named_names) + ". Those molecules "
            "are the starting point: their analogues are generated first, "
            "every round.")

    # ---- quality filters (rule-based only: a wrong guess here either
    # deletes molecules or spends minutes, so the model does not get a say)
    for pattern, name in _ALERT_RULES:
        if re.search(pattern, low):
            brief.alert_filter = name
            break
    if _OPTIMIZE_WORDS.search(text):
        brief.optimize_leads = True
        brief.notes.append(
            "The lead candidates will be relaxed at the DFT level before "
            "their single point, so the reported orbital energies come from "
            "a real minimum rather than a force-field conformer. This costs "
            "several minutes per candidate.")

    # a pattern that is not valid SMARTS can never match anything; drop it
    # loudly rather than leave a hard constraint silently unenforced
    validate_brief(brief)

    # ---- the model gets a chance to improve all of the above ----------
    if client is not None and getattr(client, "available", False):
        try:
            _refine_brief_with_llm(brief, client)
        except Exception as exc:                     # never fatal
            brief.notes.append(f"Model brief parsing unavailable ({exc}); "
                               "using the rule-based reading.")
    return brief


def _atomic_num(symbol: str) -> int:
    from ..engine import elements

    try:
        return elements.get(symbol).z
    except Exception:
        return {"N": 7, "O": 8, "S": 16, "F": 9, "P": 15}.get(symbol, 6)


# Reasoning models on this endpoint spend their budget on a chain of
# thought *and* count it against max_tokens, so a reply that is 300 tokens
# of JSON can still be cut off at 1200.  Ask for more than any plausible
# answer needs.
_REPLY_TOKENS = 4096

_BRIEF_SYSTEM = """\
You are the planning stage of a computational chemistry design tool.
Read the user's request and restate it as a machine-readable design brief.

Reply with ONE JSON object and nothing else, using this schema:
{
  "goal": "one sentence restating what is being designed and why",
  "targets": [{"property": "<key>", "aim": "max|min|near", "weight": 1.0,
               "low": <number or null>, "high": <number or null>,
               "value": <number or null>}],
  "constraints": {"max_atoms": 30, "max_molwt": null, "min_molwt": null,
                  "max_logp": null, "min_logp": null,
                  "forbidden_elements": [], "required_smarts": [],
                  "forbidden_smarts": []},
  "seeds": ["SMILES", "..."]
}

Valid property keys: molwt, logp, tpsa, hbd, hba, rotb, rings, arom_rings,
fsp3, heavy, mr, gap, homo, lumo, dipole, energy.
The last five (gap, homo, lumo, dipole, energy) are measured by DFT and are
expensive -- only include them when the user actually asked about electronics.
"low"/"high" are the boundaries where desirability reaches 0 and 1.
"seeds" are 1-6 real SMILES that are plausible starting points.
Never invent a property key that is not in the list above.

OUTPUT ONLY THE JSON OBJECT. Do not think out loud, do not explain your
reasoning, do not use markdown fences. Anything other than the object makes
the answer unusable, because it is read by a parser, not a person.
"""


def _refine_brief_with_llm(brief: DesignBrief, client) -> None:
    from .llm import loads_loose

    payload = client.json(
        [{"role": "system", "content": _BRIEF_SYSTEM},
         {"role": "user", "content": brief.text}],
        temperature=0.2, max_tokens=_REPLY_TOKENS,
        validate=lambda p: isinstance(p, dict) and bool(p))
    if not isinstance(payload, dict):
        return

    goal = payload.get("goal")
    if isinstance(goal, str) and goal.strip():
        brief.goal = goal.strip()[:400]

    raw = payload.get("targets")
    if isinstance(raw, list) and raw:
        parsed: List[Target] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = str(item.get("property", "")).strip().lower()
            if key not in PROPERTIES:
                continue
            aim = str(item.get("aim", "max")).strip().lower()
            if aim not in ("max", "min", "near"):
                aim = PROPERTIES[key].better
            parsed.append(Target(
                key=key, aim=aim,
                weight=_as_float(item.get("weight"), 1.0, 0.05, 10.0),
                low=_as_float(item.get("low"), None),
                high=_as_float(item.get("high"), None),
                value=_as_float(item.get("value"), None)))
        if parsed:
            brief.targets = parsed
            brief.has_numeric = True

    cons = payload.get("constraints")
    if isinstance(cons, dict):
        brief.max_atoms = int(_as_float(cons.get("max_atoms"), brief.max_atoms,
                                        3, 80))
        brief.max_molwt = _as_float(cons.get("max_molwt"), brief.max_molwt)
        brief.min_molwt = _as_float(cons.get("min_molwt"), brief.min_molwt)
        brief.max_logp = _as_float(cons.get("max_logp"), brief.max_logp)
        brief.min_logp = _as_float(cons.get("min_logp"), brief.min_logp)
        for key, attr in (("forbidden_elements", "forbidden_elements"),
                          ("required_smarts", "required_smarts"),
                          ("forbidden_smarts", "forbidden_smarts")):
            value = cons.get(key)
            if isinstance(value, list):
                clean = [str(v) for v in value if isinstance(v, (str, int))]
                if clean:
                    setattr(brief, attr, clean)
        # An alert filter is cheap to honour and expensive to get wrong, so
        # the model may turn one on -- but only a filter set that exists.
        wanted = str(cons.get("alert_filter", "")).strip().upper()
        if wanted in ("PAINS", "BRENK", "NIH"):
            brief.alert_filter = wanted

    seeds = payload.get("seeds")
    if isinstance(seeds, list):
        good = []
        for smi in seeds:
            if not isinstance(smi, str):
                continue
            mol = safe_mol(smi)
            if mol is not None and mol.GetNumHeavyAtoms() >= 2:
                good.append(canonical(smi))
        if good:
            brief.seeds = (brief.seeds + good)[:10]

    # A pattern RDKit cannot parse can never match, so it would leave the
    # user's constraint silently unenforced.  Drop those and say so; keep
    # every valid pattern, however exotic.
    #
    # This used to also test each pattern against the built-in molecule
    # library and drop the ones nothing there matched.  That test was both
    # wrong and inert: ``library_seeds(limit=1, ...)`` returns the
    # *non-matching* entries too, so it was always truthy, and had it worked
    # it would have deleted valid requirements the fragment assembler could
    # have satisfied.  Whether a requirement is buildable is now answered by
    # counting matches over the whole search (see ``design``).
    validate_brief(brief)

    electronic = any(PROPERTIES[t.key].source == "dft" for t in brief.targets)
    brief.dft_top = brief.dft_top if electronic else min(brief.dft_top, 1)


def _as_float(value, default, low=None, high=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    if low is not None and out < low:
        return low
    if high is not None and out > high:
        return high
    return out


# ======================================================================
#  RDKit helpers
# ======================================================================
def safe_mol(smiles: str):
    """Parse and sanitise, or None.  Never raises."""
    try:
        Chem = chem()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def canonical(smiles: str) -> str:
    mol = safe_mol(smiles)
    if mol is None:
        return smiles
    try:
        return chem().MolToSmiles(mol)
    except Exception:
        return smiles


def _attach(mol, atom_idx: int, sub_smiles: str,
            drop_idx: Optional[int] = None):
    """Bond ``sub_smiles`` onto ``mol`` at ``atom_idx``.

    Convention: a substituent is written so that **its first atom is the
    attachment point**.  ``C(C)=O`` is therefore acetyl (bond through the
    carbonyl carbon) while ``CC(=O)`` is an aldehyde fragment, which is why
    the fragment table below is written the way it is.

    RDKit's ``RemoveAtom`` does not rewire the neighbours of the atom it
    deletes, so the new bond is made to the attachment atom directly rather
    than to a dummy that is then removed.

    ``drop_idx`` removes an atom *before* sanitising.  It has to happen in
    the same edit: grafting onto a dummy site leaves the substitution centre
    with one bond too many until the dummy is gone, and sanitising in
    between rejects the molecule as over-valent.
    """
    Chem = chem()
    sub = Chem.MolFromSmiles(sub_smiles)
    if sub is None or mol.GetNumAtoms() == 0:
        return None
    combined = Chem.CombineMols(mol, sub)
    rw = Chem.RWMol(combined)
    rw.AddBond(atom_idx, mol.GetNumAtoms(), Chem.BondType.SINGLE)
    if drop_idx is not None:
        rw.RemoveAtom(drop_idx)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def _attachment_sites(mol, smarts: str = "[*;!H0]"):
    """Atoms carrying at least one hydrogen, i.e. places to substitute."""
    Chem = chem()
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return []
    # each match is a tuple of atom indices; the pattern is one atom wide
    return [match[0] for match in mol.GetSubstructMatches(pattern) if match]


def _graft(core_smiles: str, substituents: Sequence[str]):
    """Attach substituents to the dummy sites of a core, lowest map first."""
    Chem = chem()
    core = Chem.MolFromSmiles(core_smiles)
    if core is None:
        return None
    for smi in substituents:
        dummies = [a for a in core.GetAtoms() if a.GetAtomicNum() == 0]
        if not dummies:
            break
        dummies.sort(key=lambda a: a.GetAtomMapNum() or 0)
        dummy = dummies[0].GetIdx()
        neighbours = core.GetAtomWithIdx(dummy).GetNeighbors()
        if not neighbours:
            break
        site = neighbours[0].GetIdx()
        built = _attach(core, site, smi, drop_idx=dummy)
        if built is None:
            return None
        core = built
    return core


# ======================================================================
#  fragment library
# ======================================================================
# Cores carry dummy atoms with map numbers; substituents attach through
# their first atom.  Both conventions are exercised by check_design.
CORES: List[Tuple[str, str]] = [
    ("benzene", "[*:1]c1ccccc1"),
    ("benzene (para)", "[*:1]c1ccc([*:2])cc1"),
    ("benzene (meta)", "[*:1]c1cccc([*:2])c1"),
    ("benzene (ortho)", "[*:1]c1ccccc1[*:2]"),
    ("naphthalene", "[*:1]c1cccc2ccccc12"),
    ("biphenyl", "[*:1]c1ccc(-c2ccccc2)cc1"),
    ("pyridine (3-yl)", "[*:1]c1cccnc1"),
    ("pyridine (4-yl)", "[*:1]c1ccncc1"),
    ("pyridine (2-yl)", "[*:1]c1ccccn1"),
    ("pyrimidine", "[*:1]c1cncnc1"),
    ("pyrazine", "[*:1]c1cnccn1"),
    ("triazine", "[*:1]c1ncncn1"),
    ("imidazole", "[*:1]c1ncc[nH]1"),
    ("pyrazole", "[*:1]c1cn[nH]c1"),
    ("triazole", "[*:1]c1nnn(C)c1"),
    ("tetrazole", "[*:1]c1nnn[nH]1"),
    ("furan", "[*:1]c1ccco1"),
    ("thiophene", "[*:1]c1cccs1"),
    ("pyrrole", "[*:1]c1ccc[nH]1"),
    ("oxazole", "[*:1]c1ncoc1"),
    ("thiazole", "[*:1]c1nccs1"),
    ("indole", "[*:1]c1ccc2[nH]ccc2c1"),
    ("benzimidazole", "[*:1]c1ccc2[nH]cnc2c1"),
    ("quinoline", "[*:1]c1ccc2ncccc2c1"),
    ("piperidine", "[*:1]C1CCNCC1"),
    ("piperazine", "[*:1]N1CCN([*:2])CC1"),
    ("morpholine", "[*:1]N1CCOCC1"),
    ("pyrrolidine", "[*:1]C1CCNC1"),
    ("cyclohexane", "[*:1]C1CCCCC1"),
    ("cyclopropane", "[*:1]C1CC1"),
    ("ethylene", "[*:1]C=C[*:2]"),
    ("acetylene", "[*:1]C#C[*:2]"),
    ("methylene", "[*:1]C[*:2]"),
    ("ethylene glycol", "[*:1]CCO[*:2]"),
    ("propylene", "[*:1]CCC[*:2]"),
    ("amide linker", "[*:1]C(=O)N[*:2]"),
    ("reverse amide", "[*:1]N(C(=O)[*:2])"),
    ("ester linker", "[*:1]C(=O)O[*:2]"),
    ("sulfonamide linker", "[*:1]S(=O)(=O)N[*:2]"),
    ("urea linker", "[*:1]NC(=O)N[*:2]"),
    ("ketone linker", "[*:1]C(=O)[*:2]"),
    ("ether linker", "[*:1]COC[*:2]"),
    ("guanidine", "[*:1]NC(=N)N[*:2]"),
    ("amidine", "[*:1]C(N)=N[*:2]"),
]

SUBSTITUENTS: List[Tuple[str, str]] = [
    ("hydroxy", "O"),
    ("amino", "N"),
    ("methylamino", "NC"),
    ("dimethylamino", "N(C)C"),
    ("ethylamino", "NCC"),
    ("hydroxyethylamino", "NCCO"),
    ("aminoethylamino", "NCCN"),
    ("methyl", "C"),
    ("ethyl", "CC"),
    ("isopropyl", "C(C)C"),
    ("tert-butyl", "C(C)(C)C"),
    ("cyclopropyl", "C1CC1"),
    ("methoxy", "OC"),
    ("ethoxy", "OCC"),
    ("trifluoromethoxy", "OC(F)(F)F"),
    ("fluoro", "F"),
    ("chloro", "Cl"),
    ("bromo", "Br"),
    ("trifluoromethyl", "C(F)(F)F"),
    ("cyano", "C#N"),
    ("nitro", "[N+](=O)[O-]"),
    ("carboxylic acid", "C(=O)O"),
    ("methyl ester", "C(=O)OC"),
    ("carboxamide", "C(=O)N"),
    ("acetyl", "C(C)=O"),
    ("formyl", "C=O"),
    ("sulfonic acid", "S(=O)(=O)O"),
    ("sulfonamide", "S(=O)(=O)N"),
    ("methylsulfonyl", "S(C)(=O)=O"),
    ("phenyl", "c1ccccc1"),
    ("benzyl", "Cc1ccccc1"),
    ("pyridyl", "c1ccncc1"),
    ("morpholino", "N1CCOCC1"),
    ("piperazinyl", "N1CCNCC1"),
    ("pyrrolidino", "N1CCCC1"),
    ("guanidino", "NC(=N)N"),
    ("thiol", "S"),
    ("thiomethyl", "SC"),
    ("hydroxymethyl", "CO"),
    ("vinyl", "C=C"),
]

BIOISOSTERES: List[Tuple[str, str, str]] = [
    # name, SMARTS to find, SMILES replacing the single matched atom
    ("ether O -> NH", "[OD2]([#6])[#6]", "[NH]"),
    ("ether O -> S", "[OD2]([#6])[#6]", "S"),
    ("aromatic CH -> N", "[c;H1;r6]", "n"),
    ("aromatic CH -> CH3 blocked", "[c;H1;r6]", "c(C)"),
    ("carbonyl O -> S", "[CX3]=[OX1]", "C=S"),
]


def _core_sites(core_smiles: str) -> int:
    mol = chem().MolFromSmiles(core_smiles)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)


# ======================================================================
#  filtering
# ======================================================================
_SMARTS_CACHE: Dict[str, Optional[object]] = {}


def compile_smarts(smarts: str):
    """Compile a SMARTS once.  ``None`` means the pattern is not valid.

    Cached because the same handful of patterns is tested against every one
    of the hundreds of molecules a run proposes.
    """
    if smarts in _SMARTS_CACHE:
        return _SMARTS_CACHE[smarts]
    try:
        pattern = chem().MolFromSmarts(smarts)
    except Exception:
        pattern = None
    _SMARTS_CACHE[smarts] = pattern
    return pattern


def validate_brief(brief: DesignBrief) -> DesignBrief:
    """Drop substructure patterns that are not valid SMARTS, and say so.

    A pattern RDKit cannot parse can never match anything, so leaving it in
    the brief would make a *required* substructure silently unenforced and a
    *forbidden* one silently permitted.  Neither is acceptable, and neither
    is silently dropping the user's constraint -- the note is the point.
    Valid patterns are always kept, however exotic: whether the generator
    can actually build one is answered later, by counting matches over the
    whole search rather than by guessing from the size of the library.
    """
    for attr, kind in (("required_smarts", "required"),
                       ("forbidden_smarts", "forbidden")):
        patterns = getattr(brief, attr)
        if not patterns:
            continue
        kept: List[str] = []
        bad: List[str] = []
        for smarts in patterns:
            if compile_smarts(smarts) is None:
                bad.append(smarts)
            elif smarts not in kept:
                kept.append(smarts)
        if bad:
            setattr(brief, attr, kept)
            brief.notes.append(
                f"Ignored {len(bad)} {kind} substructure pattern(s) that are "
                "not valid SMARTS: " + ", ".join(f"`{s}`" for s in bad)
                + ". Rewrite them in SMARTS form (for example `C(=O)[OX2H1]` "
                "for a carboxylic acid) to have them enforced.")
    return brief


def violates(mol, brief: DesignBrief) -> Optional[str]:
    """Return a reason string when the molecule breaks a hard rule."""
    Chem = chem()
    try:
        frags = Chem.GetMolFrags(mol)
    except Exception:
        return "could not be fragmented"
    if len(frags) != 1:
        return f"{len(frags)} disconnected fragments"

    # An unfilled attachment point is not an atom.  A dummy left behind by a
    # partial graft used to reach the report as `[*:2]` -- a molecule that
    # does not exist, with an asterisk in its formula, queued for the SCF.
    # This is the invariant that keeps the whole class out.
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return "unfilled attachment point"

    atoms = mol.GetNumAtoms()
    if atoms < brief.min_atoms:
        return f"only {atoms} atoms"
    if atoms > brief.max_atoms:
        return f"{atoms} atoms (limit {brief.max_atoms})"

    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    for sym in brief.forbidden_elements:
        if sym in symbols:
            return f"contains {sym}"
    if brief.allowed_elements:
        extra = symbols - set(brief.allowed_elements) - {"H"}
        if extra:
            return "contains " + ",".join(sorted(extra))

    if brief.max_molwt is not None and _d(mol, "MolWt") > brief.max_molwt:
        return f"MW {_d(mol, 'MolWt'):.0f} over {brief.max_molwt:g}"
    if brief.min_molwt is not None and _d(mol, "MolWt") < brief.min_molwt:
        return f"MW {_d(mol, 'MolWt'):.0f} under {brief.min_molwt:g}"
    if brief.max_logp is not None:
        logp = float(_rk()["Crippen"].MolLogP(mol))
        if logp > brief.max_logp:
            return f"logP {logp:.2f} over {brief.max_logp:g}"
    if brief.min_logp is not None:
        logp = float(_rk()["Crippen"].MolLogP(mol))
        if logp < brief.min_logp:
            return f"logP {logp:.2f} under {brief.min_logp:g}"

    charge = Chem.GetFormalCharge(mol)
    if charge != 0:
        return f"net charge {charge:+d}"

    for smarts in brief.required_smarts:
        pattern = compile_smarts(smarts)
        if pattern is None:
            continue                       # reported by validate_brief
        if not mol.HasSubstructMatch(pattern):
            return f"missing required substructure {smarts}"
    for smarts in brief.forbidden_smarts:
        pattern = compile_smarts(smarts)
        if pattern is None:
            continue                       # reported by validate_brief
        if mol.HasSubstructMatch(pattern):
            return f"contains forbidden substructure {smarts}"

    # Structural alerts are reported by default and rejected only when the
    # brief asked for it, because a real molecule may legitimately trip one.
    if brief.alert_filter and brief.alert_filter != "none":
        wanted = [brief.alert_filter.upper()]
        try:
            hits = _quality().alerts(mol, sets=wanted)
        except Exception:
            hits = {}
        for name, names in hits.items():
            if names:
                return f"{name} structural alert ({names[0]})"
    return None


# ======================================================================
#  generation
# ======================================================================
def library_seeds(limit: int = 16,
                  prefer_smarts: Optional[Sequence[str]] = None) -> List[str]:
    """SMILES seeds taken from the app's own molecule library.

    When the brief demands a substructure, matching entries are returned
    first.  Without that the seeds are whatever happens to sit at the top of
    the file, the hard filter throws nearly all of them away, and the
    mutation stage -- which only ever starts from whatever survived -- never
    sees a molecule related to the goal.
    """
    from ..engine import molecule as molmod

    Chem = chem()
    out: List[str] = []
    try:
        library = molmod._load_library()
    except Exception:
        return out
    patterns = []
    for smarts in (prefer_smarts or []):
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            patterns.append(pattern)

    # the library is a {key: entry} mapping, not a list
    entries = library.values() if isinstance(library, dict) else (library or [])
    hits, misses = [], []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        smi = entry.get("smiles")
        if not smi:
            continue
        mol = safe_mol(smi)
        if mol is None or mol.GetNumHeavyAtoms() < 3:
            continue
        bucket = hits if (patterns and all(mol.HasSubstructMatch(p)
                                           for p in patterns)) else misses
        bucket.append(canonical(smi))
    out = list(dict.fromkeys(hits + misses))
    return out[:limit]


def mutate(mol, rng: random.Random, limit: int = 12):
    """Yield substituted / bioisosteric variants of one molecule."""
    Chem = chem()
    out = []
    base = canonical(Chem.MolToSmiles(mol)) if mol is not None else None
    if mol is None:
        return out

    sites = _attachment_sites(mol)
    rng.shuffle(sites)
    subs = [s for _n, s in SUBSTITUENTS]
    rng.shuffle(subs)

    # one substitution at a time
    for site in sites[:6]:
        for sub in subs[:8]:
            built = _attach(mol, site, sub)
            if built is not None and len(out) < limit:
                out.append(built)
            if len(out) >= limit:
                return out

    # bioisosteric single-atom swaps
    for _name, smarts, repl in BIOISOSTERES:
        pattern = Chem.MolFromSmarts(smarts)
        repl_mol = Chem.MolFromSmiles(repl)
        if pattern is None or repl_mol is None:
            continue
        try:
            products = Chem.ReplaceSubstructs(mol, pattern, repl_mol,
                                              replaceAll=False)
        except Exception:
            continue
        for p in products[:1]:
            try:
                Chem.SanitizeMol(p)
            except Exception:
                continue
            if canonical(Chem.MolToSmiles(p)) != base:
                out.append(p)
        if len(out) >= limit:
            break
    return out


def _hunt_required(rng: random.Random, patterns: Sequence, want: int,
                   budget: int = 4000) -> List:
    """Graft every substituent onto every core until ``want`` products match.

    Uniform draws hit a single demanded group about 2% of the time, so a
    brief like "design 4 sulfonamides" used to come back with one or two
    molecules that actually contained one.  This searches the same fragment
    space deliberately instead: a small, bounded, exhaustive pass that is
    only ever run when the brief demands a substructure.
    """
    Chem = chem()
    subs = [s for _n, s in SUBSTITUENTS]
    smi_by_name = dict(CORES)

    def matches(mol) -> bool:
        return all(mol.HasSubstructMatch(p) for p in patterns)

    out: List = []
    seen: set = set()
    tries = 0

    def take(built) -> bool:
        """Keep a matching product.  True once the quota is filled."""
        if built is None:
            return len(out) >= want
        try:
            key = Chem.MolToSmiles(built)
        except Exception:
            return len(out) >= want
        if key in seen:
            return len(out) >= want
        if matches(built):
            seen.add(key)
            out.append(built)
        return len(out) >= want

    # ---- one substituent -------------------------------------------------
    # One-site cores only.  Grafting a single substituent onto a two-site core
    # leaves its second attachment point dangling, and the product -- a
    # molecule that does not exist -- still matches the demanded substructure,
    # so it was kept and reported as e.g. `O=S(=O)(N[*:2])c1ccccc1`.  The
    # two-site cores get two substituents in the pass below.
    order = [c for c, smi in CORES if _core_sites(smi) == 1]
    rng.shuffle(order)
    for cname in order:
        csmi = smi_by_name[cname]
        for sub in subs:
            tries += 1
            if take(_graft(csmi, [sub])) or tries >= budget:
                break
        if len(out) >= want or tries >= budget:
            break

    # ---- two substituents, for briefs that demand two different groups ---
    if len(out) < want:
        order = [c for c, smi in CORES if _core_sites(smi) >= 2]
        rng.shuffle(order)
        for cname in order:
            for a in subs:
                for b in subs:
                    tries += 1
                    if take(_graft(smi_by_name[cname], [a, b])) \
                            or tries >= budget:
                        break
                if len(out) >= want or tries >= budget:
                    break
            if len(out) >= want or tries >= budget:
                break
    return out


def assemble(rng: random.Random, limit: int, two_site_budget: int = 40,
             require_smarts: Optional[Sequence[str]] = None) -> List:
    """Fragment recombination.  This is the unbounded source.

    ``require_smarts`` makes the draw requirement-aware: the demanded
    substructure is searched for first (see :func:`_hunt_required`) and the
    uniform random exploration only fills what is left, so the pool is both
    on-brief and diverse.
    """
    Chem = chem()
    subs = [s for _n, s in SUBSTITUENTS]
    smi_by_name = dict(CORES)
    out: List = []

    patterns = [p for p in (compile_smarts(s) for s in (require_smarts or ()))
                if p is not None]
    if patterns:
        out.extend(_hunt_required(rng, patterns, max(1, limit)))

    one_site = [c for c, smi in CORES if _core_sites(smi) == 1]
    two_site = [c for c, smi in CORES if _core_sites(smi) >= 2]

    order = list(one_site)
    rng.shuffle(order)
    for name in order:
        if len(out) >= limit:
            break
        for sub in rng.sample(subs, min(len(subs), 8)):
            built = _graft(smi_by_name[name], [sub])
            if built is not None:
                out.append(built)
            if len(out) >= limit:
                break

    # two-site cores: a random slice of the n^2 grid, so the space is
    # explored rather than enumerated into memory
    rng.shuffle(two_site)
    combos = [(a, b) for a in subs for b in subs]
    rng.shuffle(combos)
    for name in two_site:
        for a, b in combos[:two_site_budget]:
            if len(out) >= limit:
                return out
            built = _graft(smi_by_name[name], [a, b])
            if built is not None:
                out.append(built)
    return out


_PROPOSE_SYSTEM = """\
You are a molecular design engine. Given a design goal, propose candidate
molecules as valid SMILES strings.

Reply with ONE JSON object and nothing else:
{"candidates": [{"name": "common name", "smiles": "SMILES",
                 "rationale": "one short sentence"}]}

Rules:
- SMILES must be syntactically valid and chemically sensible (correct
  valences, neutral, a single connected fragment).
- Prefer molecules small enough for a DFT single point (under 30 atoms).
- Return distinct molecules; do not repeat anything in the "already used"
  list.
- The rationale must state why the molecule serves the stated goal.

OUTPUT ONLY THE JSON OBJECT. Do not think out loud, do not list your
reasoning, do not use markdown fences. The answer is read by a parser, so
any text outside the object wastes your token budget and loses the answer.
"""


def propose_from_llm(brief: DesignBrief, client, avoid: Sequence[str],
                     feedback: str = "") -> List[dict]:
    """Ask the model for candidates.  Returns [] on any failure."""
    from .llm import loads_loose

    if client is None or not getattr(client, "available", False):
        return []
    want = max(1, min(30, brief.n_candidates))
    user = (
        f"Goal: {brief.goal}\n"
        f"Ranking criteria: {brief.summary()}\n"
        f"Hard limits: at most {brief.max_atoms} atoms total"
        + (f", molecular weight <= {brief.max_molwt:g}" if brief.max_molwt else "")
        + (f", no {', '.join(brief.forbidden_elements)}"
           if brief.forbidden_elements else "")
        + ".\n"
        f"Already used (do not repeat): {', '.join(list(avoid)[-40:]) or 'none'}\n"
    )
    if feedback:
        user += f"\nWhat worked so far:\n{feedback}\n"
    user += f"\nPropose {want} candidates."

    def usable(p) -> bool:
        return (isinstance(p, dict)
                and isinstance(p.get("candidates"), list)
                and bool(p["candidates"]))

    try:
        payload = client.json(
            [{"role": "system", "content": _PROPOSE_SYSTEM},
             {"role": "user", "content": user}],
            temperature=0.8, max_tokens=_REPLY_TOKENS, validate=usable)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    raw = payload.get("candidates")
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        smi = item.get("smiles")
        if not isinstance(smi, str) or not smi.strip():
            continue
        mol = safe_mol(smi)
        if mol is None:
            continue
        out.append({
            "smiles": canonical(smi),
            "name": str(item.get("name") or "").strip()[:80],
            "rationale": str(item.get("rationale") or "")[:400],
            "origin": "model",
        })
    return out


# ======================================================================
#  scoring
# ======================================================================
def describe_molecule(mol) -> dict:
    """The RDKit-computed descriptor block shown on a candidate card."""
    values: Dict[str, Optional[float]] = {}
    for key, prop in PROPERTIES.items():
        if prop.source != "rdkit":
            continue
        value = property_value(mol, key)
        values[key] = None if value is None else round(value, prop.decimals)
    try:
        formula = _rk()["rdMolDescriptors"].CalcMolFormula(mol)
    except Exception:
        formula = ""
    # Structural alerts are reported, not enforced.  A molecule that trips
    # one is still a legitimate candidate; the report just must not present
    # it as if it were clean.
    try:
        found = _quality().alerts(mol)
    except Exception:
        found = {}
    return {
        "formula": formula,
        "atoms": mol.GetNumAtoms(),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "values": values,
        "alerts": found,
        "alert_count": sum(len(v) for v in found.values()),
    }


def score_candidate(mol, brief: DesignBrief, dft: Optional[dict] = None) -> dict:
    """Desirability breakdown for one molecule.

    The score is the weighted geometric mean of the per-criterion
    desirabilities, with **no floor**: a criterion that is missed
    contributes a desirability of exactly 0, and the geometric mean then
    collapses the whole score to 0.  That is the defining property of the
    Derringer-Suich construction, and it is what stops a candidate that
    ignores one hard requirement from being rescued by good numbers
    elsewhere.  ``blocking`` names the criteria that did it, so a zero score
    is never a mystery.
    """
    parts = []
    per_target = []
    blocking: List[str] = []
    for target in brief.targets:
        value = property_value(mol, target.key, dft)
        d = desirability(value, target)
        label = (PROPERTIES[target.key].label if target.key in PROPERTIES
                 else target.key)
        per_target.append({
            "key": target.key,
            "label": label,
            "unit": PROPERTIES[target.key].unit if target.key in PROPERTIES
            else "",
            "source": PROPERTIES[target.key].source if target.key in PROPERTIES
            else "rdkit",
            "aim": target.aim,
            "weight": target.weight,
            "value": None if value is None else round(value, 3),
            "desirability": None if d is None else round(d, 4),
        })
        if d is None:
            continue
        # the raw desirability goes to the geometric mean, not a clamped
        # copy of it -- clamping at 1e-6 here used to hide the zero and
        # report 0.0004 for a candidate that met none of its criteria
        parts.append((d, target.weight))
        if d <= 0.0:
            blocking.append(label)

    # A target whose value needs a calculation (the HOMO-LUMO gap, the
    # dipole) cannot be judged before that calculation exists.  With only
    # such targets ``parts`` is empty, ``overall`` returns 0.0, and the
    # report showed "score 0.000" -- which reads as "fails every criterion"
    # when it actually means "not judged yet".  An unscored candidate says
    # so instead, and carries the reason.
    if not parts:
        reason = ("no criterion could be evaluated without a calculation"
                  if per_target else "the brief set no criterion")
        return {"score": 0.0, "targets": per_target, "blocking": [],
                "measured": dft is not None, "scored": False,
                "unscored_reason": reason}
    return {"score": round(overall(parts), 4), "targets": per_target,
            "blocking": blocking, "measured": dft is not None,
            "scored": True}


# ======================================================================
#  DFT refinement
# ======================================================================
DFT_CACHE_DIR = os.environ.get("CHATDFT_CACHE_DIR") or os.path.join(
    ROOT, "data", "design_cache")

# the subset of an engine result the design report actually uses, so the
# cache stays small and every path returns the same shape
_DFT_FIELDS = (
    "energy_hartree", "homo_ev", "lumo_ev", "gap_ev", "dipole",
    "functional", "functional_label", "basis", "basis_label",
    "scf_seconds", "converged", "natoms", "geometry_source",
    "optimized_xyz", "opt_seconds", "opt_backend", "n_steps",
)


def _trim_dft(result: dict, geometry_source: str, natoms: Optional[int]) -> dict:
    out: Dict[str, Any] = {}
    for key in _DFT_FIELDS:
        value = result.get(key)
        if key == "dipole" and isinstance(value, dict):
            value = value.get("magnitude")
        out[key] = value
    out["natoms"] = natoms
    out["geometry_source"] = result.get("geometry_source") or geometry_source
    return out


def _cache_path(key: str) -> str:
    return os.path.join(DFT_CACHE_DIR, f"{key}.json")


def _cache_read(key: str) -> Optional[dict]:
    try:
        with open(_cache_path(key), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("converged") else None


def _cache_write(key: str, payload: dict) -> None:
    try:
        os.makedirs(DFT_CACHE_DIR, exist_ok=True)
        with open(_cache_path(key), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
    except OSError:
        pass


def dft_refine(smiles: str, functional: str, basis: str, progress=None,
               optimize: bool = False) -> Optional[dict]:
    """One real single point on the lead's conformer.

    The result is cached by (molecule, level of theory, relaxation) because
    an SCF is deterministic: the same input gives bit-identical numbers, and
    a 30-atom single point costs half a minute.  Repeat runs and runs whose
    shortlists overlap then cost nothing, and the report says how many
    numbers were reused rather than pretending they were recomputed.
    """
    from ..engine import molecule as molmod
    from ..engine import DFTEngine
    from ..engine.dft import JobProgress

    try:
        mol = molmod.from_smiles(smiles)
    except Exception:
        return None
    if mol.natoms() > 45:
        return None

    canonical_smiles = canonical(smiles) or smiles
    key = hashlib.sha1(
        f"{canonical_smiles}|{functional}|{basis}|{bool(optimize)}"
        .encode("utf-8")).hexdigest()
    cached = _cache_read(key)
    if cached is not None:
        cached["cached"] = True
        if progress:
            try:
                progress(JobProgress("dft", 100, "reused an identical run"))
            except Exception:
                pass
        return cached

    geometry_source = "MMFF94 (RDKit ETKDGv3)"
    try:
        engine = DFTEngine(atom_xyz=mol.to_xyz(), charge=0, multiplicity=1,
                           functional=functional, basis=basis, solvation=None)
        if optimize:
            result = engine.optimize(progress=progress)
            backend = result.get("opt_backend") or "geometric"
            geometry_source = f"DFT-optimised ({backend})"
        else:
            result = engine.run_scf(progress=progress)
    except Exception:
        return None
    if not result.get("converged"):
        return None

    payload = _trim_dft(result, geometry_source, mol.natoms())
    payload["cached"] = False
    _cache_write(key, payload)
    return payload


# ======================================================================
#  the design run
# ======================================================================
@dataclass
class DesignResult:
    id: str
    brief: dict
    candidates: List[dict]
    rejected: int = 0
    generated: int = 0
    dft_ran: int = 0
    generations: int = 1
    model_used: Optional[str] = None
    model_notes: List[str] = field(default_factory=list)
    seconds: float = 0.0
    targets: List[dict] = field(default_factory=list)
    origins: Dict[str, int] = field(default_factory=dict)
    # ---- honesty counters -------------------------------------------
    # how many of the candidates actually satisfy every required
    # substructure, and how many DFT numbers were reused from the cache
    dft_cached: int = 0
    qualified: int = 0
    requirements: List[str] = field(default_factory=list)
    unmet_requirements: List[str] = field(default_factory=list)
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    quality_checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "brief": self.brief,
            "candidates": self.candidates,
            "rejected": self.rejected,
            "generated": self.generated,
            "dft_ran": self.dft_ran,
            "dft_cached": self.dft_cached,
            "generations": self.generations,
            "origins": self.origins,
            "model_used": self.model_used,
            "model_notes": self.model_notes,
            "seconds": round(self.seconds, 2),
            "targets": self.targets,
            "qualified": self.qualified,
            "requirements": self.requirements,
            "unmet_requirements": self.unmet_requirements,
            "rejection_reasons": self.rejection_reasons,
            "quality_checks": self.quality_checks,
        }


def design(brief: DesignBrief, client=None, progress: Optional[Callable] = None,
           seed: Optional[int] = None, do_dft: bool = True) -> DesignResult:
    """Run the whole loop: propose -> filter -> rank -> measure -> reflect."""
    started = _now()
    Chem = chem()

    def step(stage: str, pct: float, msg: str) -> None:
        if progress:
            try:
                progress(stage, pct, msg)
            except Exception:
                pass

    if seed is None:
        seed = int(hashlib.sha1(brief.text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)

    ident = hashlib.sha1(
        f"{brief.text}|{seed}|{_now()}".encode("utf-8")).hexdigest()[:10]
    notes: List[str] = list(brief.notes)
    model_used: Optional[str] = None
    llm_live = client is not None and getattr(client, "available", False)

    step("brief", 3, "Reading the design brief")

    seen: Dict[str, dict] = {}          # canonical smiles -> candidate dict
    rejected = 0
    generated = 0
    origins: Dict[str, int] = {}
    reject_reasons: Dict[str, int] = {}
    # how often each demanded substructure was actually present in something
    # the generators produced.  A requirement with a count of zero after the
    # whole search is one this engine provably cannot build -- a fact about
    # this run, not a guess from the size of the library.
    req_patterns = [(s, compile_smarts(s)) for s in brief.required_smarts]
    req_hits: Dict[str, int] = {s: 0 for s, _p in req_patterns}
    req_ok = [p for _s, p in req_patterns if p is not None]

    def qualified(mol) -> bool:
        return all(mol.HasSubstructMatch(p) for p in req_ok)

    def note_reject(reason: str) -> None:
        # Bucket by category, not by exact value: "MW 412 over 300" and
        # "MW 405 over 300" are the same complaint.  Numbers become `#`
        # rather than being cut off at a bracket, because a SMARTS pattern
        # contains brackets -- splitting on "(" turned
        # "missing required substructure S(=O)(=O)N" into "missing required
        # substructure S", which named the wrong thing.
        key = re.sub(r"\d+(?:\.\d+)?", "#", reason).strip()
        if len(key) > 60:
            key = key[:57] + "..."
        reject_reasons[key] = reject_reasons.get(key, 0) + 1

    def offer(smiles: str, origin: str, name: str = "",
              rationale: str = "") -> bool:
        """Validate, filter, dedupe and store one candidate."""
        nonlocal rejected, generated
        generated += 1
        mol = safe_mol(smiles)
        if mol is None:
            rejected += 1
            note_reject("not a valid structure")
            return False
        canon = canonical(Chem.MolToSmiles(mol))
        if canon in seen:
            rejected += 1
            note_reject("duplicate of a molecule already in the pool")
            return False
        for smarts, pattern in req_patterns:
            if pattern is not None and mol.HasSubstructMatch(pattern):
                req_hits[smarts] = req_hits.get(smarts, 0) + 1
        reason = violates(mol, brief)
        if reason:
            rejected += 1
            note_reject(reason)
            return False
        seen[canon] = {
            "smiles": canon,
            "name": name or _guess_name(mol, canon),
            "rationale": rationale,
            "origin": origin,
            "descriptors": describe_molecule(mol),
            "dft": None,
            "scoring": score_candidate(mol, brief, None),
            "qualified": qualified(mol),
        }
        # counted only once the molecule is actually in the pool, so the
        # reported origin breakdown cannot claim credit for a duplicate
        origins[origin] = origins.get(origin, 0) + 1
        return True

    def n_qualified() -> int:
        return sum(1 for c in seen.values() if c["qualified"])

    # ---- generation 0 -------------------------------------------------
    step("propose", 6, "Collecting starting points")
    seeds = list(brief.seeds)
    if len(seeds) < 6:
        seeds += library_seeds(limit=20,
                               prefer_smarts=brief.required_smarts or None)
    for smi in seeds:
        offer(smi, "seed")

    if llm_live:
        step("propose", 12, "Asking the model for candidates")
        try:
            for item in propose_from_llm(brief, client, list(seen)):
                offer(item["smiles"], "model", item["name"], item["rationale"])
            model_used = getattr(client, "last_model", None) or \
                getattr(client, "model", None)
        except Exception as exc:
            notes.append(f"Model proposal failed ({exc}); "
                         "continued with the generative engine.")
    else:
        notes.append(
            "No language model is reachable, so this run is purely "
            "generative: library seeds, site substitution and fragment "
            "recombination. The design space is unaffected; only the "
            "chemistry commentary is missing.")

    # ---- generative expansion ----------------------------------------
    # The pool is grown until enough candidates have *survived* the filters,
    # not until enough have been proposed.  A tight brief can reject 95% of
    # everything, so proposing a fixed number once and stopping is how a
    # request for four molecules used to return two.
    #
    # When the brief demands a substructure the bar is higher: the pool must
    # contain enough molecules that actually carry it.  Stopping at "twelve
    # candidates" is what made "design 4 sulfonamides" return two, because
    # the demanded group turns up in only ~2% of uniform fragment draws.
    #
    # ...and the run has to *design* something.  Survivors that came straight
    # from the library are not designed molecules, so counting them let a
    # loose brief stop before the generative stage ever ran: "design 6
    # molecules with a wide HOMO-LUMO gap" returned six library entries and
    # produced nothing new at all.  The loop therefore also insists on a
    # floor of generated candidates.
    want = max(1, brief.n_candidates)
    target_pool = max(want * 3, 12)
    hard_cap = max(want * 12, 60)
    max_rounds = max(3, brief.generations * 3)
    if req_ok:
        max_rounds = max(max_rounds, 6)
    want_generated = max(want, 6)

    def n_generated() -> int:
        return sum(1 for c in seen.values() if c["origin"] != "seed")

    # The molecules the brief named, canonicalised once.
    focus: List[str] = []
    for smi in brief.seed_focus:
        canon_focus = canonical(smi)
        if canon_focus and canon_focus not in focus:
            focus.append(canon_focus)
    focus_set = set(focus)

    rounds = 0
    while rounds < max_rounds and (
            len(seen) < min(target_pool, hard_cap)
            or (n_generated() < want_generated and len(seen) < hard_cap)
            or (req_ok and n_qualified() < want and len(seen) < hard_cap)):
        rounds += 1
        pct = 18 + int(40 * (rounds - 1) / max_rounds)
        step("grow", pct, f"Generating candidates (round {rounds})")

        # A molecule the brief named is the point of the run, not one seed in
        # a shuffled pool: it is mutated first, every round, and its products
        # are labelled by where they came from.  Left to the pool draw a named
        # molecule could be offered and then never built on -- which is what
        # made "design 5 molecules similar to porphine" ignore porphine.
        for smi in focus[:6]:
            mol = safe_mol(smi)
            if mol is None:
                continue
            for variant in mutate(mol, rng, limit=16):
                offer(Chem.MolToSmiles(variant), "analogue")
            if len(seen) >= hard_cap:
                break

        pool = list(seen.keys())
        rng.shuffle(pool)
        for smi in pool[:14]:
            if smi in focus_set:
                continue            # already mutated above, with provenance
            mol = safe_mol(smi)
            if mol is None:
                continue
            for variant in mutate(mol, rng, limit=8):
                offer(Chem.MolToSmiles(variant), "mutation")
            if len(seen) >= hard_cap:
                break

        for built in assemble(rng, limit=max(40, want * 8),
                              require_smarts=brief.required_smarts or None):
            offer(Chem.MolToSmiles(built), "assembly")
            if len(seen) >= hard_cap:
                break

        # the model only gets one more turn, and only while the pool is short
        if llm_live and rounds <= 2 and len(seen) < target_pool:
            ranked = _rank(seen, brief)
            feedback = "\n".join(
                f"- {c['smiles']} ({c['scoring']['score']:.3f}) {c['name']}"
                for c in ranked[:5])
            step("reflect", pct + 6, "Asking the model for more candidates")
            try:
                for item in propose_from_llm(brief, client, list(seen),
                                             feedback=feedback):
                    offer(item["smiles"], "model", item["name"],
                          item["rationale"])
            except Exception:
                pass

    # ---- rank ---------------------------------------------------------
    step("rank", 62, "Ranking by desirability")
    ranked = _rank(seen, brief)

    # ---- DFT on the leaders ------------------------------------------
    dft_ran = 0
    dft_cached = 0
    if do_dft and brief.dft_top > 0:
        top = ranked[:max(1, brief.dft_top)]
        for k, cand in enumerate(top):
            base = 66 + int(30 * k / max(1, len(top)))
            span = max(1, int(30 / max(1, len(top))))
            step("dft", base,
                 f"DFT single point {k + 1}/{len(top)}: {cand['name']}")
            # a single point on a 30-atom molecule can run for a minute; the
            # SCF progress is mapped into this candidate's slice of the bar
            # so a slow job is never mistaken for a wedged one
            result = dft_refine(
                cand["smiles"], brief.functional, brief.basis,
                optimize=brief.optimize_leads,
                progress=lambda p: step(
                    "dft", base + int(span * (p.percent or 0) / 100.0),
                    f"{cand['name']}: {p.message}"))
            if result is None:
                cand["dft_error"] = "single point did not converge"
                continue
            if result.get("cached"):
                dft_cached += 1
            else:
                dft_ran += 1
            mol = safe_mol(cand["smiles"])
            cand["dft"] = {
                "energy_hartree": result.get("energy_hartree"),
                "homo_ev": result.get("homo_ev"),
                "lumo_ev": result.get("lumo_ev"),
                "gap_ev": result.get("gap_ev"),
                "dipole": result.get("dipole"),
                "functional": result.get("functional"),
                "functional_label": result.get("functional_label"),
                "basis": result.get("basis"),
                "basis_label": result.get("basis_label"),
                "scf_seconds": result.get("scf_seconds"),
                "converged": result.get("converged"),
                "natoms": result.get("natoms"),
                # the level of theory alone is not reproducible: the same
                # functional on an MMFF conformer and on a DFT minimum are
                # different numbers, so the geometry source travels with it
                "geometry_source": result.get("geometry_source"),
                "optimized": bool(brief.optimize_leads),
                "opt_seconds": result.get("opt_seconds"),
                "opt_backend": result.get("opt_backend"),
                "opt_steps": result.get("n_steps"),
                "cached": bool(result.get("cached")),
            }
            if mol is not None:
                cand["scoring"] = score_candidate(mol, brief, cand["dft"])
        ranked = _rank(seen, brief)

    # ---- final cut ----------------------------------------------------
    final, n_lead = _final_cut(ranked, brief, want, llm_live, bool(req_ok))
    if n_lead:
        # what leads depends on the brief: the analogues of a molecule that
        # was named, or the model's on-goal proposals for a qualitative one.
        # Saying "proposed for the stated goal" for both was simply wrong when
        # the leaders were analogues.
        if brief.seed_focus:
            notes.append(
                f"{n_lead} analogue(s) of the molecule you named are listed "
                "first: they are the answer to the question, so they lead the "
                "list even where a later candidate scores higher. The "
                "remainder are ranked by the stated criteria.")
        else:
            notes.append(
                f"{n_lead} candidate(s) were proposed for the stated goal "
                "and are listed first; the remainder come from the generative "
                "engine and are ranked by the balanced profile.")

    unmet = [s for s, _p in req_patterns
             if req_ok and req_hits.get(s, 0) == 0]
    if len(final) < want:
        # say *why*, from the tally, instead of guessing -- the old message
        # always blamed the molecular-weight limit even when the real cause
        # was a required substructure nothing could build
        why = ""
        if reject_reasons:
            top_reasons = sorted(reject_reasons.items(),
                                 key=lambda kv: -kv[1])[:3]
            why = (" The commonest rejection reasons were: "
                   + "; ".join(f"{r} ({n})" for r, n in top_reasons) + ".")
        if unmet:
            why += (" No candidate generated in this run carried the required "
                    "substructure " + ", ".join(f"`{s}`" for s in unmet)
                    + ", so that requirement is what emptied the pool.")
        notes.append(
            f"Only {len(final)} of the {want} requested candidates satisfied "
            "every hard constraint." + why +
            " Widening the pool needs a looser limit, a different "
            "substructure, or a model proposal for the specific scaffold.")
    if req_ok and not unmet and n_qualified() < want:
        notes.append(
            f"{n_qualified()} of the {want} requested molecules carry every "
            "required substructure; the rest were kept to fill the list and "
            "are marked as not matching the requirement.")

    # A measured leader that fails a criterion is the single most important
    # thing to say out loud: it is why the calculated molecules sit below
    # unmeasured ones in the list.
    measured_bad = [c for c in final
                    if c.get("dft") and c["scoring"].get("blocking")]
    if measured_bad:
        detail = "; ".join(
            f"{c['name']} misses {', '.join(c['scoring']['blocking'])}"
            for c in measured_bad[:3])
        notes.append(
            f"{len(measured_bad)} of the calculated candidates miss a "
            f"criterion the unmeasured ones were only estimated against "
            f"({detail}). They are still listed: the measured numbers are the "
            "evidence, and an unmet target is a result.")

    # ---- final commentary ---------------------------------------------
    if llm_live:
        step("explain", 98, "Writing the rationale")
        _explain(final, brief, client)

    step("done", 100, "Design complete")

    try:
        quality_checks = _quality().available()
    except Exception:
        quality_checks = {}

    result = DesignResult(
        id=ident,
        brief=brief.to_dict(),
        candidates=final,
        rejected=rejected,
        generated=generated,
        dft_ran=dft_ran,
        dft_cached=dft_cached,
        generations=max(1, brief.generations),
        model_used=model_used,
        model_notes=notes,
        seconds=_now() - started,
        targets=[t.to_dict() for t in brief.targets],
        origins=origins,
        qualified=n_qualified(),
        requirements=[s for s, _p in req_patterns],
        unmet_requirements=unmet,
        rejection_reasons=dict(sorted(reject_reasons.items(),
                                      key=lambda kv: -kv[1])),
        quality_checks=quality_checks,
    )
    _save(result)
    return result


def _final_cut(ranked: List[dict], brief: DesignBrief, want: int,
               llm_live: bool, requires: bool) -> Tuple[List[dict], int]:
    """Choose the candidates that are actually reported.

    Three rules, and they **combine** rather than override each other -- an
    earlier revision let the substructure rule replace the model rule, which
    quietly stopped a model's on-goal proposals from ever being listed:

    1. A molecule that does not carry a demanded substructure is only
       reported once the ones that do have run out.  It is not an answer to
       the question that was asked.
    1b. **The analogues of a molecule the brief named lead the list.**  They
       are not a bonus, they are the answer: "design 5 molecules similar to
       porphine" is a request for porphine analogues, and a list that opened
       with four unrelated fragments would be a different answer.
    2. For a *qualitative* brief (no number to hit) the goal itself is the
       requirement and the model is the only source that reads it, so its
       on-brief proposals take up to half the slots and lead the list.
    3. **A molecule that was measured is reported.**  The leaders are chosen
       before they are calculated and the calculation is the expensive part
       of the run; letting the post-measurement re-score drop one would
       spend a minute of SCF time and then hide the number it produced --
       including, and worst of all, when that number is the evidence that
       the target cannot be met.  It is *included*, not promoted: the list
       stays a ranking, and the note under it says which measured candidate
       missed which criterion.

    Returns the chosen candidates and how many of them led as model picks.
    """
    on_brief = ([c for c in ranked if c.get("qualified", True)]
                if requires else list(ranked))
    fallback = ([c for c in ranked if not c.get("qualified", True)]
                if requires else [])

    # 1. analogues of a named molecule lead; failing that, the model's
    #    on-goal proposals lead a qualitative brief
    lead: List[dict] = []
    if brief.seed_focus:
        lead = [c for c in on_brief if c["origin"] == "analogue"][:max(1, want // 2)]
    elif llm_live and not brief.has_numeric:
        proposed = [c for c in on_brief if c["origin"] == "model"]
        lead = proposed[:max(1, want // 2)]
    lead_ids = {id(c) for c in lead}

    # 2. the body is the ranked list, minus whatever the lead already took
    body = [c for c in on_brief if id(c) not in lead_ids][:max(0, want - len(lead))]

    # 3. a measured candidate is swapped in for the lowest-ranked unmeasured
    #    one, then the body is re-sorted so the list is still a ranking
    if body:
        for cand in on_brief:
            if not cand.get("dft") or id(cand) in lead_ids:
                continue
            if any(c is cand for c in body):
                continue
            victim = next((b for b in reversed(body) if not b.get("dft")), None)
            if victim is None:
                break
            body[next(i for i, b in enumerate(body) if b is victim)] = cand
        order = {id(c): i for i, c in enumerate(on_brief)}
        body.sort(key=lambda c: order.get(id(c), len(on_brief)))

    chosen = lead + body
    taken = {id(c) for c in chosen}
    for cand in on_brief + fallback:
        if len(chosen) >= want:
            break
        if id(cand) not in taken:
            chosen.append(cand)
            taken.add(id(cand))
    return chosen, len(lead)


def _rank(seen: Dict[str, dict], brief: DesignBrief) -> List[dict]:
    """Sort by desirability, with the hard requirements and measurement first.

    Order of precedence:

    1. **A demanded substructure.**  A molecule that does not carry it is not
       an answer to the question that was asked, however well it scores on
       the side criteria, so it can only appear once the on-brief molecules
       have run out.  Only applies when the brief demands something.
    2. **The desirability score.**
    3. **Whether the electronic properties were actually measured.**
    4. **Synthetic accessibility**, as a tie-break: between two candidates
       that are equally on-target, the one somebody can make is better.
    """
    items = list(seen.values())
    requires = bool(brief.required_smarts)

    def key(c: dict):
        values = (c.get("descriptors") or {}).get("values") or {}
        sa = values.get("sa")
        return (0 if (not requires or c.get("qualified", True)) else 1,
                -c["scoring"]["score"],
                0 if c.get("dft") else 1,
                sa if sa is not None else 99.0,
                c["smiles"])

    items.sort(key=key)
    return items


def _guess_name(mol, smiles: str) -> str:
    try:
        from ..engine import molecule as molmod

        name = molmod._library_name_for_smiles(smiles)
        if name:
            return name
    except Exception:
        pass
    try:
        return _rk()["rdMolDescriptors"].CalcMolFormula(mol)
    except Exception:
        return smiles[:24]


_EXPLAIN_SYSTEM = """\
You are writing the discussion section of a computational chemistry report.
Given the goal and the ranked candidates with their computed descriptors,
write for EACH candidate one short paragraph: why it fits, what the numbers
say, and what the main risk or trade-off is. Be specific and quantitative.
Do not invent numbers that are not in the data you were given.

Reply with ONE JSON object: {"notes": {"<smiles>": "<paragraph>"}}

OUTPUT ONLY THE JSON OBJECT -- no preamble, no reasoning, no fences.
"""


def _explain(candidates: List[dict], brief: DesignBrief, client) -> None:
    from .llm import loads_loose

    lines = [f"Goal: {brief.goal}", f"Criteria: {brief.summary()}", ""]
    for cand in candidates[:6]:
        values = cand["descriptors"]["values"]
        shown = ", ".join(f"{k}={values.get(k)}" for k in
                          ("molwt", "logp", "tpsa", "hbd", "hba", "rotb"))
        dft = cand.get("dft")
        extra = ""
        if dft:
            extra = (f", HOMO={_fmt(dft.get('homo_ev'))} eV, "
                     f"LUMO={_fmt(dft.get('lumo_ev'))} eV, "
                     f"gap={_fmt(dft.get('gap_ev'))} eV, "
                     f"dipole={_fmt(dft.get('dipole'))} D")
        lines.append(f"{cand['smiles']} | {shown}{extra} | "
                     f"score="
                     + ("not scored yet (needs a calculation)"
                        if (cand.get("scoring") or {}).get("scored") is False
                        else str(cand["scoring"]["score"])))
    try:
        payload = client.json(
            [{"role": "system", "content": _EXPLAIN_SYSTEM},
             {"role": "user", "content": "\n".join(lines)}],
            temperature=0.4, max_tokens=_REPLY_TOKENS,
            validate=lambda p: isinstance(p, dict)
            and isinstance(p.get("notes"), dict) and bool(p["notes"]))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    notes = payload.get("notes")
    if not isinstance(notes, dict):
        return
    for cand in candidates:
        text = notes.get(cand["smiles"])
        if isinstance(text, str) and text.strip():
            cand["discussion"] = text.strip()[:1200]


def _fmt(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _now() -> float:
    return time.time()


def _save(result: DesignResult) -> None:
    """Persist the run so the design history survives a restart.

    Only the newest runs are kept; the directory is bounded and nothing is
    ever deleted by the request path.

    The pruning is strictly best-effort.  It runs *after* the result has
    been written, so a failure to unlink can only leave an extra file behind
    -- it must never turn a finished design run into an error.  A restricted
    or audited filesystem can refuse the unlink, and that refusal arrives as
    whatever exception the audit layer raises, so the catch is broad on
    purpose.
    """
    try:
        os.makedirs(DESIGN_DIR, exist_ok=True)
        path = os.path.join(DESIGN_DIR, f"design-{result.id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, ensure_ascii=False, indent=1)
    except OSError:
        return
    try:
        files = sorted(
            (os.path.getmtime(os.path.join(DESIGN_DIR, f)), f)
            for f in os.listdir(DESIGN_DIR) if f.endswith(".json"))
        while len(files) > 40:
            _m, old = files.pop(0)
            try:
                os.remove(os.path.join(DESIGN_DIR, old))
            except Exception:                        # noqa: BLE001
                break
    except Exception:                                # noqa: BLE001
        pass


def list_designs(limit: int = 20) -> List[dict]:
    out: List[dict] = []
    try:
        names = os.listdir(DESIGN_DIR)
    except OSError:
        return out
    entries = []
    for name in names:
        if not name.endswith(".json"):
            continue
        path = os.path.join(DESIGN_DIR, name)
        try:
            entries.append((os.path.getmtime(path), path))
        except OSError:
            continue
    entries.sort(reverse=True)
    for _m, path in entries[:limit]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        out.append({
            "id": data.get("id"),
            "goal": (data.get("brief") or {}).get("goal", ""),
            "text": (data.get("brief") or {}).get("text", ""),
            "n": len(data.get("candidates") or []),
            "seconds": data.get("seconds"),
            "model_used": data.get("model_used"),
        })
    return out


def load_design(ident: str) -> Optional[dict]:
    if not re.match(r"^[0-9a-f]{6,32}$", ident or ""):
        return None
    path = os.path.join(DESIGN_DIR, f"design-{ident}.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ======================================================================
#  narration
# ======================================================================
def explain_design(result) -> str:
    """The written summary shown under a finished design run.

    Takes either a :class:`DesignResult` or its ``to_dict()`` form, because
    the narration is rebuilt from a stored JSON file as often as it is from
    a live object.
    """
    if not isinstance(result, dict):
        result = result.to_dict()
    data = result
    lines: List[str] = []
    brief_goal = (data.get("brief") or {}).get("goal") or \
        (data.get("brief") or {}).get("text") or "the requested molecule"
    lines.append(f"**Design goal.** {brief_goal}")
    lines.append("")

    targets = data.get("targets") or []
    if targets:
        lines.append("**Ranking criteria.** " + "; ".join(
            f"{t.get('label') or t.get('key')} ("
            + {"max": "maximise", "min": "minimise",
               "near": "target"}.get(t.get("aim"), t.get("aim"))
            + f", weight {t.get('weight', 1):g})" for t in targets))
        lines.append("")

    requirements = data.get("requirements") or []
    unmet = data.get("unmet_requirements") or []
    if requirements:
        listed = len(data.get("candidates") or [])
        listed_ok = sum(1 for c in (data.get("candidates") or [])
                        if c.get("qualified", True))
        lines.append("**Hard requirements.** Every candidate must contain "
                     + ", ".join(f"`{s}`" for s in requirements) + ". "
                     + f"The search found {data.get('qualified', 0)} "
                     "molecules carrying "
                     + ("it" if len(requirements) == 1 else "them")
                     + f"; {listed_ok} of the {listed} listed here do.")
        if unmet:
            lines.append("")
            lines.append(
                "**Unmet requirement.** Nothing the generator built in this "
                "run carried " + ", ".join(f"`{s}`" for s in unmet)
                + ", so the list below is *not* filtered on it. Widen the "
                "request, or ask the model for a specific scaffold carrying "
                "that group.")
        lines.append("")

    candidates = data.get("candidates") or []
    reused = data.get("dft_cached") or 0
    fresh = data.get("dft_ran") or 0
    if fresh and reused:
        calc = (f" {fresh} of them were then calculated at {_level(data)}, "
                f"and {reused} more reused an identical earlier calculation.")
    elif fresh:
        calc = f" {fresh} of them were then calculated at {_level(data)}."
    elif reused:
        calc = (f" {reused} of them had already been calculated at "
                f"{_level(data)}, so an identical earlier result was reused "
                "rather than recomputed.")
    else:
        calc = " No electronic-structure calculation was requested."

    # A named starting point is worth its own paragraph: the analogues lead
    # the list, and without saying so the ordering below looks like the
    # ranking is broken.
    focus = (data.get("brief") or {}).get("seed_focus") or []
    if focus:
        named = (data.get("brief") or {}).get("seed_names") or []
        lines.append(
            "**Starting point.** The brief names "
            + (", ".join(named) if named else "a molecule")
            + ", so that molecule and its analogues were generated first and "
            "lead the list below. They are the answer to the question as "
            "asked, and can therefore sit above a candidate that scores "
            "higher on the numeric criteria.")
        lines.append("")

    lines.append(
        f"**Search.** {data.get('generated', 0)} structures were proposed, "
        f"{len(candidates)} survived the hard filters and "
        f"{data.get('rejected', 0)} were rejected as duplicate, unstable or "
        f"out of bounds." + calc)
    lines.append("")

    reasons = data.get("rejection_reasons") or {}
    if reasons:
        top = list(reasons.items())[:3]
        lines.append("**Why candidates were dropped.** " + "; ".join(
            f"{reason} ({n})" for reason, n in top) + ".")
        lines.append("")

    checks = data.get("quality_checks") or {}
    if checks:
        on = [k for k, v in checks.items() if v]
        if on:
            lines.append("**Quality filters applied.** " + ", ".join(
                {"sa_score": "synthetic accessibility (Ertl 2009)",
                 "qed": "QED drug-likeness (Bickerton 2012)",
                 "alerts": "PAINS / BRENK / NIH structural alerts"}
                .get(k, k) for k in on) + ".")
            lines.append("")

    if candidates:
        lines.append("**Lead candidates.**")
        lines.append("")
        for i, cand in enumerate(candidates[:5], 1):
            values = (cand.get("descriptors") or {}).get("values") or {}
            scoring = cand.get("scoring") or {}
            # a gap-only brief cannot score anything before the SCF, and
            # "score 0.000" would read as "fails every criterion"
            score_bit = ("not scored yet"
                         if scoring.get("scored") is False
                         else f"score {scoring.get('score', 0.0):.3f}")
            head = (f"{i}. **{cand['name']}** `{cand['smiles']}` — "
                    f"{score_bit}")
            lines.append(head)
            bits = []
            for key in ("molwt", "logp", "tpsa", "hbd", "hba", "rotb",
                        "sa", "qed"):
                if values.get(key) is not None:
                    prop = PROPERTIES[key]
                    bits.append(f"{prop.label.split(' (')[0]} "
                                f"{values[key]:g}{' ' + prop.unit if prop.unit else ''}")
            alerts = (cand.get("descriptors") or {}).get("alerts") or {}
            for name, hits in alerts.items():
                bits.append(f"{name} alert: {hits[0]}"
                            + (f" (+{len(hits) - 1} more)" if len(hits) > 1
                               else ""))
            dft = cand.get("dft")
            if dft:
                bits.append(f"HOMO {_fmt(dft.get('homo_ev'))} eV")
                bits.append(f"LUMO {_fmt(dft.get('lumo_ev'))} eV")
                bits.append(f"gap {_fmt(dft.get('gap_ev'))} eV")
                bits.append(f"dipole {_fmt(dft.get('dipole'))} D")
                if dft.get("cached"):
                    bits.append("reused from an identical earlier run")
            lines.append("   " + "; ".join(bits))
            if cand.get("rationale"):
                lines.append(f"   *{cand['rationale']}*")
            if cand.get("discussion"):
                lines.append(f"   {cand['discussion']}")
            lines.append("")

    relaxed = any((c.get("dft") or {}).get("optimized")
                  for c in candidates)
    if data.get("dft_ran") or data.get("dft_cached"):
        lines.append(
            "Descriptors (logP, TPSA, counts, synthetic accessibility, QED) "
            "are computed from the 2D structure with RDKit; the orbital "
            "energies and dipole moments are measured at "
            f"{_level(data)}."
            + (" The lead geometries were relaxed at the same level of theory "
               "first, so those numbers come from a real minimum."
               if relaxed else
               " The geometry is the RDKit ETKDGv3/MMFF94 conformer, written "
               "`A//B` for a single point at A on a geometry optimised at B; "
               "relaxing it at the DFT level typically moves the gap by a few "
               "tenths of an eV. Ask to \"optimise the geometry\" to have "
               "that done."))
    else:
        lines.append(
            "All numbers above are structure-based descriptors from RDKit. "
            "Ask for the HOMO-LUMO gap (or any other electronic property) "
            "and the leads will be recalculated with DFT.")
    return "\n".join(lines)


def _level(result) -> str:
    """Level of theory of the measured numbers, geometry included.

    ``B3LYP/6-31G*`` on a force-field conformer and on a DFT minimum are
    different calculations, so the geometry source is part of the level of
    theory: ``A//B`` means "a single point at A on a geometry optimised at
    B", which is the standard way to write it.  Without that the reported
    number is not reproducible.
    """
    if not isinstance(result, dict):
        result = result.to_dict()
    for cand in result.get("candidates") or []:
        dft = cand.get("dft") or {}
        if dft.get("functional_label") and dft.get("basis_label"):
            level = f"{dft['functional_label']}/{dft['basis_label']}"
            if dft.get("optimized"):
                return level
            source = str(dft.get("geometry_source") or "MMFF94")
            short = source.split(" (")[0]
            return f"{level}//{short}"
    return "B3LYP/6-31G*"


# ======================================================================
#  self test
# ======================================================================
def _selftest() -> int:
    """Offline checks used by ``python -m backend.agent.designer``."""
    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" +
              (f"  -- {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    Chem = chem()
    from .llm import StubClient

    # 1. the fragment convention really produces the expected molecule
    benzoic = _graft("[*:1]c1ccccc1", ["C(=O)O"])
    check("graft benzoic acid",
          benzoic is not None and canonical(Chem.MolToSmiles(benzoic))
          == canonical("O=C(O)c1ccccc1"),
          "" if benzoic is None else Chem.MolToSmiles(benzoic))

    paba = _graft("[*:1]c1ccc([*:2])cc1", ["C(=O)O", "N"])
    check("graft two sites -> PABA",
          paba is not None and canonical(Chem.MolToSmiles(paba))
          == canonical("Nc1ccc(C(=O)O)cc1"),
          "" if paba is None else Chem.MolToSmiles(paba))

    # 2. every core parses and carries the expected number of sites
    bad_cores = [n for n, s in CORES if Chem.MolFromSmiles(s) is None]
    check("all cores parse", not bad_cores, ",".join(bad_cores))
    bad_subs = [n for n, s in SUBSTITUENTS if Chem.MolFromSmiles(s) is None]
    check("all substituents parse", not bad_subs, ",".join(bad_subs))

    # 3. grafting every substituent onto every core must give valid chemistry
    broken: List[str] = []
    for cname, csmi in CORES:
        sites = _core_sites(csmi)
        for sname, ssmi in SUBSTITUENTS:
            if sites >= 2:
                built = _graft(csmi, [ssmi, "C"])
            else:
                built = _graft(csmi, [ssmi])
            if built is None:
                broken.append(f"{cname}+{sname}")
            else:
                try:
                    Chem.SanitizeMol(built)
                    Chem.MolToSmiles(built)
                except Exception:
                    broken.append(f"{cname}+{sname}(smiles)")
    check("core x substituent grafting is valid",
          not broken, ",".join(broken[:6]))

    # 3b. ...and a core that is given fewer substituents than it has sites
    # comes back with an unfilled attachment point, which is not an atom and
    # not a molecule.  The section above never produced one because it hands
    # two-site cores two substituents -- but `_hunt_required` did, and the
    # report came back with `O=S(=O)(N[*:2])c1ccccc1`: an asterisk in the
    # formula, queued for the SCF.
    dummy_mol = safe_mol("O=S(=O)(N[*:2])c1ccccc1")
    check("a molecule with an unfilled attachment point is rejected",
          violates(dummy_mol, DesignBrief()) is not None,
          str(violates(dummy_mol, DesignBrief())))
    leaked = [cname for cname, csmi in CORES
              if _core_sites(csmi) >= 2
              and _graft(csmi, ["O"]) is not None
              and any(a.GetAtomicNum() == 0
                      for a in _graft(csmi, ["O"]).GetAtoms())]
    check("handing one substituent to a two-site core leaves a dummy",
          bool(leaked), str(leaked[:3]))
    hunt_run = design(
        parse_brief("design 4 molecules containing a sulfonamide",
                    client=None),
        client=None, seed=13, do_dft=False)
    stars = [c["smiles"] for c in hunt_run.candidates
             if "*" in c["smiles"]]
    check("no reported candidate carries a dummy atom", not stars,
          str(stars[:3]))
    check("and every one still carries what was demanded",
          all(c["qualified"] for c in hunt_run.candidates),
          str([c["smiles"] for c in hunt_run.candidates
               if not c["qualified"]]))

    # 4. desirability behaves
    t_max = Target("logp", "max", 1.0, 0.0, 4.0)
    check("desirability max: below low = 0", desirability(-1.0, t_max) == 0.0)
    check("desirability max: above high = 1", desirability(9.0, t_max) == 1.0)
    check("desirability max: midway",
          abs(desirability(2.0, t_max) - 0.5) < 1e-9)
    t_min = Target("logp", "min", 1.0, 0.0, 4.0)
    check("desirability min is reversed",
          abs(desirability(1.0, t_min) - 0.75) < 1e-9)
    check("one zero criterion zeroes the overall score",
          overall([(0.0, 1.0), (1.0, 1.0)]) == 0.0)
    check("geometric mean of 0.5 and 0.5",
          abs(overall([(0.5, 1.0), (0.5, 1.0)]) - 0.5) < 1e-9)

    # 4b. and the score a candidate is actually given goes through the same
    # path.  This is the check that was missing: `overall` was tested with
    # raw zeros while `score_candidate` clamped each desirability to 1e-6
    # first, so a molecule that met none of its criteria reported 0.0004
    # instead of 0.0.
    missed = DesignBrief(text="t", targets=[
        Target("molwt", "near", 1.0, 400.0, 600.0, 500.0),
        Target("logp", "max", 1.0)])
    ethanol = safe_mol("CCO")
    scored = score_candidate(ethanol, missed, None)
    check("a missed criterion zeroes the reported score",
          scored["score"] == 0.0, str(scored["score"]))
    check("and the zero score names what caused it",
          scored["blocking"] == ["Molecular weight"],
          str(scored["blocking"]))
    met = score_candidate(ethanol, DesignBrief(
        text="t", targets=[Target("logp", "max", 1.0)]), None)
    check("a met criterion still scores above zero", met["score"] > 0.0,
          str(met["score"]))
    check("a brief with no missed criterion blocks nothing",
          met["blocking"] == [])

    # 4c. a hand-built brief must survive a run.  `has_numeric` used to be
    # attached in parse_brief rather than declared, so this raised
    # AttributeError from inside design() whenever a model was reachable.
    check("a hand-built brief carries the flags design() reads",
          DesignBrief(text="x").has_numeric is False
          and DesignBrief(text="x").alert_filter == "none")
    hand = DesignBrief(text="x", targets=[Target("logp", "max", 1.0)])
    hand.n_candidates = 3
    hand.dft_top = 0
    hand_run = design(hand, client=StubClient([]), seed=2, do_dft=False)
    check("a hand-built brief runs to completion",
          len(hand_run.candidates) >= 1, str(len(hand_run.candidates)))

    # 4d. substructure patterns: an invalid one is reported, a valid one is
    # kept.  A pattern RDKit cannot parse can never match, so a *required*
    # one would silently go unenforced.
    patched = DesignBrief(text="t")
    patched.required_smarts = ["[ZZZ", "C(=O)[OX2H1]"]
    patched.forbidden_smarts = ["[[["]
    validate_brief(patched)
    check("an invalid required SMARTS is dropped",
          patched.required_smarts == ["C(=O)[OX2H1]"],
          str(patched.required_smarts))
    check("the drop is reported, with the offending pattern",
          any("[ZZZ" in n and "not valid SMARTS" in n for n in patched.notes),
          str(patched.notes[-1:]))
    check("an invalid forbidden SMARTS is dropped too",
          patched.forbidden_smarts == [], str(patched.forbidden_smarts))

    # 5. rule parsing
    brief = parse_brief("设计5个小分子吸附CO2，分子量小于300，含氮，无卤素",
                        client=None)
    check("parses candidate count", brief.n_candidates == 5,
          str(brief.n_candidates))
    check("parses no-halogen", set(["F", "Cl", "Br", "I"]).issubset(
        set(brief.forbidden_elements)), str(brief.forbidden_elements))
    check("parses nitrogen requirement", "[#7]" in brief.required_smarts,
          str(brief.required_smarts))
    brief2 = parse_brief("design something with logP below 2 and gap above 5 eV",
                         client=None)
    keys = {t.key for t in brief2.targets}
    check("parses logP and gap", {"logp", "gap"} <= keys, str(keys))
    check("electronic target raises the DFT budget", brief2.dft_top >= 3,
          str(brief2.dft_top))

    # 5b. the clause / sign / direction rules
    b = parse_brief("a molecule with HOMO above -6 eV and dipole below 2 D",
                    client=None)
    got = {t.key: t for t in b.targets}
    check("negative bound keeps its sign",
          got["homo"].low == -6.0, str(got.get("homo")))
    check("'above' means maximise, whatever the property default",
          got["homo"].aim == "max", str(got.get("homo")))
    check("the dipole bound does not leak into HOMO",
          got["homo"].high is None and got["dipole"].high == 2.0,
          f"{got.get('homo')} / {got.get('dipole')}")

    b = parse_brief("gap above 5 eV, logP between 1 and 3", client=None)
    got = {t.key: t for t in b.targets}
    check("a range belongs to its own clause",
          got["gap"].low == 5.0 and got["gap"].aim == "max",
          str(got.get("gap")))
    check("'between 1 and 3' is read as a range",
          got["logp"].aim == "near" and (got["logp"].low, got["logp"].high)
          == (1.0, 3.0), str(got.get("logp")))

    b = parse_brief("rigid aromatic with TPSA around 70 and at most 3 "
                    "rotatable bonds", client=None)
    got = {t.key: t for t in b.targets}
    check("'at most N' keeps the bound", got["rotb"].high == 3.0,
          str(got.get("rotb")))
    check("'around N' gives a window", got["tpsa"].low == 59.5,
          str(got.get("tpsa")))

    b = parse_brief("design a small amine with a wide HOMO-LUMO gap",
                    client=None)
    check("'HOMO-LUMO gap' is one target, not three",
          [t.key for t in b.targets] == ["gap"], str([t.key for t in b.targets]))
    b = parse_brief("water soluble molecule with logP below 1", client=None)
    check("the same property is not targeted twice",
          len(b.targets) == 1, str([t.key for t in b.targets]))

    # 5c. a bound must not leak across a clause boundary.  The full-width
    # comma `，` was missing from the break set *and* the preceding-window
    # search was not cut at a boundary either, so every Chinese brief that
    # wrote "logP 小于 2，能隙大于 4 eV" gave the gap logP's bound and
    # minimised it -- the exact opposite of the request.
    zh = parse_brief("设计4个含磺酰胺的分子，logP 小于 2，能隙大于 4 eV",
                     client=None)
    got = {t.key: t for t in zh.targets}
    check("a bound does not leak past a full-width comma",
          got["logp"].aim == "min" and got["logp"].high == 2.0
          and got["gap"].aim == "max" and got["gap"].low == 4.0,
          f"logp={got.get('logp')} gap={got.get('gap')}")
    zh2 = parse_brief("能隙大于 4 eV，logP 小于 2", client=None)
    got2 = {t.key: t for t in zh2.targets}
    check("and the order of the clauses does not matter",
          got2["gap"].aim == "max" and got2["gap"].low == 4.0
          and got2["logp"].aim == "min",
          f"gap={got2.get('gap')} logp={got2.get('logp')}")

    # 5d. substructure keywords overlap: "磺酰胺" contains "酰胺", so asking
    # for a sulfonamide used to also demand an amide and the pool filled
    # with sulfonyl amides nobody asked for.
    sulfonamide = parse_brief("设计4个含磺酰胺的分子", client=None)
    check("a longer substructure keyword wins over the shorter one inside it",
          sulfonamide.required_smarts == ["S(=O)(=O)N"],
          str(sulfonamide.required_smarts))
    amide = parse_brief("含酰胺的分子", client=None)
    check("the shorter keyword still works on its own",
          amide.required_smarts == ["C(=O)N"], str(amide.required_smarts))
    both = parse_brief("含氨基和羧基的分子", client=None)
    check("two different demands are both kept",
          set(both.required_smarts) == {"[NX3;H2,H1,H0;!$(N=*)]",
                                        "C(=O)[OX2H1]"},
          str(both.required_smarts))

    # 5e. a molecule the brief names is the starting point.  "design 5
    # molecules similar to porphine" used to be read as a request about
    # nothing in particular: the named compound was dropped without a word
    # and the run began from the library in general, so no porphine-like
    # molecule was ever built.
    named = parse_brief("design 4 molecules similar to porphine", client=None)
    check("a molecule named in the brief becomes the starting point",
          bool(named.seed_focus), str(named.seed_focus))
    check("and it leads the seed list",
          named.seeds[:1] == named.seed_focus[:1], str(named.seeds[:1]))
    zh_named = parse_brief("设计4个类似阿司匹林的分子", client=None)
    check("a Chinese molecule name resolves too",
          bool(zh_named.seed_focus), str(zh_named.seed_focus))
    # a hyphenated compound word is ordinary prose, not a name
    druglike = parse_brief("design a polar, drug-like amine", client=None)
    check("'drug-like' does not name anything",
          druglike.seed_focus == [], str(druglike.seed_focus))
    # ...nor does a chemical word that merely sits inside a library key:
    # a fuzzy lookup read "amine" as histamine
    amines = parse_brief("design 4 amines", client=None)
    check("a word that is only part of a library key names nothing",
          amines.seed_focus == [], str(amines.seed_focus))
    named_run = design(zh_named, client=None, seed=11, do_dft=False)
    check("the named molecule's analogues are generated",
          (named_run.origins.get("analogue") or 0) > 0,
          str(named_run.origins))
    check("and they lead the report",
          named_run.candidates[:1] and
          named_run.candidates[0]["origin"] == "analogue",
          str([c["origin"] for c in named_run.candidates]))

    # 6. a full offline run produces ranked, distinct, valid candidates
    offline = parse_brief("design a polar, drug-like amine", client=None)
    offline.n_candidates = 8
    offline.dft_top = 0
    res = design(offline, client=StubClient([]), seed=7, do_dft=False)
    check("offline run produced candidates", len(res.candidates) >= 5,
          f"{len(res.candidates)}")
    smiles = [c["smiles"] for c in res.candidates]
    check("candidates are distinct", len(set(smiles)) == len(smiles))
    check("candidates are valid SMILES",
          all(safe_mol(s) is not None for s in smiles))
    scores = [c["scoring"]["score"] for c in res.candidates]
    check("candidates are ranked", all(a >= b for a, b in
                                       zip(scores, scores[1:])),
          str(scores[:4]))
    check("scores are in [0,1]", all(0.0 <= s <= 1.0 for s in scores))
    check("every candidate has descriptors",
          all(c["descriptors"]["values"]["molwt"] for c in res.candidates))
    check("hard filters were applied", res.rejected > 0,
          f"rejected {res.rejected}")

    # 6b. the pool grows until enough candidates have *survived*, not merely
    # been proposed -- a tight brief used to return 2 of 4 requested
    for txt, want in (("design 6 amines", 6),
                      ("设计8个含羧基的分子", 8),
                      ("design 10 molecules with logP below 1 and TPSA above 60", 10)):
        bb = parse_brief(txt, client=None)
        bb.n_candidates = want
        bb.dft_top = 0
        rr = design(bb, client=StubClient([]), seed=4, do_dft=False)
        check(f"'{txt[:26]}' returns {want} candidates",
              len(rr.candidates) == want,
              f"{len(rr.candidates)} of {want}")

    # 6c. the run has to *design* something.  Survivors that came straight
    # from the library are not designed molecules, and counting them let a
    # loose brief stop before the generative stage ever ran: "design 6
    # molecules with a wide HOMO-LUMO gap" returned six library entries and
    # produced nothing new at all.
    loose = parse_brief("design 6 molecules with logP below 2", client=None)
    loose.n_candidates = 6
    loose.dft_top = 0
    loose_run = design(loose, client=None, seed=17, do_dft=False)
    check("a loose brief still generates candidates",
          sum(v for k, v in loose_run.origins.items() if k != "seed") > 0,
          str(loose_run.origins))

    # 6d. a target that only a calculation can supply cannot be scored before
    # that calculation exists, and 0.000 there reads as "fails every
    # criterion" -- a different and untrue statement.
    gap_only = parse_brief(
        "design 4 molecules with a wide HOMO-LUMO gap", client=None)
    gap_only.n_candidates = 4
    gap_only.dft_top = 0
    gap_run = design(gap_only, client=None, seed=19, do_dft=False)
    check("a brief only a calculation can judge is marked unscored",
          all(c["scoring"].get("scored") is False
              for c in gap_run.candidates),
          str([c["scoring"].get("scored") for c in gap_run.candidates]))
    check("and says why, so a zero is never unexplained",
          all(c["scoring"].get("unscored_reason")
              for c in gap_run.candidates),
          str([c["scoring"].get("unscored_reason")
               for c in gap_run.candidates]))

    # 7. the stub client path is exercised end to end
    # Three amines that are not in the library, so all three are genuinely
    # new; a proposal that merely repeats a library seed is a duplicate and
    # must not be counted twice.
    stub = StubClient([
        json.dumps({"candidates": [
            {"name": " Monoethanolamine ", "smiles": "NCCO",
             "rationale": "A primary amine that chemisorbs CO2."},
            {"name": "Diethanolamine", "smiles": "OCCNCCO",
             "rationale": "Two hydroxyls raise the CO2 loading."},
            {"name": "N-Methyldiethanolamine", "smiles": "CN(CCO)CCO",
             "rationale": "A tertiary amine with low regeneration cost."},
        ]}),
    ])
    brief3 = parse_brief("design an amine for CO2 capture", client=None)
    brief3.n_candidates = 6
    brief3.dft_top = 0
    res3 = design(brief3, client=stub, seed=3, do_dft=False)
    check("stub model candidates entered the pool",
          res3.origins.get("model", 0) >= 2, str(res3.origins))
    check("model proposals are surfaced",
          any(c["origin"] == "model" for c in res3.candidates),
          str([c["origin"] for c in res3.candidates]))
    check("model names are kept", any("Monoethanolamine" in (c["name"] or "")
                                      for c in res3.candidates),
          str([c["name"] for c in res3.candidates][:4]))
    check("model rationale is kept",
          any("chemisorbs" in (c.get("rationale") or "")
              for c in res3.candidates))

    # a numeric brief must be ranked purely by score, origin notwithstanding
    numeric = parse_brief("logP below 1 and TPSA above 80", client=None)
    numeric.n_candidates = 6
    numeric.dft_top = 0
    res4 = design(numeric, client=StubClient([
        json.dumps({"candidates": [{"name": "Hexane", "smiles": "CCCCCC",
                                    "rationale": "very lipophilic"}]})]),
        seed=5, do_dft=False)
    order = [c["scoring"]["score"] for c in res4.candidates]
    check("numeric brief is ranked purely by score",
          all(a >= b for a, b in zip(order, order[1:])), str(order[:5]))
    check("numeric brief rejects hexane (logP target unmet)",
          all(c["smiles"] != "CCCCCC" for c in res4.candidates)
          or order[0] < 0.3, str(order[:2]))
    hexane = [c for c in res4.candidates if c["smiles"] == "CCCCCC"]
    check("hexane scores exactly zero, not a clamped near-zero",
          all(c["scoring"]["score"] == 0.0 for c in hexane),
          str([c["scoring"]["score"] for c in hexane]))

    # 8. every substructure the brief parser can demand must be reachable by
    # the generator.  This is the check that was missing: a uniform fragment
    # draw puts a sulfonamide in about 2% of products, and the pool-filling
    # loop stopped as soon as *any* twelve candidates had survived, so
    # "design 4 sulfonamides" returned two molecules that had one.
    wanted = 4
    unreachable: List[str] = []
    dishonest: List[str] = []
    for label, smarts in _REQUIRED_SUBSTRUCTURES:
        pattern = compile_smarts(smarts)
        if pattern is None:
            unreachable.append(f"{label}(unparsable)")
            continue
        probe = parse_brief("design molecules", client=None)
        probe.n_candidates = wanted
        probe.dft_top = 0
        probe.required_smarts = [smarts]
        run = design(probe, client=StubClient([]), seed=21, do_dft=False)
        if len(run.candidates) < wanted:
            unreachable.append(f"{label}({len(run.candidates)}/{wanted})")
            continue
        for cand in run.candidates:
            mol = safe_mol(cand["smiles"])
            if mol is None or not mol.HasSubstructMatch(pattern):
                dishonest.append(f"{label}:{cand['smiles']}")
    check("every demandable substructure can be built",
          not unreachable, ",".join(unreachable[:6]))
    check("every returned candidate really carries what was demanded",
          not dishonest, ",".join(dishonest[:4]))

    # 8b. a requirement nothing can build is reported as the cause, and the
    # message names it.  The old note blamed the molecular-weight limit.
    hopeless = parse_brief("design 4 molecules", client=None)
    hopeless.n_candidates = 4
    hopeless.dft_top = 0
    hopeless.required_smarts = ["[U]"]
    hopeless_run = design(hopeless, client=StubClient([]), seed=9,
                          do_dft=False)
    check("an unsatisfiable requirement is named, not guessed at",
          hopeless_run.unmet_requirements == ["[U]"],
          str(hopeless_run.unmet_requirements))
    check("the note blames the requirement, not an unrelated limit",
          any("[U]" in n and "emptied the pool" in n
              for n in hopeless_run.model_notes),
          str(hopeless_run.model_notes[-1:]))
    check("the rejection tally says why molecules were dropped",
          any("required substructure" in r
              for r in hopeless_run.rejection_reasons),
          str(hopeless_run.rejection_reasons))

    # 8c. quality filters: reported for every candidate, and enforceable
    quinone = safe_mol("O=C1C=CC(=O)C=C1")
    described = describe_molecule(quinone)
    check("the descriptor block carries the quality scores",
          described["values"].get("sa") is not None
          and described["values"].get("qed") is not None,
          str({k: described["values"].get(k) for k in ("sa", "qed")}))
    check("a known promiscuous molecule is flagged",
          described["alert_count"] > 0
          and "PAINS" in (described.get("alerts") or {}),
          str(described.get("alerts")))
    check("aspirin's alerts are reported too (BRENK phenol ester)",
          "BRENK" in (describe_molecule(
              safe_mol("CC(=O)Oc1ccccc1C(=O)O")).get("alerts") or {}))
    strict = DesignBrief(text="t", alert_filter="PAINS")
    check("an alert filter rejects a flagged molecule",
          violates(quinone, strict) is not None,
          str(violates(quinone, strict)))
    check("and leaves a clean molecule alone",
          violates(safe_mol("CCO"), strict) is None)
    check("no filter means no rejection",
          violates(quinone, DesignBrief(text="t")) is None)

    # 8c-bis. the reported list must keep the molecules that were measured.
    # The leaders are picked before they are calculated and the calculation
    # is the expensive part of the run; the post-measurement re-score used to
    # push a disappointing leader out of the list, so a minute of SCF time
    # produced nothing visible -- including, and worst of all, when the
    # measured number was the evidence that the target could not be met.
    def _fake(smi, score, measured=False, qualified_=True, origin="mutation"):
        cand = {"smiles": smi, "name": smi, "origin": origin,
                "qualified": qualified_, "dft": {"gap_ev": 1.0} if measured
                else None,
                "descriptors": {"values": {"sa": 3.0}, "alerts": {}},
                "scoring": {"score": score, "blocking": []}}
        return cand

    pool = [_fake("CC", 0.10, measured=True), _fake("CCC", 0.20),
            _fake("CCCC", 0.30), _fake("CCCCC", 0.40)]
    ranked_pool = _rank({c["smiles"]: c for c in pool}, DesignBrief(text="t"))
    cut, _m = _final_cut(ranked_pool, DesignBrief(text="t"), 3, False, False)
    check("a measured candidate is never displaced by an unmeasured one",
          any(c["dft"] for c in cut), str([c["smiles"] for c in cut]))
    check("but the list is still exactly the requested length", len(cut) == 3,
          str(len(cut)))
    check("inclusion is not promotion: the list stays a ranking",
          [c["smiles"] for c in cut] == ["CCCCC", "CCCC", "CC"],
          str([c["smiles"] for c in cut]))
    cut_all, _m2 = _final_cut(ranked_pool, DesignBrief(text="t"), 4, False,
                              False)
    check("and with room for everything the order is by score",
          [c["smiles"] for c in cut_all]
          == ["CCCCC", "CCCC", "CCC", "CC"],
          str([c["smiles"] for c in cut_all]))
    mixed = [_fake("N", 0.90, qualified_=False), _fake("O", 0.10)]
    mixed_ranked = _rank({c["smiles"]: c for c in mixed},
                         DesignBrief(text="t", required_smarts=["[#7]"]))
    cut_req, _m3 = _final_cut(mixed_ranked, DesignBrief(text="t"), 1, False,
                              True)
    check("a demanded substructure outranks a higher score",
          [c["smiles"] for c in cut_req] == ["O"],
          str([c["smiles"] for c in cut_req]))

    # 8d. the DFT cache round-trips and keys on the level of theory and the
    # relaxation, so a cached number can never be attributed to a different
    # calculation than the one that produced it.  It writes into the scratch
    # directory `_run_selftest` installed, never into the live cache.
    sample = {"energy_hartree": -154.0, "gap_ev": 7.5, "converged": True}
    _cache_write("probe", sample)
    check("the DFT cache round-trips", _cache_read("probe") == sample)
    _cache_write("probe", dict(sample, converged=False))
    check("a non-converged entry is never reused", _cache_read("probe") is None)
    keys = {
        hashlib.sha1(f"CCO|b3lyp|6-31g*|{flag}".encode()).hexdigest()
        for flag in (False, True)}
    check("the cache key separates relaxed from unrelaxed", len(keys) == 2)

    # 9. narration never crashes and never claims a measurement it lacks
    text = explain_design(res.to_dict())
    check("narration mentions the search size", "Search" in text)
    check("narration does not claim DFT when none ran",
          "measured at" not in text)
    req_run = hopeless_run.to_dict()
    req_text = explain_design(req_run)
    check("narration states the hard requirement and who met it",
          "Hard requirements" in req_text, req_text[:80])
    check("narration admits an unmet requirement",
          "Unmet requirement" in req_text)
    check("narration lists why candidates were dropped",
          "Why candidates were dropped" in req_text)
    check("narration names the quality filters that ran",
          "Quality filters applied" in req_text)

    # 9b. a named starting point is stated, because the analogues lead the
    # list and an ordering that does not rank by score looks broken otherwise
    named_text = explain_design(named_run.to_dict())
    check("narration states the molecule the brief named",
          "Starting point" in named_text
          and "Aspirin" in named_text, named_text[:160])
    check("and says the named molecule's analogues lead the list",
          "lead the list" in named_text)

    print()
    if failures:
        print(f"{len(failures)} FAILURES: " + ", ".join(failures))
        return 1
    print("designer self test: all checks passed")
    return 0


def _run_selftest() -> int:
    """Run the self test against a scratch directory, not the live one.

    The self test calls :func:`design` a few dozen times, and every run saves
    its result and prunes the history by unlinking the oldest files.  Pointing
    that at `data/designs` has two consequences, both bad: the gate leaves
    dozens of files in the user's design history, and the unlink calls can be
    refused on a restricted or audited filesystem -- which then aborts the
    gate in the middle with a file-permission error that has nothing to do
    with the code under test.  Both go away if the run has its own directory.
    """
    global DESIGN_DIR, DFT_CACHE_DIR
    import shutil
    import tempfile

    original = (DESIGN_DIR, DFT_CACHE_DIR)
    scratch = tempfile.mkdtemp(prefix="chatdft-selftest-")
    DESIGN_DIR = os.path.join(scratch, "designs")
    DFT_CACHE_DIR = os.path.join(scratch, "cache")
    try:
        return _selftest()
    finally:
        DESIGN_DIR, DFT_CACHE_DIR = original
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    # deliberately no bootstrap: this self test is offline and must run
    # wherever RDKit does, whether or not the compiled PySCF libs are present
    raise SystemExit(_run_selftest())
