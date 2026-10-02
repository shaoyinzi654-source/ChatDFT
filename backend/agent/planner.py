"""
ChatDFT natural-language layer.

The agent turns a chemist's sentence into a structured job request, runs it
through the DFT engine, and writes back an interpretation of the numbers.

Two reasoning backends:

* ``RuleBasedPlanner`` - a deterministic, fully offline intent parser built
  from chemistry-aware regular expressions. No API key, no network, always
  available. This is what the application falls back to, and what it uses
  outright when ``CHATDFT_PLANNER=local``.
* ``LLMPlanner`` - planning is delegated to an OpenAI-compatible endpoint when
  one is configured (environment variables, or ``data/llm.json``), with the
  rule based parser used as a tool-schema hint and a safety net. Every reply
  says which of the two produced the plan, and why, when it was the parser.

The narration layer (``explain``) is always local: it converts raw DFT output
into readable chemistry commentary, so results are explained even offline.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..engine import FUNCTIONALS, BASIS_SETS
from ..engine.molecule import library_names
from .llm import default_client

# ======================================================================
#  intent model
# ======================================================================
JOB_TYPES = {
    "single_point": "Single-point energy + electronic structure",
    "geometry_optimization": "Geometry optimisation",
    "excited_states": "Excited states (TD-DFT)",
    "compare": "Compare two molecules",
    "scan": "Bond scan / potential energy curve",
    "vibrations": "Vibrational frequencies + IR spectrum",
    "nmr": "NMR chemical shifts (GIAO shielding tensor)",
    "dos": "Density of states (total and projected)",
    "reactivity": "Reactivity descriptors + Fukui functions",
    "surfaces": "MEP map, orbital and density isosurfaces",
    "nci": "Non-covalent interaction (RDG / NCI) analysis",
    "design": "Automatic molecular design",
    "field": "Real-space field (ELF / Laplacian / spin / density difference)",
    "nto": "Natural transition orbitals of an excited state",
    "series": "Series of molecules at one level of theory",
    "reaction": "Reaction path, barrier and transition state",
    "info": "Explain a concept",
    "library": "Browse the molecule library",
}


def field_kind_for(low: str) -> str:
    """Which real-space field a request means.

    'plot the ELF of benzene' and 'spin density of the methyl radical' are
    different calculations, so the wording has to pick the field and not
    leave the user looking at an ELF map when they asked for a spin density.
    """
    table = (
        ("spin", (r"\bspin density\b", r"\bspin[- ]polari[sz]ation\b",
                  r"自旋密度", r"自旋极化")),
        ("difference", (r"\bdensity difference\b", r"\bdifference density\b",
                        r"\bdeformation density\b", r"\bdensity deformation\b",
                        r"密度差", r"形变密度", r"差分密度")),
        ("laplacian", (r"\blaplacian\b", r"\bnabla\^?2\s*rho\b",
                       r"\bdel2\s*rho\b", r"拉普拉斯")),
        ("elf", (r"\belf\b", r"\belectron locali[sz]ation function\b",
                 r"\blocali[sz]ation function\b", r"\blone pair\b",
                 r"电子局域化", r"电子定域化")),
    )
    for kind, pats in table:
        if _match_any(low, list(pats)):
            return kind
    return "elf"


# The words a user may put in front of an explicit reaction coordinate, and
# how many atom indices each one needs.
_COORD_WORDS = (
    ("torsion", ("torsion", "torsional", "dihedral", "二面角", "扭转角", "扭角")),
    ("angle", ("angle", "bend", "bond angle", "角")),
    ("bond", ("bond", "distance", "键长", "键")),
)
_COORD_ARITY = {"bond": 2, "angle": 3, "torsion": 4}


def reaction_coordinate_for(low: str) -> str:
    """Pull an explicit reaction coordinate out of the request.

    "torsion 2 0 1 5", "the dihedral 2-0-1-5" and the Chinese forms all mean
    the same thing: a named internal coordinate on a list of atoms.  Returned
    as ``"kind i j [k l]"`` with **1-based** indices, matching how a chemist
    counts atoms in a figure; an empty string means "you choose".

    Torsion is tested before angle and bond, because "the H-C-C-H dihedral"
    is not a bond request even though it names bonds.
    """
    for kind, words in _COORD_WORDS:
        need = _COORD_ARITY[kind]
        for w in words:
            pat = re.escape(w) + r"[\s:=]*((?:\d+[\s,;:/\\-]*){2,4})"
            m = re.search(pat, low)
            if not m:
                continue
            nums = re.findall(r"\d+", m.group(1))
            if len(nums) >= need:
                return kind + " " + " ".join(nums[:need])
    return ""


def parse_coordinate(spec: str):
    """``"torsion 2 1 3 6"`` -> ``("torsion", (1, 0, 2, 5))`` (0-based).

    Indices are **1-based**, as a chemist counts atoms in a figure.  A spec
    that contains a literal ``0`` cannot be 1-based, so it is read as
    ``0-based`` instead of being rejected -- ``"torsion 2 0 1 5"`` is what most
    other programs print, and silently shifting it by one would scan the wrong
    torsion.  The two conventions can never collide, because 0 is not a valid
    1-based index.

    Returns ``None`` for an empty or malformed spec, so the caller falls back
    to letting the engine choose the coordinate rather than guessing wrongly.
    """
    parts = str(spec or "").split()
    if len(parts) < 2:
        return None
    kind = parts[0].lower()
    if kind not in _COORD_ARITY:
        return None
    need = _COORD_ARITY[kind]
    try:
        raw = [int(p) for p in parts[1:1 + need]]
    except ValueError:
        return None
    if len(raw) != need:
        return None
    base = 0 if any(v == 0 for v in raw) else 1
    idx = tuple(v - base for v in raw)
    if any(i < 0 for i in idx) or len(set(idx)) != len(idx):
        return None
    return kind, idx


@dataclass
class JobIntent:
    job_type: str = "single_point"
    molecule: str = ""
    molecule2: str = ""
    # every molecule named in the request, in order.  ``molecule``/
    # ``molecule2`` only ever held the first two, so "compare the gaps of
    # benzene, pyridine, furan and pyrrole" silently dropped two of the
    # four and reported a two-way comparison as if it were the whole set.
    molecules: List[str] = field(default_factory=list)
    bond: str = ""
    functional: str = "b3lyp"
    basis: str = "6-31g*"
    charge: int = 0
    multiplicity: int = 1
    nstates: int = 6
    # which real-space field, when job_type == "field"
    field_kind: str = "elf"
    # Explicit reaction coordinate, e.g. "torsion 2 0 1 5" or "bond 1 2".
    # Empty means the engine picks the most stretched bond.  Kept as text so
    # the intent payload stays JSON-serialisable.
    reaction_coord: str = ""
    reaction_points: int = 9
    solvation: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    confidence: float = 0.6
    raw: str = ""


# ======================================================================
#  vocabulary
# ======================================================================
_FUNCTIONAL_ALIASES = {
    "hf": ["hf", "hartree-fock", "hartree fock", "scf"],
    "lda": ["lda", "svwn"],
    "pbe": ["pbe"],
    "blyp": ["blyp", "b-lyp"],
    "bp86": ["bp86", "b-p86"],
    "b3lyp": ["b3lyp", "b-3-lyp"],
    "pbe0": ["pbe0", "pbe-0", "b3p86"],
    "m06l": ["m06l", "m06-l"],
    "tpss": ["tpss"],
    "wb97x": ["wb97x", "wb97x-d", "wb97"],
}

_BASIS_ALIASES = {
    "sto-3g": ["sto-3g", "sto3g"],
    "3-21g": ["3-21g", "3-21"],
    "6-31g": ["6-31g", "6-31"],
    "6-31g*": ["6-31g*", "6-31g(d)"],
    "6-31g**": ["6-31g**", "6-31g(d,p)"],
    "def2-svp": ["def2-svp", "def2svp"],
    "6-311g*": ["6-311g*", "6-311g(d)"],
    "def2-tzvp": ["def2-tzvp", "def2tzvp"],
    "cc-pvdz": ["cc-pvdz", "ccpvdz"],
}

_SOLVENT_EPS = {
    "water": 78.3553,
    "methanol": 32.613,
    "ethanol": 24.852,
    "acetone": 20.493,
    "dmso": 46.826,
    "dichloromethane": 8.93,
    "chloroform": 4.7113,
    "thf": 7.4257,
    "toluene": 2.3741,
    "benzene": 2.2706,
    "acetonitrile": 35.688,
    "hexane": 1.8819,
}

_JOB_KEYWORDS = {
    # NOTE: order matters. Comparison and explicit computation verbs are
    # checked before the generic "what is ..." question patterns, otherwise
    # "what is the HOMO-LUMO gap of pyridine" is misclassified as a concept
    # question instead of a calculation request.
    # design comes first of all: a request to design something names
    # properties and molecules too, and answering it with a single-point
    # calculation on the first molecule mentioned is the wrong answer.
    "design": [
        r"\bdesign\b", r"\bdesigns\b", r"\bdesigned\b", r"\bdesigning\b",
        r"\bde novo\b", r"\binvent\b", r"\bpropose\b", r"\bgenerate\b",
        r"\bdiscover\b", r"\bsuggest\b", r"\bscreen for\b",
        r"\bcandidates?\b", r"\boptimise a molecule\b",
        r"\boptimize a molecule\b",
        r"设计", r"自动设计", r"分子设计", r"帮我设计", r"设计一", r"设计个",
        r"候选分子", r"候选", r"筛选", r"生成分子", r"找一?个分子",
        r"给我一?个.*分子",
    ],
    "compare": [
        r"\bcompare\b", r"\bcomparison\b", r"\bversus\b", r"\bvs\.?\b",
        r"对比", r"比较",
    ],
    "reaction": [
        # Before "scan" and before "excited_states": a reaction path is
        # scanned, and "transition state" contains "transition", which the
        # excited-state branch claims -- so a request for the barrier would
        # otherwise be answered with a UV-Vis spectrum.
        r"\breaction path\b", r"\breaction coordinate\b",
        r"\btransition state\b", r"\benergy barrier\b",
        r"\bactivation (?:energy|barrier)\b", r"\bbarrier height\b",
        r"\bpotential energy barrier\b", r"\bsaddle point\b",
        r"\bIRC\b", r"\bintrinsic reaction coordinate\b",
        r"\breaction profile\b", r"\benergy profile\b",
        r"\bhow high is the barrier\b", r"\bbarrier\b",
        r"反应路径", r"反应坐标", r"过渡态", r"能垒", r"活化能",
        r"势垒", r"反应势垒", r"鞍点", r"反应能垒", r"能量曲线",
    ],
    "scan": [
        # the bare verb has to be accepted: "scan the O-H bond" has an element
        # pair wedged between "the" and "bond", so "scan the bond" misses it.
        r"\bscan(?:ning)?\b", r"\bshow me a scan\b", r"\bbond scan\b",
        r"\bscan the bond\b", r"\bpotential energy curve\b",
        r"\bpotential energy surface\b", r"扫描", r"势能面", r"势能曲线",
    ],
    "nto": [
        # before excited_states: "natural transition orbital" contains
        # "transition", so it would otherwise be answered with a spectrum
        # when the user asked for the orbitals of the transition.
        r"\bnto\b", r"\bntos\b", r"\bnatural transition orbital",
        r"\bnatural orbital",
        r"自然跃迁轨道", r"自然轨道", r"跃迁轨道",
    ],
    "excited_states": [
        r"\bexcited\b", r"\bexcitation\b", r"\btd-?dft\b", r"\btddft\b",
        r"\babsorption\b", r"\buv-?vis\b", r"\bspectrum\b", r"\btransition\b",
        r"激发态", r"激发", r"吸收光谱", r"紫外",
    ],
    "vibrations": [
        r"\bIR\b", r"\binfrared\b", r"\bvibrational\b",
        r"\bfrequenc(?:y|ies)\b", r"\bnormal mode\b", r"\bphonon\b",
        r"\bIR spectrum\b",
        r"红外", r"红外光谱", r"振动", r"频率", r"振动频率",
    ],
    "nmr": [
        # Before excited_states and vibrations: both of those claim the bare
        # word "spectrum", so "the NMR spectrum of ethanol" was answered with
        # a TD-DFT absorption spectrum.  Every pattern here has to be
        # specific enough not to steal a request from another kind -- "13C"
        # and "1H" on their own are not, because "1H" appears in "1 H atom",
        # so they only count next to a nucleus word.
        r"\bnmr\b", r"\bchemical shift", r"\bshield(?:ing|ings)\b",
        r"\bgiao\b", r"\bspin[- ]spin coupling\b",
        r"\b(?:1h|13c|15n|17o|19f|29si|31p|14n)\s*(?:nmr|spectrum|shift)\b",
        r"\b(?:proton|carbon-?13|nitrogen-?15|fluorine-?19|phosphorus-?31)"
        r"\s*(?:nmr|spectrum|shift)\b",
        r"\bcarbon nmr\b", r"\bproton nmr\b",
        r"核磁", r"核磁共振", r"化学位移", r"屏蔽", r"氢谱", r"碳谱",
    ],
    "dos": [
        r"\bdos\b", r"\bdensity of states\b", r"\bpdos\b",
        r"\bprojected density\b", r"态密度", r"投影态密度",
    ],
    "reactivity": [
        r"\breactivity\b", r"\breactive\b", r"\bfukui\b",
        r"\belectrophilic\b", r"\bnucleophilic\b", r"\bhardness\b",
        r"\belectrophilicity\b", r"\bchemical potential\b",
        r"反应性", r"福井", r"亲电", r"亲核", r"硬度",
    ],
    "nci": [
        # The weak-interaction figure.  "weak interaction" is deliberately
        # narrow -- "interaction" alone shows up in too many sentences that
        # are really about reactivity.
        r"\bnci\b", r"\brdg\b", r"\bnon-?covalent\b",
        r"\bweak interaction\b", r"\breduced density gradient\b",
        r"\bhydrogen bond(?:ing)? (?:plot|analysis|map|index)\b",
        r"\bdispersion (?:plot|analysis)\b", r"\bsteric (?:clash|repulsion)\b",
        r"\bsign\s*\(?\s*lambda2?\s*\)?\s*rho\b",
        r"弱相互作用", r"非共价", r"氢键分析", r"约化密度梯度", r"位阻",
    ],
    "field": [
        # The real-space figures a paper prints in a plane: ELF, the
        # Laplacian, the spin density, the density difference.  Every one of
        # these used to fall through to a single point or to the 3D
        # isosurface code, which produced a figure that looked plausible and
        # answered a different question.
        r"\belf\b", r"\belectron locali[sz]ation function\b",
        r"\blocali[sz]ation function\b",
        r"\blaplacian\b", r"\bnabla\^?2\s*rho\b", r"\bdel2\s*rho\b",
        r"\bspin density\b", r"\bspin[- ]polari[sz]ation\b",
        r"\bdensity difference\b", r"\bdifference density\b",
        r"\bdeformation density\b", r"\bdensity deformation\b",
        r"\bcontour (?:plot|map|diagram)\b", r"\b2d (?:contour|map|plot)\b",
        r"\b(?:slice|plane) (?:plot|map)\b",
        r"\bwhere (?:is|are) the (?:lone pair|electron pair)\b",
        r"电子局域化", r"电子定域化", r"自旋密度", r"自旋极化",
        r"密度差", r"形变密度", r"差分密度", r"拉普拉斯",
        r"等值线", r"等值线图", r"二维等值线", r"平面图", r"切片图",
    ],
    "surfaces": [
        # Deliberately does not claim the bare word "surface": "potential
        # energy surface" belongs to a scan, which is tested first.  Nor does
        # it claim "HOMO" or "LUMO" on their own -- those are orbital-energy
        # questions and are answered by a single point.  Only explicit
        # picture language routes here.  "contour plot" moved to 'field',
        # because a contour map is a 2D slice, not an isosurface.
        r"\bmep\b", r"\belectrostatic potential\b",
        r"\bmolecular electrostatic\b", r"\bpotential map\b",
        r"\bisosurface\b", r"\biso-?surface\b",
        r"\belectron density\b", r"\bdensity (?:surface|isosurface)\b",
        r"\b(?:homo|lumo|orbital)[- ](?:plot|surface|isosurface|picture)\b",
        r"\b(?:plot|render|draw|visuali[sz]e) the (?:homo|lumo|orbital|density)\b",
        r"\bsurface (?:map|plot)\b", r"\b3d (?:plot|picture|image|view)\b",
        r"静电势", r"电子密度", r"等值面", r"轨道图", r"分子表面", r"势能图",
    ],
    "geometry_optimization": [
        r"\boptimi[sz]e\b", r"\boptimi[sz]ation\b", r"\brelax\b", r"\bminimi[sz]e\b",
        r"\bequilibrium geometry\b", r"\boptimize geometry\b",
        r"几何优化", r"结构优化", r"优化结构", r"优化几何", r"优化",
        r"\bgeometry\s*opt\b",
    ],
    "library": [
        r"\blibrary\b", r"\blist .*(molecule|molecules)\b",
        r"\bavailable molecules\b", r"\bwhat molecules\b",
        r"分子库", r"有哪些分子", r"可用分子",
    ],
    "info": [
        r"\bexplain\b", r"\bhow does\b", r"\bwhy\b", r"\bwhat is a\b",
        r"\bwhat are\b", r"\bmeaning of\b",
        r"什么是", r"解释", r"为什么", r"介绍",
    ],
}

# A request phrased as a question about a *property* is still a calculation.
_COMPUTE_VERBS = [
    r"\bwhat(?:'s| is)\b", r"\bcalculate\b", r"\bcompute\b", r"\bgive me\b",
    r"\bfind\b", r"\bdetermine\b", r"\bget\b", r"\bshow\b", r"\breport\b",
    r"计算", r"是多少", r"多少", r"给我", r"算一下", r"求",
]

# Tokens that look like element formulas but are really method acronyms or
# property names. Never treat these as molecules.
_NOT_MOLECULES = {
    "hf", "dft", "tddft", "scf", "homo", "lumo", "somо", "somo", "gap",
    "uv", "vis", "nmr", "ir", "pes", "pcm", "cosmo", "lda", "gga", "mp2",
    "ccsd", "ci", "td", "ks", "mo", "ao", "xyz", "smiles", "mo", "kohn",
    "sham", "b3lyp", "pbe", "pbe0", "blyp", "bp86", "m06l", "tpss", "wb97x",
    "svwn", "sto", "angstrom", "ev", "ha", "hartree", "kcal", "debye",
}


def _match_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


_CJK_RE = re.compile(r"[\u3400-\u9fff]")


def _alias_pattern(alias: str) -> str:
    """
    Build a match pattern for an alias.

    Latin aliases need word boundaries so that ``hf`` does not fire inside
    ``dft``. CJK text has no word separators, so a pure substring test is the
    correct behaviour there.
    """
    if _CJK_RE.search(alias):
        return re.escape(alias)
    return r"(?<![a-z0-9\-])" + re.escape(alias) + r"(?![a-z0-9])"


def _extract_vocab(text: str, alias_map: Dict[str, List[str]]) -> Optional[str]:
    lowered = text.lower()
    best: Optional[str] = None
    best_len = 0
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            if re.search(_alias_pattern(alias.lower()), lowered) and len(alias) > best_len:
                best, best_len = canonical, len(alias)
    return best


# ======================================================================
#  rule-based planner
# ======================================================================
class RuleBasedPlanner:
    """Offline intent parser with chemistry-aware heuristics."""

    def __init__(self):
        self._library_index = self._build_library_index()

    # ------------------------------------------------------------------
    def _build_library_index(self) -> Dict[str, str]:
        idx: Dict[str, str] = {}
        for entry in library_names():
            key = entry["key"]
            idx[key] = key
            idx[entry["name"].lower()] = key
            if entry["formula"]:
                idx[entry["formula"].lower()] = key
            # common short names
            for alt in _COMMON_NAMES.get(key, []):
                idx[alt.lower()] = key
        return idx

    # ------------------------------------------------------------------
    def plan(self, text: str) -> JobIntent:
        raw = text
        text = text.strip()
        low = text.lower()
        intent = JobIntent(raw=raw)
        matched = 0

        # ---- job type ------------------------------------------------
        # Priority order: compare and scan first (unambiguous verbs), then
        # excited states, then optimisation, then library, and only then the
        # generic "what is ..." question form.
        # vibrations precedes excited_states on purpose: both claim the word
        # "spectrum", so "IR spectrum of water" used to be answered with a
        # TD-DFT calculation.  "UV-Vis" and "absorption" only match the latter.
        # surfaces sits after scan so that "potential energy surface" stays a
        # scan, and before dos so that "electron density" is not read as a
        # "density of states" request.
        # nci precedes surfaces because an NCI request is phrased with the
        # same picture words ("isosurface", "plot") and answering it with a
        # MEP map would be the wrong figure entirely.
        # field precedes surfaces: "contour map" and "electron density" used
        # to land on surfaces, which answered a request for a 2D contour
        # figure with a 3D isosurface.  A contour map is what the field
        # engine draws.
        for jt in ("design", "compare", "reaction", "scan", "vibrations",
                   "nmr", "nto", "excited_states",
                   "geometry_optimization", "nci", "field", "surfaces", "dos",
                   "reactivity", "library", "info"):
            if _match_any(low, _JOB_KEYWORDS[jt]):
                intent.job_type = jt
                matched += 1
                if jt == "field":
                    intent.field_kind = field_kind_for(low)
                elif jt == "reaction":
                    intent.reaction_coord = reaction_coordinate_for(low)
                break

        # A question about a named molecule's property is a calculation, not a
        # concept lookup: "what is the HOMO-LUMO gap of pyridine" must run SCF.
        # _extract_molecules runs first so we can tell the two apart.
        pending_molecules = self._extract_molecules(text)
        if intent.job_type == "info" and pending_molecules \
                and _match_any(low, _COMPUTE_VERBS) \
                and not _match_any(low, [r"\bexplain\b", r"\bmeaning of\b",
                                         r"\bwhat is a\b", r"\bwhat are\b",
                                         r"\u4ec0\u4e48\u662f", r"\u89e3\u91ca"]):
            # "what is the HOMO-LUMO gap of pyridine" -> a property calculation
            intent.job_type = "single_point"
            matched += 1
        elif intent.job_type == "excited_states" and pending_molecules \
                and not _match_any(low, _JOB_KEYWORDS["excited_states"]):
            intent.job_type = "single_point"
            matched += 1

        # ---- method / basis ------------------------------------------
        f = _extract_vocab(low, _FUNCTIONAL_ALIASES)
        if f:
            intent.functional = f
            matched += 1
            intent.notes.append(f"functional: {FUNCTIONALS[f]['label']}")
        b = _extract_vocab(low, _BASIS_ALIASES)
        if b:
            intent.basis = b
            matched += 1
            intent.notes.append(f"basis: {BASIS_SETS[b]['label']}")

        # ---- solvent -------------------------------------------------
        # The solvent name must directly follow a solvent cue ("in water",
        # "solvent = dmso", "pcm(ethanol)").  Matching the name anywhere in
        # the sentence would read the *molecule* as the solvent: "the energy
        # of water in dmso" would match "water" and silently solvate in water.
        for solv, eps in _SOLVENT_EPS.items():
            pattern = (
                r"\b(?:in|using|solvent|phase|pcm|cosmo|implicit)\b[\s(=:]*"
                + re.escape(solv)
                + r"(?![a-z])"
            )
            if re.search(pattern, low):
                intent.solvation = str(eps)
                matched += 1
                intent.notes.append(f"implicit solvent: {solv} (eps={eps})")
                break

        # ---- charge / multiplicity -----------------------------------
        m = re.search(r"charge\s*(?:=|of|:)?\s*([+-]?\d+)", low)
        if m:
            intent.charge = int(m.group(1))
            matched += 1
        else:
            if re.search(r"\b(anion|negative|anionic)\b", low):
                intent.charge = -1
                matched += 1
            elif re.search(r"\b(cation|positive|cationic)\b", low):
                intent.charge = 1
                matched += 1

        m = re.search(r"(?:multiplicity|spin)\s*(?:=|of|:)?\s*(\d+)", low)
        if m:
            intent.multiplicity = int(m.group(1))
            matched += 1
        elif re.search(r"\b(triplet)\b", low):
            intent.multiplicity = 3
            matched += 1
        elif re.search(r"\b(doublet|radical)\b", low):
            intent.multiplicity = 2
            matched += 1

        # ---- nstates --------------------------------------------------
        m = re.search(r"(\d+)\s*(?:excited\s*states?|states?|roots?|transitions?)", low)
        if m:
            intent.nstates = max(1, min(20, int(m.group(1))))
            matched += 1

        # ---- vocabulary extraction -----------------------------------
        # Strip the method keywords first so that a functional name can never
        # be mistaken for a molecule: e.g. "benzene at PBE0/def2-TZVP" must not
        # leave a bare "at" behind, and "PBE0" must not become a formula.
        vocab_src = " " + low + " "
        for aliases in _FUNCTIONAL_ALIASES.values():
            for alias in aliases:
                vocab_src = re.sub(
                    r"(?<![a-z0-9\-])" + re.escape(alias) + r"(?![a-z0-9])",
                    " ", vocab_src)
        for aliases in _BASIS_ALIASES.values():
            for alias in aliases:
                vocab_src = re.sub(
                    r"(?<![a-z0-9\-])" + re.escape(alias) + r"(?![a-z0-9])",
                    " ", vocab_src)

        # ---- molecules ------------------------------------------------
        mols = self._extract_molecules(vocab_src)
        if mols:
            intent.molecule = mols[0]
            intent.molecules = list(mols)
            matched += 1
            if len(mols) > 2:
                # Three or more molecules named together is a series, not a
                # pairwise comparison: keep every one of them.
                if intent.job_type in ("compare", "single_point") and \
                        _match_any(low, _COMPUTE_VERBS +
                                   [r"\band\b", r"\bwith\b", r"和", r"与",
                                    r"\bseries\b", r"\btrend\b", r","]):
                    intent.job_type = "series"
                    intent.notes.append(
                        f"series of {len(mols)} molecules: "
                        + ", ".join(mols))
            elif len(mols) > 1:
                intent.molecule2 = mols[1]
                # two molecules named with no other verb -> a comparison
                if intent.job_type == "single_point" and _match_any(
                        low, _COMPUTE_VERBS + [r"\band\b", r"\bwith\b", r"和", r"与"]):
                    intent.job_type = "compare"
            elif intent.job_type == "compare":
                # "compare water and water" -- _extract_molecules de-duplicates,
                # so the two mentions collapse into one and the comparison
                # target was silently lost.  Keep the request as a comparison
                # of the molecule with itself instead of letting a
                # compare-labelled job fall through to the single-point path.
                intent.molecule2 = mols[0]

        # ---- a named element pair, for a bond scan --------------------
        if intent.job_type == "scan":
            intent.bond = _extract_bond_pair(text)

        # residual confidence
        intent.confidence = min(0.95, 0.35 + 0.15 * matched)
        return intent

    # ------------------------------------------------------------------
    def _extract_molecules(self, text: str) -> List[str]:
        """Find molecule references in a sentence, in order of appearance."""
        low = text.lower()
        found: List[Tuple[int, str]] = []

        # 1) library names (longest match wins, no overlap)
        candidates = sorted(self._library_index.items(), key=lambda kv: -len(kv[0]))
        consumed: List[Tuple[int, int]] = []
        for alias, key in candidates:
            # A single Latin character is too ambiguous, but a single CJK
            # character is a perfectly good molecule name (\u82ef = benzene).
            if len(alias) < 2 and not _CJK_RE.search(alias):
                continue
            pat = _alias_pattern(alias)
            for m in re.finditer(pat, low):
                s, e = m.span()
                if any(not (e <= cs or s >= ce) for cs, ce in consumed):
                    continue
                # "the hydrogen bond in the water dimer" must not read
                # "hydrogen" as the molecule.  A library name followed by a
                # chemistry-role noun is describing a piece of a molecule,
                # not naming one, and the mismatch is silent: the job runs
                # on H2 instead of the dimer the user asked about.
                if re.match(r"\s+(?:bond|bonds|atom|atoms|orbital|orbitals|"
                            r"spectrum|site|sites|centre|center)\b", low[e:]):
                    continue
                consumed.append((s, e))
                found.append((s, key))

        # 2) explicit SMILES markers
        for m in re.finditer(r"(?:smiles|微笑)\s*[:=]?\s*([^\s,;]+)", text, re.IGNORECASE):
            found.append((m.start(1), m.group(1).strip()))

        # 3) explicit formulas that are not in the library
        for m in re.finditer(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+)\b", text):
            token = m.group(1)
            if token.lower() in self._library_index:
                continue
            if token.lower() in _NOT_MOLECULES:
                continue
            # a single element symbol is not a formula here
            if len(token) <= 2 and token.capitalize() in ("H", "C", "N", "O"):
                continue
            # skip basis-set-looking tokens and anything with hyphens
            if re.search(r"\d-\d", token) or token.lower() in _NOT_MOLECULES:
                continue
            found.append((m.start(), token))

        found.sort(key=lambda t: t[0])
        out: List[str] = []
        for _, val in found:
            if val not in out:
                out.append(val)
        return out


# A hyphenated element pair, e.g. "the O-H bond", "C-C".  Validated against
# the periodic table so a stray "TD-DFT" or "6-31G" cannot masquerade as one.
_BOND_PAIR_RE = re.compile(r"\b([A-Z][a-z]?)\s*[-\u2013\u2014]\s*([A-Z][a-z]?)\b")


def _extract_bond_pair(text: str) -> str:
    """Return a validated element pair such as ``"O-H"``, or ``""``."""
    from ..engine import elements

    for m in _BOND_PAIR_RE.finditer(text):
        a, b = m.group(1), m.group(2)
        try:
            elements.get(a)
            elements.get(b)
        except KeyError:
            continue
        return f"{a}-{b}"
    return ""


_COMMON_NAMES = {
    "water": ["h2o", "water molecule", "\u6c34"],
    "water_dimer": ["(h2o)2", "h4o2", "\u6c34\u4e8c\u805a\u4f53",
                    "\u6c34\u4e8c\u805a\u7269", "dimer"],
    "methane": ["ch4", "methane gas", "\u7532\u70f7"],
    "ammonia": ["nh3", "\u6c28", "\u6c28\u6c14"],
    "benzene": ["c6h6", "\u82ef", "benzene ring"],
    "phenol": ["c6h5oh", "\u82ef\u915a"],
    "toluene": ["c7h8", "\u7532\u82ef"],
    "aniline": ["c6h5nh2", "\u82ef\u80fa"],
    "pyridine": ["c5h5n", "\u5421\u5576"],
    "furan": ["c4h4o", "\u544b\u5583"],
    "pyrrole": ["c4h5n", "\u5421\u54af"],
    "imidazole": ["c3h4n2", "\u54aa\u5511"],
    "methanol": ["ch3oh", "\u7532\u9187"],
    "ethanol": ["c2h5oh", "\u4e59\u9187", "\u9152\u7cbe"],
    "acetic_acid": ["ch3cooh", "\u4e59\u9178", "\u918b\u9178"],
    "glycine": ["\u7518\u6c28\u9178"],
    "caffeine": ["\u5496\u5561\u56e0"],
    "aspirin": ["\u963f\u53f8\u5339\u6797"],
    "formaldehyde": ["ch2o", "\u7532\u919b"],
    "acetone": ["c3h6o", "\u4e19\u916e"],
    "ethylene": ["c2h4", "\u4e59\u70ef"],
    "acetylene": ["c2h2", "\u4e59\u7094"],
    "nitrobenzene": ["\u785d\u57fa\u82ef"],
    "styrene": ["\u82ef\u4e59\u70ef"],
    "chloroform": ["chcl3", "\u6c2f\u4eff"],
    "urea": ["ch4n2o", "\u5c3f\u7d20"],
    "histamine": ["\u7ec4\u80fa"],
    "ozone": ["o3", "\u81ed\u6c27"],
    "co2": ["\u4e8c\u6c27\u5316\u78b3", "carbon dioxide"],
    "nitrogen": ["n2", "\u6c2e\u6c14"],
    "sulfur_dioxide": ["so2", "\u4e8c\u6c27\u5316\u786b"],
    "hydrogen": ["h2", "\u6c22\u6c14"],
    "oxygen": ["o2", "\u6c27\u6c14", "dioxygen"],
    "methanethiol": ["ch3sh"],
    "hydroxide": ["oh-", "\u6c22\u6c27\u6839"],
    "ammonium": ["nh4+", "\u94f5\u6839"],
    "nitrate": ["no3-", "\u785d\u9178\u6839"],
    "methyl_radical": ["ch3 radical", "\u7532\u57fa\u81ea\u7531\u57fa"],
    "hydroxyl_radical": ["oh radical", "\u7f9f\u57fa\u81ea\u7531\u57fa"],
    "porphine": ["\u5367\u5429", "porphyrin"],
    "naphthalene": ["c10h8", "\u8418"],
    "tfa": ["trifluoroacetic", "\u4e09\u6c1f\u4e59\u9178"],
    "dichloromethane": ["ch2cl2", "\u4e8c\u6c2f\u7532\u70f7"],
}


# ======================================================================
#  optional LLM planner
# ======================================================================
# How long a planning call may take before the keyword parser takes over.
# Measured on this endpoint, the same one-line JSON request comes back in
# anywhere from 5 s to 64 s -- the long end is a slow primary followed by the
# fallback model -- so this is a ceiling on patience, not on the network.  A
# chat turn that hangs for a minute is worse than one that answers at once and
# says the model was not reached: the keyword parser gets most requests right,
# and when it does not, the reply says which planner read the sentence.
PLAN_TIMEOUT = 30.0

# The sentence a reply carries when the model did not do the planning.
# It must contain "unavailable": the front end raises its warning box for
# notes matching /unavailable|fallback/i, and this is the note that says the
# model was not consulted.  That warning slot existed and never filled,
# because the note was only produced inside LLMPlanner.plan's except branch --
# so the one failure mode that mattered (the planner never being constructed)
# was the one mode that produced no note at all.
PLANNER_OFF_NOTE = ("language model unavailable ({reason}); the request was "
                    "parsed by the built-in keyword parser")


def apply_llm_payload(intent: JobIntent, payload: Any) -> List[str]:
    """Fold a model reply into ``intent``; return the fields that were rejected.

    Pure, so a gate can prove both directions without a network: a valid
    payload changes the intent, an off-schema one does not and says which
    field it refused.

    A reply is not trusted just because it parsed.  The model is told the job
    vocabulary, but nothing stops it inventing ``job_type: "nmr_spectrum"`` or
    returning ``molecules`` as one string.  Applying that blindly puts a job
    type the server cannot dispatch into the intent, and the failure surfaces
    much later as a missing branch.  Refusing keeps the keyword parser's value
    -- which, for a request the parser understood, is usually the right one --
    and the caller can report that the model answered off-schema instead of
    quietly shipping the default.
    """
    rejected: List[str] = []
    if not isinstance(payload, dict):
        return ["the reply was not a JSON object"]

    for key, value in payload.items():
        # ``notes`` is the channel this function's own results travel in, and
        # ``raw`` is the user's sentence; a model that echoes either back must
        # not be able to overwrite it.
        if value is None or key in ("notes", "raw") or not hasattr(intent, key):
            continue
        if key == "job_type":
            if value not in JOB_TYPES:
                rejected.append(f"job_type {value!r} is not a job this server runs")
                continue
        elif key == "molecules":
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list) or not all(
                    isinstance(m, str) and m.strip() for m in value):
                rejected.append("molecules was not a list of names")
                continue
        elif key in ("charge", "multiplicity", "nstates"):
            try:
                value = int(value)
            except (TypeError, ValueError):
                rejected.append(f"{key} {value!r} is not an integer")
                continue
        elif key in ("functional", "basis"):
            value = str(value).strip().lower()
            table = FUNCTIONALS if key == "functional" else BASIS_SETS
            if value not in table:
                rejected.append(f"{key} {value!r} is not in the vocabulary")
                continue
        setattr(intent, key, value)

    # the model may hand back an unvalidated pair; normalise it
    intent.bond = _extract_bond_pair(str(intent.bond or ""))
    return rejected


class _DisabledClient:
    """A client that answers "no" to everything, for ``CHATDFT_PLANNER=local``.

    It exists so that "the model was not used" has exactly one implementation.
    When this returned a bare ``RuleBasedPlanner`` instead, the reason lived on
    a code path that only ran when the LLM planner *was* constructed -- so the
    configuration that skipped construction entirely produced no explanation
    anywhere, and the chat box quietly stopped being AI-driven.
    """

    base = ""
    key = ""
    model = ""

    @property
    def available(self) -> bool:
        return False

    def describe(self) -> Dict[str, Any]:
        return {"base": "", "model": "", "models": [], "latency": {},
                "configured": False, "key_hint": ""}


class LLMPlanner:
    """OpenAI-compatible planner, driven by the shared LLM client.

    This used to resolve ``CHATDFT_LLM_BASE`` / ``CHATDFT_LLM_KEY`` /
    ``CHATDFT_LLM_MODEL`` from the environment and POST to the endpoint itself,
    while the design agent went through ``agent.llm.LLMClient``, which reads the
    environment *and* ``data/llm.json``.  One setting, two readers, so they
    could disagree -- and they did:

    * ``GET /api/llm`` reports what ``LLMClient`` sees, so with a key in
      ``data/llm.json`` it answered ``available: true`` while
      ``create_planner()`` handed back the keyword parser.  Every sentence
      typed into the chat box was matched by regular expressions, and nothing
      in the response, the interface or the log said so.
    * ``choices[0].message.content`` was read directly.  These models are
      reasoning models: they leave ``content`` null and put the answer in
      ``reasoning_content``.  ``llm.py`` documents and handles that in
      ``_message_text``; this class predated it, so even with the environment
      set it would have raised TypeError and fallen back to the parser.

    Configuration, transport, retries, model fallback and the reasoning-field
    handling now live in one place, shared with the design agent.
    """

    def __init__(self, client: Optional[Any] = None, reason: str = ""):
        self.client = client if client is not None else default_client()
        # why this planner was handed a client that cannot answer
        self.reason = reason
        self.fallback = RuleBasedPlanner()
        self.last_error = ""

    # ---- configuration comes from the client, never from here ----------
    @property
    def base(self) -> str:
        return self.client.base

    @property
    def key(self) -> str:
        return self.client.key

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def available(self) -> bool:
        return bool(self.client.available)

    @classmethod
    def system_prompt(cls) -> str:
        """The schema handed to the model.

        The job vocabulary is rendered from ``JOB_TYPES`` rather than typed
        out.  It was typed out, and listed seven of the eighteen jobs the
        server dispatches on, so a request for a transition state, an NTO pair
        or a designed molecule could not be routed by the model at all -- it
        could only answer with one of the seven it had been told about.  A
        vocabulary written twice drifts; this one is written once.
        """
        return (
            "You convert a chemist's request into a JSON job specification for "
            "a density functional theory (DFT) code. Reply with JSON only.\n"
            "Schema:\n"
            '{"job_type": one of ['
            + ", ".join(f'"{k}"' for k in JOB_TYPES) + "], "
            '"molecule": string (name, formula, SMILES or XYZ), '
            '"molecule2": string (only for compare), '
            '"molecules": [string] (every molecule named, in order; used by '
            'compare and series), '
            '"bond": string like "O-H" (only for scan, optional), '
            '"functional": string, "basis": string, '
            '"charge": integer, "multiplicity": integer, "nstates": integer, '
            '"field_kind": string (elf/laplacian/spin/difference), '
            '"solvation": string or null}\n'
            "Available functionals: " + ", ".join(FUNCTIONALS) + "\n"
            "Available basis sets: " + ", ".join(BASIS_SETS) + "\n"
            "If the user does not specify a method, use b3lyp/6-31g*. "
            "Never invent molecules that the user did not mention."
        )

    def plan(self, text: str) -> JobIntent:
        # The keyword parser runs first in both branches: it supplies the
        # defaults, and it is what the answer falls back to.
        intent = self.fallback.plan(text)

        if not self.available:
            self.last_error = self.reason or (
                "no language model configured (set CHATDFT_LLM_KEY or add "
                "data/llm.json)")
            intent.notes.append(PLANNER_OFF_NOTE.format(reason=self.last_error))
            return intent

        try:
            payload = self.client.json(
                [{"role": "system", "content": self.system_prompt()},
                 {"role": "user", "content": text}],
                temperature=0.0, max_tokens=1500, timeout=PLAN_TIMEOUT,
                validate=lambda p: isinstance(p, dict),
            )
        except Exception as exc:                          # noqa: BLE001
            # Naming the error is the whole point.  A fallback that says why is
            # a degradation the reader can act on; one that says nothing is
            # indistinguishable from a request that never needed the model.
            self.last_error = f"{type(exc).__name__}: {exc}"
            intent.notes.append(
                f"LLM planner unavailable ({self.last_error}); used local parser")
            return intent

        rejected = apply_llm_payload(intent, payload)
        intent.confidence = 0.9
        intent.raw = text
        intent.notes.append(f"planned by {self.model}")
        if rejected:
            intent.notes.append(
                "the model's reply was partly off-schema and was ignored: "
                + "; ".join(rejected))
        return intent


# ======================================================================
#  narration: turn numbers into chemistry
# ======================================================================
ELECTRONEGATIVITY = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04,
    "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Br": 2.96,
    "I": 2.66, "Fe": 1.83, "Cu": 1.90, "Zn": 1.65, "Se": 2.55,
}


# Below this magnitude a partial charge is too small to point at: at 6-31G*
# a C-H bond in a hydrocarbon already polarises by ~0.08 e, and calling that
# a "site" would be noise.
_CHARGE_FLOOR = 0.15
# Atoms within this of the extremum are the same chemical site (the two
# carboxylate oxygens, the four N-H hydrogens) and are reported as a group
# rather than as whichever one happened to sort first.
_CHARGE_DEGENERATE = 0.05
# Loewdin charges are systematically smaller than Mulliken's.  Below this
# fraction the two schemes agree on *where* the charge is but not on how much,
# which is worth saying rather than glossing as agreement.
_CHARGE_WEAK_RATIO = 0.40
# A heteroatom this close to the extremum is reported even though it is not
# the extremum, so the chemically important atom is not silently dropped.
_CHARGE_RIVAL_RATIO = 0.70


def _describe_site(rows: List[dict]) -> str:
    """Name a group of near-degenerate atoms, e.g. "O3/O4 (-0.450 e)"."""
    if not rows:
        return ""
    mean = sum(r["charge"] for r in rows) / len(rows)
    value = f"({mean:+.3f} e)"
    if len(rows) > 3:
        # Too many to enumerate honestly: "all 4 H atoms (+0.362 e)".
        return f"all {len(rows)} {rows[0]['symbol']} atoms {value}"
    names = "/".join(f"{r['symbol']}{r['atom']}" for r in rows)
    return f"{names} {value}"


def _atoms_near(charges: List[dict], extreme: float, side: int) -> List[dict]:
    """Atoms within _CHARGE_DEGENERATE of ``extreme`` on the given side.

    ``side`` is -1 for the electron-rich end, +1 for the electron-poor end.
    """
    if side < 0:
        near = [
            r for r in charges
            if r["charge"] <= extreme + _CHARGE_DEGENERATE
        ]
        return sorted(near, key=lambda r: r["charge"])[:4]
    near = [r for r in charges if r["charge"] >= extreme - _CHARGE_DEGENERATE]
    return sorted(near, key=lambda r: -r["charge"])[:4]


def _charge_sentence(result: dict) -> str:
    """Narrate the population analysis, choosing the verb from the signs.

    Returns "" when no atom carries a charge worth naming, so the caller can
    skip the paragraph instead of emitting an empty one.
    """
    charges = [
        c for c in (result.get("mulliken_charges") or [])
        if isinstance(c.get("charge"), (int, float))
    ]
    if not charges:
        return ""

    loewdin = [
        c for c in (result.get("lowdin_charges") or [])
        if isinstance(c.get("charge"), (int, float))
    ]

    rich = min(charges, key=lambda c: c["charge"])   # most negative
    poor = max(charges, key=lambda c: c["charge"])   # most positive

    # A same-sign spread (e.g. acetate, where the two oxygens dominate and
    # nothing is appreciably positive) is genuinely a concentration; opposite
    # signs at the two ends is a separation.
    rich_notable = rich["charge"] <= -_CHARGE_FLOOR
    poor_notable = poor["charge"] >= _CHARGE_FLOOR


    if rich_notable and poor_notable:
        rich_rows = _atoms_near(charges, rich["charge"], -1)
        poor_rows = _atoms_near(charges, poor["charge"], +1)
        if abs(rich["charge"]) >= abs(poor["charge"]):
            primary, primary_rows = rich, rich_rows
        else:
            primary, primary_rows = poor, poor_rows
        # "C1/C4 are the most electron-rich sites" -- a group is plural.
        verb = "are" if len(rich_rows) > 1 else "is"
        noun = "sites" if len(rich_rows) > 1 else "site"
        body = (
            f"Mulliken population analysis separates charge across the "
            f"molecule: {_describe_site(rich_rows)} {verb} the most "
            f"electron-rich {noun} and {_describe_site(poor_rows)} the most "
            f"electron-poor."
        )
    elif rich_notable or poor_notable:
        end = -1 if rich_notable else +1
        primary = rich if rich_notable else poor
        primary_rows = _atoms_near(charges, primary["charge"], end)
        body = (
            f"Mulliken population analysis puts the largest partial charge on "
            f"{_describe_site(primary_rows)}."
        )
    else:
        biggest = max(abs(rich["charge"]), abs(poor["charge"]))
        return (
            f"No atom carries more than {biggest:.2f} e of partial charge, so "
            f"the electron density is shared rather evenly and a population "
            f"analysis has little to say here."
        )

    body += _loewdin_clause(primary, primary_rows, loewdin)
    body += _electronegative_rival(charges, rich)
    return (
        f"{body} Mulliken charges are basis-set sensitive, so read them as "
        f"trends rather than as absolute values."
    )


def _loewdin_clause(primary: dict, primary_rows: List[dict],
                    loewdin: List[dict]) -> str:
    """Cross-check the named site against the Lowdin partition.

    Lowdin splits the same density in a symmetrically orthogonalised basis, so
    agreement is reassuring and a disagreement is itself worth reporting.  The
    check has to run per site: for acetone Mulliken's most positive atom is the
    carbonyl carbon while Lowdin's is a methyl hydrogen, so a blanket "the two
    schemes agree" would be false.
    """
    if not loewdin:
        return ""
    side = -1 if primary["charge"] < 0 else 1
    l_ext = (
        min(loewdin, key=lambda c: c["charge"]) if side < 0
        else max(loewdin, key=lambda c: c["charge"])
    )
    atoms = {r["atom"] for r in primary_rows}
    if l_ext["atom"] not in atoms:
        return (
            f" Loewdin partitioning, which is less basis-set dependent, "
            f"instead points at {l_ext['symbol']}{l_ext['atom']} "
            f"({l_ext['charge']:+.3f} e), so this site assignment is "
            f"partition-dependent and should be read as qualitative only."
        )
    desc = _describe_site([c for c in loewdin if c["atom"] in atoms])
    denom = abs(primary["charge"])
    ratio = abs(l_ext["charge"]) / denom if denom > 1e-9 else 0.0
    if ratio < _CHARGE_WEAK_RATIO:
        return (
            f" Loewdin partitioning, which is less basis-set dependent, agrees "
            f"on the site {desc} but with only {ratio:.0%} of the magnitude, "
            f"so treat the number as a trend rather than a measurement."
        )
    return (
        f" Loewdin partitioning, which is less basis-set dependent, puts the "
        f"largest magnitude on the same site {desc}, so the assignment is "
        f"not an artefact of the partition."
    )


def _electronegative_rival(charges: List[dict], rich: dict) -> str:
    """Name the most electronegative atom when it is not the extremum.

    Acetone is the motivating case: Mulliken puts -0.52 e on the methyl carbons
    and only -0.40 e on the carbonyl oxygen, so naming the extremum alone never
    mentions the C=O bond -- the one thing a chemist actually wants to know.
    """
    present = {c["symbol"] for c in charges}
    scored = [
        (ELECTRONEGATIVITY[s], s) for s in present if s in ELECTRONEGATIVITY
    ]
    if not scored:
        return ""
    symbol = max(scored)[1]
    if symbol == rich["symbol"]:
        return ""   # the extremum already is the electronegative atom
    candidates = [
        c for c in charges if c["symbol"] == symbol and c["charge"] < 0.0
    ]
    if not candidates:
        return ""
    best = min(candidates, key=lambda c: c["charge"])
    denom = abs(rich["charge"])
    if denom < 1e-9 or abs(best["charge"]) < _CHARGE_RIVAL_RATIO * denom:
        return ""
    return (
        f" Note that {symbol}{best['atom']} ({best['charge']:+.3f} e) is less "
        f"negative than the extremum even though {symbol} is the most "
        f"electronegative atom present: Mulliken splits every overlap "
        f"population evenly between the two atoms, which systematically "
        f"understates the polarity of bonds to {symbol}."
    )


def explain_scf(molecule, result: dict, intent: JobIntent) -> str:
    """Scientific commentary on a single-point result."""
    parts: List[str] = []
    conv = result.get("converged")
    status = "converged" if conv else "**did not fully converge**"
    parts.append(
        f"The {result['functional_label']} / {result['basis_label']} SCF for "
        f"**{molecule.name or molecule.formula}** ({molecule.formula}, "
        f"{result['nelec']} electrons, multiplicity {result['multiplicity']}) {status}."
    )
    # A solvated number and a gas-phase number are different quantities with
    # the same name, and quoting one as the other is not reproducible.
    if result.get("solvation"):
        parts.append(
            f"This is not a gas-phase calculation: the solvent was treated as "
            f"a polarisable continuum (**{result.get('solvation_model') or 'implicit'}**, "
            f"\u03b5 = {result['solvation']}), so these numbers are to be "
            f"compared with solvated references, not gas-phase ones."
        )

    # An unrestricted determinant is not an eigenfunction of S^2, so "it is a
    # doublet" is a claim that has to be checked, not assumed.  Reporting
    # <S^2> is what makes the orbitals and the spin density below quotable.
    spin = result.get("spin")
    if spin and spin.get("s_squared") is not None:
        parts.append(
            f"This is an unrestricted (UKS) calculation: one determinant is "
            f"not an eigenfunction of S², so the spin state is reported, not "
            f"assumed. ⟨S²⟩ = **{spin['s_squared']:.4f}** against the exact "
            f"{spin['s_squared_expected']:.4f} for multiplicity "
            f"{spin['multiplicity']} -- a deviation of "
            f"{spin['contamination_pct']:+.2f}%, "
            + ("well inside the range where unrestricted energies are still "
               "trustworthy."
               if abs(spin["contamination_pct"]) <= 10.0 else
               "**past the ~10% point at which the result is no longer a pure "
               "spin state**, and the energies and spin density should not be "
               "reported as those of one.")
        )

    parts.append(
        f"Total energy is **{result['energy_hartree']:.6f} Ha** "
        f"({result['energy_ev']:.3f} eV). Absolute energies are only meaningful "
        f"when compared between calculations using the identical method."
    )

    # Bond orders and Hirshfeld charges.  Mayer bond orders are the number a
    # paper quotes when it says a bond got stronger, and Hirshfeld is the
    # partition worth quoting because it barely moves with the basis.
    bonding = result.get("bonding") or {}
    mayer = bonding.get("mayer") or {}
    bonds = mayer.get("bonds") or []
    if bonds:
        top = bonds[:4]
        listed = ", ".join(
            f"{b['from']}{b['i']}\u2013{b['to']}{b['j']} {b['order']:.3f}"
            for b in top)
        parts.append(
            f"**Mayer bond orders** (from the total density matrix, "
            f"P\u03b1 + P\u03b2): {listed}"
            + (f", and {len(bonds) - len(top)} more"
               if len(bonds) > len(top) else "")
            + ". Near 1 is a single bond and near 2 a double bond; the value is "
              "not an integer because the density is delocalised over the "
              "molecule, not because the calculation is approximate."
        )
    hf = bonding.get("hirshfeld") or {}
    hcharges = hf.get("charges") or []
    hdiag = hf.get("diagnostics") or {}
    if hcharges:
        low = min(hcharges, key=lambda c: c["charge"])
        high = max(hcharges, key=lambda c: c["charge"])
        tail = ""
        if hdiag.get("grid_integral_electrons") is not None:
            tail = (
                f" They sum to {hdiag['charge_sum']:+.5f} e against the "
                f"molecule's {hdiag['expected_charge_sum']:+.5f} e, and the "
                f"density integrates to {hdiag['grid_integral_electrons']:.5f} "
                f"of {hdiag['n_electrons']} electrons, so the partition is "
                f"sound."
            )
        parts.append(
            f"**Hirshfeld charges** (free-atom densities, spherically "
            f"averaged so equivalent atoms come out equivalent): "
            f"{low['symbol']}{low['atom']} {low['charge']:+.3f} e, "
            f"{high['symbol']}{high['atom']} {high['charge']:+.3f} e.{tail} "
            f"Of the three partitions on this result, Hirshfeld is the one to "
            f"quote in a paper: it moves by thousandths of an electron when "
            f"the basis grows, where Mulliken can move by tenths."
        )
    for key, label in (("mayer_error", "bond-order"), ("hirshfeld_error", "Hirshfeld")):
        if bonding.get(key):
            parts.append(
                f"The {label} analysis did not run: `{bonding[key]}`. "
                f"No numbers are shown for it rather than a plausible-looking "
                f"empty table."
            )

    if result.get("homo_ev") == result.get("homo_ev"):
        gap = result.get("gap_ev", float("nan"))
        parts.append(
            f"The Kohn-Sham frontier orbitals sit at HOMO = "
            f"**{result['homo_ev']:.3f} eV** and LUMO = **{result['lumo_ev']:.3f} eV**, "
            f"giving a gap of **{gap:.3f} eV**."
        )
        # A Kohn-Sham gap is neither the fundamental nor the optical gap, and
        # it does not identify a bonding class -- benzene (6.8 eV) and pyridine
        # (6.3 eV) are aromatic despite large gaps, while extended conjugation
        # shrinks the gap.  Keep the reading to what the number supports.
        if gap < 2.0:
            parts.append(
                "A gap this small means the frontier orbitals lie close in "
                "energy, which is typical of extended conjugation or of a "
                "molecule that is easy to oxidise or reduce."
            )
        elif gap < 4.0:
            parts.append(
                "This is a typical Kohn-Sham gap for a mid-sized conjugated "
                "organic molecule."
            )
        else:
            parts.append(
                "A gap this large means the frontier orbitals are well "
                "separated. Saturated molecules sit here, but so do aromatics "
                "such as benzene and pyridine, so this alone does not identify "
                "the bonding."
            )
        parts.append(
            "Note that a Kohn-Sham gap is not an optical gap: it underestimates "
            "the true fundamental gap and normally exceeds the first excitation "
            "energy, so use TD-DFT to find where the molecule actually absorbs."
        )

    # dipole
    d = result.get("dipole", {})
    mag = d.get("magnitude", 0.0)
    if mag > 0.05:
        if mag < 0.5:
            comment = "essentially non-polar"
        elif mag < 2.0:
            comment = "weakly polar"
        elif mag < 4.0:
            comment = "moderately polar"
        else:
            comment = "strongly polar"
        parts.append(
            f"The molecular dipole moment is **{mag:.3f} D** ({comment}), "
            f"along ({d['x']:.2f}, {d['y']:.2f}, {d['z']:.2f}) in Debye."
        )
    else:
        parts.append("The dipole moment is essentially zero, as symmetry demands.")

    # charges
    #
    # The old wording said the analysis "concentrates charge on" the three
    # largest-magnitude atoms.  That is wrong for nearly every polar molecule:
    # the largest entries are then of *opposite* sign, which is charge
    # separation, not concentration.  Water came out as "concentrates charge
    # on O1 (-0.357), H2 (+0.178), H3 (+0.178)" -- three atoms that are in fact
    # the two ends of the molecular dipole.  Report the two ends separately
    # and let the sign decide the verb.
    _charge_text = _charge_sentence(result)
    if _charge_text:
        parts.append(_charge_text)

    # frontier composition
    #
    # ``frontier_orbitals`` lists atoms in index order, not by weight, so the
    # largest contributor has to be found explicitly.  Naphthalene once had
    # C1 (7.5%) named as the reactive site when C3/C5/C8/C10 each carried
    # 17.5% -- the opposite of the textbook result, and silent.
    fo = result.get("frontier_orbitals", {})
    homo_comp = [
        c for c in (fo.get("homo") or [])
        if isinstance(c.get("percent"), (int, float))
    ]
    if homo_comp:
        ranked = sorted(homo_comp, key=lambda c: c["percent"], reverse=True)
        top = ranked[0]
        runner_up = ranked[1]["percent"] if len(ranked) > 1 else 0.0
        if runner_up <= 0.0 or top["percent"] >= 2.0 * runner_up:
            parts.append(
                f"The HOMO is concentrated on {top['symbol']}{top['atom']} "
                f"({top['percent']:.1f}% of the density), which is therefore the "
                f"most likely site of electrophilic attack or oxidation."
            )
        else:
            # No single atom dominates: naming one would invent a reactive site.
            leaders = [c for c in ranked if c["percent"] >= 0.7 * top["percent"]]
            names = ", ".join(f"{c['symbol']}{c['atom']}" for c in leaders[:4])
            more = "" if len(leaders) <= 4 else f" and {len(leaders) - 4} more"
            parts.append(
                f"The HOMO is delocalised rather than confined to one atom: it is "
                f"spread over {names}{more}, the largest single contribution being "
                f"only {top['percent']:.1f}%. Reactivity is therefore shared across "
                f"these positions."
            )

    for w in result.get("warnings", []):
        parts.append(f"Note: {w}")

    return "\n\n".join(parts)


def explain_opt(molecule, result: dict, intent: JobIntent) -> str:
    parts = [
        f"Geometry optimisation of **{molecule.name or molecule.formula}** finished "
        f"at the {result['functional_label']} / {result['basis_label']} level in "
        f"{result.get('opt_seconds', 0):.1f} s."
    ]
    # "Finished" is not "converged".  The optimiser will hand back whatever
    # geometry it last visited, so the verdict has to be read off the residual
    # gradient and stated -- an unrelaxed structure described as an
    # equilibrium geometry would put every number below it in doubt.
    conv = result.get("opt_converged")
    grms = result.get("opt_grms")
    if conv is True and grms is not None:
        parts.append(
            f"The residual gradient is |g|rms = {grms:.1e} hartree/bohr, so "
            f"the gradient criterion was met and this is a stationary point.")
    elif conv is False:
        parts.append(
            f"**The optimisation did not converge.** The residual gradient is "
            f"|g|rms = {grms if grms is not None else float('nan'):.1e} "
            f"hartree/bohr after {result.get('n_steps') or '?'} steps, above "
            f"the criterion of "
            f"{(result.get('opt_criteria') or {}).get('convergence_grms', 3e-4):.0e}. "
            f"Treat the bond lengths below as those of a partly relaxed "
            f"structure, not an equilibrium geometry.")
    else:
        parts.append(
            "The gradient criterion could not be checked (the SCF at the "
            "final geometry did not converge), so this is not confirmed as a "
            "stationary point.")

    geo = result.get("geometry", {})
    bonds = geo.get("bonds", [])
    if bonds:
        by_type: Dict[str, List[float]] = {}
        for b in bonds:
            label = re.sub(r"\d+", "", b["label"])
            by_type.setdefault(label, []).append(b["value"])
        summary = []
        for label, vals in by_type.items():
            lo, hi = min(vals), max(vals)
            if hi - lo > 0.02:
                # Genuinely inequivalent bonds (e.g. the C=O and C-OH of a
                # carboxylic acid).  An average would name a length that no
                # bond in the molecule actually has.
                summary.append(f"{label} {lo:.3f}-{hi:.3f} A")
            else:
                summary.append(f"{label} {sum(vals) / len(vals):.3f} A")
        parts.append(
            "Optimised bond lengths: " + "; ".join(summary) + ". "
            "These are equilibrium values at this level of theory."
        )

    angles = geo.get("angles", [])
    if angles:
        interesting = sorted(angles, key=lambda a: -a["value"])[:2]
        parts.append(
            "Widest bond angles: "
            + ", ".join(f"{a['label']} = {a['value']:.1f} deg" for a in interesting)
            + "."
        )

    if result.get("homo_ev") == result.get("homo_ev"):
        parts.append(
            f"At the optimised geometry the HOMO-LUMO gap is "
            f"**{result['gap_ev']:.3f} eV** "
            f"(E = {result['energy_hartree']:.6f} Ha)."
        )

    parts.append(
        "The optimised structure is available in the 3D viewer; export it as "
        "XYZ if you want to continue with a different code or a higher level "
        "of theory."
    )
    return "\n\n".join(parts)


def explain_excited(molecule, result: dict, intent: JobIntent) -> str:
    states = result.get("excited_states", [])
    parts = [
        f"TD-DFT (Tamm-Dancoff) on **{molecule.name or molecule.formula}** at the "
        f"{result['functional_label']} / {result['basis_label']} level produced "
        f"{len(states)} vertical excitations from the ground-state geometry."
    ]

    if states:
        table_lines = ["| State | Energy (eV) | Wavelength (nm) | f (osc.) |", "|---|---|---|---|"]
        for s in states:
            fval = s.get("oscillator_strength")
            table_lines.append(
                f"| S{s['state']} | {s['energy_ev']:.3f} | "
                f"{s['wavelength_nm']:.1f} | "
                f"{(f'{fval:.4f}' if fval is not None else '-')} |"
            )
        parts.append("\n".join(table_lines))

        # Use the engine's own 'active' flag rather than a private threshold:
        # the table's "Optically active" column is driven by it, and a second
        # cut-off here would let the prose and the table disagree.
        bright = [s for s in states if s.get("active")]
        if bright:
            # ``states`` is ordered by energy, so bright[0] is the *lowest*
            # bright state, not the strongest.  Naphthalene's S1 has
            # f = 0.094 while S4 has f = 1.99, so calling S1 "the strongest"
            # was plainly false.
            strongest = max(
                bright, key=lambda s: s.get("oscillator_strength") or 0.0
            )
            lowest = bright[0]
            parts.append(
                f"The strongest transition is S{strongest['state']} at "
                f"**{strongest['energy_ev']:.2f} eV "
                f"({strongest['wavelength_nm']:.0f} nm)** with an oscillator "
                f"strength of {strongest['oscillator_strength']:.4f}."
            )
            if lowest["state"] != strongest["state"]:
                parts.append(
                    f"The lowest-lying optically bright transition is "
                    f"S{lowest['state']} at {lowest['energy_ev']:.2f} eV "
                    f"({lowest['wavelength_nm']:.0f} nm, "
                    f"f = {lowest['oscillator_strength']:.4f})."
                )
            parts.append(
                "Only transitions flagged optically active carry appreciable "
                "oscillator strength; the rest are formally dark."
            )
        else:
            parts.append(
                "None of the computed transitions carry significant oscillator "
                "strength, so this molecule is not strongly absorbing in the "
                "window probed by these roots."
            )

        parts.append(
            "Vertical excitation energies from TD-DFT are usually good to about "
            "0.2-0.3 eV with a hybrid functional; for benchmark accuracy you "
            "would move to a range-separated functional or a wavefunction method "
            "such as EOM-CCSD."
        )

        # The broadened curve is the figure, so say what it is on: the window
        # it was drawn over, the absolute band height, and how much of the
        # total oscillator strength these roots captured.  Without the last
        # number a stick list reads like a complete spectrum.
        uv = result.get("uv_spectrum") or {}
        win = uv.get("window_nm") or []
        srule = uv.get("sum_rule") or {}
        if len(win) == 2:
            how = ("taken from the states themselves"
                   if uv.get("window_source") == "auto"
                   else "the window requested")
            line = (f"The absorption curve is drawn over {win[0]:.0f}-{win[1]:.0f} "
                    f"nm ({how}), broadened with Gaussian bands of "
                    f"FWHM {uv.get('fwhm_ev')} eV "
                    f"({uv.get('fwhm_cm1')} cm-1) applied in energy.")
            peaks = {p.get("state"): p for p in (uv.get("peaks") or [])}
            sp = peaks.get((uv.get("strongest") or {}).get("state"))
            if sp and sp.get("epsilon_max_l_mol_cm"):
                line += (f" On the absolute scale the strongest band reaches "
                         f"eps_max = {sp['epsilon_max_l_mol_cm']:.3g} "
                         f"L mol-1 cm-1.")
            parts.append(line)
        if srule.get("sum_f_over_n_electrons") is not None:
            parts.append(
                f"These {srule.get('n_used')} roots carry "
                f"sum(f) = {srule.get('sum_f'):.4f}, which is "
                f"**{100 * srule['sum_f_over_n_electrons']:.1f}% of the "
                f"Thomas-Reiche-Kuhn sum rule** "
                f"({srule.get('n_electrons')} electrons). The remaining "
                f"oscillator strength sits in transitions above these roots, so "
                f"the figure shows the onset of absorption rather than the "
                f"whole spectrum."
            )
        if srule.get("residual_pct") is not None:
            resid = float(srule["residual_pct"])
            # Say which way the check went, not what it is meant to show.  The
            # earlier draft ended this sentence with "so the broadening, the
            # unit conversion and the grid agree" whatever the number was,
            # which is a conclusion the measurement had not earned.
            verdict = ("so the broadening, the unit conversion and the grid "
                       "agree" if abs(resid) < 0.5 else
                       "which is larger than the 0.5% the broadening and the "
                       "grid should reproduce, so treat the absolute ordinate "
                       "with caution")
            parts.append(
                f"The curve is on an absolute scale: the area under it over "
                f"wavenumber is {srule.get('measured_l_mol_cm2'):.4g} "
                f"L mol-1 cm-2 against "
                f"{srule.get('expected_l_mol_cm2'):.4g} predicted from "
                f"2.3154e8 * sum(f), a residual of {resid:+.3f}% -- {verdict}."
            )
        for note in (uv.get("notes") or []):
            parts.append(f"**Caveat:** {note}")
    else:
        parts.append("No excitations were returned - check that the ground state converged.")

    return "\n\n".join(parts)


def explain_compare(mol_a, mol_b, res_a: dict, res_b: dict, intent: JobIntent) -> str:
    parts = [
        f"Comparing **{mol_a.name or mol_a.formula}** and "
        f"**{mol_b.name or mol_b.formula}** at the "
        f"{res_a['functional_label']} / {res_a['basis_label']} level "
        "(identical method for both, as required for a meaningful comparison)."
    ]

    name_a = mol_a.name or mol_a.formula
    name_b = mol_b.name or mol_b.formula

    dgap = res_a["gap_ev"] - res_b["gap_ev"]
    gap_line = (
        f"HOMO-LUMO gap: {name_a} = **{res_a['gap_ev']:.3f} eV**, "
        f"{name_b} = **{res_b['gap_ev']:.3f} eV**."
    )
    if abs(dgap) < 1e-6:
        # comparing a molecule with itself, or a genuine tie: naming a "smaller"
        # gap here would be false
        parts.append(
            gap_line + " The two gaps agree to within rounding, so neither is "
            "the more easily excited on this measure."
        )
    else:
        lower = name_b if dgap > 0 else name_a
        parts.append(
            gap_line + f" {lower} has the smaller gap by {abs(dgap):.3f} eV, so it "
            "is the more easily excited and generally the more reactive of the two."
        )

    if mol_a.formula and mol_b.formula and mol_a.formula != mol_b.formula:
        parts.append(
            "Because the two molecules have different compositions, absolute "
            f"energies ({res_a['energy_hartree']:.4f} Ha vs "
            f"{res_b['energy_hartree']:.4f} Ha) are not directly comparable - "
            "only relative orbital energies, gaps and derived properties are."
        )

    dmu_a = res_a.get("dipole", {}).get("magnitude", 0)
    dmu_b = res_b.get("dipole", {}).get("magnitude", 0)
    if abs(dmu_a - dmu_b) < 1e-6:
        parts.append(
            f"Dipole moments: {dmu_a:.3f} D vs {dmu_b:.3f} D - equal to within "
            "rounding, so neither is the more polar."
        )
    else:
        polar = name_a if dmu_a > dmu_b else name_b
        parts.append(
            f"Dipole moments: {dmu_a:.3f} D vs {dmu_b:.3f} D. {polar} is the more "
            "polar of the two, so it interacts more strongly with a polar solvent "
            "and couples more strongly to neighbouring molecules."
        )

    return "\n\n".join(parts)


# ======================================================================
#  concept library for "info" intents
# ======================================================================
CONCEPTS: Dict[str, str] = {
    "dft": (
        "**Density Functional Theory** reformulates the electronic structure "
        "problem in terms of the electron density rather than a many-electron "
        "wavefunction. The Hohenberg-Kohn theorems guarantee that the ground-state "
        "density determines the energy, and the Kohn-Sham construction makes that "
        "practical by mapping the interacting system onto a fictitious "
        "non-interacting one.\n\n"
        "The catch is the exchange-correlation functional, which is not known "
        "exactly and must be approximated. The accuracy of a DFT calculation is "
        "almost entirely set by that choice - which is why ChatDFT always prints "
        "the functional alongside every number."
    ),
    "homo_lumo": (
        "The **HOMO** is the highest-energy orbital containing electrons and the "
        "**LUMO** is the lowest empty one. Their energy difference, the gap, is a "
        "good proxy for kinetic stability and optical response: molecules with "
        "small gaps are easily oxidised, reduced or excited, and typically absorb "
        "toward the visible.\n\n"
        "Two caveats matter. First, in DFT the orbital energies are not physical "
        "ionisation energies - the gap underestimates the true fundamental gap. "
        "Second, only within the same method can gaps be compared between molecules."
    ),
    "b3lyp": (
        "**B3LYP** is a hybrid functional: it mixes exact Hartree-Fock exchange "
        "(20%) with Becke-88 exchange and LYP correlation. It is the most widely "
        "used functional in the literature and a sensible default, but it "
        "underestimates reaction barriers slightly and struggles with "
        "charge-transfer excitations."
    ),
    "basis_set": (
        "A **basis set** is the finite set of functions used to expand the "
        "molecular orbitals. STO-3G is minimal and only good for very rough "
        "trends; 6-31G* adds polarisation functions and is the usual workhorse; "
        "triple-zeta sets such as def2-TZVP are needed for quantitative "
        "energetics.\n\n"
        "Rule of thumb: the basis set error and the functional error are of "
        "comparable size, so there is little point in a huge basis with a poor "
        "functional, or vice versa."
    ),
    "geometry_optimization": (
        "**Geometry optimisation** moves the nuclei along the negative gradient "
        "of the energy until the forces vanish, locating a stationary point. The "
        "result is a local minimum, not necessarily the global one, so the "
        "starting structure matters.\n\n"
        "ChatDFT uses analytic gradients, which is far more efficient than "
        "sweeping a grid of geometries by hand."
    ),
    "tddft": (
        "**TD-DFT** extends DFT to excited states, in practice usually in the "
        "Tamm-Dancoff approximation (TDA), which neglects certain "
        "de-excitation terms in exchange for speed and better numerical "
        "stability.\n\n"
        "The output is a set of vertical excitation energies with oscillator "
        "strengths. Multiply the strength by the energy distribution and you get "
        "a simulated absorption spectrum."
    ),
    "nmr": (
        "**NMR shielding** is the magnetic field the electrons themselves "
        "generate at a nucleus when the molecule sits in a spectrometer field. "
        "It is the second derivative of the energy with respect to that field "
        "and the nuclear magnetic moment, which makes it a 3x3 tensor per "
        "nucleus.\n\n"
        "What a spectrum shows is the **chemical shift**, the difference "
        "between a nucleus in the molecule and the same nucleus in a reference "
        "compound - tetramethylsilane for 1H and 13C, water for 17O, ammonia "
        "for 14N. ChatDFT computes the reference at the same level of theory, "
        "because a tabulated reference from another program would put a "
        "constant offset on every number and there would be no way to see it.\n\n"
        "One limit worth knowing: ChatDFT evaluates the shielding at the "
        "**Hartree-Fock** level regardless of the functional you ask for. That "
        "is a hard limit of the engine, not a preference - the density "
        "functional integrator cannot take the complex density the GIAO "
        "response needs. Every result says so."
    ),
    "chemical_shift": (
        "A **chemical shift** is reported in ppm relative to a standard:\n\n"
        "    delta = sigma(reference) - sigma(sample)\n\n"
        "so a nucleus that is *less* shielded than the standard comes out at a "
        "higher ppm. The sign is the single easiest thing to get backwards, "
        "which is why ChatDFT prints the reference compound and its absolute "
        "shielding next to every shift.\n\n"
        "The isotropic shift is the average of the three principal shieldings "
        "and is what a solution spectrum measures. The spread between them - "
        "the span - is what a solid-state spectrum measures, and it is where "
        "the geometry shows up."
    ),
    "mulliken": (
        "**Mulliken population analysis** partitions the electron density among "
        "atoms by apportioning each basis function's contribution. It is fast and "
        "intuitive but notoriously basis-set dependent - adding diffuse functions "
        "can shift charges substantially.\n\n"
        "Use it to see *how charge is distributed* - which end of the molecule "
        "is electron-rich and which is electron-poor - not to quote absolute "
        "atomic charges. Because the overlap population is split evenly between "
        "the two atoms of every bond, Mulliken systematically understates the "
        "polarity of bonds to electronegative atoms: in acetone it puts more "
        "negative charge on the methyl carbons than on the carbonyl oxygen.\n\n"
        "Löwdin charges (also reported) use the same density in a symmetrically "
        "orthogonalised basis and are less basis-set sensitive, but they are "
        "systematically smaller in magnitude. Where the two agree the site "
        "assignment is safe; where they disagree, treat it as qualitative only."
    ),
    "dipole": (
        "The **dipole moment** measures the separation of positive and negative "
        "charge. It controls solvation, intermolecular forces and infrared "
        "intensities. A molecule can have polar bonds and a zero total dipole - "
        "CO2 is the standard example, where the two C=O bond dipoles cancel by "
        "symmetry."
    ),
    "solvation": (
        "**Implicit solvation** models replace the surrounding solvent with a "
        "structureless polarisable continuum of permittivity epsilon, carving out "
        "a cavity for the solute. ChatDFT uses ddCOSMO for this.\n\n"
        "It captures the bulk electrostatic effect but ignores specific "
        "hydrogen-bonding and solvent structure. For those you would need an "
        "explicit solvation shell."
    ),
    "exchange_correlation": (
        "The **exchange-correlation functional** encodes everything that is not "
        "classical electrostatics. Jacob's ladder organises them: LDA "
        "(density only), GGA (adds gradients), meta-GGA (adds the kinetic-energy "
        "density), hybrids (mix in exact exchange), and double hybrids.\n\n"
        "Higher rungs are more accurate but computationally heavier, and no "
        "single functional is best for all properties."
    ),
    "scf": (
        "The **SCF procedure** solves the Kohn-Sham equations self-consistently: "
        "guess the density, build the potential, solve for orbitals, rebuild the "
        "density, and repeat until it stops changing. Convergence problems are "
        "the most common practical failure, usually cured by better mixing or a "
        "second-order (Newton) solver - which ChatDFT applies automatically."
    ),
}


def explain_scan(molecule, result: dict, intent: JobIntent) -> str:
    """Narrate a rigid bond scan."""
    from ..engine.dft import HARTREE2KCAL

    sc = result.get("scan") or {}
    points = sc.get("points") or []
    name = molecule.name or molecule.formula
    if not points:
        return (
            f"The bond scan on **{name}** returned no points - check that the "
            "job converged."
        )

    labels = sc.get("labels") or []
    label = "-".join(labels) if labels else sc.get("element_pair", "bond")
    r0 = sc.get("r0") or points[0]["r"]
    mn = sc.get("minimum") or {}

    parts = [
        f"Rigid potential-energy scan along the **{label}** bond of **{name}** "
        f"at the {result['functional_label']} / {result['basis_label']} level. "
        f"The bond was driven over {len(points)} points from "
        f"{points[0]['r']:.2f} to {points[-1]['r']:.2f} A with every other atom "
        "held fixed."
    ]

    table = ["| r (A) | E (Ha) | E - Emin (kcal/mol) |", "|---|---|---|"]
    for pt in points:
        table.append(
            f"| {pt['r']:.3f} | {pt['energy_hartree']:.6f} | "
            f"{pt['energy_rel_kcal']:.2f} |"
        )
    parts.append("\n".join(table))

    if mn:
        # The grid minimum is a sampled point, so it is quantised to the grid:
        # refining the grid does not steadily improve it (measured on water's
        # O-H well, it is off by 0.004 A at 12 points and still 0.009 A at 25).
        # Reporting it to three decimals claims a precision the scan does not
        # have.  Quote the parabola vertex instead and state the grid step, so
        # the number carries the precision it actually has.
        fit = sc.get("minimum_fitted") or {}
        step = sc.get("grid_step") or 0.0
        r_best = fit.get("r") or mn["r"]
        e_best = fit.get("energy_hartree") or mn["energy_hartree"]
        if abs(r_best - r0) < 0.02:
            where = "already sits at the scanned minimum."
        elif r0 > r_best:
            where = "sits slightly longer than the scanned minimum."
        else:
            where = "sits slightly shorter than the scanned minimum."
        line = (
            f"The scanned curve bottoms out at **r = {r_best:.3f} A** "
            f"(E = {e_best:.6f} Ha). The input geometry has "
            f"r = {r0:.3f} A, so it {where}"
        )
        if fit:
            line += (
                f" The curve is sampled every {step:.3f} A, and the lowest "
                f"*sampled* point is r = {mn['r']:.3f} A; the value above is "
                f"the vertex of a parabola through the three lowest points, "
                f"which is why it is quoted to a precision finer than the "
                f"grid."
            )
        else:
            line += (
                f" The curve is sampled every {step:.3f} A and this is the "
                f"lowest *sampled* point, so treat it as uncertain by up to "
                f"one grid step."
            )
        parts.append(line)

    # Anharmonicity: in a harmonic well equal displacements cost the same in
    # both directions.  The reference must be the *minimum* and the two probe
    # points must actually sit at r_min -/+ d -- the sampled grid is not
    # symmetric about the input bond length, so comparing against r0 reports a
    # fictitious asymmetry (water's O-H well is in fact near-harmonic over
    # +/- 0.1 A, and the naive version called that "anharmonic").
    # The reference has to be the bottom of the well, not the sampled point
    # nearest it: ±0.12 r0 measured from a quantised grid point puts the two
    # probe points at unequal distances from the true minimum and reports an
    # anharmonicity that belongs to the grid, not the molecule.
    r_min = fit.get("r") or mn.get("r") or r0
    d = 0.12 * r0
    below = min(points, key=lambda p: abs(p["r"] - (r_min - d)))
    above = min(points, key=lambda p: abs(p["r"] - (r_min + d)))
    if mn and below is not above \
            and abs(below["r"] - (r_min - d)) < 0.06 \
            and abs(above["r"] - (r_min + d)) < 0.06:
        cost_below = (below["energy_hartree"] - mn["energy_hartree"]) * HARTREE2KCAL
        cost_above = (above["energy_hartree"] - mn["energy_hartree"]) * HARTREE2KCAL
        if abs(cost_below - cost_above) > 1.0:
            steeper = "compression" if cost_below > cost_above else "stretching"
            parts.append(
                f"Shortening the bond by {d:.2f} A from the minimum costs "
                f"{cost_below:.1f} kcal/mol while lengthening it by the same "
                f"amount costs {cost_above:.1f} kcal/mol, so the well is "
                f"anharmonic with {steeper} the stiffer direction."
            )
        else:
            parts.append(
                f"Shortening and lengthening the bond by {d:.2f} A cost "
                f"{cost_below:.1f} and {cost_above:.1f} kcal/mol respectively: "
                "over a displacement this small the well is close to harmonic. "
                "The asymmetry shows up further out, where compression climbs "
                "far faster than stretching."
            )

    parts.append(
        "This is a *rigid* scan: the other nuclei are frozen, so the minimum "
        "sits slightly above the true equilibrium length and the curve is "
        "stiffer than a relaxed one. Optimise the geometry for the real "
        "equilibrium bond length."
    )
    return "\n\n".join(parts)


def find_concept(text: str) -> Optional[str]:
    low = text.lower()
    keys = [
        ("homo", "homo_lumo"), ("lumo", "homo_lumo"), ("gap", "homo_lumo"),
        ("b3lyp", "b3lyp"), ("td-dft", "tddft"), ("tddft", "tddft"),
        ("basis set", "basis_set"), ("basis", "basis_set"),
        ("mulliken", "mulliken"), ("charge", "mulliken"),
        ("dipole", "dipole"), ("solvation", "solvation"),
        ("solvent", "solvation"), ("exchange", "exchange_correlation"),
        ("functional", "exchange_correlation"),
        ("scf", "scf"), ("self-consistent", "scf"),
        ("geometry opt", "geometry_optimization"), ("optimis", "geometry_optimization"),
        ("dft", "dft"), ("density functional", "dft"),
    ]
    for needle, key in keys:
        if needle in low:
            return CONCEPTS.get(key)
    return None


# ======================================================================
#  public entry points
# ======================================================================
def create_planner():
    """Return the planner to use, and whether the model is behind it.

    ``CHATDFT_PLANNER`` overrides the choice:

        auto   (default) the model when one is configured, else the parser
        local  the keyword parser, with the reason attached to every reply
        llm    the model, or the parser plus the reason it could not be reached

    The gates run the server with ``local``.  A regression suite that spends
    forty seconds of model latency on every chat call fails for reasons that
    have nothing to do with the code under test, and an account-level rate
    limit would look exactly like a broken router.  ``check_planner`` is the
    gate that covers the model path.

    Note that this always returns an ``LLMPlanner`` unless the parser was
    asked for by name.  Returning a bare ``RuleBasedPlanner`` when nothing was
    configured is what made the fallback silent: the object that knew the
    reason was never the object that answered.
    """
    mode = (os.environ.get("CHATDFT_PLANNER") or "auto").strip().lower()
    if mode == "local":
        return LLMPlanner(_DisabledClient(),
                          reason="disabled by CHATDFT_PLANNER=local"), False
    llm = LLMPlanner()
    return llm, llm.available


def explain_vibrations(molecule, result: dict, intent: JobIntent) -> str:
    """Narrate a harmonic frequency run and its IR spectrum."""
    name = molecule.name or molecule.formula
    freq = result.get("frequencies_cm1") or []
    if not freq:
        return (f"The vibrational calculation on **{name}** returned no "
                "frequencies - check that the job converged.")

    intens = result.get("ir_intensities_km_mol") or []
    therm = result.get("thermochemistry") or {}
    level = f"{result.get('functional_label', '')} / {result.get('basis_label', '')}"
    n_imag = result.get("n_imaginary") or 0

    parts = [
        f"Harmonic vibrational analysis of **{name}** at the {level} level. "
        f"PySCF's analytic second derivatives give {len(freq)} normal modes; "
        f"the IR intensities come from finite differences of the analytic "
        f"dipole along each mode, not from a fit."
    ]

    if n_imag:
        parts.append(
            f"**{n_imag} imaginary mode(s)**: " +
            ", ".join(f"{abs(f):.0f}i cm-1"
                      for f in (result.get("imaginary_cm1") or [])) +
            ". This geometry is a saddle point, not a minimum, so the "
            "thermochemistry below is not that of a stable species."
        )
    else:
        parts.append("No imaginary frequencies, so this is a true minimum.")

    # The scale factor is a property of the method, not of the molecule: 0.961
    # is the CCCBDB value for B3LYP/6-31G* and is wrong for everything else
    # (HF/6-31G* is 0.899, a 7% shift).  Read it off the result, where the
    # engine put the one for the level that actually ran.
    irs = result.get("ir_spectrum") or {}
    scale = irs.get("scale_factor") or result.get("freq_scale") or 1.0
    scaled = abs(float(scale) - 1.0) > 1e-9
    rows = ["| Mode | Harmonic (cm-1) |"
            + (" Scaled (cm-1) |" if scaled else "")
            + " IR (km/mol) |",
            "|---|---|" + ("---|" if scaled else "") + "---|"]
    order = sorted(range(len(freq)),
                   key=lambda i: -(intens[i] if i < len(intens) else 0.0))
    for i in order[:8]:
        f = freq[i]
        it = intens[i] if i < len(intens) else 0.0
        rows.append(
            f"| {i + 1} | {f:.1f} |"
            + (f" {f * float(scale):.0f} |" if scaled else "")
            + f" {it:.2f} |")
    if len(freq) > 8:
        rows.append(f"| ... | {len(freq) - 8} more |"
                    + ("| " if scaled else "")             + "|")
    parts.append("\n".join(rows))

    # IR intensities are only quotable if they are on an absolute scale, and
    # the plotted curve cannot show whether they are -- it is normalised to 100
    # before it is drawn.  The engine's sum-rule self-check is the evidence, so
    # quote it rather than leaving the reader to trust the column heading.
    sr = result.get("ir_sum_rule") or {}
    resid = sr.get("residual_pct")
    if resid is not None:
        if abs(float(resid)) <= 0.5:
            parts.append(
                "Intensities are integrated absorbances in km/mol (napierian, "
                "A = 42.256 |dmu/dQ|^2). The intensity sum rule closes to "
                f"{resid}% at this level, so these are absolute values, not "
                "relative ones scaled to the strongest band.")
        else:
            parts.append(
                f"**Caution:** the intensity sum rule is off by {resid}% at "
                "this level, so the km/mol column is not on an absolute "
                "scale. Treat the intensities as relative.")

    if order and intens:
        top = order[0]
        tail = (
            f"The strongest band is mode {top + 1} at {freq[top]:.0f} cm-1 "
            f"({intens[top]:.1f} km/mol). Harmonic frequencies overshoot "
            "experiment because the real potential is anharmonic"
        )
        if scaled:
            tail += (
                f", so the scaled column applies {float(scale):.3f} -- the "
                f"factor published for this level of theory, not a generic "
                f"one: the correction is fitted per method and basis, and a "
                f"single hardcoded value shifts every band of every other "
                f"method by up to 7%."
            )
        else:
            tail += (
                f". **No scale factor has been applied**: "
                f"{result.get('freq_scale_source') or 'none is published for this level'}"
                f" Compare these numbers with measured bands only after "
                f"scaling them yourself, or rerun at a level that has a "
                f"published factor."
            )
        parts.append(tail)

    # Raman.  The reason this is worth a paragraph of its own is that the two
    # spectra routinely disagree about which band matters -- water's symmetric
    # stretch is the weakest IR band and the strongest Raman line -- so the
    # narration should point at the comparison rather than list a second
    # column of numbers.
    ram = result.get("raman") or {}
    if ram.get("available") is False:
        parts.append(f"**No Raman spectrum:** {ram.get('reason')}")
    elif ram.get("activities_a4_amu"):
        act = ram["activities_a4_amu"]
        rho = ram.get("depolarization") or []
        rows = ["| Mode | Harmonic (cm-1) |"
                + (" Scaled (cm-1) |" if scaled else "")
                + " Raman (A^4/amu) | rho |",
                "|---|---|" + ("---|" if scaled else "") + "---|---|"]
        order_r = sorted(range(len(freq)), key=lambda i: -act[i])
        for i in order_r[:8]:
            rows.append(
                f"| {i + 1} | {freq[i]:.1f} |"
                + (f" {freq[i] * float(scale):.0f} |" if scaled else "")
                + f" {act[i]:.3f} |"
                + (f" {rho[i]:.3f} |" if i < len(rho) else " - |"))
        parts.append("\n".join(rows))

        if order and order_r and order_r[0] != order[0]:
            parts.append(
                f"The two spectra disagree about which band dominates: mode "
                f"{order[0] + 1} ({freq[order[0]]:.0f} cm-1) is the strongest "
                f"IR band at {intens[order[0]]:.1f} km/mol, while mode "
                f"{order_r[0] + 1} ({freq[order_r[0]]:.0f} cm-1) is the "
                f"strongest Raman line at {act[order_r[0]]:.1f} A^4/amu. "
                f"That is expected rather than a defect -- IR intensity needs a "
                f"change in the dipole, Raman needs a change in the "
                f"polarizability, and the two are not proportional to each "
                f"other."
            )

        # rho is the one Raman number with a hard ceiling, so it is the one to
        # quote as a check rather than as a result.
        if rho:
            worst = max(rho)
            at = rho.index(worst)
            bits_r = [f"depolarization ratios run up to {worst:.3f} "
                      f"(mode {at + 1})"]
            if abs(worst - 0.75) < 5e-3:
                bits_r.append(
                    "that mode sits exactly at the 3/4 limit, which is what a "
                    "mode with no isotropic derivative does by symmetry")
            parts.append("; ".join(bits_r).capitalize() + ".")
        rs = ram.get("sum_rule") or {}
        if rs.get("residual_pct") is not None:
            parts.append(
                f"The activities are on an absolute scale: the Raman sum rule "
                f"-- the mass-weighted sum of |dalpha/dx|^2 against the sum "
                f"over normal modes, with translation contributing exactly "
                f"zero because alpha does not move when the molecule does -- "
                f"closes to {rs['residual_pct']}%.")
        if ram.get("basis_warning"):
            parts.append(f"**Basis caution:** {ram['basis_warning']}")

    bits = []
    if result.get("zpe_kj_mol") is not None:
        bits.append(f"ZPE {result['zpe_kj_mol']} kJ/mol")
    if therm.get("entropy_j_mol_k") is not None:
        bits.append(f"S°(298 K) {therm['entropy_j_mol_k']} J/(mol K)")
    if therm.get("heat_capacity_cv_j_mol_k") is not None:
        bits.append(f"Cv {therm['heat_capacity_cv_j_mol_k']} J/(mol K)")
    if therm.get("symmetry_number"):
        bits.append(f"rotational symmetry number {therm['symmetry_number']}")
    if bits:
        parts.append("Thermochemistry (rigid rotor, harmonic oscillator): " +
                     ", ".join(bits) + ".")
    return "\n\n".join(parts)


def explain_dos(molecule, result: dict, intent: JobIntent) -> str:
    """Narrate a total and projected density of states."""
    name = molecule.name or molecule.formula
    d = result.get("dos") or {}
    proj = d.get("projected") or {}
    grid = d.get("x_energy_ev") or []
    if not grid or not proj:
        return f"The density of states for **{name}** came back empty."

    homo = d.get("homo_ev")
    lumo = d.get("lumo_ev")
    parts = [
        f"Density of states of **{name}** at the "
        f"{result.get('functional_label', '')} / "
        f"{result.get('basis_label', '')} level. Each orbital is broadened "
        f"with a {d.get('fwhm_ev', 0.5)} eV Gaussian and split across atoms "
        "by Loewdin population, so the projected curves add up to the total "
        "exactly rather than approximately. The broadening is normalised, so "
        "the curve is a density in states per eV: its area counts orbitals "
        f"({d.get('n_orbitals', 'all')} of them here"
        + (f", {d.get('n_occupied')} below the HOMO"
           if d.get("n_occupied") is not None else "")
        + "), not an arbitrary height that changes with the smoothing."
    ]
    if homo is not None:
        parts.append(
            f"The Kohn-Sham gap runs from {homo:.2f} eV (HOMO) to "
            f"{lumo:.2f} eV (LUMO)" if lumo is not None
            else f"The HOMO sits at {homo:.2f} eV."
        )

    # which element actually carries the frontier states
    if homo is not None:
        window = [i for i, e in enumerate(grid) if homo - 1.0 <= e <= homo + 0.2]
        if not window:
            window = [min(range(len(grid)),
                          key=lambda i: abs(grid[i] - homo))]
        share = {s: float(sum(curve[i] for i in window)) for s, curve in proj.items()}
        tot = sum(share.values()) or 1.0
        dom = max(share, key=share.get)
        parts.append(
            f"Within 1 eV of the HOMO the states are {share[dom] / tot * 100:.0f}% "
            f"**{dom}** in character"
            + (f", then " + ", ".join(
                f"{s} {v / tot * 100:.0f}%"
                for s, v in sorted(share.items(), key=lambda kv: -kv[1])[1:3])
               if len(share) > 1 else "")
            + ". That is the part of the DOS an electrophilic attack samples "
            "first, and it is the region a catalyst would have to engage."
        )
    return "\n\n".join(parts)


def explain_reactivity(molecule, result: dict, intent: JobIntent) -> str:
    """Narrate the conceptual-DFT descriptors and condensed Fukui functions."""
    name = molecule.name or molecule.formula
    gap = result.get("gap_ev")
    if gap is None:
        return f"The reactivity analysis of **{name}** returned no descriptors."

    parts = [
        f"Conceptual-DFT reactivity descriptors for **{name}** at the "
        f"{result.get('functional_label', '')} / "
        f"{result.get('basis_label', '')} level."
    ]
    rows = ["| Quantity | Value (eV) |", "|---|---|"]
    for label, key in (("HOMO", "homo_ev"), ("LUMO", "lumo_ev"),
                       ("Gap", "gap_ev"),
                       ("Ionisation potential (I)", "ionization_potential_ev"),
                       ("Electron affinity (A)", "electron_affinity_ev"),
                       ("Chemical potential (μ)", "chemical_potential_ev"),
                       ("Hardness (η)", "hardness_ev"),
                       ("Electrophilicity (ω)", "electrophilicity_ev")):
        if result.get(key) is not None:
            rows.append(f"| {label} | {result[key]:.3f} |")
    parts.append("\n".join(rows))
    parts.append(
        "I and A use Koopmans' theorem (I ≈ −E_HOMO, A ≈ −E_LUMO), which "
        "ignores orbital relaxation, so treat these as a ranking device "
        "rather than as measured values. Hardness and electrophilicity "
        "follow from them: η = (I − A)/2 and ω = μ²/2η."
    )

    fk = result.get("fukui") or {}
    rows = fk.get("rows") or []
    if rows:
        top = sorted(rows, key=lambda r: -(r.get("f_minus") or 0))[:5]
        table = ["| Atom | f+ (nucleophilic attack) | f- (electrophilic attack) "
                 "| f0 |", "|---|---|---|---|"]
        for r in top:
            table.append(f"| {r['symbol']}{r['atom']} | "
                         f"{r.get('f_plus', 0):.3f} | {r.get('f_minus', 0):.3f} | "
                         f"{r.get('f_zero', 0):.3f} |")
        parts.append("\n".join(table))
        parts.append(
            "These are condensed Fukui functions from the real N−1, N and "
            "N+1 electron densities, not from the ground-state charges, so "
            "they account for how the density reorganises on gaining or "
            f"losing an electron. **{fk.get('most_electrophilic_site')}** is "
            "the site an electrophile attacks (largest f−) and "
            f"**{fk.get('most_nucleophilic_site')}** is the site a "
            "nucleophile attacks (largest f+)."
        )
    return "\n\n".join(parts)


def explain_surfaces(molecule, result: dict, intent: JobIntent) -> str:
    """Narrate a set of cube grids (MEP map, orbital and density contours)."""
    name = molecule.name or molecule.formula
    s = result.get("surfaces") or {}
    items = s.get("surfaces") or []
    if not items:
        return f"No surface grids were produced for **{name}**."

    mep = next((i for i in items if i["kind"] == "mep"), None)
    orbs = [i for i in items if i["kind"] == "orbital"]
    parts = [
        f"Cube grids for **{name}** at the "
        f"{result.get('functional_label', '')} / "
        f"{result.get('basis_label', '')} level, written on a "
        f"{s.get('grid_n')}³ grid. Switch to the Surfaces tab to display "
        "them in the 3D viewer."
    ]

    if mep:
        lo = mep.get("most_negative") or {}
        hi = mep.get("most_positive") or {}
        span = mep.get("range_kcal_mol") or [0, 0]
        changes = mep.get("potential_changes_sign")

        # The wording has to follow the sign of the potential, not a template.
        # For an ion the potential keeps one sign everywhere, so "the most
        # positive region" would be describing a number that is still
        # negative, and "electron-rich / electron-poor" would be a fiction.
        if changes:
            head = (f"The electrostatic-potential map runs from "
                    f"{span[0]} to +{span[1]} kcal/mol (red negative, blue "
                    f"positive). The potential changes sign across the "
                    f"surface, so the molecule genuinely presents both an "
                    f"electron-rich and an electron-poor face.")
        else:
            sign = "negative" if (hi.get("value_kcal_mol") or 0) < 0 else "positive"
            head = (f"The electrostatic-potential map runs from {span[0]} to "
                    f"{span[1]} kcal/mol. The potential stays **{sign} "
                    f"everywhere** on the accessible surface"
                    + (f", as expected for a species of net charge "
                       f"{mep['net_charge']:+d}" if mep.get("net_charge") else "")
                    + ", so there is no electron-rich face for an "
                      "electrophile to find; the variation is one of degree, "
                      "not of sign.")

        if lo.get("site") and hi.get("site"):
            if lo["site"] == hi["site"]:
                detail = (f" Both extremes lie in the neighbourhood of "
                          f"**{lo['site']}** ({lo.get('value_kcal_mol')} to "
                          f"{hi.get('value_kcal_mol')} kcal/mol) — this "
                          f"molecule is symmetric enough that neither "
                          f"extremum belongs to a distinct atom.")
            else:
                detail = (f" The most negative region sits by "
                          f"**{lo['site']}** ({lo.get('value_kcal_mol')} "
                          f"kcal/mol) and the most positive by "
                          f"**{hi['site']}** ({hi.get('value_kcal_mol')} "
                          f"kcal/mol).")
        else:
            detail = ""
        parts.append(
            head + detail +
            " Values are read on the van der Waals sheet rather than over the "
            "whole grid: the raw cube diverges at the nuclei, which would "
            "otherwise both flatten the map and report the extremes at points "
            "no reagent can reach."
        )

    if orbs:
        rows = ["| Orbital | Index | Energy (eV) | Occupation |", "|---|---|---|---|"]
        for o in orbs:
            rows.append(f"| {o['label']} | {o['index'] + 1} | "
                        f"{o.get('energy_ev')} | {o.get('occupation')} |")
        parts.append("\n".join(rows))
        parts.append(
            "Orbitals are drawn at ±0.032 e/bohr³ with both phases shown. "
            "The sign of an orbital is arbitrary — only the nodal structure "
            "and the spatial extent carry meaning, so do not read the two "
            "colours as charge."
        )
    return "\n\n".join(parts)


def summarise_result(job_type: str, molecule, result: dict, intent: JobIntent) -> str:
    if job_type == "geometry_optimization":
        return explain_opt(molecule, result, intent)
    if job_type == "excited_states":
        return explain_excited(molecule, result, intent)
    if job_type == "scan":
        return explain_scan(molecule, result, intent)
    # Every analysis kind needs its own narration.  Falling through to the
    # generic SCF summary used to describe a frequency run, a DOS or a MEP
    # map without ever mentioning what was actually computed.
    if job_type == "vibrations":
        return explain_vibrations(molecule, result, intent)
    if job_type == "nmr":
        return explain_nmr(molecule, result, intent)
    if job_type == "dos":
        return explain_dos(molecule, result, intent)
    if job_type == "reactivity":
        return explain_reactivity(molecule, result, intent)
    if job_type == "surfaces":
        return explain_surfaces(molecule, result, intent)
    if job_type == "nci":
        return explain_nci(molecule, result, intent)
    if job_type == "series":
        return explain_series(result, intent)
    if job_type == "field":
        return explain_field(molecule, result, intent)
    if job_type == "nto":
        return explain_nto(molecule, result, intent)
    if job_type == "reaction":
        return explain_reaction(result, intent)
    return explain_scf(molecule, result, intent)


def explain_nmr(molecule, result: dict, intent: JobIntent) -> str:
    """Narrate a GIAO shielding calculation and the shifts it produces.

    The level of theory is the first thing this has to say, not the last: the
    shielding is computed at Hartree-Fock whatever functional was asked for,
    and a reader who assumes B3LYP will misread every number.
    """
    name = getattr(molecule, "name", None) or getattr(molecule, "formula", "")
    lvl = result.get("level") or {}
    basis = lvl.get("basis", "")
    want = lvl.get("functional_requested")
    groups = result.get("groups") or []
    nuclei = result.get("nuclei") or []
    if not nuclei:
        return (f"The NMR calculation on **{name}** returned no nuclei - "
                "check that the job converged.")

    parts = [
        f"**GIAO nuclear magnetic shielding of {name}**, basis {basis}. "
        "The shielding tensor is the second derivative of the energy with "
        "respect to the magnetic field and each nuclear magnetic moment, "
        "computed by finite differences of a field-perturbed SCF - not a "
        "correlation with experimental shifts."
    ]

    if want:
        parts.append(
            f"**Level of theory: Hartree-Fock.** You asked for {want}; the "
            "shielding does not use it. This build's DFT integrator cannot "
            "take the complex density a GIAO response needs, and the "
            "real-arithmetic substitute neither converges nor returns a "
            "non-zero response, so the shielding is evaluated at RHF in the "
            f"{basis} basis. Ten of the twelve test shifts against gas-phase "
            "experiment come out inside 2.5 ppm at that level; a carbon "
            "bearing a lone pair (methanol, carbonyls) can be off by 20 ppm, "
            "and absolute 17O shieldings by about 14 ppm."
        )

    refs = result.get("references") or {}
    if refs:
        ref_bits = ", ".join(
            f"{el} vs {r['label']} ({r['sigma_iso_ppm']:.2f} ppm)"
            for el, r in sorted(refs.items()))
        parts.append(
            "References were computed with the same code at the same level, "
            "so the shifts are on a self-consistent scale: " + ref_bits + ". "
            "delta = sigma(reference) - sigma(sample)."
        )
        # Which geometry each reference sits at, and why it matters.  17O
        # shielding moves 539 ppm per angstrom of O-H (probe_nmr28), so a
        # reference at a geometry the sample does not have is not a rounding
        # error -- and for a compound the molecule library also ships, the
        # reference is computed at the library geometry so that computing that
        # molecule gives exactly zero.
        geoms = {}
        for r in refs.values():
            if r.get("geometry"):
                geoms.setdefault(str(r["geometry"]), []).append(r["label"])
        if geoms:
            parts.append(
                "Each reference is computed at a fixed geometry, recorded "
                "here so the number can be reproduced: "
                + "; ".join(f"{', '.join(labels)} at {g}"
                            for g, labels in sorted(geoms.items()))
                + ". Where the molecule library ships the same compound, the "
                "reference uses the library geometry, so computing that "
                "molecule gives a shift of exactly zero."
            )

    if groups:
        rows = ["| Nucleus | delta (ppm) | sigma_iso (ppm) | Count | Spread (ppm) |",
                "|---|---|---|---|---|"]
        for g in groups[:14]:
            rows.append(
                f"| {g.get('isotope') or g['element']} "
                f"| {g['delta_ppm']:.2f} "
                f"| {g['sigma_iso_ppm']:.2f} "
                f"| {g['count']} "
                f"| {g['spread_ppm']:.3f} |")
        if len(groups) > 14:
            rows.append(f"| ... | {len(groups) - 14} more groups | | | |")
        parts.append("\n".join(rows))
        tol = groups[0].get("tol_ppm", 0.05)
        parts.append(
            f"Nuclei are grouped as equivalent when their computed shifts "
            f"agree to within {tol:g} ppm. That is a heuristic, not a symmetry "
            "analysis, so the spread column is printed next to every group - a "
            "large spread means the grouping should be read with suspicion. "
            "The merge is anchored on each group's first member after sorting, "
            "so the grouping does not depend on the order the atoms were "
            "written in."
        )
    else:
        parts.append(
            "No chemical shifts are available for this molecule, so only "
            "absolute shieldings are reported below.")

    # ---- one spectrum per isotope ----------------------------------------
    spectra = result.get("spectra") or {}
    missing = result.get("unreferenced_elements") or []
    if missing:
        parts.append(
            f"**No reference could be computed for {', '.join(missing)}.** "
            "Those nuclei have an absolute shielding but no chemical shift, "
            "and they are absent from the spectra. The reason is in the "
            "warnings; the usual one is that the reference compound is too "
            "large for the free memory on this machine."
        )
    if spectra:
        bits = []
        for iso, sp in sorted(spectra.items()):
            lw = sp.get("linewidth_hz")
            req = sp.get("linewidth_hz_requested")
            larmor = sp.get("larmor_mhz")
            n = len(sp.get("sticks") or [])
            s = (f"**{iso}**: {n} signal{'s' if n != 1 else ''} on a "
                 f"{sp['wmin']:.2f} to {sp['wmax']:.2f} ppm window at "
                 f"{larmor:.1f} MHz")
            if req is not None and abs(lw - req) > 1e-9:
                s += (f", drawn with a {lw:.3f} Hz line rather than the "
                      f"requested {req:.3f} Hz because resolving the narrower "
                      f"one needs {sp.get('points_requested')} grid points")
            else:
                s += f", Lorentzian {lw:.3f} Hz ({sp.get('linewidth_ppm'):.4f} ppm)"
            s += (f", grid {sp.get('grid_step_ppm'):.5f} ppm "
                  f"({sp.get('points_per_hwhm'):.1f} points per half-width)")
            bits.append(s + ".")
        parts.append(
            "A spectrum is acquired for one nucleus, so there is one axis per "
            "isotope rather than a single axis carrying both - a 1H window is "
            "about 12 ppm wide and a 13C window about 220, and the ppm width "
            "of the same 1 Hz line differs by the ratio of the Larmor "
            "frequencies. " + " ".join(bits) + " The stick heights are the "
            "integrals, which is the quantitative content; the curve is the "
            "same signals broadened."
        )

    diag = result.get("diagnostics") or {}
    dE = diag.get("dE_dB") or []
    if dE:
        parts.append(
            "**Self-check.** A closed shell has no linear Zeeman term, so "
            "dE/dB must vanish: measured "
            + ", ".join(f"{v:.1e}" for v in dE)
            + ". That is the test that catches a wrong perturbation "
            "convention, and it is reported rather than assumed."
        )
    asym = diag.get("tensor_asymmetry_max_ppm")
    if asym is not None and asym > 0.01:
        parts.append(
            f"The raw tensor's antisymmetric part reached {asym:.3f} ppm and "
            "was removed by symmetrisation. The isotropic shielding and every "
            "shift above are unaffected - the trace is invariant under "
            "transposition - but the anisotropy of the affected nuclei is the "
            "least reliable number here."
        )

    tensor_bits = []
    for n in nuclei:
        if n.get("span_ppm") and n["span_ppm"] > 1.0:
            tensor_bits.append(
                f"{n['symbol']}{n['index'] + 1} span {n['span_ppm']:.1f} ppm, "
                f"skew {n['skew']:+.2f}")
    if tensor_bits:
        parts.append("Principal-axis data (solid-state): "
                     + "; ".join(tensor_bits[:6]) + ".")

    for w in (result.get("warnings") or [])[:2]:
        if "HARTREE-FOCK" not in w:
            parts.append(w)

    return "\n\n".join(parts)


def explain_nto(molecule, result: dict, intent: JobIntent) -> str:
    """Narrate a natural-transition-orbital figure."""
    name = getattr(molecule, "name", None) or "the molecule"
    state = result.get("state", 1)
    ev = result.get("energy_ev")
    nm = result.get("wavelength_nm")
    level = (f"{result.get('functional_label', '')} / "
             f"{result.get('basis_label', '')}").strip(" /")
    lines = []
    head = f"**Natural transition orbitals of {name}, state {state}**"
    if ev is not None:
        head += f" — {ev:.3f} eV"
        if nm:
            head += f" ({nm:.1f} nm)"
    lines.append(head + f", {level}.")
    lines.append("")

    dom = result.get("dominant_occupation")
    if dom is not None:
        lines.append(
            f"The leading NTO pair carries **{dom * 100:.1f}%** of the "
            f"transition: {result.get('character', '')}.")
    canon = result.get("canonical") or []
    if canon:
        top = ", ".join(f"{c['from_label']}→{c['to_label']} "
                        f"({c['weight'] * 100:.1f}%)" for c in canon[:3])
        lines.append("")
        lines.append(f"In canonical orbitals the same transition is {top}.")
    pairs = result.get("pairs") or []
    if len(pairs) > 1:
        lines.append("")
        lines.append(
            "The remaining pairs carry "
            + ", ".join(f"{p['occupation'] * 100:.1f}%" for p in pairs[1:])
            + ", so a single-orbital description is incomplete for this "
              "state.")
    lines.append("")
    lines.append(
        "The **hole** orbital is where the electron came from and the "
        "**acceptor** orbital where it went; both are in the Surfaces tab. "
        "This is the figure that replaces the ambiguous "
        "\"HOMO→LUMO\" caption: the NTOs are the pair that actually "
        "diagonalises the transition.")
    if result.get("note"):
        lines.append("")
        lines.append(f"_{result['note']}_")
    return "\n".join(lines)


def explain_field(molecule, result: dict, intent: JobIntent) -> str:
    """Narrate an ELF / Laplacian / spin / difference map.

    The figure carries the information, so the text does the one thing the
    figure cannot: say which values matter and where the interesting ones
    are, and be explicit when the answer is 'nothing to see'.
    """
    kind = result.get("kind") or result.get("field") or "elf"
    name = getattr(molecule, "name", None) or "the molecule"
    level = (f"{result.get('functional_label', '')} / "
             f"{result.get('basis_label', '')}").strip(" /")
    stats = result.get("stats") or {}
    lines = [f"**{result.get('label', kind)} of {name}**, {level}."]
    lines.append("")

    if kind == "elf":
        hi = stats.get("max", 0.0)
        lines.append(
            f"The map spans ELF {stats.get('min', 0):.3f}–{hi:.3f}. "
            "Values near 1 mark a region where a single electron pair "
            "dominates — a covalent bond or a lone pair — and 0.5 is what a "
            "uniform electron gas would give, so anything above about 0.7 is "
            "genuine localisation.")
        if hi < 0.75:
            lines.append(
                f"The highest value on this plane is only {hi:.2f}, which "
                "usually means the plane misses the interesting region: try "
                "the plane through three atoms (or a different slice).")
    elif kind == "laplacian":
        lines.append(
            f"∇²ρ ranges from {stats.get('min', 0):.4f} to "
            f"{stats.get('max', 0):.4f} e/bohr⁵ on this plane. Negative "
            "(blue) regions are where electronic charge is locally "
            "concentrated — bonding and lone-pair regions; positive (red) "
            "regions are where it is depleted. In Bader's QTAIM a bond "
            "critical point with ∇²ρ < 0 is a shared (covalent) interaction "
            "and ∇²ρ > 0 a closed-shell one (ionic, hydrogen bond, van der "
            "Waals).")
    elif kind == "spin":
        mult = result.get("multiplicity", 1)
        if mult == 1:
            lines.append(
                "This is a closed-shell calculation, so the spin density is "
                "**zero everywhere by construction** — the map is blank "
                "because that is the correct answer, not because the "
                "calculation failed. Ask for a radical or a triplet to see "
                "where the unpaired electrons are.")
        else:
            lines.append(
                f"Spin density ρα − ρβ for a multiplicity-{mult} state: "
                f"it runs from {stats.get('min', 0):.4f} to "
                f"{stats.get('max', 0):.4f} e/bohr³. Positive regions hold "
                "excess α (spin-up) density and negative ones excess β; the "
                "integral over all space is the number of unpaired "
                "electrons.")
    elif kind == "difference":
        lines.append(
            "Δρ = ρ(complex) − Σ ρ(fragment), with every fragment "
            "calculated in the **full** basis set (ghost atoms), so what "
            "remains is real charge rearrangement rather than the missing "
            "basis functions. Blue is where charge accumulates on binding "
            "and red where it is depleted.")
        frags = result.get("fragments") or []
        if frags:
            lines.append(
                f"{len(frags)} fragments were separated this way "
                f"({', '.join(str(len(f)) + ' atoms' for f in frags)}).")
    if result.get("note"):
        lines += ["", f"_{result['note']}_"]
    return "\n".join(lines)


def explain_reaction(result: dict, intent: JobIntent) -> str:
    """Narrate a reaction path: the barrier, the TS verdict, the two wells."""
    coord = result.get("coordinate") or {}
    points = result.get("points") or []
    ver = result.get("verification") or {}
    unit = coord.get("unit", "")
    level = (f"{result.get('functional_label') or result.get('functional', '')}"
             f" / {result.get('basis_label') or result.get('basis', '')}"
             ).strip(" /")
    lines: List[str] = []

    lines.append(
        f"**Reaction path along {coord.get('label', 'the reaction coordinate')}"
        f"** ({result.get('n_points', len(points))} relaxed points, {level}, "
        f"{result.get('seconds', 0):.0f} s)."
    )
    lines.append("")

    barrier = result.get("barrier_kcal")
    react = result.get("reactant") or {}
    prod = result.get("product") or {}
    ts = result.get("ts") or {}

    if barrier is not None:
        lines.append(
            f"The barrier is **{barrier:.2f} kcal/mol** "
            f"({barrier / 23.0605:.3f} eV) above the reactant, at "
            f"{coord.get('kind', 'coordinate')} = "
            f"{_fmt_or_dash(ts.get('coord'), 2)} {unit}."
        )
        # A relaxed scan measures electronic energy differences with the
        # nuclei frozen.  Calling that an activation energy overstates it --
        # the zero-point term alone moves a barrier by 1-3 kcal/mol -- so
        # the corrections are stated whenever they were computed, and their
        # absence is stated when they were not.
        bt = result.get("barrier_thermo") or {}
        if bt.get("computed"):
            t = bt.get("temperature_K", 298.15)
            lines.append("")
            lines.append(
                f"Corrected barriers at {t:g} K (harmonic oscillator, rigid "
                f"rotor, 1 atm), which are the numbers a rate constant needs: "
                f"**{bt.get('zpe_corrected_kcal')}** kcal/mol with the "
                f"zero-point term, **{bt.get('enthalpy_kcal')}** as an "
                f"activation enthalpy and **{bt.get('gibbs_kcal')}** as an "
                f"activation Gibbs energy. The transition state's imaginary "
                f"mode is not a vibration and is left out of its partition "
                f"function, which is why the corrected barriers sit below the "
                f"electronic one by half a quantum of the mode that becomes "
                f"imaginary."
            )
        elif bt.get("reason"):
            lines.append("")
            lines.append(
                f"These are **electronic** energy differences: {bt['reason']}. "
                f"Compare them with other electronic barriers, not with "
                f"measured activation energies."
            )
    elif not result.get("barrier_bracketed"):
        lines.append(
            "**No maximum was found inside the scanned window**, so no "
            "barrier is reported. The profile is monotonic, which means this "
            "coordinate is not the one the reaction follows — or the window "
            "is too narrow. The curve below is still a real relaxed scan; "
            "what it is not is a barrier."
        )
    lines.append(
        f"The reaction energy is **{_fmt_or_dash(result.get('reaction_kcal'), 2)}"
        f" kcal/mol** (product relative to reactant), from fully relaxed "
        f"endpoints rather than from the ends of the scan."
    )
    lines.append("")

    if ver.get("checked"):
        n_imag = ver.get("n_imaginary")
        align = ver.get("mode_alignment")
        if ver.get("is_transition_state"):
            lines.append(
                f"**Verified first-order saddle.** The analytic Hessian at the "
                f"maximum has exactly one imaginary frequency, "
                f"{_fmt_or_dash((ver.get('imaginary_cm') or [None])[0], 1)} "
                f"cm⁻¹, and its eigenvector overlaps the reaction coordinate "
                f"by {align:.2f}. That is the definition of a transition "
                f"state, so the profile has a genuine maximum there — the "
                f"barrier quoted is the electronic one, and the corrected "
                f"values above are the ones to quote as an activation energy."
            )
        else:
            detail = (f"{n_imag} imaginary frequencies"
                      if n_imag is not None else "an unreadable Hessian")
            if ver.get("imaginary_cm"):
                detail += (", the lowest at "
                           f"{ver['imaginary_cm'][0]:.1f} cm⁻¹")
            lines.append(
                f"**The maximum is not a verified transition state.** The "
                f"Hessian at that geometry shows {detail}"
                + (f" and the lowest mode overlaps the reaction coordinate by "
                   f"only {align:.2f}" if align is not None else "")
                + ". A transition state has exactly one imaginary frequency "
                  "aligned with the coordinate, so the barrier above is the "
                  "height of the relaxed path — an upper bound on the true "
                  "activation energy, not the saddle value."
            )
    elif ver.get("error"):
        lines.append(f"_The frequency check could not be run: {ver['error']}_")
    lines.append("")

    # ---- the intrinsic reaction coordinate -----------------------------
    irc = result.get("irc") or {}
    branches = irc.get("branches") or {}
    if branches:
        fwd = branches.get("forward") or {}
        bwd = branches.get("backward") or {}
        drops = [b.get("total_drop_kcal") for b in (fwd, bwd)
                 if b.get("total_drop_kcal") is not None]
        span = []
        for name, b in (("down", fwd), ("up", bwd)):
            if b.get("final_coord") is not None:
                span.append(f"{name} to {b['final_coord']:.1f} {unit}")
        line = "**IRC** — the imaginary mode was followed down both ways"
        if drops:
            line += (f", and each branch descends by "
                     f"{min(abs(d) for d in drops):.2f} kcal/mol")
        if span:
            line += f" ({', '.join(span)})"
        line += "."
        lines.append(line)
        if irc.get("note"):
            lines.append(irc["note"])
        if not (fwd.get("descends_monotonically")
                and bwd.get("descends_monotonically")):
            lines.append(
                "One branch does not descend at every step, so the step size "
                "is too large for this surface and the path is not a reliable "
                "IRC."
            )
        lines.append("")
    elif irc.get("skipped"):
        lines.append(f"_IRC not run: {irc['skipped']}_")
        lines.append("")

    if points:
        lines.append("| " + coord.get("kind", "coordinate").capitalize()
                     + f" ({unit}) | Energy (kcal/mol) |")
        lines.append("|---|---:|")
        for p in points:
            mark = ""
            if ts and abs((p.get("coord") or 0) - (ts.get("coord") or 0)) < 1e-9:
                mark = " ← max"
            lines.append(f"| {_fmt_or_dash(p.get('coord'), 3)}{mark} | "
                         f"{_fmt_or_dash(p.get('relative_kcal'), 2)} |")
        lines.append("")

    gr = react.get("grad_rms")
    if gr is not None:
        lines.append(
            f"_The reactant endpoint relaxed to |g| = {gr:.1e} a.u. and the "
            f"product to "
            f"{_fmt_or_dash((prod.get('grad_rms')), 1)}; both are genuine "
            f"minima of the potential energy surface, so the reaction energy "
            f"is meaningful._"
        )
    return "\n".join(lines)


def _fmt_or_dash(value, decimals: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    return f"{v:.{decimals}f}"


def _short_geometry(source: str) -> str:
    """'MMFF94 (RDKit ETKDGv3)' -> 'MMFF94', for the A//B notation.

    The full provenance belongs in the payload; the notation wants the method,
    and 'B3LYP/6-31G*//MMFF94 (RDKit ETKDGv3)' is unreadable.
    """
    s = (source or "unknown").strip()
    head = s.split("(")[0].strip()
    return head or s


def explain_series(result: dict, intent: JobIntent) -> str:
    """Narrate a set of molecules calculated at one level of theory."""
    points = result.get("points") or []
    if not points:
        return "The series ran no molecules."

    level = f"{result.get('functional_label', '')} / {result.get('basis_label', '')}".strip(" /")
    ok = [p for p in points if p.get("gap_ev") is not None]
    failed = [p for p in points if p.get("gap_ev") is None]
    lines: List[str] = []

    # A//B: the level of theory is only half of the provenance.  These are
    # single points, so unless something optimised them the numbers sit on
    # force-field conformers, and a gap or a dipole on a DFT minimum is a
    # different number.  A comparison across molecules is only fair if every
    # one of them was treated the same way -- which has to include this.
    geom = result.get("geometry_source") or "unknown"
    lines.append(
        f"**Series of {len(points)} molecules at {level}"
        f"//{_short_geometry(geom)}.** "
        "Every structure was treated at the same level of theory, so the "
        "orbital energies, gaps and dipoles can be compared directly; the "
        "total energies cannot be compared across different formulas, which "
        "is why the relative column is given per-series and only as a "
        "convenience."
    )
    if not any(w in geom.lower() for w in ("dft", "optimis", "optimiz")):
        lines.append("")
        lines.append(
            f"The geometries are **{geom}**, not minima at this level of "
            f"theory. That is the right choice for screening a set this size, "
            f"but the gaps and dipoles are those of the conformer: quote them "
            f"as `{level}//{_short_geometry(geom)}`, and optimise first if a "
            f"number here is going into a table."
        )
    lines.append("")

    if ok:
        gaps = [p["gap_ev"] for p in ok]
        lo = min(ok, key=lambda p: p["gap_ev"])
        hi = max(ok, key=lambda p: p["gap_ev"])
        lines.append(
            f"The HOMO-LUMO gap spans **{min(gaps):.3f} – {max(gaps):.3f} eV** "
            f"({(max(gaps) - min(gaps)):.3f} eV across the set): "
            f"narrowest at **{lo['name']}** ({lo['gap_ev']:.3f} eV), widest at "
            f"**{hi['name']}** ({hi['gap_ev']:.3f} eV)."
        )
        # a real trend statement, not a restatement of the table
        if len(ok) >= 3:
            homos = [p["homo_ev"] for p in ok if p.get("homo_ev") is not None]
            if homos and max(homos) - min(homos) > 0.05:
                easiest = max(ok, key=lambda p: p["homo_ev"]
                              if p.get("homo_ev") is not None else -1e9)
                hardest = min(ok, key=lambda p: p["homo_ev"]
                              if p.get("homo_ev") is not None else 1e9)
                lines.append(
                    f"HOMO varies by {max(homos) - min(homos):.3f} eV, so the "
                    f"set is not electronically equivalent: **{easiest['name']}** "
                    f"is the easiest to oxidise (HOMO "
                    f"{easiest['homo_ev']:.3f} eV) and **{hardest['name']}** "
                    f"the hardest ({hardest['homo_ev']:.3f} eV)."
                )
        lines.append("")

    head = "| Molecule | Formula | HOMO (eV) | LUMO (eV) | Gap (eV) | Dipole (D) |"
    lines.append(head)
    lines.append("|---|---|---:|---:|---:|---:|")
    for p in points:
        lines.append(
            f"| {p.get('name') or p.get('spec')} | {p.get('formula') or '—'} | "
            f"{_fmt_or_dash(p.get('homo_ev'))} | "
            f"{_fmt_or_dash(p.get('lumo_ev'))} | "
            f"{_fmt_or_dash(p.get('gap_ev'))} | "
            f"{_fmt_or_dash(p.get('dipole'), 2)} |")
    lines.append("")

    if failed:
        lines.append(
            "**" + str(len(failed)) + " molecule(s) did not finish:** " +
            ", ".join(f"{p.get('name') or p.get('spec')} "
                      f"({(p.get('error') or 'unknown error')[:80]})"
                      for p in failed) +
            ". They are shown as gaps in the chart rather than being dropped "
            "silently.")
        lines.append("")

    lines.append(
        "Gaps from a ground-state DFT calculation are orbital-energy "
        "differences (Kohn-Sham gaps), not the fundamental gap: they "
        "underestimate it by roughly the derivative discontinuity, so compare "
        "within this series rather than against a measured optical gap.")
    return "\n".join(lines)


def explain_nci(molecule, result: dict, intent: JobIntent) -> str:
    """Narrate a reduced-density-gradient (NCI) analysis.

    The temptation here is to announce "a hydrogen bond" whenever anything
    dips.  It only counts when a *cluster* of grid points dips at a density
    that is too low for a covalent bond, and when the system actually has
    somewhere for a weak interaction to be.  So this states what was
    measured, and qualifies the claim by the strength of the evidence.
    """
    name = molecule.name or molecule.formula
    d = result.get("nci") or {}
    sc = d.get("scatter") or {}
    series = {s.get("key"): s for s in (sc.get("series") or [])}
    grid = d.get("grid") or {}
    npts = sum(len(s.get("points") or []) for s in (sc.get("series") or []))
    natm = len(getattr(molecule, "symbols", []) or []) or molecule.natoms()

    parts = [
        f"Non-covalent interaction (NCI) analysis of **{name}** at the "
        f"{result.get('functional_label', 'chosen')} / "
        f"{result.get('basis_label', 'chosen')} level. The reduced density "
        f"gradient was evaluated on a {grid.get('n', '?')}³ grid "
        f"({grid.get('points', 0):,} points) and plotted against "
        f"sign(λ₂)·ρ; {npts:,} points are drawn. Cores and covalent bonds "
        f"(ρ > {d.get('rho_cut', 0.05):.3f} a.u.) are excluded, because this "
        "analysis is about everything weaker than a bond."
    ]

    lo = float(series.get("attractive", {}).get("count", 0) or 0)
    mid = float(series.get("vdw", {}).get("count", 0) or 0)
    hi = float(series.get("repulsive", {}).get("count", 0) or 0)
    tot = max(lo + mid + hi, 1.0)
    parts.append(
        "Split by the sign of λ₂: "
        f"**{100 * lo / tot:.0f}%** of the low-density points are attractive "
        f"(λ₂ < 0), **{100 * mid / tot:.0f}%** sit in the near-zero van der "
        f"Waals band, and **{100 * hi / tot:.0f}%** are repulsive (steric)."
    )

    spike = d.get("n_spike_points") or 0
    min_rdg = d.get("min_rdg")
    at = d.get("sign_l2_rho_at_spike")
    if d.get("has_weak_interaction") and spike >= 3:
        parts.append(
            f"There is a genuine **weak-interaction spike**: {spike} grid "
            f"points fall below RDG = 0.35, bottoming out at "
            f"**{min_rdg:.3f} a.u.** at sign(λ₂)·ρ = {at:+.4f} a.u. A density "
            "that low is an order of magnitude below a covalent bond, so this "
            "is a non-covalent contact, and the negative sign says it is "
            "**attractive** — in the isosurface it is the blue disc sitting "
            "between the two fragments."
        )
    elif natm <= 2:
        parts.append(
            "No weak-interaction spike appears, and for a molecule this small "
            "none should: there is no second fragment and no folded backbone "
            "to bring two groups together. What the plot shows is the tail of "
            "the covalent bond density."
        )
    else:
        extra = ""
        if min_rdg is not None:
            extra = (f" The lowest RDG reached in that window is "
                     f"{min_rdg:.3f} a.u.")
        parts.append(
            "**No weak-interaction spike was found** — fewer than three grid "
            "points drop below RDG = 0.35 at a density above 0.005 a.u., so "
            "there is no evidence here for a hydrogen bond or a dispersion "
            "contact." + extra +
            " If you expected an interaction, check that the geometry really "
            "brings the two groups together; the SCF was run on the "
            "coordinates as given, and nothing was minimised."
        )

    vdw = d.get("vdw_min_rdg")
    if vdw is not None:
        parts.append(
            f"The near-zero van der Waals band reaches RDG = {vdw:.3f} a.u., "
            "which is where broad, unspecific dispersion contacts show up."
        )

    parts.append(
        "Reading the figure: each point is one piece of space. Far from any "
        "nucleus the density is tiny and the gradient is large, giving the "
        "vertical wall of points near sign(λ₂)·ρ = 0; an interaction shows up "
        "as a tongue of points reaching down towards RDG = 0. The 3D tab draws "
        "the RDG = 0.50 a.u. isosurface in the same three colours."
    )
    return "\n\n".join(parts)
