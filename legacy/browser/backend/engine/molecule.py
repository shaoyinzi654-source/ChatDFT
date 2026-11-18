"""
Molecule input handling for ChatDFT.

Three routes into a valid 3D structure:

1. ``from_xyz``    - raw XYZ text (most precise, user-supplied geometry)
2. ``from_smiles`` - SMILES string -> RDKit -> ETKDG conformer -> MMFF94
3. ``from_name``   - common-molecule database lookup (built-in library)

Everything returns a :class:`Molecule` with cartesian coordinates in
Angstrom, element symbols and a charge/multiplicity pair ready for the
DFT driver.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from . import elements


@dataclass
class Atom:
    symbol: str
    x: float
    y: float
    z: float
    index: int = 0

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Molecule:
    atoms: List[Atom]
    charge: int = 0
    multiplicity: int = 1
    name: str = ""
    smiles: str = ""
    source: str = "unknown"
    # Where the *coordinates* came from, which is not what ``source`` means
    # ("library", "smiles", "xyz" -- how the molecule was named).  A property
    # computed at B3LYP/6-31G* on a force-field conformer and on a DFT minimum
    # are different numbers, and a result that does not say which one it is
    # cannot be compared with anything.  Written the standard way at the point
    # of use: A//B for a single point at A on a geometry optimised at B.
    geometry_source: str = "unknown"
    formula: str = ""
    bonds: List[Tuple[int, int, float]] = field(default_factory=list)

    def __post_init__(self):
        for i, atom in enumerate(self.atoms):
            atom.index = i
        if not self.formula:
            self.formula = self.compute_formula()
        if not self.bonds:
            self.bonds = self.perceive_bonds()

    # ------------------------------------------------------------------
    def compute_formula(self) -> str:
        counts: Dict[str, int] = {}
        for atom in self.atoms:
            counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
        # Hill notation: C first, H second, then alphabetical
        order = sorted(
            counts,
            key=lambda s: (0 if s == "C" else 1 if s == "H" else 2, s),
        )
        out = ""
        for sym in order:
            n = counts[sym]
            out += sym + (str(n) if n > 1 else "")
        return out

    def mass(self) -> float:
        return sum(elements.get(a.symbol).mass for a in self.atoms)

    def natoms(self) -> int:
        return len(self.atoms)

    def total_electrons(self) -> int:
        return sum(elements.get(a.symbol).z for a in self.atoms) - self.charge

    # ------------------------------------------------------------------
    def perceive_bonds(self) -> List[Tuple[int, int, float]]:
        """Distance-based bond perception using covalent radii."""
        bonds = []
        n = len(self.atoms)
        for i in range(n):
            ai = self.atoms[i]
            for j in range(i + 1, n):
                aj = self.atoms[j]
                cutoff = elements.bond_cutoff(ai.symbol, aj.symbol)
                d = distance(ai, aj)
                if 0.4 < d <= cutoff:
                    bonds.append((i, j, round(d, 4)))
        return bonds

    def to_xyz(self) -> str:
        lines = [str(len(self.atoms)), self.name or self.formula]
        for a in self.atoms:
            lines.append(f"{a.symbol:<3s} {a.x:16.8f} {a.y:16.8f} {a.z:16.8f}")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["formula"] = self.formula
        d["mass"] = round(self.mass(), 4)
        d["natoms"] = self.natoms()
        d["nelectrons"] = self.total_electrons()
        d["bonds"] = [{"i": i, "j": j, "length": L} for i, j, L in self.bonds]
        return d

    def geometry_summary(self) -> dict:
        """Bond lengths/angles table for the report."""
        rows = []
        for i, j, L in sorted(self.bonds, key=lambda b: b[2]):
            rows.append(
                {
                    "type": "bond",
                    "atoms": f"{self.atoms[i].symbol}{i + 1}-{self.atoms[j].symbol}{j + 1}",
                    "value": round(L, 4),
                    "unit": "Angstrom",
                }
            )
        return {"bonds": rows, "centroid": self.centroid()}

    def centroid(self) -> Tuple[float, float, float]:
        if not self.atoms:
            return (0.0, 0.0, 0.0)
        n = len(self.atoms)
        return (
            sum(a.x for a in self.atoms) / n,
            sum(a.y for a in self.atoms) / n,
            sum(a.z for a in self.atoms) / n,
        )

    def center_in_place(self) -> None:
        cx, cy, cz = self.centroid()
        for a in self.atoms:
            a.x -= cx
            a.y -= cy
            a.z -= cz


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def distance(a: Atom, b: Atom) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


# How a geometry came to be, in the A//B sense a reader needs: these are the
# right-hand half of "B3LYP/6-31G* // <this>".  Naming them once here beats
# repeating strings that drift out of step with the code that builds them.
#
# Which one applies is decided by ``_relax_conformer``, not assumed.  MMFF94
# only covers the elements it was parametrised for -- boron, the bare
# diatomics and several ions have no parameters -- and it reports that by
# returning -1 rather than raising, so a geometry can come out of this module
# having been relaxed by nothing at all.  Calling that MMFF94 would put a
# false claim on every report built from it.
MMFF_GEOMETRY = "MMFF94 (RDKit ETKDGv3)"
UFF_GEOMETRY = "UFF (RDKit ETKDGv3; MMFF94 has no parameters for this molecule)"
UNRELAXED_GEOMETRY = ("RDKit ETKDGv3 embedding; no available force field would "
                      "relax this molecule, so the coordinates are unrelaxed")
MMFF_UNCONVERGED = "MMFF94 (RDKit ETKDGv3; the relaxation did not converge)"

# A library entry whose coordinates were not built by anything this program
# can repeat.  Saying so is worth more than a plausible guess: a reader who
# wants to reproduce the numbers knows to ask.
LIBRARY_UNVERIFIED = ("coordinates from the built-in library; the library does "
                      "not record where they came from")

# The level this program relaxes library geometries at when a force field
# cannot vouch for them.  Written by the optimiser itself -- see
# backend/verify_library_geometry.py -- so the string cannot drift from the
# calculation that produced the coordinates.
DFT_METHOD = "b3lyp"
DFT_BASIS = "6-31g*"
# Deliberately not the string a job writes for a geometry it optimised itself
# ("DFT-optimised (B3LYP / 6-31G*)", in server.py), and deliberately free of
# the word "optimised": the contract fails any job that did not run an
# optimisation yet reports one, and it is right to.  These coordinates are a
# stationary point of that level -- verified with a gradient, which is a claim
# about the coordinates -- but the job reporting them did not find it, and the
# parenthetical says whose geometry it is.  "Stationary point" rather than
# "minimum" because that is what the gradient test establishes: it cannot tell
# a minimum from a saddle.
DFT_GEOMETRY = "B3LYP / 6-31G* stationary point (built-in library geometry)"


def _relax_conformer(mol) -> str:
    """Relax ``mol`` in place and return the provenance string it earned.

    The return value of ``MMFFOptimizeMolecule`` is the only way to learn
    whether a force field was applied at all: it answers -1 when MMFF94 has no
    parameters, and 1 when it ran but did not converge, and it neither raises
    nor warns.  A caller that ignores it cannot tell a relaxed conformer from
    a raw embedding, and every molecule MMFF94 cannot describe -- boranes, the
    diatomics, bare ions -- would go out labelled with a force field that was
    never used.

    When MMFF94 cannot be used the conformer is relaxed with UFF instead and
    the label says so; when nothing works the label says the coordinates are
    unrelaxed.  An honest, weaker claim beats a stronger false one.
    """
    from rdkit.Chem import AllChem

    if AllChem.MMFFGetMoleculeProperties(mol) is not None:
        try:
            rc = AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        except Exception:
            rc = -1
        if rc == 0:
            return MMFF_GEOMETRY
        if rc == 1:
            # MMFF94 ran to the iteration limit: the coordinates are part
            # relaxed and part whatever the embedding produced.
            return MMFF_UNCONVERGED

    try:
        if AllChem.UFFOptimizeMolecule(mol, maxIters=1000) == 0:
            return UFF_GEOMETRY
    except Exception:
        pass
    return UNRELAXED_GEOMETRY


def parse_xyz(text: str, name: str = "") -> Molecule:
    """Parse XYZ text, tolerating blank lines and extra columns."""
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("XYZ block is too short - need a count line, a comment and atoms.")
    try:
        n = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise ValueError("First line of an XYZ file must be the atom count.")
    body = lines[2 : 2 + n]
    if len(body) < n:
        raise ValueError(f"XYZ declares {n} atoms but only {len(body)} were found.")
    atoms = []
    for ln in body:
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ atom line: '{ln}'")
        sym = parts[0]
        elements.get(sym)  # validate
        atoms.append(Atom(sym, float(parts[1]), float(parts[2]), float(parts[3])))
    title = lines[1].strip() if len(lines) > 1 else ""
    return Molecule(atoms=atoms, name=name or title or "molecule", source="xyz",
                    geometry_source="coordinates as supplied")


# ----------------------------------------------------------------------
# SMILES -> 3D via RDKit
# ----------------------------------------------------------------------
def _canonical_smiles(smiles: str) -> str:
    """RDKit's canonical form of a SMILES string ('' when it will not parse)."""
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else ""
    except Exception:
        return ""


_SMILES_NAME_INDEX: Optional[dict] = None


def _library_name_for_smiles(smiles: str) -> str:
    """Display name of the library molecule whose SMILES matches ``smiles``.

    Without this a SMILES input is labelled with the raw string, so ``CCO``
    would be reported as "cco" instead of "Ethanol".
    """
    global _SMILES_NAME_INDEX
    canon = _canonical_smiles(smiles)
    if not canon:
        return ""
    if _SMILES_NAME_INDEX is None:
        index: dict = {}
        for entry in _load_library().values():
            lib_smiles = entry.get("smiles")
            if not lib_smiles:
                continue
            lib_canon = _canonical_smiles(lib_smiles)
            if lib_canon:
                index.setdefault(lib_canon, _entry_name(entry))
        _SMILES_NAME_INDEX = index
    return _SMILES_NAME_INDEX.get(canon, "")


def from_smiles(smiles: str, name: str = "", seed: int = 0xC0FFEE) -> Molecule:
    """Generate a 3D conformer from SMILES with RDKit (ETKDG + MMFF94)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse the SMILES string '{smiles}'.")

    mol = Chem.AddHs(mol)

    # RDKit stores an isotope label beside the element and returns the plain
    # element from GetSymbol(), so [2H]C([2H])([2H])[2H] arrived here as CH4
    # and was computed as methane: a different molecule from the one asked
    # for, with nothing said.  That is the worst of the available outcomes --
    # the user gets a number, it is a real number, and it answers a question
    # they did not ask.
    #
    # This build has no isotope handling anywhere: masses come from the
    # element table (``elements.get(symbol).mass``), and every shielding is
    # evaluated at the clamped-nucleus level, where 2H and 1H are the same
    # electronic problem.  So a "deuterium spectrum" here would be a proton
    # spectrum wearing a deuterium label, and a "CD4 IR spectrum" would be
    # methane's -- the C-H stretch near 3000 cm-1 is a C-D stretch near 2200.
    # Refusing is the only honest option; approximating it would be exactly
    # the failure the 15N and 31P notes exist to prevent.
    labelled = sorted({(a.GetSymbol(), a.GetIsotope())
                       for a in mol.GetAtoms() if a.GetIsotope()})
    if labelled:
        shown = ", ".join(f"[{iso}{sym}]" for sym, iso in labelled)
        raise ValueError(
            f"this SMILES labels isotopes ({shown}), and this build does not "
            "support them.  A mass number is not a decoration: it changes "
            "the vibrational frequencies, and for deuterium it is a "
            "different nucleus with its own NMR scale and its own reference "
            "compound.  Computing the unlabelled molecule instead would "
            "answer a different question without saying so, so this is "
            "refused rather than approximated.")
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # fall back to a randomised embedding attempt
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError("Failed to generate a 3D conformer for this molecule.")

    geometry_source = _relax_conformer(mol)

    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(Atom(atom.GetSymbol(), p.x, p.y, p.z))

    charge = Chem.GetFormalCharge(mol)
    # electron parity fixes the multiplicity guess
    n_elec = sum(elements.get(a.symbol).z for a in atoms) - charge
    mult = 1 if n_elec % 2 == 0 else 2

    # Adopt the library display name when the SMILES is a known molecule, so
    # the report reads "Ethanol" rather than "CCO".
    if not name:
        name = _library_name_for_smiles(smiles)

    m = Molecule(
        atoms=atoms,
        charge=charge,
        multiplicity=mult,
        name=name or smiles,
        smiles=smiles,
        source="smiles",
        geometry_source=geometry_source,
    )
    m.center_in_place()
    return m


# ----------------------------------------------------------------------
# built-in molecule library
# ----------------------------------------------------------------------
_LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "molecules",
    "library.json",
)

_LIBRARY_CACHE: Optional[dict] = None


def _load_library() -> dict:
    global _LIBRARY_CACHE
    if _LIBRARY_CACHE is None:
        try:
            with open(_LIB_PATH, "r", encoding="utf-8") as fh:
                _LIBRARY_CACHE = json.load(fh)
        except (OSError, json.JSONDecodeError):
            _LIBRARY_CACHE = {}
    return _LIBRARY_CACHE


# ----------------------------------------------------------------------
# library entry fields
# ----------------------------------------------------------------------
# library.json was written by two different scripts with two different
# schemas: the original entries carry ``display_name`` / ``description``, and
# the seventeen that ``extend_library`` appended carried ``name`` / ``desc``
# (plus a redundant ``key``).  Every reader looked for the first pair, so
# those seventeen reached the UI with their lowercase key as their name and a
# blank description -- while the real strings sat in the same dict under a
# name nothing read.
#
# The file has been normalised and ``extend_library`` now emits the canonical
# pair.  These accessors accept both anyway: a schema that a reader silently
# tolerates cannot make a molecule invisible, and the contract asserts that no
# entry is.
def _entry_name(entry: dict, key: str = "") -> str:
    """The display name of a library entry, whichever schema wrote it."""
    name = entry.get("display_name") or entry.get("name") or ""
    return str(name) if name else key


def _entry_desc(entry: dict) -> str:
    """The description of a library entry, whichever schema wrote it."""
    return str(entry.get("description") or entry.get("desc") or "")


def library_names() -> List[dict]:
    lib = _load_library()
    out = []
    for key, entry in lib.items():
        out.append(
            {
                "key": key,
                "name": _entry_name(entry, key),
                "formula": entry.get("formula", ""),
                "smiles": entry.get("smiles", ""),
                "category": entry.get("category", "general"),
                "xyz": entry.get("xyz", ""),
                "desc": _entry_desc(entry),
            }
        )
    return sorted(out, key=lambda d: (d["category"], d["name"]))


# ----------------------------------------------------------------------
# molecule-name aliases
# ----------------------------------------------------------------------
# The app is used in Chinese and every library key is English, so a molecule
# the library plainly holds was unresolvable when it was named the way the
# user actually writes it: "阿司匹林" fell through to a SMILES parse and
# failed, and a design brief that said "类似阿司匹林" silently started from
# nothing at all -- the named molecule was dropped without a word.
#
# Matched *exactly*, never as a substring.  苯 is benzene, but 苯酚 is phenol,
# 苯胺 is aniline and 硝基苯 is nitrobenzene, so a "contains" rule would
# resolve most of the aromatic names to the wrong molecule.  Extra spellings
# simply go in the list.
_LIBRARY_ALIASES: Dict[str, List[str]] = {
    "water": ["水", "水分子"],
    "water_dimer": ["水二聚体", "水二聚物"],
    "hydrogen": ["氢气", "氢分子"],
    "methane": ["甲烷"],
    "ammonia": ["氨", "氨气", "氨分子"],
    "benzene": ["苯", "苯环"],
    "naphthalene": ["萘"],
    "ethylene": ["乙烯"],
    "acetylene": ["乙炔"],
    "formaldehyde": ["甲醛"],
    "acetone": ["丙酮"],
    "phenol": ["苯酚"],
    "pyridine": ["吡啶"],
    "furan": ["呋喃"],
    "pyrrole": ["吡咯"],
    "imidazole": ["咪唑"],
    "methanol": ["甲醇"],
    "ethanol": ["乙醇", "酒精"],
    "acetic_acid": ["乙酸", "醋酸"],
    "glycine": ["甘氨酸"],
    "caffeine": ["咖啡因"],
    "aspirin": ["阿司匹林"],
    "nitrobenzene": ["硝基苯"],
    "aniline": ["苯胺"],
    "styrene": ["苯乙烯"],
    "toluene": ["甲苯"],
    "chloroform": ["氯仿", "三氯甲烷"],
    "dichloromethane": ["二氯甲烷"],
    "tfa": ["三氟乙酸"],
    "urea": ["尿素"],
    "histamine": ["组胺"],
    "ozone": ["臭氧"],
    "co2": ["二氧化碳"],
    "carbon_monoxide": ["一氧化碳"],
    "nitrogen": ["氮气", "氮分子"],
    "nitric_oxide": ["一氧化氮"],
    "sulfur_dioxide": ["二氧化硫"],
    "methanethiol": ["甲硫醇"],
    "hydroxide": ["氢氧根", "氢氧根离子"],
    "ammonium": ["铵根", "铵离子", "铵根离子"],
    "nitrate": ["硝酸根", "硝酸根离子"],
    "methyl_radical": ["甲基自由基"],
    "hydroxyl_radical": ["羟基自由基"],
    "oxygen": ["氧气", "氧分子"],
    "porphine": ["卟吩"],
    "ethane": ["乙烷"],
    "propane": ["丙烷"],
    "butane": ["丁烷"],
    "isobutane": ["异丁烷"],
    "pentane": ["戊烷"],
    "hexane": ["己烷"],
    "cyclohexane": ["环己烷"],
    "butadiene": ["丁二烯"],
    "cyclohexene": ["环己烯"],
    "thiophene": ["噻吩"],
    "pyrazine": ["吡嗪"],
    "pyrimidine": ["嘧啶"],
    "hydrogen_fluoride": ["氟化氢"],
    "hydrogen_chloride": ["氯化氢"],
    "silane": ["硅烷"],
    "borane": ["硼烷"],
    "ammonia_borane": ["氨硼烷"],
}

# alias -> key.  A duplicate alias is a mistake (it would let the table's
# order decide the answer), so the self test asserts there are none.
_ALIAS_TO_KEY: Dict[str, str] = {
    alias: key
    for key, names in _LIBRARY_ALIASES.items()
    for alias in names
}


def _find_library_entry(name: str, fuzzy: bool = True):
    """Locate a library entry by key, display name or formula.

    ``fuzzy`` additionally allows a "contains" match, which is what makes
    "caffeine" find the entry keyed ``caffeine`` even if the user typed
    "Caffeine ".  It must be switched off when the caller is guessing between
    several input syntaxes: a SMILES string like ``CO`` would otherwise match
    the key ``co2``.
    """
    lib = _load_library()
    raw = name.strip()
    needle = raw.lower()
    key = needle.replace(" ", "_")

    entry = lib.get(key)
    if entry is not None:
        return entry, key

    # Chinese (and other) names resolve exactly, and *before* the display-name
    # and formula passes so nothing else can claim them first.
    alias = _ALIAS_TO_KEY.get(raw)
    if alias is not None:
        entry = lib.get(alias)
        if entry is not None:
            return entry, alias

    # Display names are ordinary words and are matched case-insensitively.
    for k, v in lib.items():
        if _entry_name(v).lower() == needle:
            return v, k

    # Chemical formulae are case-sensitive.  "CO" is carbon monoxide while
    # "Co" is the element cobalt, so a case-folded comparison would make the
    # two indistinguishable.
    for k, v in lib.items():
        if v.get("formula", "") == raw:
            return v, k

    # Convenience: accept a fully lower-cased formula ("h2o", "c6h6").  The
    # all-lowercase requirement stops mixed-case tokens being guessed at --
    # "No" is not the formula NO, and element symbols always start uppercase.
    if raw == raw.lower():
        for k, v in lib.items():
            if v.get("formula", "").lower() == needle:
                return v, k

    if fuzzy and needle:
        for k, v in lib.items():
            if needle in k or needle in _entry_name(v).lower():
                return v, k

    return None, None


def from_name(name: str) -> Molecule:
    """Look up a built-in molecule by key, display name or formula."""
    entry, key = _find_library_entry(name, fuzzy=True)
    if entry is None:
        raise KeyError(f"'{name}' is not in the ChatDFT molecule library.")

    m = parse_xyz(entry["xyz"], name=_entry_name(entry, key))
    m.name = _entry_name(entry, key)
    m.smiles = entry.get("smiles", "")
    m.source = "library"
    # Whatever the entry records for itself.  The library is a mixture: some
    # geometries are ETKDG+MMFF94 conformers this program can rebuild on
    # demand, some were relaxed here at B3LYP/6-31G*, and each has to be
    # reported as what it is.  Stamping one string over all of them would
    # misdescribe the majority.
    m.geometry_source = str(entry.get("geometry_source") or LIBRARY_UNVERIFIED)
    m.charge = entry.get("charge", 0)
    m.multiplicity = entry.get("multiplicity", 1)
    m.formula = entry.get("formula", m.compute_formula())
    return m


def resolve(spec: str, kind: str = "auto", charge: int = 0, multiplicity: int = 1) -> Molecule:
    """
    Smart resolver used by the agent layer.

    ``kind`` may be 'auto', 'name', 'smiles', 'xyz' or 'formula'.
    """
    spec = spec.strip()

    if kind == "name":
        return from_name(spec)
    if kind == "smiles":
        return from_smiles(spec)
    if kind == "xyz":
        return parse_xyz(spec)

    # auto-detect
    if re.search(r"^\s*\d+\s*\n", spec) and len(spec.splitlines()) > 2:
        return parse_xyz(spec)

    # An exact match on a built-in entry wins over every other reading: a
    # user who types "water" or "C6H6" means the library molecule, and the
    # strict (non-fuzzy) lookup cannot be fooled by a SMILES string.
    entry, key = _find_library_entry(spec, fuzzy=False)
    if entry is not None:
        return from_name(key)

    # A bare element symbol or element name means the isolated atom.  The
    # symbol test is case-sensitive on purpose: "Co" is cobalt, whereas "CO"
    # is carbon monoxide (or, read as SMILES, methanol) -- never cobalt.
    el = elements.ELEMENTS.get(spec)
    if el is None:
        low = spec.lower()
        el = next((e for e in elements.ELEMENTS.values() if e.name.lower() == low), None)
    if el is not None:
        m = Molecule(atoms=[Atom(el.symbol, 0.0, 0.0, 0.0)], name=el.name, source="atom")
        m.charge = charge
        if charge == 0:
            # Hund's rules give the free-atom ground state (atomic O is a
            # triplet, atomic N a quartet); parity alone would get these wrong.
            m.multiplicity = elements.ground_state_multiplicity(el.symbol)
        else:
            m.multiplicity = 2 if m.total_electrons() % 2 else 1
        return m

    # fall back to SMILES
    try:
        return from_smiles(spec)
    except Exception as exc:
        raise ValueError(
            f"Could not interpret '{spec}' as a molecule name, formula, SMILES or XYZ. ({exc})"
        )
