"""
Periodic table data used throughout ChatDFT.

Provides atomic numbers, masses, covalent/van der Waals radii, standard
CPK colours (Jmol palette) and a small library of common monatomic and
polyatomic fragments that the geometry builder can use.

Sources: IUPAC 2021 atomic weights, Cordero et al. (2008) covalent radii,
Alvarez (2013) vdW radii, Jmol colour scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Element:
    symbol: str
    z: int
    name: str
    mass: float
    covalent_radius: float  # Angstrom
    vdw_radius: float  # Angstrom
    color: str  # hex, Jmol/CPK
    group: int
    period: int
    valence: int  # typical bonding valence


_RAW = [
    # symbol, z, name, mass, cov, vdw, color, group, period, valence
    ("H", 1, "Hydrogen", 1.008, 0.31, 1.20, "#FFFFFF", 1, 1, 1),
    ("He", 2, "Helium", 4.0026, 0.28, 1.40, "#D9FFFF", 18, 1, 0),
    ("Li", 3, "Lithium", 6.94, 1.28, 1.82, "#CC80FF", 1, 2, 1),
    ("Be", 4, "Beryllium", 9.0122, 0.96, 1.53, "#C2FF00", 2, 2, 2),
    ("B", 5, "Boron", 10.81, 0.84, 1.92, "#FFB5B5", 13, 2, 3),
    ("C", 6, "Carbon", 12.011, 0.76, 1.70, "#909090", 14, 2, 4),
    ("N", 7, "Nitrogen", 14.007, 0.71, 1.55, "#3050F8", 15, 2, 3),
    ("O", 8, "Oxygen", 15.999, 0.66, 1.52, "#FF0D0D", 16, 2, 2),
    ("F", 9, "Fluorine", 18.998, 0.57, 1.47, "#90E050", 17, 2, 1),
    ("Ne", 10, "Neon", 20.180, 0.58, 1.54, "#B3E3F5", 18, 2, 0),
    ("Na", 11, "Sodium", 22.990, 1.66, 2.27, "#AB5CF2", 1, 3, 1),
    ("Mg", 12, "Magnesium", 24.305, 1.41, 1.73, "#8AFF00", 2, 3, 2),
    ("Al", 13, "Aluminium", 26.982, 1.21, 1.84, "#BFA6A6", 13, 3, 3),
    ("Si", 14, "Silicon", 28.085, 1.11, 2.10, "#F0C8A0", 14, 3, 4),
    ("P", 15, "Phosphorus", 30.974, 1.07, 1.80, "#FF8000", 15, 3, 3),
    ("S", 16, "Sulfur", 32.06, 1.05, 1.80, "#FFFF30", 16, 3, 2),
    ("Cl", 17, "Chlorine", 35.45, 1.02, 1.75, "#1FF01F", 17, 3, 1),
    ("Ar", 18, "Argon", 39.948, 1.06, 1.88, "#80D1E3", 18, 3, 0),
    ("K", 19, "Potassium", 39.098, 2.03, 2.75, "#8F40D4", 1, 4, 1),
    ("Ca", 20, "Calcium", 40.078, 1.76, 2.31, "#3DFF00", 2, 4, 2),
    ("Sc", 21, "Scandium", 44.956, 1.70, 2.15, "#E6E6E6", 3, 4, 3),
    ("Ti", 22, "Titanium", 47.867, 1.60, 2.11, "#BFC2C7", 4, 4, 4),
    ("V", 23, "Vanadium", 50.942, 1.53, 2.07, "#A6A6AB", 5, 4, 5),
    ("Cr", 24, "Chromium", 51.996, 1.39, 2.06, "#8A99C7", 6, 4, 3),
    ("Mn", 25, "Manganese", 54.938, 1.39, 2.05, "#9C7AC7", 7, 4, 4),
    ("Fe", 26, "Iron", 55.845, 1.32, 2.04, "#E06633", 8, 4, 3),
    ("Co", 27, "Cobalt", 58.933, 1.26, 2.00, "#F090A0", 9, 4, 3),
    ("Ni", 28, "Nickel", 58.693, 1.24, 1.97, "#50D050", 10, 4, 2),
    ("Cu", 29, "Copper", 63.546, 1.32, 1.96, "#C88033", 11, 4, 2),
    ("Zn", 30, "Zinc", 65.38, 1.22, 2.01, "#7D80B0", 12, 4, 2),
    ("Ga", 31, "Gallium", 69.723, 1.22, 1.87, "#C28F8F", 13, 4, 3),
    ("Ge", 32, "Germanium", 72.630, 1.20, 2.11, "#668F8F", 14, 4, 4),
    ("As", 33, "Arsenic", 74.922, 1.19, 1.85, "#BD80E3", 15, 4, 3),
    ("Se", 34, "Selenium", 78.971, 1.20, 1.90, "#FFA100", 16, 4, 2),
    ("Br", 35, "Bromine", 79.904, 1.20, 1.85, "#A62929", 17, 4, 1),
    ("Kr", 36, "Krypton", 83.798, 1.16, 2.02, "#5CB8D1", 18, 4, 0),
    ("Rb", 37, "Rubidium", 85.468, 2.20, 3.03, "#702EB0", 1, 5, 1),
    ("Sr", 38, "Strontium", 87.62, 1.95, 2.49, "#00FF00", 2, 5, 2),
    ("Y", 39, "Yttrium", 88.906, 1.90, 2.32, "#94FFFF", 3, 5, 3),
    ("Zr", 40, "Zirconium", 91.224, 1.75, 2.23, "#94E0E0", 4, 5, 4),
    ("Mo", 42, "Molybdenum", 95.95, 1.54, 2.17, "#54B5B5", 6, 5, 4),
    ("Ag", 47, "Silver", 107.868, 1.45, 1.72, "#C0C0C0", 11, 5, 1),
    ("Cd", 48, "Cadmium", 112.414, 1.44, 1.58, "#FFD98F", 12, 5, 2),
    ("In", 49, "Indium", 114.818, 1.42, 1.93, "#A67573", 13, 5, 3),
    ("Sn", 50, "Tin", 118.710, 1.39, 2.17, "#668080", 14, 5, 4),
    ("Sb", 51, "Antimony", 121.760, 1.39, 2.06, "#9E63B5", 15, 5, 3),
    ("Te", 52, "Tellurium", 127.60, 1.38, 2.06, "#D47A00", 16, 5, 2),
    ("I", 53, "Iodine", 126.904, 1.39, 1.98, "#940094", 17, 5, 1),
    ("Xe", 54, "Xenon", 131.293, 1.40, 2.16, "#429EB0", 18, 5, 0),
    ("Pt", 78, "Platinum", 195.084, 1.36, 1.75, "#D0D0E0", 10, 6, 2),
    ("Au", 79, "Gold", 196.967, 1.36, 1.66, "#FFD123", 11, 6, 1),
    ("Hg", 80, "Mercury", 200.592, 1.32, 1.55, "#B8B8D0", 12, 6, 2),
    ("Pb", 82, "Lead", 207.2, 1.46, 2.02, "#575961", 14, 6, 2),
    ("Bi", 83, "Bismuth", 208.980, 1.48, 2.07, "#9E4FB5", 15, 6, 3),
]

ELEMENTS: Dict[str, Element] = {}
BY_Z: Dict[int, Element] = {}
for row in _RAW:
    el = Element(*row)
    ELEMENTS[el.symbol] = el
    BY_Z[el.z] = el


# Ground-state spin multiplicities (2S+1) of the free atoms, from Hund's
# rules.  The electron-count parity alone is not enough: atomic oxygen has 8
# electrons but a 3P ground state, and atomic nitrogen has 7 electrons but a
# 4S ground state.
GROUND_STATE_MULT: Dict[str, int] = {
    "H": 2, "He": 1,
    "Li": 2, "Be": 1, "B": 2, "C": 3, "N": 4, "O": 3, "F": 2, "Ne": 1,
    "Na": 2, "Mg": 1, "Al": 2, "Si": 3, "P": 4, "S": 3, "Cl": 2, "Ar": 1,
    "K": 2, "Ca": 1,
    "Sc": 2, "Ti": 3, "V": 4, "Cr": 7, "Mn": 6, "Fe": 5, "Co": 4, "Ni": 3,
    "Cu": 2, "Zn": 1,
    "Ga": 2, "Ge": 3, "As": 4, "Se": 3, "Br": 2, "Kr": 1,
    "Rb": 2, "Sr": 1, "Y": 2, "Zr": 3, "Mo": 7,
    "Ag": 2, "Cd": 1, "In": 2, "Sn": 3, "Sb": 4, "Te": 3, "I": 2, "Xe": 1,
    "Pt": 3, "Au": 2, "Hg": 1, "Pb": 3, "Bi": 4,
}


def ground_state_multiplicity(symbol: str) -> int:
    """Spin multiplicity of the free atom's ground state (falls back to parity)."""
    el = get(symbol)
    if el.symbol in GROUND_STATE_MULT:
        return GROUND_STATE_MULT[el.symbol]
    # Fall back to the electron-count parity rule.
    n = el.z
    return 2 if n % 2 else 1


def get(symbol: str) -> Element:
    """Look up an element; raises a helpful error for unknown symbols."""
    sym = symbol.strip().capitalize() if len(symbol.strip()) > 1 else symbol.strip().upper()
    if sym not in ELEMENTS:
        # try case-insensitive match
        for k, v in ELEMENTS.items():
            if k.lower() == symbol.strip().lower():
                return v
        raise KeyError(f"Element '{symbol}' is not supported by ChatDFT.")
    return ELEMENTS[sym]


def z_to_symbol(z: int) -> str:
    return BY_Z[z].symbol


def symbol_to_z(sym: str) -> int:
    return get(sym).z


# --------------------------------------------------------------------------
# Bond perception thresholds: sum of covalent radii * factor per pair type
# --------------------------------------------------------------------------
BOND_FACTORS = {
    # H-H is the one pair the "sum of radii x 1.15" rule gets wrong.  Two
    # hydrogens bonded to each other sit 0.74 A apart -- H2 is 0.7414
    # experimentally, 0.743 at B3LYP/6-31G* -- and the covalent radii add to
    # 0.62, so 1.15 draws the cutoff at 0.713 A and the bond in dihydrogen is
    # not a bond.  At 1.25 the cutoff is 0.775 A: H2 is perceived, and the
    # next closest H...H contact in the whole library is 1.53 A (water), so
    # nothing is invented.  A shorter H...H pair than H2 is not a structure
    # any molecule has.
    frozenset({"H", "H"}): 1.25,
    frozenset({"C", "H"}): 1.15,
    frozenset({"N", "H"}): 1.15,
    frozenset({"O", "H"}): 1.15,
    frozenset({"C", "C"}): 1.25,
    frozenset({"C", "N"}): 1.25,
    frozenset({"C", "O"}): 1.25,
    frozenset({"C", "S"}): 1.30,
    frozenset({"C", "F"}): 1.20,
    frozenset({"C", "Cl"}): 1.25,
}

DEFAULT_BOND_FACTOR = 1.25


def bond_cutoff(sym_a: str, sym_b: str) -> float:
    """Maximum distance (Angstrom) for considering two atoms bonded."""
    key = frozenset({sym_a, sym_b})
    factor = BOND_FACTORS.get(key, DEFAULT_BOND_FACTOR)
    ra = get(sym_a).covalent_radius
    rb = get(sym_b).covalent_radius
    return (ra + rb) * factor
