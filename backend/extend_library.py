"""Add missing molecules to the library.

The library is how names become structures, so a gap in it is a gap in the
whole program: asking for butane used to fail with "could not interpret",
which is the answer a user would read as "this software does not know what
butane is".  The alkanes -- the first homologous series anyone reaches for
when they want a trend -- were missing entirely.

Geometries are not typed in by hand.  Each one is embedded with RDKit
(ETKDG) and relaxed with MMFF94, the same pipeline ``from_smiles`` uses, so
``smiles``, ``formula`` and ``xyz`` cannot drift apart.  The coordinates are
a starting structure for the DFT code, not a result, which is exactly what a
force-field geometry is good for.

Usage:
    python -m backend.extend_library            # add everything below
    python -m backend.extend_library --dry-run  # report, change nothing
"""

from __future__ import annotations

import json
import os
import sys

from backend.engine import molecule as molmod

_LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "molecules", "library.json",
)

# (key, display name, SMILES, category, one-line description)
NEW = [
    ("ethane", "Ethane", "CC", "alkanes",
     "The simplest C-C single bond; staggered D3d ground state."),
    ("propane", "Propane", "CCC", "alkanes",
     "Three-carbon alkane; the first member with a distinct conformer."),
    ("butane", "Butane", "CCCC", "alkanes",
     "Linear C4H10; anti conformer, the reference for gauche strain."),
    ("isobutane", "Isobutane", "CC(C)C", "alkanes",
     "Branched C4H10; the textbook isomerisation partner of butane."),
    ("pentane", "Pentane", "CCCCC", "alkanes",
     "Linear C5H12; the middle of the alkane trend."),
    ("hexane", "Hexane", "CCCCCC", "alkanes",
     "Linear C6H14; common non-polar solvent."),
    ("cyclohexane", "Cyclohexane", "C1CCCCC1", "alkanes",
     "Chair C6H12, D3d; the standard saturated ring."),
    ("butadiene", "1,3-Butadiene", "C=CC=C", "alkenes",
     "Conjugated diene; the four-electron Huckel test case."),
    ("cyclohexene", "Cyclohexene", "C1CCC=CC1", "alkenes",
     "Half-chair ring with one double bond."),
    ("thiophene", "Thiophene", "c1ccsc1", "heterocycles",
     "Sulfur heteroaromatic; completes the furan/pyrrole series."),
    ("pyrazine", "Pyrazine", "c1cnccn1", "heterocycles",
     "Para diazine; two equivalent ring nitrogens."),
    ("pyrimidine", "Pyrimidine", "c1cncnc1", "heterocycles",
     "Meta diazine; the ring in cytosine, thymine and uracil."),
    ("hydrogen_fluoride", "Hydrogen Fluoride", "F", "basics",
     "The most polar neutral diatomic; a harsh test of a basis set."),
    ("hydrogen_chloride", "Hydrogen Chloride", "Cl", "basics",
     "Polar diatomic hydride; compare against HF."),
    ("silane", "Silane", "[SiH4]", "inorganic",
     "Tetrahedral SiH4; the silicon analogue of methane."),
    ("borane", "Borane", "B", "inorganic",
     "Trigonal planar BH3; the empty p orbital makes it a Lewis acid."),
    ("ammonia_borane", "Ammonia Borane", "[NH3+][BH3-]",
     "inorganic", "Dative N->B bond; a hydrogen-storage material."),
    # Phosphorus.  The library had 62 entries and not one of them contained P,
    # so 31P was reachable only by typing a SMILES: no gate, no library entry,
    # no real job had ever run it, and only the reference layer was checked, in
    # isolation.  These two close that, and they are deliberately *not*
    # phosphine: PH3 is the 31P reference compound, and the rule that a
    # reference is computed at the library geometry when the library ships it
    # would have replaced the experimental r_e reference with an MMFF94 one and
    # moved the whole scale -- this build measured 31P shielding moving by
    # 2.819 ppm between two geometries of PH3 alone.
    #
    # They are also a pair on purpose.  Successive methylation, PH2Me then
    # PHMe2, is the classic 31P trend, and a program whose selling point is
    # trends should be able to show one on the nucleus it just learned.
    # Both are small enough to be affordable in a gate (0.21 and 0.87 GB by
    # the memory estimate, against a 7.3 GB budget).
    ("methylphosphine", "Methylphosphine", "CP", "inorganic",
     "Primary phosphine, CH3PH2; the first rung of the P-methyl series."),
    ("dimethylphosphine", "Dimethylphosphine", "CPC", "inorganic",
     "Secondary phosphine, (CH3)2PH; the second rung, and the P-H "
     "stretch that distinguishes it from the tertiary phosphine."),
    # Selenium, for the same reason phosphorus was added: the nucleus is
    # supported and the library had no way to reach it.  This one is chosen to
    # pair with an entry that already exists -- methanethiol, CH3SH -- so the
    # library carries a genuine S/Se comparison rather than an isolated atom.
    #
    # Not the reference compound.  77Se is quoted against neat Me2Se, and
    # shipping Me2Se would make the reference use the library's UFF geometry
    # and move the whole scale, exactly as shipping phosphine would have.
    #
    # 0.57 GB by the memory estimate (54 basis functions), so a gate can run
    # 77Se end to end on any machine.  MMFF94 has no selenium parameters, so
    # RDKit relaxes it with UFF and says so -- which is the point of asking the
    # molecule rather than assuming.
    ("methaneselenol", "Methaneselenol", "C[SeH]", "inorganic",
     "Selenol, CH3SeH; the selenium analogue of methanethiol, so the two "
     "can be compared directly."),
]


def build(key: str, name: str, smiles: str, category: str, desc: str):
    """One library entry, in the schema ``library.json`` actually uses.

    This used to emit ``key`` / ``name`` / ``desc``, while the entries already
    in the file used ``display_name`` / ``description``.  The readers all look
    for the second pair, so every one of the seventeen entries this script
    added arrived in the UI with its key as its name and a blank description
    -- and the strings were right there in the file, under a name nothing
    read.  The dict key is the key; a second copy inside the entry is how the
    two schemas drifted apart in the first place.
    """
    try:
        mol = molmod.from_smiles(smiles, name=name)
    except Exception as exc:  # noqa: BLE001
        return None, f"RDKit could not embed: {exc}"
    return {
        "display_name": name,
        "formula": mol.compute_formula(),
        "smiles": smiles,
        "category": category,
        "description": desc,
        # Charge and multiplicity used to be left out, so the seventeen entries
        # this script added fell back to the reader's defaults (0 and 1).
        # That is right for these seventeen and wrong the moment a charged
        # molecule is added through the same path -- it would be computed as a
        # neutral singlet, silently.  They are properties of the molecule, so
        # they come from the molecule.
        "charge": int(mol.charge),
        "multiplicity": int(mol.multiplicity),
        # What actually relaxed these coordinates, asked of the molecule
        # rather than assumed.  from_smiles falls back to UFF when MMFF94 has
        # no parameters for a molecule and says which it used, so recording
        # it here is what keeps a boron hydride from being filed as an MMFF94
        # geometry it never was.
        "geometry_source": mol.geometry_source,
        "xyz": mol.to_xyz().rstrip("\n"),
    }, f"{mol.natoms()} atoms, {mol.compute_formula()}"


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    with open(_LIB_PATH, encoding="utf-8") as fh:
        text = fh.read()
    data = json.loads(text)
    trailing_newline = text.endswith("\n")

    added, skipped, failed = [], [], []
    for key, name, smiles, category, desc in NEW:
        if key in data:
            skipped.append(key)
            print(f"[SKIP] {key}: already present")
            continue
        entry, msg = build(key, name, smiles, category, desc)
        if entry is None:
            failed.append(key)
            print(f"[FAIL] {key}: {msg}")
            continue
        if not dry:
            data[key] = entry
        added.append(key)
        print(f"[ ADD] {key}: {msg}")

    if not dry and added:
        out = json.dumps(data, indent=2, ensure_ascii=False)
        if trailing_newline:
            out += "\n"
        with open(_LIB_PATH, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"\nwrote {_LIB_PATH}")

    print(f"\n{len(added)} added, {len(skipped)} already present, "
          f"{len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
