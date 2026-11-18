"""Round-17 mutation tests for the molecule library.

The rule this project holds itself to is that an assertion which cannot fail
guards nothing.  ``contract.library_sections`` is new, and it exists because
of a real defect: ``library.json`` was written by two scripts with two
different schemas, and seventeen entries reached the UI with their lowercase
key as their name and a blank description while the real strings sat in the
same dict under ``desc``, a name nothing read.

Round 18 added the assertions about what an entry claims its geometry *is*.
Until then every entry was stamped "MMFF94 (RDKit ETKDGv3)" whatever its
coordinates were -- and re-embedding showed 37 of the 62 were not that at
all, including two that were not molecules: the stored methanol closed an
H-C-H angle of 62 degrees.  L12-L19 put those states back, including the
reader that stamped one string over the whole library and the helper that
credited MMFF94 for a relaxation that never ran.

Each mutation below puts one of those states back, or breaks something else
the section claims to check, and the harness reports whether a failure
actually fired.  Every mutation is applied to a copy and the file is restored
byte-for-byte afterwards, including on failure.

Run:  PYTHONPATH=C:/Users/frddx/Desktop/ChatDFT <conda python> -m backend.mutate_library
"""
from __future__ import annotations

import collections
import io
import json
import os
import shutil
import sys
import tempfile

import backend.bootstrap as bootstrap
from backend import mutate_common

bootstrap.setup()

LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "molecules", "library.json")
MOL = "backend/engine/molecule.py"
ELE = "backend/engine/elements.py"

# Which source file a code mutation is written to.  Everything before round 18
# lived in molecule.py; the bond-perception mutation lives in elements.py.
#
# The key is the mutation's *full label*, and it used to be the bare code
# "L20" while the loop looked up the whole string "L20
# bond-perception-misses-dihydrogen".  So the lookup missed, `dict.get` fell
# back to molecule.py, the pattern was not in that file, and the harness
# reported "mutation not applicable" -- on every run, for as long as the
# mutation had existed.  It never once ran.
#
# That is the same shape as the NMR harness's crash and the assertion in the
# wrong loop: the report said "nothing was caught" when the truth was "nothing
# was run".  A default that quietly absorbs a key mismatch is what made it
# survivable; _check_mutation_files below now refuses to start in that state.
MUTATION_FILE = {"L20 bond-perception-misses-dihydrogen": ELE}


def _load():
    with open(LIB, encoding="utf-8") as fh:
        return json.load(fh, object_pairs_hook=collections.OrderedDict)


def _save(data):
    with open(LIB, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ----------------------------------------------------------------------
#  the mutations: each takes the parsed library and returns a new one
# ----------------------------------------------------------------------
def m_legacy_schema(d):
    """Put one entry back under the schema extend_library used to emit."""
    k = "propane"
    v = d[k]
    out = collections.OrderedDict()
    out["key"] = k
    out["name"] = v["display_name"]
    for f in ("formula", "smiles", "category"):
        out[f] = v[f]
    out["desc"] = v["description"]
    out["charge"] = v["charge"]
    out["multiplicity"] = v["multiplicity"]
    out["xyz"] = v["xyz"]
    d[k] = out
    return d


def m_blank_description(d):
    d["thiophene"]["description"] = ""
    return d


def m_display_name_is_key(d):
    d["ethane"]["display_name"] = "ethane"
    return d


def m_drop_charge(d):
    del d["propane"]["charge"]
    return d


def m_wrong_atom_count(d):
    xyz = d["water"]["xyz"].splitlines()
    xyz[0] = "4"
    d["water"]["xyz"] = "\n".join(xyz)
    return d


def m_wrong_formula(d):
    d["benzene"]["formula"] = "C6H12"
    return d


def m_description_claims_wrong_angle(d):
    d["water"]["description"] = ("The classic test case. Bent C2v geometry, "
                                 "~104.5 deg H-O-H angle.")
    return d


def m_description_claims_wrong_length(d):
    d["water"]["description"] = ("Bent C2v geometry, O-H 1.1000 A, "
                                 "H-O-H 104.00 deg.")
    return d


def m_trailing_line(d):
    d["methane"]["xyz"] = d["methane"]["xyz"] + "\n   stray line"
    return d


# ----------------------------------------------------------------------
#  round 18: the provenance a library entry claims for its own geometry
# ----------------------------------------------------------------------
def m_provenance_claimed_but_false(d):
    """Call a DFT-relaxed geometry an MMFF94 conformer."""
    d["water"]["geometry_source"] = "MMFF94 (RDKit ETKDGv3)"
    return d


def m_provenance_dft_on_a_force_field_entry(d):
    """Call a plain force-field conformer a DFT-optimised structure."""
    d["ethanol"]["geometry_source"] = (
        "B3LYP / 6-31G* stationary point (built-in library geometry)")
    return d


def m_provenance_missing(d):
    d["benzene"].pop("geometry_source", None)
    return d


def m_provenance_hand_typed(d):
    """A plausible string that is not one the program can stand behind."""
    d["propane"]["geometry_source"] = "MMFF94 (RDKit ETKDGv3, relaxed)"
    return d


def m_impossible_angle(d):
    """Close an H-N-H angle to about 40 degrees -- what methanol used to be."""
    lines = d["ammonia"]["xyz"].splitlines()
    hs = [i for i, ln in enumerate(lines[2:], start=2) if ln.split()[0] == "H"]
    first = lines[hs[0]].split()
    second = lines[hs[1]].split()
    second[1] = f"{float(first[1]) + 0.7:.8f}"
    second[2], second[3] = first[2], first[3]
    lines[hs[1]] = " ".join(second)
    d["ammonia"]["xyz"] = "\n".join(lines)
    return d


def m_phosphorus_removed(d):
    """Take the only two phosphorus entries back out.

    This is not a hypothetical defect: it is the state the library was in
    until this round.  Sixty-two entries, not one containing phosphorus, while
    31P had a reference compound, a scale note, a place in the isotope list
    and a spectrum channel -- so the nucleus was fully supported and
    completely unreachable, unless you already knew a SMILES.  Nothing said
    the coverage rule out loud, so nothing could notice it.  The mutation
    restores exactly that silence.
    """
    for k in ("methylphosphine", "dimethylphosphine"):
        d.pop(k, None)
    return d


DATA_MUTATIONS = [
    ("L1 legacy-schema-reintroduced",
     "put propane back under the name/desc schema extend_library used to "
     "emit, so the UI shows 'propane' and no description",
     m_legacy_schema),
    ("L2 description-emptied",
     "empty thiophene's description, which is a blank row in the panel",
     m_blank_description),
    ("L3 display-name-is-the-key",
     "make ethane's display name its lowercase key",
     m_display_name_is_key),
    ("L4 charge-missing",
     "drop propane's charge, so a charged molecule added later would be "
     "computed as a neutral singlet",
     m_drop_charge),
    ("L5 atom-count-wrong",
     "declare four atoms in a three-atom XYZ block",
     m_wrong_atom_count),
    ("L6 formula-wrong",
     "give benzene the formula of cyclohexane",
     m_wrong_formula),
    ("L7 angle-claim-wrong",
     "put water's description back to the ~104.5 deg it never had",
     m_description_claims_wrong_angle),
    ("L8 length-claim-wrong",
     "claim an O-H of 1.10 A for a geometry whose O-H is 0.9686",
     m_description_claims_wrong_length),
    ("L9 trailing-line",
     "append a stray line after methane's atoms",
     m_trailing_line),
    ("L12 provenance-claimed-but-false",
     "label the DFT-relaxed water geometry an MMFF94 conformer, the claim "
     "that was on all 62 entries whatever their coordinates were",
     m_provenance_claimed_but_false),
    ("L13 provenance-dft-on-a-force-field-entry",
     "label ethanol, a plain force-field conformer, a DFT-optimised "
     "structure",
     m_provenance_dft_on_a_force_field_entry),
    ("L14 provenance-missing",
     "drop benzene's geometry source, so nothing says where its coordinates "
     "came from and no result on it can be written as A//B",
     m_provenance_missing),
    ("L15 provenance-hand-typed",
     "give propane a plausible-looking source string that is not one the "
     "program defines",
     m_provenance_hand_typed),
    ("L16 impossible-bond-angle",
     "close ammonia's H-N-H angle to about 40 degrees, the defect the stored "
     "methanol had (62 degrees)",
     m_impossible_angle),
    ("L21 phosphorus-unreachable",
     "remove the only two phosphorus entries, so 31P can be shifted but no "
     "molecule in the library contains it -- the state the library was in "
     "until this round",
     m_phosphorus_removed),
]

# ---- and one in the reader, which is what makes the schema tolerance real
CODE_MUTATIONS = [
    ("L10 reader-ignores-legacy-name",
     "make the name accessor read only display_name, so a legacy entry is "
     "invisible again",
     [('    name = entry.get("display_name") or entry.get("name") or ""',
       '    name = entry.get("display_name") or ""')]),
    ("L11 reader-ignores-legacy-desc",
     "make the description accessor read only description",
     [('    return str(entry.get("description") or entry.get("desc") or "")',
       '    return str(entry.get("description") or "")')]),
    ("L17 loader-stamps-one-source-again",
     "have from_name stamp MMFF_GEOMETRY on every entry, so a user is told "
     "the same provenance for a conformer and for a DFT-relaxed structure",
     [('    m.geometry_source = str(entry.get("geometry_source") or LIBRARY_UNVERIFIED)',
       '    m.geometry_source = MMFF_GEOMETRY')]),
    ("L18 from_smiles-credits-mmff-it-did-not-run",
     "restore the call that ignored the optimiser's return code, so a "
     "molecule MMFF94 has no parameters for is handed out as MMFF94",
     [('    geometry_source = _relax_conformer(mol)',
       '    try:\n        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)\n'
       '    except Exception:\n        pass\n'
       '    geometry_source = MMFF_GEOMETRY')]),
    ("L20 bond-perception-misses-dihydrogen",
     "put the H-H bond cutoff back to 1.15 x the covalent radii, which draws "
     "it at 0.713 A against a bond of 0.743 A, so the program reports the H2 "
     "entry as two unconnected atoms",
     [('    frozenset({"H", "H"}): 1.25,', '    frozenset({"H", "H"}): 1.15,')]),
    ("L19 relax-helper-takes-mmff-on-trust",
     "make _relax_conformer report MMFF94 whether or not the optimiser said "
     "it did anything (-1 means 'no parameters', and used to be ignored)",
     [('    if AllChem.MMFFGetMoleculeProperties(mol) is not None:',
       '    if True:'),
      ('        if rc == 0:', '        if rc in (0, -1, 1):')]),
]


def reload_molecule():
    import importlib

    # `elements` has to be in this list, and it was not.  L20 rewrites the H-H
    # bond cutoff in elements.py, and this function reloaded only molecule and
    # contract -- so the mutated file was never read: the contract does
    # `from backend.engine.elements import bond_cutoff`, which returns whatever
    # is in sys.modules, and that was still the unmutated function.  The
    # mutation could not fire however it was written, which is why it reported
    # success on three failures that all belonged to the legacy-schema entry
    # the harness injects for the *reader* mutations.
    #
    # That is the third time this round that a harness defeated its own test --
    # after the NMR harness catching the wrong exception class because of its
    # reload order, and MUTATION_FILE keyed by a code the lookup never passed.
    # Order matters here: reload the leaf first, so the modules that import
    # from it rebind to the new object.
    for mod in ("backend.engine.elements", "backend.engine.molecule",
                "backend.contract"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
        else:
            importlib.import_module(mod)
    import backend.contract as contract

    return contract


def run_library(contract) -> list:
    fail: list = []
    buf, real = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        contract.library_sections(fail)
    finally:
        sys.stdout = real
    return fail


# Which code mutations need a legacy-schema entry present to mean anything.
#
# The harness used to inject one for *every* code mutation.  That is right for
# the two reader mutations, which exist to prove a legacy entry is still
# readable, and wrong for the other four: it put three failures about propane's
# fields at the top of every list, so "4 failures raised" looked like success
# whether or not the mutation's own failure was in there.  L20 spent its whole
# life looking like that.
NEEDS_LEGACY_ENTRY = {
    "L10 reader-ignores-legacy-name",
    "L11 reader-ignores-legacy-desc",
}


def _check_mutation_files() -> list:
    """Refuse to run if a code mutation's file cannot be resolved.

    The lookup in the loop is ``MUTATION_FILE.get(label, MOL)``, so a label
    that is not a key does not raise -- it quietly edits the wrong file and
    the mutation reports "not applicable" for ever.  That is what happened to
    L20.  The default is still wanted (five mutations really do live in
    molecule.py), so the fix is to check the keys up front rather than to
    remove the default.

    The substitution and syntax half of this is the same rule every other
    harness needs, so it lives in ``mutate_common`` and is called here rather
    than written twice.  This function keeps only the part that is specific to
    this harness: which file each label actually resolves to.
    """
    problems = []
    labels = {t[0] for t in CODE_MUTATIONS}
    for key in MUTATION_FILE:
        if key not in labels:
            problems.append(
                f"MUTATION_FILE is keyed {key!r}, which is not a mutation "
                f"label; the lookup would fall back to {MOL} and the mutation "
                "would silently never run")
    for label, _what, _subs in CODE_MUTATIONS:
        path = MUTATION_FILE.get(label, MOL)
        if not os.path.exists(path):
            problems.append(f"{label}: {path} does not exist")
    if problems:
        return problems

    sources: dict = {}

    def source_of(entry):
        path = MUTATION_FILE.get(entry[0], MOL)
        if path not in sources:
            sources[path] = io.open(path, encoding="utf-8").read()
        return sources[path]

    return mutate_common.dead_mutations(
        CODE_MUTATIONS, source_of, lambda e: MUTATION_FILE.get(e[0], MOL))


def main() -> int:
    stuck = _check_mutation_files()
    if stuck:
        print("=" * 78)
        print("REFUSING TO RUN -- a code mutation cannot be applied")
        print("=" * 78)
        for s in stuck:
            print(f"  - {s}")
        return 2

    original_lib = io.open(LIB, encoding="utf-8").read()
    original_mol = io.open(MOL, encoding="utf-8").read()
    original_ele = io.open(ELE, encoding="utf-8").read()

    print("=" * 78)
    print("BASELINE -- unmutated")
    print("=" * 78)
    contract = reload_molecule()
    base = run_library(contract)
    if base:
        for b in base[:8]:
            print(f"       {b[:150]}")
        print(f"  !! baseline already failing: {len(base)} failure(s)")
        return 1
    print("  baseline: 0 failures, as required\n")

    problems = []

    for label, what, fn in DATA_MUTATIONS:
        print("=" * 78)
        print(f"MUTATION {label}: {what}")
        print("=" * 78)
        try:
            _save(fn(_load()))
            contract = reload_molecule()
            fail = run_library(contract)
        finally:
            io.open(LIB, "w", encoding="utf-8").write(original_lib)
        if not fail:
            problems.append(f"{label}: FIRED NOTHING")
            print("  !! NO FAILURE RAISED")
        else:
            print(f"  -> {len(fail)} failure(s) raised:")
            for f in fail[:8]:
                print(f"       {f[:150]}")
        print()

    for label, what, subs in CODE_MUTATIONS:
        path = MUTATION_FILE.get(label, MOL)
        original_src = io.open(path, encoding="utf-8").read()
        print("=" * 78)
        print(f"MUTATION {label}: {what}")
        print("=" * 78)
        text = original_src
        ok = True
        for old, new in subs:
            if old not in text:
                # Name the file.  The message used to print only the pattern,
                # which sent the reader looking in the right file when the
                # lookup had gone to the wrong one.
                print(f"  !! mutation not applicable: {old[:70]!r} is not in "
                      f"{path}")
                ok = False
                break
            text = text.replace(old, new, 1)
        if not ok:
            problems.append(f"{label}: mutation not applicable")
            continue
        try:
            io.open(path, "w", encoding="utf-8").write(text)
            # Only the reader mutations need a legacy entry to read.  For the
            # rest it was noise on top of the answer (see NEEDS_LEGACY_ENTRY).
            if label in NEEDS_LEGACY_ENTRY:
                _save(m_legacy_schema(_load()))
            contract = reload_molecule()
            fail = run_library(contract)
        finally:
            io.open(path, "w", encoding="utf-8").write(original_src)
            io.open(LIB, "w", encoding="utf-8").write(original_lib)
        if not fail:
            problems.append(f"{label}: FIRED NOTHING")
            print("  !! NO FAILURE RAISED")
        else:
            print(f"  -> {len(fail)} failure(s) raised:")
            for f in fail[:8]:
                print(f"       {f[:150]}")
        print()

    print("=" * 78)
    print("RESTORED -- unmutated again")
    print("=" * 78)
    contract = reload_molecule()
    back = run_library(contract)
    if back:
        problems.append(f"restore: {back[:3]}")
        print(f"  !! not clean after restore: {back[:3]}")
    else:
        print("  restored: 0 failures\n")

    # byte-for-byte, because a mutation harness that leaves the data file
    # reformatted is a harness that has changed the thing it was testing
    if io.open(LIB, encoding="utf-8").read() != original_lib:
        problems.append("restore: library.json is not byte-identical")
    if io.open(MOL, encoding="utf-8").read() != original_mol:
        problems.append("restore: molecule.py is not byte-identical")
    if io.open(ELE, encoding="utf-8").read() != original_ele:
        problems.append("restore: elements.py is not byte-identical")

    print("=" * 78)
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1
    total = len(DATA_MUTATIONS) + len(CODE_MUTATIONS)
    print(f"All {total} mutations fired; files restored byte-for-byte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
