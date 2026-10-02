"""Round-16 mutation tests: every NMR assertion has to be able to fail.

The rule this project holds itself to is that an assertion which cannot fail
guards nothing.  Each mutation below breaks one specific thing, the *real*
checks from ``contract`` are run against the result, and the harness reports
whether a failure actually fired.

Two groups:

* **Pure functions** -- the cache reader/writer, the grouping rule, the grid,
  the Larmor frequency.  These need no SCF, so they run through
  ``contract.nmr_sections``, which is the same code the gate runs.
* **The payload** -- the per-isotope split, the sticks, the default axis.  These
  only mean anything on a finished job, so the harness runs ``nmr.compute`` on
  water and feeds the payload to ``contract.nmr_payload_sections``.  Water needs
  the TMS reference for its protons, so when TMS is not in the cache the payload
  group is SKIPPED and said to be skipped rather than quietly dropped.

The round's headline defect is worth restating here because three of the
mutations below exist only for it: the reference cache had never once worked.
``mol.nao_nr()`` returns a ``numpy.int64``, ``json.dump`` raises ``TypeError``
on one *after* writing the key, and the write was wrapped in
``except Exception: pass`` -- so all five cache files were truncated at
``"nao": `` and every job recomputed every reference.  TMS is 80 minutes on a
busy machine.

Run:  PYTHONPATH=C:/Users/frddx/Desktop/ChatDFT <conda python> -m backend.mutate_nmr
"""
from __future__ import annotations

import importlib
import io
import sys
import time

import backend.bootstrap as bootstrap
from backend import mutate_common

bootstrap.setup()

NMR = "backend/engine/nmr.py"

def _water_xyz() -> str:
    """The water the contract's server job computes, resolved from the library.

    It used to be a literal here, and that literal was a different molecule
    from the library's water: O-H 0.95792 A / H-O-H 104.42 deg against
    O-H 0.96857 A / 104.00 deg.  Probe 28 measured the cost of that gap --
    17O shielding moves by 539 ppm per angstrom of O-H, so the two waters are
    5.74 ppm apart and neither is the molecule a user gets by typing "water".

    Resolving it means the harness tests the geometry the product ships.
    """
    from backend.engine.molecule import resolve

    return resolve("water", kind="name").to_xyz()

# (label, what it simulates, [(old, new), ...], needs_payload)
MUTATIONS = [
    # ---- the cache -----------------------------------------------------
    ("N1 json-default-removed",
     "stop _json_default from converting numpy scalars",
     [("    if isinstance(obj, np.generic):\n        return obj.item()",
       "    if False:\n        return obj.item()")],
     False),

    ("N2 write-silently-skipped",
     "report success without writing anything, as the old code did",
     [('    tmp = f"{path}.{os.getpid()}.tmp"',
       '    return None  # noqa: this is the mutation\n'
       '    tmp = f"{path}.{os.getpid()}.tmp"')],
     False),

    ("N3 write-not-atomic",
     "drop the rename, so a crash leaves a half file that looks like an entry",
     [("        os.replace(tmp, path)", "        pass")],
     False),

    ("N4 read-ignores-format",
     "accept an entry from any layout version",
     [('    if cached.get("format") != CACHE_FORMAT:',
       "    if False:")],
     False),

    ("N5 read-ignores-required",
     "accept an entry that is missing the fields the caller uses",
     [("    for field in _CACHE_REQUIRED:",
       "    for field in ():")],
     False),

    # ---- the scale -----------------------------------------------------
    ("N6 ppm-sign",
     "flip the sign of the ppm scale",
     [("PPM = ALPHA ** 2 * 1e6", "PPM = -ALPHA ** 2 * 1e6")],
     False),

    # ---- the grouping --------------------------------------------------
    ("N7 grouping-running-mean",
     "compare against the running mean instead of the group's anchor",
     [('            if abs(g["_anchor"] - float(e["delta_ppm"])) <= tol_ppm:',
       '            if abs(g["delta_ppm"] - float(e["delta_ppm"])) <= tol_ppm:')],
     False),

    ("N8 grouping-unsorted",
     "walk the entries in the order they arrive",
     [("    ordered = sorted(usable, key=lambda e: (-float(e[\"delta_ppm\"]),\n"
       "                                            int(e.get(\"index\", 0))))",
       "    ordered = list(usable)")],
     False),

    # ---- the grid ------------------------------------------------------
    ("N9 fixed-2000-points",
     "go back to a fixed 2000-point grid whatever the linewidth",
     [("    n = int(min(max_points, max(int(points), need)))",
       "    n = int(points)")],
     False),

    ("N10 coarse-per-hwhm",
     "ask for only half a point per half-width",
     [("POINTS_PER_HWHM = 5.0", "POINTS_PER_HWHM = 0.5")],
     False),

    ("N11 tiny-cap",
     "cap the curve so low that the line has to be widened",
     [("MAX_SPECTRUM_POINTS = 30000", "MAX_SPECTRUM_POINTS = 100")],
     False),

    # ---- the Larmor frequency ------------------------------------------
    ("N12 larmor-is-1H",
     "use the 1H frequency for every nucleus",
     [('    return float(abs(info["gamma"]) / g1 * float(spectrometer_mhz))',
       "    return float(spectrometer_mhz)")],
     False),

    # ---- the sticks ----------------------------------------------------
    ("N13 stick-height-is-count",
     "draw the sticks at the raw count instead of relative to the tallest",
     [('                    "rel_intensity": float(g["count"]) / float(max(1, biggest)),',
       '                    "rel_intensity": float(g["count"]),')],
     False),

    ("N14 sticks-dropped",
     "return no sticks at all",
     [('                    "isotope": g.get("isotope")} for g in groups],',
       '                    "isotope": g.get("isotope")} for g in []],')],
     False),

    # ---- the payload ---------------------------------------------------
    ("N15 one-axis-for-everything",
     "put every isotope back on a single axis",
     [('        by_iso.setdefault(str(g.get("isotope") or g["element"]), '
       '[]).append(g)',
       '        by_iso.setdefault("all", []).append(g)')],
     "water"),

    ("N16 larmor-ignored-per-isotope",
     "build every spectrum at the 1H Larmor frequency",
     [("            isotope=iso, larmor_mhz_=larmor_mhz(el, spectrometer_mhz))",
       "            isotope=iso, larmor_mhz_=spectrometer_mhz)")],
     "water"),

    ("N17 default-not-1H",
     "open the panel on something other than the protons",
     [('    if "1H" in spectra:\n        default_iso: Optional[str] = "1H"',
       '    if "1H" in spectra:\n        default_iso: Optional[str] = sorted(spectra)[0]')],
     "water"),

    ("N18 unreferenced-hidden",
     "claim nothing is missing a reference, on a molecule where something is",
     [('        "unreferenced_elements": missing,',
       '        "unreferenced_elements": [],')],
     "thiophene"),

    # ---- the reference geometry (round 17) ------------------------------
    # 17O shielding moves 539 ppm per angstrom of O-H (probe_nmr28), so a
    # reference computed at a geometry the molecule does not have is 6 ppm of
    # error on every shift in the output -- and it made water's own 17O shift
    # 6.11 ppm instead of 0.00.
    ("N19 reference-ignores-library",
     "keep the module's own geometry for a compound the library defines",
     [("    hit = _library_geometry_index().get(sig)\n"
       "    return list(hit[\"atoms\"]) if hit else atoms",
       "    return atoms")],
     False),

    ("N20 reference-source-lies",
     "claim the built-in geometry even where the library has the compound",
     [("    return \"library\" if sig in _library_geometry_index() else "
       "\"builtin\"",
       "    return \"builtin\"")],
     False),

    ("N21 reference-note-typed",
     "go back to a hand-written geometry string",
     [("    where = (\"the molecule library\" if src == \"library\"\n"
       "             else \"the module's experimental geometry\")\n"
       "    return f\"{'; '.join(parts)} ({where})\"",
       "    return \"O-H 0.9572 A, H-O-H 104.52 deg (experimental r_e)\"")],
     False),

    ("N22 fingerprint-ignores-coordinates",
     "fingerprint only the element symbols, not where the atoms are",
     [("    blob = json.dumps([[str(s), round(float(x), 6), round(float(y), 6),\n"
       "                        round(float(z), 6)] for s, x, y, z in atoms],\n"
       "                      sort_keys=True)",
       "    blob = json.dumps([str(s) for s, *_ in atoms], sort_keys=True)")],
     False),

    ("N23 library-name-unreported",
     "use the library geometry but refuse to say which entry it came from",
     [("    hit = _library_geometry_index().get(sig)\n"
       "    return hit[\"name\"] if hit else None",
       "    return None")],
     False),

    ("N24 19F-reference-is-chloroform",
     "let the 19F reference become the library's CHCl3 instead of CFCl3",
     [('    if key == "chloroform_f":\n'
       '        # CFCl3: F up the z axis, three Cl at the tetrahedral angle to it.\n'
       '        out = [("C", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, 1.330)]',
       '    if key == "chloroform_f":\n'
       '        out = [("C", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.090)]')],
     False),

    ("N25 cache-key-drops-geometry",
     "leave the geometry out of the cache key again",
     # The target text has to quote the blob as it is *now*: this mutation
     # stopped matching the moment the method signature was added to it, and
     # the preflight is what caught that rather than a "not applicable" line
     # at the end of a run.  See mutate_common.
     [("    blob = json.dumps([key, basis, conv_tol, dB, geometry, method],\n"
       "                      sort_keys=True)",
       "    blob = json.dumps([key, basis, conv_tol, dB, method],\n"
       "                      sort_keys=True)")],
     False),

    # ---- the memory guard (round 18) -----------------------------------
    # The estimate is a constant times nao^4 and for rounds the constant was
    # the 56 that counting the live arrays gives.  probe_mem1 measured the
    # peak resident set of shielding_tensor alone at 68.8 bytes per nao^4
    # over nao = 18..54 (R^2 = 0.99997), so 56 under-bids by 23% -- and the
    # TMS job that was estimated at 5165 MB against a 6000 MB budget, passed,
    # and then took 4829 s swapping instead of 285, was really 6346 MB and
    # should have been refused.  Going back to the counted value has to be
    # caught, and the number the mutation puts back is not arbitrary: it is
    # the value the old argument still looks like it implies.
    ("N26 memory-constant-counted-not-measured",
     "put 56 bytes per nao^4 back, the number array-counting gives",
     [("_BYTES_PER_NAO4 = 69", "_BYTES_PER_NAO4 = 56")],
     False),

    # ---- the scale a shift is printed on (round 18) ----------------------
    # Five of the seven referenced nuclei sit on their IUPAC primary
    # reference; 15N (nitromethane, 1.3 GB we do not spend) and 31P (85%
    # H3PO4, not a molecule this build can compute) do not, and the gap is
    # 380 ppm for nitrogen.  Naming the compound is not the same as naming
    # the scale, so both notes have to survive.
    ("N27 scale-note-suppressed",
     "say nothing about the scale for any nucleus",
     [("    prim = IUPAC_PRIMARY.get(el)\n    if prim is None:\n        return None",
       "    if True:\n        return None")],
     False),

    ("N28 wrong-15N-conversion",
     "get the ammonia-to-nitromethane conversion wrong by a factor of ten",
     [('"conversion": "subtract 380.2 ppm to move the shift onto the "\n'
       '                        "nitromethane scale"',
       '"conversion": "subtract 38.02 ppm to move the shift onto the "\n'
       '                        "nitromethane scale"')],
     False),

    # The budget is the one refusal free memory cannot argue with.  At the
    # counted constant TMS was 5165 MB and 6000 MB was comfortable; at the
    # measured one it is 6364 MB and 6000 MB refuses it outright, on any
    # machine, for ever.  That is what this mutation restores, and it is the
    # regression the constant change actually introduced -- no gate saw it,
    # because they all run with the TMS cache warm and the cache is read
    # before the budget is ever consulted.
    ("N29 budget-refuses-the-reference",
     "put the 6000 MB budget back, which TMS no longer fits",
     [('os.environ.get("CHATDFT_NMR_MEMORY_MB", "7500")',
       'os.environ.get("CHATDFT_NMR_MEMORY_MB", "6000")')],
     False),

    # ---- why each unreferenced nucleus has no reference (round 18) -------
    # Sulfur is the one a user actually meets: thiophene carries it.  Drop
    # its reason and the element is still correctly reported as unreferenced,
    # but the *why* has silently gone missing -- which is the state the list
    # was in before this round, for every one of them.
    ("N30 reason-for-sulfur-dropped",
     "keep the unreferenced list but lose the reason for one element",
     [('    "S": "its standard is saturated (NH4)2SO4 in D2O, a solution",\n',
       '')],
     False),

    # The scale note has to reach the object the panel reads the reference
    # out of, not just the per-nucleus list.  Dropping it here leaves the
    # warning intact and the per-nucleus field intact, so a pure-function
    # check would still pass -- this one only fails on a real payload.
    ("N31 scale-note-missing-from-references",
     "keep the note on the nuclei but lose it beside the reference name",
     [('            "scale_note": scale_note(el),',
       '            "scale_note_dropped": scale_note(el),')],
     "water"),

    # ---- the accuracy ledger (round 18, second pass) --------------------
    # ``SCALE_CHECK`` is a table of measurements, so every property the
    # contract asserts about it has to be breakable.  These eight each break
    # one, and each is written so that the *only* thing that changes is the
    # property under test -- a mutation that trips three assertions teaches
    # you nothing about which one works.
    #
    # The headline is N33.  The ledger shipped with a 2.5 ppm tolerance on
    # 1H, inherited from the docstring's twelve-shift table where a single
    # band covered 1H and 13C together.  Since the whole 1H shift range is
    # about 12 ppm, that band could not be failed by any degradation the
    # ledger exists to catch: the assertion was there and the claim it made
    # was empty.  N33 restores exactly that number, so the run proves the
    # rule that now forbids it is enforced rather than intended.
    ("N32 scale-H-value-falsified",
     "move a recorded accuracy number off the value it was measured at",
     [('        "computed_ppm": 0.66, "experimental_ppm": 0.23,',
       '        "computed_ppm": 9.66, "experimental_ppm": 0.23,')],
     False),

    ("N33 scale-H-tolerance-inflated",
     "widen a tolerance back to the 2.5 ppm band that nothing could fail",
     [('        "usable": True, "tolerance_ppm": 1.0,',
       '        "usable": True, "tolerance_ppm": 2.5,')],
     False),

    ("N34 scale-F-tolerance-deflated",
     "shrink a tolerance below the error it was measured at (19F is 59.6 ppm "
     "out, so 20 ppm cannot hold it)",
     [('        "usable": True, "tolerance_ppm": 80.0,',
       '        "usable": True, "tolerance_ppm": 20.0,')],
     False),

    ("N35 scale-Se-usable-flipped",
     "report a 77Se shift on a scale the method does not reproduce",
     [('        "usable": False, "tolerance_ppm": 60.0,',
       '        "usable": True, "tolerance_ppm": 60.0,')],
     False),

    # N36 has now been rewritten twice, and both rewrites were the same lesson
    # arriving by a different road.  It was written against a hardcoded
    # ``note`` string and quoted that string; the round-19 rewrite of the three
    # SCALE_CHECK sentences replaced ``note`` with a placeholder-rendered
    # ``why``, and the quoted text stopped existing -- the preflight reported
    # ``1 dead`` before a single check ran.  That is the fourth mutation in this
    # file retired by an edit elsewhere (N34, N25, N37, N36), and it is why the
    # preflight is run first: a dead mutation is indistinguishable from a live
    # one in the final "63 of 63 fired" line.
    #
    # The target is deliberately the same line N64 acts on, and the two produce
    # different failures on purpose: N64 types the *value* back in and must be
    # caught by the literal check, while this one deletes the *placeholders*
    # and must be caught by the why_quotes check.  A single substitution
    # feeding two independent assertions is the point -- if only one of the two
    # fired, one of the assertions would be decoration.
    ("N36 scale-Se-why-loses-its-measurement",
     "keep the 77Se warning but take the measurement out of it, so the "
     "sentence asserts a verdict with nothing behind it",
     [('               "{computed:.0f} ppm against a {range:.0f} ppm shift '
       'range, and "\n',
       '               "a small fraction of its shift range, and "\n')],
     False),

    # N37 had to be rewritten twice, and the second time taught the lesson the
    # first one only hinted at.  Its original substitution named the whole "Si"
    # entry as it stood, so changing the table from strings to dicts retired it;
    # the rewrite named the whole entry again, and adding two fields to that
    # entry retired it a second time inside the same round.
    #
    # The fix is to depend on as little shape as possible: renaming the key
    # leaves a duplicate "O" (Python keeps the last) and removes "Si" from the
    # table, which is the property under test.  One line, and no field of the
    # entry is mentioned -- so this mutation survives any edit to what the entry
    # contains.  A mutation that quotes the thing it is trying to delete will
    # keep breaking every time the thing changes.
    ("N37 scale-unmeasured-entry-dropped",
     "delete an unmeasured nucleus from the ledger, leaving a gap that "
     "reads as a decision",
     [('    "Si": {\n', '    "O": {\n')],
     False),

    ("N38 scale-geometry-aliased",
     "measure the 77Se scale at a geometry name the product does not use",
     [('        "reference_geometry": "dimethyl_selenide",',
       '        "reference_geometry": "me2se",')],
     False),

    # Two assertions fire on this one -- the unreferenced set stops matching
    # NO_REFERENCE_WHY, and the ledger's own check that selenium is not in
    # it.  The second is therefore not proven *alone* by this run; it is
    # proven reachable by a direct call in the note above the mutation, and
    # co-firing is what a real regression would look like anyway.
    ("N39 scale-Se-back-in-no-reference",
     "put selenium back in the table of elements with no reference",
     [('    "Al": "its standard is Al(NO3)3 in D2O, a solution",',
       '    "Se": "no reference is provided yet; (CH3)2Se would be computable '
       'here",\n'
       '    "Al": "its standard is Al(NO3)3 in D2O, a solution",')],
     False),

    # ---- deuterium, the nucleus no job can contain (round 19) ------------
    # The reason recorded for 2H used to be "no reference is provided yet;
    # TMS-d12 would be computable here" -- a to-do note about a *compound*,
    # for a nucleus that cannot enter a calculation at all because the
    # molecule builder's element table has no deuterium.  Worse, the SMILES
    # path silently turned [2H] into 1H, so a user asking for CD4 got CH4 and
    # methane's numbers.
    #
    # N41 is the exact regression: it puts the old sentence back, so the
    # user-facing reason and the structural one become two different stories
    # about the same nucleus.  Nothing else in the build would notice.
    ("N40 unbuildable-record-stale",
     "leave a nucleus in the unbuildable table after it becomes buildable",
     [('UNBUILDABLE_NUCLEI: Dict[str, str] = {\n'
       '    "D": "no job can contain it: the molecule builder\'s element '
       'table has no "',
       'UNBUILDABLE_NUCLEI: Dict[str, str] = {\n'
       '    "S": "its standard is a solution",\n'
       '    "D": "no job can contain it: the molecule builder\'s element '
       'table has no "')],
     False),

    ("N41 unbuildable-reason-diverges",
     "put the old to-do reason back where the user reads it",
     [('    "D": UNBUILDABLE_NUCLEI["D"],',
       '    "D": "no reference is provided yet; TMS-d12 would be computable '
       'here",')],
     False),

    # ---- the recorded gaps, now that they carry values (round 19) --------
    # SCALE_UNMEASURED used to be a table of sentences.  It is now a table of
    # *measurements* -- a named pair and the number this build computes for it
    # -- because "unmeasured" and "never computed" are different states needing
    # different work, and a sentence cannot tell them apart.  That adds six
    # properties the contract can check, so each gets a mutation that breaks
    # exactly one of them.
    #
    # N42 is the one that matters most.  Erasing the value is precisely the
    # state the table was in before this round: an entry that reads like a
    # decision but could equally be an oversight, because it names no number to
    # be missing.
    ("N42 scale-gap-value-erased",
     "keep a gap in the ledger but take its computed value out",
     [('        "computed_ppm": 716.94,\n',
       '        "computed_ppm": None,\n')],
     False),

    # N43 has to replace the *whole* string literal, not its first line.  The
    # first version of this mutation swapped the opening line only, which left
    # six continuation lines attached to an empty string and made nmr.py
    # unparseable -- and the harness died on the reload rather than reporting a
    # mutation.  Both halves of that are fixed: the substitution below is the
    # full literal, and the harness now survives a mutation that does not
    # compile (see the crash handling in main()).
    ("N43 scale-gap-reason-erased",
     "keep a gap and its value but say nothing about why it is still open",
     [('        "why": "the obstacle is the phase, not a number.  The IUPAC '
       'primary "\n'
       '               "reference is liquid water and every 17O shift is '
       'measured in "\n'
       '               "solution, where it is strongly solvent-dependent; this '
       'build "\n'
       '               "is gas-phase, so a comparison would confound the method '
       'with "\n'
       '               "the solvent.  It needs either a gas-phase standard or '
       'an "\n'
       '               "explicit solvent model -- a literature number alone '
       'would not "\n'
       '               "make it sound",\n',
       '        "why": "",\n')],
     False),

    ("N44 scale-gap-reference-swapped",
     "record a gap against a reference compound the product does not use",
     [('        "nucleus": "17O", "reference": "H2O", "sample": "formaldehyde",',
       '        "nucleus": "17O", "reference": "D2O", "sample": "formaldehyde",')],
     False),

    ("N45 scale-gap-geometry-aliased",
     "name a geometry key the product does not resolve",
     [('        "reference_geometry": "water", "sample_geometry": "formaldehyde",',
       '        "reference_geometry": "h2o", "sample_geometry": "formaldehyde",')],
     False),

    ("N46 scale-gap-outside-its-own-range",
     "record a gap value larger than the nucleus's whole shift range",
     [('        "computed_ppm": 66.41,\n', '        "computed_ppm": 9999.0,\n')],
     False),

    ("N47 scale-gap-without-a-range",
     "keep a gap value but drop the range that says whether it is a shift",
     [('        "range_ppm": 519.0,\n', '        "range_ppm": 0.0,\n')],
     False),

    # ---- the reason text: a number written twice (round 19, third pass) --
    # The gap reasons used to type their own numbers -- "silane 64.9 ppm
    # upfield ... against a 519 ppm range" -- duplicating computed_ppm and
    # range_ppm, which the gate recomputes.  A number written twice is a number
    # that can disagree with itself, and this module has already had one ledger
    # go stale exactly that way.  The sentences now interpolate from the entry's
    # own fields, and these seven mutations are what makes that a rule rather
    # than a style preference.
    ("N48 scale-gap-restates-its-own-number",
     "type a recomputed number back into the sentence that should quote it",
     [('               "a {range:.0f} ppm range), so what is missing is one '
       'verified "',
       '               "a 519 ppm range), so what is missing is one verified "')],
     False),

    ("N49 scale-gap-reason-kind-invalid",
     "declare a reason as neither a measurement nor an argument",
     [('        "reason_kind": "argument",\n',
       '        "reason_kind": "note",\n')],
     False),

    ("N50 scale-gap-measurement-hides-its-number",
     "call a reason a measurement while quoting nothing",
     [('        "reason_kind": "measurement",\n'
       '        "why_quotes": ("computed", "range"),\n'
       '        "why": "the method produces a real 29Si',
       '        "reason_kind": "measurement",\n'
       '        "why_quotes": (),\n'
       '        "why": "the method produces a real 29Si')],
     False),

    ("N51 scale-gap-placeholder-shows-the-wrong-form",
     "keep the placeholder but make it render a number nobody reads",
     [('"the method produces a real 29Si scale (silane {computed:+.1f} "',
       '"the method produces a real 29Si scale (silane {computed:+.1e} "')],
     False),

    # Two assertions fire on this one -- the placeholder is gone, and so the
    # rendered text no longer carries the value.  The second is a consequence of
    # the first rather than an independent property, and N51 above is the
    # mutation that proves the second one *alone*.
    ("N52 scale-gap-placeholder-deleted",
     "drop the placeholder, so the sentence states the verdict without the "
     "measurement",
     [('"the method produces a real 29Si scale (silane {computed:+.1f} "',
       '"the method produces a real 29Si scale (silane upfield "')],
     False),

    ("N53 scale-gap-trend-sign-invalid",
     "claim a trend direction that is not a direction",
     [('        "trend": [("dimethylphosphine", +1)],\n',
       '        "trend": [("dimethylphosphine", 0)],\n')],
     False),

    ("N54 scale-gap-trend-geometry-unknown",
     "name a trend step the build cannot resolve, so the claim is never "
     "recomputed",
     [('        "trend": [("dimethylphosphine", +1)],\n',
       '        "trend": [("me2ph", +1)],\n')],
     False),

    # ---- the twelve-shift ledger and the tolerances that rest on it --------
    # These five break the ledger *data*, so they are caught by the contract.
    # The other half of the ledger check -- that ``check_shift_scales``
    # recomputes the rows and the band still holds -- cannot be broken by a file
    # mutation cheaply: it is one SCF per row, six rows, and benzene alone is
    # six minutes.  That half is the normal run, and the band rule itself is
    # falsified on every run by ``check_shift_scales.prove()``, which is the
    # same treatment ``check_one``'s band has always had.
    ("N55 shift-ledger-one-row",
     "cut the 1H ledger to a single row, so its tolerance is measured on one "
     "molecule again",
     [('        "rows": [\n'
       '            ("acetylene", "acetylene", 1.997, 1.80),\n'
       '            ("methane", "methane", 0.655, 0.23),\n'
       '            ("ethane", "ethane", 1.130, 0.86),\n'
       '            ("ethylene", "ethylene", 5.737, 5.40),\n'
       '            ("benzene", "benzene", 7.666, 7.26),\n'
       '            ("methanol", "methanol", 2.668, 3.35),\n'
       '        ],\n',
       '        "rows": [\n'
       '            ("acetylene", "acetylene", 1.997, 1.80),\n'
       '        ],\n')],
     False),

    ("N56 shift-ledger-exception-orphan",
     "name an exception that is not one of the rows, so it excuses nothing "
     "while reading as if it excused something",
     [('            "acetylene": "the known RHF/6-31G* weakness on a '
       'carbon-carbon "',
       '            "me2se": "the known RHF/6-31G* weakness on a '
       'carbon-carbon "')],
     False),

    ("N57 tolerance-unbacked-dropped",
     "drop a nucleus from TOLERANCE_UNBACKED without giving it a ledger, so "
     "its tolerance is checked against nothing",
     [('    "F": "the tolerance is set from the single pair CFCl3 vs CH3F, '
       'where CH3F "\n'
       '         "is constructed rather than taken from the library.  A '
       'second row is "\n'
       '         "available -- the library carries hydrogen fluoride and "\n'
       '         "trifluoroacetic acid, both fluorinated -- but it has not '
       'been "\n'
       '         "measured.  Recorded as work, not as done",\n'
       '}',
       '}')],
     False),

    ("N58 carbon-tolerance-too-tight",
     "put the 13C tolerance back to 2.5 ppm, which methanol falsifies at 3.04",
     [('        "usable": True, "tolerance_ppm": 3.2,',
       '        "usable": True, "tolerance_ppm": 2.5,')],
     False),

    ("N59 hydrogen-tolerance-vacuous",
     "widen the 1H tolerance to 5.0 ppm, more than three times the worst error "
     "in its ledger",
     [('        "usable": True, "tolerance_ppm": 1.0,',
       '        "usable": True, "tolerance_ppm": 5.0,')],
     False),

    # ---- the cache key, and the code version it must carry ----------------
    # data/nmr_cache ships as data, so these three are about a number that is
    # read back rather than recomputed: the failure they produce is a constant
    # offset on every shift the product prints, and it is invisible in the
    # output.  All three are caught without an SCF.
    ("N60 cache-key-ignores-the-method",
     "drop the method signature from the cache key, so a reference computed by "
     "other code is read as if it were this code's",
     [("    blob = json.dumps([key, basis, conv_tol, dB, geometry, method],\n"
       "                      sort_keys=True)",
       "    blob = json.dumps([key, basis, conv_tol, dB, geometry],\n"
       "                      sort_keys=True)")],
     False),

    ("N61 method-signature-is-a-constant",
     "make the method signature a constant, so editing the shielding code "
     "invalidates nothing",
     [('    return hashlib.sha1("\\n".join(parts).encode("utf-8")).hexdigest()[:16]',
       '    return "constant"')],
     False),

    ("N62 method-not-a-required-cache-field",
     "stop requiring the method field, so a file from another code version is "
     "read instead of rejected",
     [('                   "geometry", "geometry_source", "geometry_signature",\n'
       '                   "method")',
       '                   "geometry", "geometry_source", "geometry_signature")')],
     False),

    ("N63 reader-stops-checking-the-method",
     "let the reader accept an entry stamped with a different method, so the "
     "filename becomes the only guard",
     [('    if cached.get("method") != _method_signature():\n'
       '        return None\n'
       '    return cached',
       '    return cached')],
     False),

    # ---- the reason sentences -------------------------------------------
    # Three SCALE_CHECK entries used to carry a hardcoded ``note`` with their
    # own field values typed into it, and the contract *required* those copies
    # -- it asserted the note carried the computed and experimental numbers.  So
    # the duplicate was enforced rather than overlooked, and the fix was to
    # render from placeholders and check the rendered text instead.
    #
    # These three are what proves the new assertions are load-bearing.  Before
    # them, typing the number back into the selenium sentence produced zero
    # failures: the shape assertions existed, but only in the SCALE_UNMEASURED
    # loop, so nothing applied them to a measured entry.
    ("N64 scale-Se-why-types-the-number",
     "put the computed value back into the selenium reason as a literal, so "
     "the table and the sentence can disagree",
     [('               "{computed:.0f} ppm against a {range:.0f} ppm shift '
       'range, and "\n',
       '               "-101 ppm against a {range:.0f} ppm shift range, and "\n')],
     False),

    ("N65 scale-Se-why-loses-its-quotes",
     "stop declaring which fields the selenium reason quotes, so a placeholder "
     "can be deleted without anything objecting",
     [('        "why_quotes": ("computed", "range", "geometry_spread"),',
       '        "why_quotes": (),')],
     False),

    # N66 was first written against the nitrogen sentence, and it fired
    # NOTHING.  The reason is worth keeping, because the mutation was wrong
    # rather than the assertion, and it was wrong in a way that is easy to
    # repeat.
    #
    # It replaced ``"puts it {experimental:.0f}. The scale note ..."`` with
    # ``"puts it. The scale note ..."`` -- one of the *two* places the nitrogen
    # sentence carries the experimental value, the other being ``subtract
    # {experimental:.1f} ppm``.  So the field still had a placeholder, the
    # placeholder check is an existence check, and it passed.  The rendered-text
    # check passed too, for a second and unrelated reason: it looks for the
    # token ``"380"``, and ``"380.2"`` contains ``"380"``.
    #
    # Neither of those is a defect.  The property is "the sentence gives the
    # user the experimental value", and after that substitution it still does.
    # What was wrong is that the mutation had been written from the diff -- the
    # line just edited -- instead of from the assertion it was meant to prove.
    # That is the same error as the to-do invented from a naming convention in
    # round 19, arriving by a different road: a test aimed at the wrong thing
    # passes for ever.
    #
    # It is replaced by one that targets an assertion nothing else pins.  Seven
    # assertions live in _reason_shape; four had a mutation (the literal check
    # via N48/N64, an invalid reason_kind via N49, a measurement that quotes
    # nothing via N50/N65, a missing placeholder via N52) and three had none.
    # The three below are those three, so every assertion in the shared helper
    # is now falsified by a file mutation rather than only by a one-off probe.
    ("N66 scale-Se-why-quotes-declaration-dropped",
     "remove the selenium reason's declaration instead of emptying it, so an "
     "omission and a decision become the same entry",
     [('        "why_quotes": ("computed", "range", "geometry_spread"),\n',
       '')],
     False),

    ("N67 scale-Se-why-quotes-a-field-that-is-not-one",
     "misspell one of the quoted field names, so the placeholder check for the "
     "real field silently never runs",
     [('        "why_quotes": ("computed", "range", "geometry_spread"),',
       '        "why_quotes": ("computed", "range", "geometryspread"),')],
     False),

    ("N68 scale-Si-reason-mislabelled-as-argument",
     "call a sentence an argument while it cites two numbers, so a reason that "
     "rests on a measurement is filed as a decision",
     [('        "reason_kind": "measurement",\n'
       '        "why_quotes": ("computed", "range"),\n'
       '        "why": "the method produces a real 29Si ',
       '        "reason_kind": "argument",\n'
       '        "why_quotes": ("computed", "range"),\n'
       '        "why": "the method produces a real 29Si ')],
     False),
]


def reload_engine():
    for mod in ("backend.engine.nmr", "backend.engine.dft"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
        else:
            importlib.import_module(mod)
    import backend.contract as contract
    importlib.reload(contract)
    return contract


def run_checks(payload, contract, thio=None) -> list:
    fail: list = []
    buf, real = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        contract.nmr_sections(fail)
        contract.nmr_scale_sections(fail)
        # The cache-key assertions live in check_shift_scales, the slowest
        # gate in the suite because of the SCF behind the rest of it -- but
        # this section of that gate costs no SCF and is the only thing in the
        # repository that catches two mutations in this file.
        #
        # N61 (make the signature a constant) and N62 (drop method from the
        # required fields) both reported FIRED NOTHING on a run where those
        # assertions existed and passed.  The harness had simply never called
        # them.  The lesson is the one this file keeps re-learning: an
        # assertion that is never invoked and a mutation that catches nothing
        # produce the same line in the log, so the check has to be wired in
        # rather than merely written down somewhere else.
        from backend import check_shift_scales as _css
        fail.extend(_css.check_cache_key())
        if payload is not None:
            contract.nmr_payload_sections(payload, fail)
        if thio is not None:
            contract.nmr_unreferenced_sections(thio, fail)
    finally:
        sys.stdout = real
    for line in buf.getvalue().split("\n"):
        if line.startswith("["):
            print(f"    {line[:100]}")
    return fail


def _carriers():
    """The carrier list, from the one place that defines it.

    Thiophene is what a user meets the unreferenced-nucleus case with, but it
    is nao = 82 at 6-31G* and the memory guard refuses it on a machine with
    about 1.4 GB free -- which is this one, so the group reported SKIPPED and
    the assertion inside it was never run.  A skip is honest but it is not
    coverage.  Methanethiol (CH3SH, 6 atoms) carries the same unreferenced
    nucleus for a fraction of the cost, so the group is exercised on any
    machine; the substitution is printed rather than hidden, because a
    fallback nobody mentions is how a gate ends up testing something other
    than what its name says.

    The list itself lives in ``contract`` (the gate that has the same problem)
    rather than here, so the two cannot drift apart.  Imported inside the
    function because ``reload_engine`` reloads that module.
    """
    import backend.contract as contract

    return contract.UNREFERENCED_CARRIERS


def make_payload_thiophene():
    """A real NMR job on a molecule with an unreferenced nucleus.

    Water has nothing unreferenced once TMS is cached, so the
    "unreferenced is named, not hidden" assertion cannot fail on it -- the
    mutation N18 fired nothing on the first run for exactly that reason.
    Thiophene carries sulfur and 33S has no reference in this build, so it is
    the dataset the assertion wants; methanethiol is the fallback for when the
    memory guard refuses it.
    """
    import backend.engine.nmr as N

    from backend.engine.molecule import resolve

    carriers = _carriers()
    refused = []
    for name in carriers:
        t0 = time.time()
        xyz = resolve(name, kind="name").to_xyz()
        # N.DFTError, deliberately, not `from backend.engine.dft import DFTError`.
        # reload_engine() reloads nmr and then dft, so dft's DFTError becomes a
        # *new* class object while nmr still holds the old one -- and `except`
        # on the new class misses the old one entirely.  The first version of
        # this handler did exactly that and the harness still crashed, which
        # looked like the fix had not been applied.  Catching the class the
        # raising module actually holds is immune to the reload order.
        try:
            res = N.compute(xyz, basis="6-31g*", use_cache=True)
        except N.DFTError as exc:
            # The refusal a user gets, not a defect: thiophene at 6-31G* is
            # nao = 82, the guard prices that at about 3 GB, and a machine with
            # 1.4 GB free cannot do it without swapping.  It used to propagate
            # out of here and kill the whole harness, which is the worst
            # available outcome -- a crash that stops every other mutation
            # from being tested reads like "nothing was caught" when the truth
            # is "nothing was run".  Falling through to the next carrier keeps
            # the group tested; returning None would report it SKIPPED, which
            # is the difference between a lie and an untested assertion, but a
            # skip is still not coverage while a cheaper carrier exists.
            refused.append(f"{name}: {str(exc)[:90]}")
            continue
        if name != carriers[0]:
            print(f"  (substituted {name} for {carriers[0]}, "
                  f"refused by the memory guard: {refused[0][:70]})")
        print(f"  ({name} payload built in {time.time() - t0:.1f} s: "
              f"unreferenced {res.get('unreferenced_elements')})")
        return res
    print("  !! every carrier was refused by the memory guard:")
    for r in refused:
        print(f"       {r}")
    print("     the unreferenced-nucleus group is SKIPPED on this machine")
    return None


def make_payload():
    """One real NMR job, reused by every payload mutation.

    Returns None when a reference could not be computed -- water needs the TMS
    reference for its protons and TMS is 17 atoms / nao = 98 at 6-31G*, so on a
    machine that is short of physical memory it is refused.  Returning None
    rather than a half-payload means the payload mutations are reported as
    SKIPPED, which is honest: an assertion that was never run is not an
    assertion that passed.
    """
    import backend.engine.nmr as N

    t0 = time.time()
    res = N.compute(_water_xyz(), basis="6-31g*", use_cache=True)
    missing = res.get("unreferenced_elements") or []
    print(f"  (payload built in {time.time() - t0:.1f} s: "
          f"spectra {sorted(res.get('spectra') or {})}, "
          f"missing {missing})")
    if missing:
        print(f"  !! no reference for {missing}; the payload group cannot be "
              "tested on this machine.")
        print("     generate the cache with "
              "`python -m backend.gen_nmr_cache` when it has the RAM")
        return None
    return res


def _live_server() -> bool:
    """Is something already answering on the gate port?

    Mutating nmr.py in place is safe only if nothing else is reading it: a
    server that starts during a mutation run imports the mutated module and
    then serves corrupted numbers to anything that talks to it.  Round 15 hit
    exactly that -- a browser gate reported a sum-rule residual of -93.33%,
    which was this harness's own mutation leaking through the server.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", 8000)) == 0


def main() -> int:
    if _live_server():
        print("REFUSING TO RUN: something is listening on 127.0.0.1:8000.")
        print("This harness rewrites backend/engine/nmr.py in place.  A server")
        print("that starts while it runs will import a mutated module and serve")
        print("corrupted numbers to any browser gate run against it.  Stop the")
        print("server first.")
        return 2

    original = io.open(NMR, encoding="utf-8").read()

    # Applicability first, before a single SCF.  A mutation whose target text
    # has moved is not a mutation that found nothing -- it is a mutation that
    # was never attempted, and the two print the same way at the end.  This has
    # cost real time twice: N37 stopped matching after the ledger entries were
    # restructured, and N34 stopped matching when the 13C tolerance was
    # corrected from 2.5 to 3.2 ppm.  Both times the harness had to finish its
    # baseline and walk the whole list before saying so, which is minutes of
    # SCF for an answer available in milliseconds.  See mutate_common.
    dead = mutate_common.dead_mutations(MUTATIONS, lambda _e: original,
                                        lambda _e: NMR)
    if mutate_common.report("NMR", MUTATIONS, dead):
        return 1
    print()

    print("=" * 78)
    print("BASELINE -- unmutated")
    print("=" * 78)
    contract = reload_engine()
    payload = make_payload()
    # Thiophene is three minutes of SCF, so it is built only if some mutation
    # actually needs it -- and the baseline has to include it, or a mutation
    # could be "detected" by a failure that was already there.
    thio = (make_payload_thiophene()
            if any(m[3] == "thiophene" for m in MUTATIONS) else None)
    if any(m[3] == "thiophene" for m in MUTATIONS) and thio is None:
        print("  note: no thiophene, so the unreferenced-nucleus assertions "
              "are not part of this run at all")
    base = run_checks(payload, contract, thio)
    if base:
        print(f"  !! baseline already failing: {base}")
        return 1
    print("  baseline: 0 failures, as required\n")

    problems = []
    skipped = []
    for label, what, subs, needs_payload in MUTATIONS:
        # The header goes first, before anything can go wrong.  It used to be
        # printed only once the mutated source was on disk, so a mutation whose
        # substitution broke the syntax produced an "!! INVALID SOURCE" line
        # with no section above it -- the log did not say which mutation it had
        # been attempting.  Announcing the attempt before the checks also means
        # a "not applicable" report sits under the mutation it belongs to.
        print("=" * 78)
        print(f"MUTATION {label}: {what}")
        print("=" * 78)
        if needs_payload == "water" and payload is None:
            skipped.append(f"{label}: no payload available")
            print("  SKIPPED: no payload available")
            print()
            continue
        if needs_payload == "thiophene" and thio is None:
            # Without that payload `run_checks` never reaches
            # nmr_unreferenced_sections, so no failure can fire -- and a
            # mutation reported as "not caught" when it was never run is
            # worse than one reported as skipped.
            skipped.append(f"{label}: thiophene payload unavailable")
            print("  SKIPPED: thiophene payload unavailable")
            print()
            continue
        text = original
        ok = True
        for old, new in subs:
            if old not in text:
                print(f"  !! mutation not applicable ({old[:70]!r})")
                ok = False
                break
            text = text.replace(old, new, 1)
        if not ok:
            problems.append(f"{label}: mutation not applicable")
            print()
            continue

        # Compile before writing.  A substitution that breaks the syntax is a
        # defect in the *mutation*, not a defect the mutation found, and there
        # is no reason to put it on disk: writing it means that if this process
        # is killed between the write and the restore, the repository is left
        # holding a file that does not parse.  Checking here also turns the
        # loudest possible outcome (a traceback that ends the run) into an
        # ordinary reported problem.
        try:
            compile(text, NMR, "exec")
        except SyntaxError as exc:
            problems.append(
                f"{label}: the substitution does not produce valid Python "
                f"({exc.msg} at line {exc.lineno}); not applied, so the run "
                "continues")
            print(f"  !! INVALID SOURCE, not written: {exc.msg} "
                  f"(line {exc.lineno})")
            print()
            continue

        io.open(NMR, "w", encoding="utf-8").write(text)
        crashed = None
        try:
            # The reload is inside the crash handling too, and that is the fix
            # for a real abort: N43's first version replaced one line of a
            # seven-line string literal, leaving nmr.py unparseable, and the
            # SyntaxError came out of importlib.reload -- *outside* the
            # try/except that guards run_checks -- so it propagated out of
            # main() and ended the run.  43 of the 47 mutations had been tested
            # and the remaining four were never reached, with the log ending in
            # a traceback instead of a verdict.  That is the same failure this
            # harness was written to prevent (see the N15 note below), arriving
            # through a different door: catching exceptions from the checks does
            # not catch exceptions from loading the code the checks run on.
            try:
                contract = reload_engine()
                p = make_payload() if needs_payload == "water" else None
                t = (make_payload_thiophene()
                     if needs_payload == "thiophene" else None)
                fail = run_checks(p, contract, t)
            except Exception as exc:                     # noqa: BLE001
                # A mutation that breaks the code structurally raises instead
                # of letting an assertion fail.  That is NOT the assertion
                # firing -- the assertion never ran -- so it is reported as a
                # problem, not as a detection.  (N15 did exactly this on its
                # first run: it renamed the spectrum's key without renaming
                # the lookup, and the harness died mid-run, leaving nine
                # mutations untested.)
                crashed = f"{type(exc).__name__}: {exc}"
                fail = []
        finally:
            io.open(NMR, "w", encoding="utf-8").write(original)
        if crashed is not None:
            problems.append(
                f"{label}: raised {crashed[:110]} instead of failing an "
                "assertion")
            print(f"  !! CRASHED, assertion never reached: {crashed[:160]}")
            print()
            continue
        if not fail:
            problems.append(
                f"{label}: FIRED NOTHING -- the assertion guards nothing")
            print("  !! NO FAILURE RAISED")
        else:
            print(f"  -> {len(fail)} failure(s) raised:")
            for f in fail[:3]:
                print(f"       {f[:150]}")
        print()

    print("=" * 78)
    print("RESTORED -- unmutated again")
    print("=" * 78)
    contract = reload_engine()
    payload = make_payload()
    back = run_checks(payload, contract)
    if back:
        problems.append(f"restore: {back}")
        print(f"  !! not clean after restore: {back}")
    else:
        print("  clean\n")

    if io.open(NMR, encoding="utf-8").read() != original:
        problems.append(f"{NMR} was not restored byte-for-byte")
        print(f"  !! {NMR} differs from the original after restore")

    for s in skipped:
        print(f"SKIPPED {s}")

    if problems:
        print("MUTATION PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    fired = len(MUTATIONS) - len(skipped)
    print(f"All {fired} mutations fired; source restored cleanly.")
    if skipped:
        print(f"{len(skipped)} mutation(s) SKIPPED -- not proven this run:")
        for s in skipped:
            print("  -", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
