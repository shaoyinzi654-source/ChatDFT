"""Recompute the accuracy ledger -- does the method still reproduce each scale?

    python -m backend.check_shift_scales          # all entries, plus the proof
    python -m backend.check_shift_scales F Se     # a subset
    python -m backend.check_shift_scales --prove  # the drift check only, no SCF

Why this exists.  ``engine/nmr.py`` claims, in prose, that its 1H and 13C
shifts are good to about 2 ppm.  For four rounds that claim lived in a
docstring, attributed to a probe, recomputed by nothing -- and it has already
gone stale: two of its numbers measured against the current molecule library
come out differently from the ones written down, because the library's
geometries were re-derived after the ledger was written.  A number in prose
cannot be wrong, so it cannot be trusted.

``SCALE_CHECK`` turns that ledger into data: per nucleus, the reference the
product actually uses, a molecule whose shift against it is known, and the
computed value.  This gate recomputes every entry and fails if the answer has
moved, so the ledger is a measurement rather than a memory.

``SCALE_UNMEASURED`` records the nuclei whose scale has never been measured,
each with the pair it is about and the value this build computes for it.  Those
are recomputed too: the computed side is the half that rots, and pinning it
means that when the experimental value finally arrives it is compared against a
verified number rather than a remembered one.

``SHIFT_LEDGER`` is the twelve 1H/13C shifts the 1H and 13C tolerances are set
from.  It lived in the module docstring for seven rounds with nothing
recomputing it, and by the time anyone looked, two of its numbers had moved with
the library geometries and its *mean* had rotted from 1.89 to 1.736 ppm -- a
derived quantity written into prose.  The rows are data now, the derived
quantities are printed by this gate rather than written down, and each tolerance
is checked against every row of its nucleus instead of the single molecule it
was measured on.  That check is what raised the 13C tolerance from 2.5 to 3.2
ppm: methane is 2.09 out, methanol 3.04.

It also re-checks the *verdict*.  A ``usable`` entry has to be inside its own
tolerance; an unusable one has to be unusable by the numbers recorded for it
(the computed response small against the nucleus's shift range, and the
reference geometry spread comparable to it).  A method that improves should
fail this gate -- that is the point, because 77Se is currently suppressed on
the strength of these numbers and the suppression has to be re-decided rather
than inherited.

And it falsifies its own drift check first, because that check is the one thing
here that cannot be broken by a mutation: a band that never fires prints the
same PASS as a band that works.  See ``prove()``.

The molecules are the ledger's own geometries, resolved through
``nmr.scale_geometry``, which for a reference compound is ``reference_geometry``
and therefore the library geometry where the library has one -- the same
geometry a real job uses.  Reference shieldings go through
``reference_shieldings`` so the on-disk cache is used.
"""
from __future__ import annotations

import sys
import time

import backend.bootstrap as bootstrap

bootstrap.setup()

from backend.engine import nmr  # noqa: E402
from backend.engine import elements  # noqa: E402

# How large a trend step has to be before it counts as evidence that the method
# carries a scale.  A sign test alone is satisfied by numerical noise: the claim
# "each methyl deshields phosphorus" would pass on a step of +0.001 ppm, and a
# claim that cannot be false is not a claim.
TREND_FLOOR_PPM = 5.0


# One geometry, one shielding tensor per process run.  The twelve-shift section
# walks the same six molecules once per nucleus, and benzene at 6-31G* is 96
# basis functions: computing it twice costs seven minutes for a number that is
# identical by construction.  This is a memo, not the on-disk cache -- it lives
# and dies with the run, so a stale value cannot survive into the next one.
_SIGMA_CACHE: dict = {}


def _sigma_key(atoms, label: str):
    return (label, tuple(tuple(a) for a in atoms))


def _shielding_raw(atoms, label: str):
    """sigma_iso per element, in ppm, for a geometry given as atom tuples."""
    import numpy as np

    xyz = nmr._xyz(atoms)
    mol = nmr._build_mol(xyz, "6-31g*", 0, 1)
    ops = nmr.build_operators(mol)
    sigma, _diag = nmr.shielding_tensor(mol, ops)
    syms = [a[0] for a in atoms]
    out = {}
    for el in sorted(set(syms)):
        idx = [i for i, s in enumerate(syms) if s == el]
        vals = [float(np.trace(sigma[i]) / 3.0) * nmr.PPM for i in idx]
        out[el] = sum(vals) / len(vals)
    return out, mol.nao_nr(), syms


def shielding(atoms, label: str):
    """``_shielding_raw``, computed at most once per geometry per run.

    Callers that print a duration must ask ``_sigma_key(...) in _SIGMA_CACHE``
    first: "0.0 s" and "we did not recompute this" are different claims, and a
    gate that reports the second as the first is reporting a recomputation it
    did not do.
    """
    key = _sigma_key(atoms, label)
    if key not in _SIGMA_CACHE:
        _SIGMA_CACHE[key] = _shielding_raw(atoms, label)
    return _SIGMA_CACHE[key]


def band_for(recorded: float) -> float:
    """How far the recomputation may move without the ledger being wrong.

    2 ppm, or 5% of the shift when the shift is large -- the 5% is there
    because a 700 ppm shift carried to 2 ppm would be a tolerance tighter than
    the basis set's own error, and the 2 ppm floor is there because a shift near
    zero would otherwise have a band narrower than the numerical noise in the
    finite-difference-free GIAO solve.

    Pulled out as a function of the recorded value alone so that the rule can be
    tested without an SCF: see ``prove()``.  A band that no synthetic number can
    cross is a band that guards nothing, and before this round there was no way
    to tell whether the one in here could be crossed at all.
    """
    return max(2.0, abs(recorded) * 0.05)


def band_failures(tag: str, delta: float, recorded: float, sample: str,
                  reference: str, where: str) -> list:
    """The drift check on its own, as a pure function of three numbers.

    ``where`` says which sentence to use -- the ledger's ("Either the method
    changed or the recorded number was never measured") or the gaps'
    ("The recorded gap describes a method this build no longer has") -- because
    the consequence differs: a stale ledger entry invalidates a verdict, while a
    stale gap means the experimental value it is waiting for would be compared
    against the wrong number.
    """
    band = band_for(recorded)
    if abs(delta - recorded) <= band:
        return []
    if where == "ledger":
        tail = ("Either the method changed or the recorded number was never "
                "measured; either way the verdict that rests on it has to be "
                "re-decided")
    else:
        tail = ("The recorded gap describes a method this build no longer has, "
                "so the experimental value it is waiting for would be compared "
                "against the wrong number")
    return [
        f"{tag}: the {where} records {recorded:+.2f} ppm for {sample} against "
        f"{reference} and the recomputation gives {delta:+.2f} -- "
        f"{abs(delta - recorded):.2f} ppm apart, outside the {band:.2f} ppm "
        f"band.  {tail}"]


def trend_failures(tag: str, step: float, want_sign: int, label: str,
                   rng: float = 0.0) -> list:
    """The directional claim, as a pure function of the step it is about.

    Pulled out for the same reason ``band_failures`` was: the floor is a number,
    and a number that nothing can violate is not a rule.  ``prove()`` drives
    this directly with steps on both sides of the floor and both signs, which is
    the only way to show that the floor is where it says it is.
    """
    if want_sign > 0 and step < TREND_FLOOR_PPM:
        return [f"{tag}: the gap claims that {label}, but the recomputed step "
                f"is {step:+.2f} ppm -- below the {TREND_FLOOR_PPM:.0f} ppm "
                "floor, so the claim is either false or too small to be "
                "evidence of a scale"]
    if want_sign < 0 and step > -TREND_FLOOR_PPM:
        return [f"{tag}: the gap claims that {label}, but the recomputed step "
                f"is {step:+.2f} ppm -- above the {-TREND_FLOOR_PPM:.0f} ppm "
                "floor, so the claim is either false or too small to be "
                "evidence of a scale"]
    if rng and abs(step) > rng:
        return [f"{tag}: the trend step {step:+.1f} ppm is outside the "
                f"{rng:.0f} ppm range of the nucleus"]
    return []


def geometry_sane(atoms, label: str) -> list:
    """Every atom bonded to something, every hydrogen bonded exactly once.

    A constructed geometry is easy to get subtly wrong -- a first version of
    the 19F entry placed a hydrogen 1.2656 A from its carbon because the local
    frame it used was not orthonormal -- and the symptom is a shielding that is
    wrong by a thousand ppm, which reads like a defect in the method.  These
    two checks catch the geometry instead, with a message that says so.
    """
    import numpy as np

    out = []
    syms = [a[0] for a in atoms]
    coords = np.array([a[1:] for a in atoms], dtype=float)
    deg = [0] * len(atoms)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if 0.4 < d <= elements.bond_cutoff(syms[i], syms[j]):
                deg[i] += 1
                deg[j] += 1
    for i, sym in enumerate(syms):
        if deg[i] == 0:
            out.append(f"{label}: atom {i} ({sym}) is not bonded to anything")
        elif sym == "H" and deg[i] != 1:
            out.append(f"{label}: hydrogen {i} has {deg[i]} bonds, not 1")
    return out


def _recompute(fn, el: str, refused: list) -> list:
    """Run one recomputation, turning a refused calculation into a skip.

    The memory guard refuses a molecule when the machine is short of physical
    memory -- ``dimethyl_selenide`` is nao = 82 and priced at 3.0 GB, and this
    machine has been sitting between 1.9 and 2.8 GB free.  Until this wrapper
    existed that refusal came out of ``check_one`` as an uncaught ``DFTError``
    and ended the run with a traceback, which is the worst of the available
    outcomes: a gate that crashes reports neither pass nor fail, and the
    reason it crashed (a busy desktop) is indistinguishable from the reason a
    real defect would stop it.

    A skip is not a pass.  The assertion behind this recomputation was never
    run, so the caller has to count it and say so in the verdict line --
    "4 of 5 recomputed" and "5 of 5 recomputed" must not print the same way.
    """
    try:
        return fn(el)
    except nmr.DFTError as exc:
        refused.append(f"{el}: {str(exc).split('.')[0][:110]}")
        return []


def check_one(el: str) -> list:
    entry = nmr.SCALE_CHECK[el]
    fail: list = []
    tag = f"scale/{el}"

    ref_key = entry["reference_geometry"]
    sample_key = entry["sample_geometry"]
    ref_atoms = nmr.scale_geometry(ref_key)
    sample_atoms = nmr.scale_geometry(sample_key)

    fail += geometry_sane(ref_atoms, f"{tag} reference {ref_key}")
    fail += geometry_sane(sample_atoms, f"{tag} sample {sample_key}")

    t0 = time.time()
    if ref_key in nmr.REFERENCES:
        res = nmr.reference_shieldings(ref_key, "6-31g*", use_cache=True)
        sref = {e: v["sigma_iso_ppm"]
                for e, v in res["element_shielding_ppm"].items()}
        nao_ref = res["nao"]
        cached = bool(res.get("cached"))
    else:
        sref, nao_ref, _ = shielding(ref_atoms, ref_key)
        cached = False
    sample_reused = _sigma_key(sample_atoms, sample_key) in _SIGMA_CACHE
    ssam, nao_sam, _ = shielding(sample_atoms, sample_key)
    dt = time.time() - t0
    # Two different reuses are possible and the old line reported both as a
    # bare duration: the reference may have come off the on-disk cache
    # (``cached``), and the sample's tensor may have been computed a moment ago
    # for another nucleus in this same run.  ``cached (0.0 s)`` reads as
    # "everything here was instant", which is not what happened.
    spent = (f"cached ref, sample tensor reused ({dt:.1f} s total)"
             if sample_reused else
             f"cached ref ({dt:.1f} s)" if cached else f"{dt:.1f} s")

    if el not in sref or el not in ssam:
        return [f"{tag}: the nucleus {el} is not in "
                f"{ref_key} ({sorted(sref)}) or {sample_key} ({sorted(ssam)})"]

    delta = sref[el] - ssam[el]
    # A missing recorded value is a defect in the table, not in the method, and
    # it has to be reported as such.  ``float(None)`` raises, and a traceback out
    # of a gate reads as "the gate is broken" when the truth is "the ledger is"
    # -- the contract catches it too, but this gate must not depend on another
    # gate having run first to stay legible.
    if entry.get("computed_ppm") is None:
        fail.append(f"{tag}: the ledger entry has no computed value, so there is "
                    "nothing for the recomputation to be checked against")
        return fail
    recorded = float(entry["computed_ppm"])
    fail += band_failures(tag, delta, recorded, str(entry["sample"]),
                          str(entry["reference"]), "ledger")

    exp = entry.get("experimental_ppm")
    tol = float(entry.get("tolerance_ppm", 0.0))
    if entry.get("usable"):
        if exp is None:
            fail.append(f"{tag}: usable with no experimental value")
        elif abs(delta - float(exp)) > tol:
            fail.append(
                f"{tag}: marked usable, but the recomputed shift {delta:+.2f} "
                f"is {abs(delta - float(exp)):.2f} ppm from the experimental "
                f"{float(exp):+.2f} and the tolerance is {tol:.2f}")
    else:
        rng = float(entry.get("range_ppm", 0.0))
        if rng and abs(delta) > rng / 10.0:
            fail.append(
                f"{tag}: marked UNUSABLE, but the recomputed response is "
                f"{abs(delta):.1f} ppm against a {rng:.0f} ppm shift range -- "
                "the method covers enough of the scale to report a shift, so "
                "the suppression is no longer justified")

    print(f"  {tag:10s} {entry['sample']:8s} vs {entry['reference']:12s} "
          f"computed {delta:+9.2f}  recorded {recorded:+9.2f}  "
          f"experimental {('     n/a' if exp is None else f'{float(exp):+9.2f}')}"
          f"  nao {nao_ref}/{nao_sam}  ({spent})")
    return fail


def check_unmeasured(el: str) -> list:
    """Recompute a pair that has no experimental value to be checked against.

    The computed side is still a measurement, and it is the half that rots:
    the module docstring's twelve-shift ledger went stale because the library's
    geometries were re-derived underneath it and nothing recomputed the
    numbers.  Pinning these here means that when the experimental value finally
    arrives, the computed side is already verified rather than remembered --
    and that a change in the method shows up as a failure instead of silently
    invalidating a recorded gap.
    """
    entry = nmr.SCALE_UNMEASURED[el]
    fail: list = []
    tag = f"scale/{el}"

    ref_key = entry["reference_geometry"]
    sample_key = entry["sample_geometry"]
    ref_atoms = nmr.scale_geometry(ref_key)
    sample_atoms = nmr.scale_geometry(sample_key)
    fail += geometry_sane(ref_atoms, f"{tag} reference {ref_key}")
    fail += geometry_sane(sample_atoms, f"{tag} sample {sample_key}")

    t0 = time.time()
    if ref_key in nmr.REFERENCES:
        res = nmr.reference_shieldings(ref_key, "6-31g*", use_cache=True)
        sref = {e: v["sigma_iso_ppm"]
                for e, v in res["element_shielding_ppm"].items()}
        nao_ref = res["nao"]
        cached = bool(res.get("cached"))
    else:
        sref, nao_ref, _ = shielding(ref_atoms, ref_key)
        cached = False
    sample_reused = _sigma_key(sample_atoms, sample_key) in _SIGMA_CACHE
    ssam, nao_sam, _ = shielding(sample_atoms, sample_key)
    dt = time.time() - t0
    # Two different reuses are possible and the old line reported both as a
    # bare duration: the reference may have come off the on-disk cache
    # (``cached``), and the sample's tensor may have been computed a moment ago
    # for another nucleus in this same run.  ``cached (0.0 s)`` reads as
    # "everything here was instant", which is not what happened.
    spent = (f"cached ref, sample tensor reused ({dt:.1f} s total)"
             if sample_reused else
             f"cached ref ({dt:.1f} s)" if cached else f"{dt:.1f} s")

    if el not in sref or el not in ssam:
        return [f"{tag}: the nucleus {el} is not in {ref_key} "
                f"({sorted(sref)}) or {sample_key} ({sorted(ssam)})"]

    delta = sref[el] - ssam[el]
    if entry.get("computed_ppm") is None:
        fail.append(f"{tag}: the recorded gap has no computed value, so 'not "
                    "measured' and 'not computed' are the same entry -- and "
                    "they need different work")
        return fail
    recorded = float(entry["computed_ppm"])
    fail += band_failures(tag, delta, recorded, str(entry["sample"]),
                          str(entry["reference"]), "gap")
    rng = float(entry.get("range_ppm", 0.0))
    if rng and abs(delta) > rng:
        fail.append(f"{tag}: the recomputed shift {abs(delta):.1f} ppm is "
                    f"outside the {rng:.0f} ppm range of the nucleus")

    # The directional claims.  "Each methyl deshields phosphorus" is the
    # evidence that the method carries a real scale for the nucleus rather than
    # producing one arbitrary number, so it is a claim and it gets recomputed
    # like any other.  The floor is the part that keeps it from being vacuous:
    # a step of 0.001 ppm has the right sign and says nothing, so a trend
    # assertion without a magnitude is an assertion that cannot fail.
    for i, (geom, want_sign) in enumerate(entry.get("trend") or []):
        t_atoms = nmr.scale_geometry(geom)
        fail += geometry_sane(t_atoms, f"{tag} trend[{i}] {geom}")
        s_t, nao_t, _syms = shielding(t_atoms, geom)
        if el not in s_t:
            fail.append(f"{tag}: the trend step {geom} does not contain {el}")
            continue
        step = (sref[el] - s_t[el]) - delta
        word = "deshields" if want_sign > 0 else "shields"
        fail += trend_failures(tag, step, int(want_sign),
                               f"each additional substituent {word} {el} "
                               f"(here {entry['sample']} -> {geom})", rng)
        print(f"  {tag:10s} trend    {geom:8s} vs {entry['sample']:12s} "
              f"step {step:+9.2f}  (claims {word})  nao {nao_t}")

    print(f"  {tag:10s} {entry['sample']:8s} vs {entry['reference']:12s} "
          f"computed {delta:+9.2f}  recorded {recorded:+9.2f}  "
          f"experimental {('     n/a'):>9s}"
          f"  nao {nao_ref}/{nao_sam}  ({spent})")
    return fail


def tolerance_failures(tag: str, tolerance: float, errors: dict,
                       exceptions: dict) -> list:
    """A thin wrapper over ``nmr.tolerance_rule``, for the proof's sake.

    The rule itself lives in the engine, next to the two tables it reads,
    because the contract checks it too -- against the numbers the ledger
    *records* rather than the ones this gate *recomputes*.  One implementation,
    two callers.  What this adds is the ability to drive the rule with synthetic
    numbers, which is what ``prove()`` needs: a rule that can only be exercised
    by running six SCFs is a rule that will never be exercised.
    """
    return nmr.tolerance_rule(tolerance, errors, exceptions, tag=tag)


def ledger_summary(tag: str, rows: list, band: float, exceptions: dict) -> str:
    """One line describing a nucleus's twelve-shift rows.

    Everything here is *derived* from the rows -- how many are inside the band,
    the worst error, the mean.  None of it is written down in the module, on
    purpose: the docstring used to carry the count and the mean, and the mean
    had rotted from 1.89 to 1.736 with nothing to notice.  A derived number
    belongs where it is recomputed.
    """
    errs = [(name, abs(c - e)) for name, _k, c, e in rows]
    inside = sum(1 for _n, d in errs if d <= band)
    worst_name, worst = max(errs, key=lambda kv: kv[1])
    mean = sum(d for _n, d in errs) / len(errs)
    inl = [(n, d) for n, d in errs if n not in exceptions]
    wl_name, wl = max(inl, key=lambda kv: kv[1]) if inl else ("-", 0.0)
    note = (f", worst inlier {wl:.3f} ({wl_name})" if exceptions else "")
    exc = (f", {len(exceptions)} exception(s) {sorted(exceptions)}"
           if exceptions else "")
    return (f"  {tag:10s} {len(rows)} rows, {inside} inside {band:.1f} ppm, "
            f"worst {worst:.3f} ({worst_name}), mean {mean:.3f}{note}{exc}")


def check_ledger(el: str) -> list:
    """Recompute the twelve-shift ledger for one nucleus.

    This is the measurement the 1H and 13C tolerances are set from, and until
    now nothing recomputed it.  Two of its numbers had already moved when the
    library's geometries were re-derived, and a third -- the mean -- had rotted
    without anyone noticing, because a mean is derived and the docstring wrote
    it down anyway.
    """
    entry = nmr.SHIFT_LEDGER[el]
    fail: list = []
    tag = f"ledger/{el}"

    ref_key = entry["reference_geometry"]
    if ref_key in nmr.REFERENCES:
        res = nmr.reference_shieldings(ref_key, "6-31g*", use_cache=True)
        sref = {e: v["sigma_iso_ppm"]
                for e, v in res["element_shielding_ppm"].items()}
    else:
        sref, _nao, _s = shielding(nmr.scale_geometry(ref_key), ref_key)

    if el not in sref:
        return [f"{tag}: the reference {ref_key} does not contain {el}"]

    errors: dict = {}
    for name, geom, recorded, exp in entry["rows"]:
        atoms = nmr.scale_geometry(geom)
        fail += geometry_sane(atoms, f"{tag} {name}")
        # The shielding tensor belongs to the molecule, not to the nucleus, so
        # the second nucleus reads the tensor the first one computed.  That is
        # the same measurement, not a weaker one -- but it must not be printed
        # as "0.0 s", which reads as a recomputation that happened.
        reused = _sigma_key(atoms, geom) in _SIGMA_CACHE
        t0 = time.time()
        s, nao, _syms = shielding(atoms, geom)
        spent = "tensor reused from this run" if reused else f"{time.time() - t0:.1f} s"
        if el not in s:
            fail.append(f"{tag}: {name} ({geom}) does not contain {el}")
            continue
        # Both sides are element -> sigma_iso in ppm; the element index is not
        # optional on either side.  Writing ``sref[el] - s`` here subtracted a
        # whole dict from a float and the section died on its first row -- which
        # is the failure mode this gate exists to make impossible, one line
        # inside the gate itself.
        delta = sref[el] - s[el]
        fail += band_failures(tag, delta, float(recorded), name,
                              str(entry["reference"]), "ledger")
        errors[name] = abs(delta - float(exp))
        print(f"  {tag:10s} {name:10s} computed {delta:+9.3f}  recorded "
              f"{float(recorded):+9.3f}  exptl {float(exp):+8.2f}  "
              f"error {delta - float(exp):+7.3f}  nao {nao:4d}"
              f"  ({spent})")

    fail += nmr.tolerance_holds(el, errors)
    print(ledger_summary(tag, entry["rows"], float(entry["band_ppm"]),
                         dict(entry.get("exceptions") or {})))
    return fail


def prove() -> list:
    """Can the drift check actually fire?  Returns its failures, prints nothing.

    Every other assertion in this build is locked behind a mutation that proves
    it can fail.  This one could not be, cheaply: the honest way to falsify it is
    to change the method until the shielding moves, which is an SCF per trial,
    and the gate would then take three times as long on every run for a proof
    that only has to be made once.

    So the rule was pulled out as ``band_for``/``band_failures``, functions of
    three numbers, and this exercises them directly.  What is proven is the
    *rule*: that a drift just inside the band passes, a drift just outside it
    fails in both directions, and the band scales with the shift.  What is not
    proven here is that the recomputation feeds them the right numbers -- that
    half is the normal run, which is checked every time.

    This matters because the failure it rules out is invisible.  A band check
    that never fires prints the same PASS as one that works, so the state
    "the gate recomputes everything and can never object" looks exactly like
    "the gate recomputes everything and it is all still right".

    Verified to fail: replacing ``band_for`` with a constant 1e9 ppm makes this
    report nine problems, with 1e-12 eight, and collapsing the two sentences
    into one reports one.
    """
    fail: list = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            fail.append(msg)

    # The rule itself, on numbers chosen to straddle the boundary in both
    # directions rather than to be far from it: a proof that only uses absurd
    # values proves the comparison exists, not that it is at the right place.
    for recorded in (0.0, -2.3, 66.41, -212.27, 716.94, -101.38):
        band = band_for(recorded)
        want = max(2.0, abs(recorded) * 0.05)
        expect(abs(band - want) < 1e-9,
               f"prove: the band for {recorded:+.2f} is {band:.3f} ppm, not "
               f"{want:.3f}")
        if abs(recorded) < 40.0:
            # The floor has to be the thing that applies here, not the 5%: a
            # shift of -2.3 ppm would otherwise get a 0.115 ppm band, which is
            # tighter than the noise in the solve it is tolerating.
            expect(band == 2.0,
                   f"prove: a shift of {recorded:+.2f} got a {band:.3f} ppm "
                   "band instead of the 2 ppm floor, so it would fail on "
                   "numerical noise")
        inside = band_failures("prove", recorded + 0.5 * band, recorded,
                               "s", "r", "ledger")
        expect(inside == [],
               f"prove: a drift of half the band fired "
               f"{len(inside)} failure(s); the band is narrower than it says "
               "and the ledger would fail on numerical noise")
        for sign in (+1.0, -1.0):
            outside = band_failures("prove", recorded + sign * 1.5 * band,
                                    recorded, "s", "r", "ledger")
            expect(len(outside) == 1,
                   f"prove: a drift of {sign * 1.5:.1f}x the band produced "
                   f"{len(outside)} failures, not 1 -- the check does not fire "
                   "on a shift that has moved out of its band, which is the "
                   "only thing it exists to catch")
            if outside:
                expect(f"{band:.2f} ppm" in outside[0],
                       "prove: the failure does not quote the band it used, so "
                       "a reader cannot tell how far the answer had to move")

    # The two sentences have to differ, because the consequences do: a stale
    # ledger entry invalidates a verdict, a stale gap misdirects a future
    # comparison.  Sharing one message would hide which of the two happened.
    led = band_failures("t", 100.0, 0.0, "s", "r", "ledger")
    gap = band_failures("t", 100.0, 0.0, "s", "r", "gap")
    expect(bool(led) and bool(gap) and led[0] != gap[0],
           "prove: the ledger and the gaps report a stale number with the same "
           "sentence, so the reader cannot tell which consequence applies")

    # The trend floor, on both sides and both signs.  A directional claim whose
    # magnitude is never tested passes on numerical noise, so the interesting
    # assertions are the two just inside the floor -- a proof that only uses a
    # step of 100 ppm shows the comparison exists, not where it sits.
    just_under = TREND_FLOOR_PPM * 0.5
    just_over = TREND_FLOOR_PPM * 2.0
    for sign, good, bad in ((+1, just_over, -just_over), (-1, -just_over, just_over)):
        if trend_failures("t", good, sign, "each substituent does the thing"):
            fail.append(
                f"prove: a trend step of {good:+.1f} ppm failed a claim whose "
                f"direction it has ({'deshielding' if sign > 0 else 'shielding'}"
                f"); the floor rejects real evidence")
        if not trend_failures("t", bad, sign, "each substituent does the thing"):
            fail.append(
                f"prove: a trend step of {bad:+.1f} ppm passed a claim whose "
                f"direction it contradicts -- the sign is not being checked at "
                "all")
    if not trend_failures("t", just_under, +1, "each substituent deshields"):
        fail.append(
            f"prove: a deshielding step of {just_under:+.2f} ppm passed as "
            f"evidence of a scale against a {TREND_FLOOR_PPM:.0f} ppm floor; "
            "the floor is not being applied, so the claim can be satisfied by "
            "noise")
    if trend_failures("t", just_over, +1, "each substituent deshields"):
        fail.append(
            f"prove: a deshielding step of {just_over:+.2f} ppm was rejected "
            f"against a {TREND_FLOOR_PPM:.0f} ppm floor; the floor is stricter "
            "than it says")
    # And the range sanity check, which is a different rule in the same function.
    if not trend_failures("t", 999.0, +1, "each substituent deshields", rng=430.0):
        fail.append("prove: a 999 ppm trend step passed inside a 430 ppm range")

    # The tolerance rule.  It is the assertion that turns "the 1H tolerance is
    # 1.0 ppm" from a memory into a claim about a nucleus, so it has to be
    # exercised in every direction: too tight, too loose, and the two ways a
    # label can make it vacuous.  Only the second case below may pass; every
    # other one is a failure the rule exists to produce, so each is asserted in
    # the direction that would catch the rule being absent.
    errs = {"a": 1.0, "b": 3.0}
    if not tolerance_failures("t", 2.0, errs, {}):
        fail.append("prove: a 2.0 ppm tolerance passed errors of 1.0 and 3.0 -- "
                    "a tolerance below the worst error it is meant to cover is "
                    "exactly the claim that cannot hold, and it was accepted")
    if tolerance_failures("t", 3.0, errs, {}):
        fail.append("prove: a 3.0 ppm tolerance was rejected against errors of "
                    "1.0 and 3.0, which it covers; the rule rejects a claim "
                    "that holds")
    if not tolerance_failures("t", 20.0, errs, {}):
        fail.append("prove: a 20.0 ppm tolerance passed a worst error of 3.0 -- "
                    "more than three times the error it was measured at, so the "
                    "rule is not bounding how loose a tolerance may be")
    if not tolerance_failures("t", 3.0, errs, {"b": "known weakness"}):
        fail.append("prove: an exception whose error is *inside* the tolerance "
                    "passed; a stale exception label hides a row that now "
                    "passes and has to be reported")
    if not tolerance_failures("t", 3.0, errs, {"a": "x", "b": "y"}):
        fail.append("prove: a ledger with every row excepted passed; the "
                    "tolerance is then a claim about nothing")
    if not tolerance_failures("t", 3.0, {}, {}):
        fail.append("prove: an empty ledger passed, so a nucleus whose rows "
                    "were all deleted would keep its tolerance and never be "
                    "checked")
    return fail


def check_cache_key() -> list:
    """Does the reference cache key carry the code that filled it?

    ``data/nmr_cache`` ships as data, so a user who upgrades keeps whatever
    numbers were computed by the previous version.  The key already carried the
    basis, the tolerance, the field step and a fingerprint of the geometry,
    each added after a concrete failure; the code itself was the last thing
    missing.  Edit the GIAO solve and every cached reference stays put,
    computed by the old code, and every shift inherits the difference as a
    constant offset -- which is the failure the module's own docstring calls
    out about ``conv_tol`` and which nothing in the output can reveal.

    This runs first and costs nothing, because the failure it guards against is
    invisible in exactly the same way whether or not the rest of the gate
    passes.
    """
    fail: list = []
    tag = "cachekey"

    method = nmr._method_signature()
    if not method:
        fail.append(f"{tag}: the method signature is empty, so every key "
                    "collides and the cache is unversioned again")
    if nmr._method_signature() != method:
        fail.append(f"{tag}: the method signature is not stable within one "
                    "process, so no entry can ever be read back")

    # The only way to show a key really carries something is to vary that thing
    # and watch the key move.  Both directions: the method must move it, and
    # the geometry -- which was added for the same reason, one round earlier --
    # must still move it.
    base = nmr._cache_key("tms", "6-31g*", nmr.CONV_TOL, nmr.FIELD_STEP, "sig")
    if nmr._cache_key("tms", "6-31g*", nmr.CONV_TOL, nmr.FIELD_STEP, "sig",
                      method) != base:
        fail.append(f"{tag}: passing the signature explicitly changed the key, "
                    "so the default is not the one the reader uses")
    if nmr._cache_key("tms", "6-31g*", nmr.CONV_TOL, nmr.FIELD_STEP, "sig",
                      method + "x") == base:
        fail.append(f"{tag}: two different methods produced the same key -- a "
                    "reference computed by other code would be read as if it "
                    "were this code's")
    if nmr._cache_key("tms", "6-31g*", nmr.CONV_TOL, nmr.FIELD_STEP,
                      "other-sig", method) == base:
        fail.append(f"{tag}: two different geometries produced the same key")

    # And the signature must actually be reading the shielding code, not a
    # constant that happens to look like a hash.  Swapping one of the functions
    # it claims to cover has to change it.
    for name in ("shielding_tensor", "build_operators", "_build_mol", "_xyz"):
        original = getattr(nmr, name)
        try:
            def _stub(*a, **k):
                return None
            _stub.__name__ = name
            setattr(nmr, name, _stub)
            moved = nmr._method_signature() != method
        finally:
            setattr(nmr, name, original)
        if not moved:
            fail.append(f"{tag}: replacing {name} left the signature unchanged, "
                        "so editing that function would not invalidate the "
                        "cache it feeds")
        if nmr._method_signature() != method:
            fail.append(f"{tag}: the signature did not come back after "
                        f"restoring {name}")

    if "method" not in nmr._CACHE_REQUIRED:
        fail.append(f"{tag}: 'method' is not a required cache field, so a file "
                    "written at a matching path by another code version would "
                    "be read instead of rejected")

    # The path is the first guard; the reader is the second.  A foreign entry
    # is normally at a different filename, but a directory walk, a repair
    # script or a person can hand one straight to the reader -- so the reader
    # has to refuse it on its own, without the key's help.
    import json as _json
    import os as _os
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".json")
    _os.close(fd)
    try:
        entry = {"format": nmr.CACHE_FORMAT}
        for field in nmr._CACHE_REQUIRED:
            entry[field] = "x"
        entry["method"] = "some other code"
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(entry, fh)
        if nmr._read_cache(tmp) is not None:
            fail.append(f"{tag}: the reader accepted an entry stamped with a "
                        "different method -- the only thing standing between a "
                        "foreign value and being used would be the filename")

        entry["method"] = method
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(entry, fh)
        if nmr._read_cache(tmp) is None:
            fail.append(f"{tag}: the reader rejected an entry that carries "
                        "every required field and the current method, so the "
                        "check above proves nothing -- a reader that refuses "
                        "everything refuses the wrong method for the wrong "
                        "reason")
    finally:
        try:
            _os.remove(tmp)
        except OSError:
            pass

    # And the orphans already on disk must be inert.  They are not read today
    # because the key moved; this is the assertion that they would not be read
    # even if something did open them.
    inventory = nmr.cache_inventory()
    for path in inventory["stale"]:
        if nmr._read_cache(path) is not None:
            fail.append(f"{tag}: {_os.path.basename(path)} is not reachable "
                        "from the current key but the reader accepts it, so it "
                        "is only one directory walk away from being used")
    if inventory["stale"]:
        print(f"  note: {len(inventory['live'])} entries are reachable from "
              f"the current key, {len(inventory['stale'])} are not")
        print("        (the unreachable ones are inert -- asserted above -- "
              "but a person")
        print("         listing this directory would see both; see "
              "nmr.cache_inventory)")
    return fail


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    only_proof = "--prove" in argv
    wanted = [a for a in argv if not a.startswith("-")]
    known = sorted(nmr.SCALE_CHECK)
    gaps = sorted(nmr.SCALE_UNMEASURED)
    shift_rows = sorted(nmr.SHIFT_LEDGER)
    unknown = [w for w in wanted
               if w not in known and w not in gaps and w not in shift_rows]
    if unknown:
        print(f"unknown nucleus/nuclei: {', '.join(unknown)}")
        print(f"the scale ledger covers: {', '.join(known)}")
        print(f"the gaps cover:          {', '.join(gaps)}")
        print(f"the shift rows cover:    {', '.join(shift_rows)}")
        return 2
    todo = wanted or known
    todo_gaps = [el for el in gaps if not wanted or el in wanted]
    todo_rows = [el for el in shift_rows if not wanted or el in wanted]

    fail: list = []

    print("=" * 74)
    print("  the drift check, falsified on purpose")
    print("=" * 74)
    print("  half a band   -> no failure   (numerical noise is tolerated)")
    print("  1.5 bands     -> one failure  (a moved answer is caught)")
    print("  both signs, six recorded values, two sentences")
    print(f"  trend floor   -> {TREND_FLOOR_PPM * 0.5:.1f} ppm rejected, "
          f"{TREND_FLOOR_PPM * 2.0:.0f} ppm accepted, both signs")
    print()
    proof = prove()
    if proof:
        print(f"  !! {len(proof)} problem(s) in the drift check itself")
        for p in proof:
            print("     -", p)
    else:
        print("  ok: the drift check fires where it claims to and nowhere else")
    print()
    fail += proof
    print("=" * 74)
    print("  the reference cache key, which decides whether any of the")
    print("  numbers below are recomputed or read")
    print("=" * 74)
    cache_fail = check_cache_key()
    if cache_fail:
        for c in cache_fail:
            print(f"  !! {c}")
    else:
        print(f"  ok: the key carries the method ({nmr._method_signature()}), "
              "the geometry, the basis, the tolerance and the field step;")
        print("      changing any one of them moves it, and the signature "
              "reads the")
        print("      four functions that produce the number")
    print()
    fail += cache_fail

    # The cache check is part of the cheap half: it needs no SCF, and the
    # failure it guards against is exactly as invisible here as it is in a full
    # run, so --prove must not skip it.
    if only_proof:
        if fail:
            print(f"=== FAIL: {len(fail)} problem(s) in the drift check or the "
                  "cache key ===")
            for f in fail:
                print("  -", f)
            return 1
        print("=== PASS: the drift check and the cache key both fire where "
              "they claim to and nowhere else ===")
        return 0

    print("=" * 74)
    print("  the scale ledger, recomputed")
    print("=" * 74)
    refused: list = []
    n_scale = 0
    for el in todo:
        before = len(refused)
        fail += _recompute(check_one, el, refused)
        n_scale += len(refused) == before

    print()
    print("=" * 74)
    print("  the recorded gaps, recomputed (no experimental value to check)")
    print("=" * 74)
    n_gaps = 0
    for el in todo_gaps:
        before = len(refused)
        fail += _recompute(check_unmeasured, el, refused)
        n_gaps += len(refused) == before

    print()
    print("=" * 74)
    print("  the twelve-shift rows, recomputed -- the tolerances rest on these")
    print("=" * 74)
    n_rows = 0
    for el in todo_rows:
        before = len(refused)
        fail += _recompute(check_ledger, el, refused)
        if len(refused) == before:
            n_rows += len(nmr.SHIFT_LEDGER[el]["rows"])

    print()
    # The denominators are what was *asked for*; the numerators are what
    # actually ran.  They used to be the same expression -- `len(todo) of
    # len(known)` -- which meant a nucleus the memory guard refused would
    # still be counted as recomputed, and the verdict would claim coverage it
    # did not have.  A skip has to move the numerator, or it is not a skip,
    # it is a quieter kind of pass.
    n_rows_total = sum(len(nmr.SHIFT_LEDGER[e]["rows"]) for e in shift_rows)
    if refused:
        print(f"=== FAIL: {len(fail)} scale problem(s), and {len(refused)} "
              "recomputation(s) were REFUSED BY THE MEMORY GUARD, so the "
              "assertions behind them never ran ===")
        for r in refused:
            print(f"  - NOT RUN: {r}")
        for f in fail:
            print("  -", f)
        print("  (the refusal is the machine being short of memory, not the "
              "method: generate the reference with `python -m "
              "backend.gen_nmr_cache`, or free memory and re-run)")
        return 1
    if fail:
        print(f"=== FAIL: {len(fail)} scale problem(s) ===")
        for f in fail:
            print("  -", f)
        return 1
    # The proof is named in the counted line on purpose.  It is cheap enough to
    # run every time, but "cheap and always run" and "cheap and quietly skipped"
    # produce identical output unless the verdict itself says it ran -- and a
    # gate that recomputes everything and can never object is the failure mode
    # this whole section exists to rule out.
    print(f"=== PASS: drift check falsified; {n_scale} of {len(known)} scale "
          f"entries, {n_gaps} of {len(gaps)} gaps and {n_rows} of "
          f"{n_rows_total} shift rows recomputed and consistent ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
