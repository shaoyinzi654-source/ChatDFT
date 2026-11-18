"""Contract audit: every field the front end reads must be in the API payload.

The selector check (`check_selectors.py`) proves the DOM wiring is consistent;
this one proves the *data* wiring is. Both catch bugs that API-level tests miss,
because a render function reading an absent field silently produces `undefined`
in the UI rather than an error.

Runs against a live server.  Usage:  python -m backend.contract
"""

from __future__ import annotations

import argparse
import json
import json as _json
import math
import os as _os
import re as _re
import sys
import time
import urllib.error
import urllib.request

import numpy as np

BASE = "http://127.0.0.1:8000"
FAILURES: list[str] = []

# Molecules that carry an NMR-active element with no reference, wanted first.
# Thiophene is the one a user meets this case with, but it is nao = 82 at
# 6-31G* and the memory guard prices that at about 3.0 GB against the free
# physical memory *when the gate runs*, so on a busy machine the check went
# red for an environmental reason.  Methanethiol (CH3SH, 6 atoms) carries the
# same unreferenced nucleus -- sulfur -- for a fraction of the cost.
# backend/mutate_nmr.py imports this list rather than keeping its own copy:
# two lists that are meant to stay equal is the same defect as two conventions
# that differ by a constant.
UNREFERENCED_CARRIERS = ("thiophene", "methanethiol")


def _req(method: str, path: str, payload: dict | None = None, timeout: float = 60.0):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _wait(job_id: str, timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    job: dict = {}
    while time.time() < deadline:
        _, job = _req("GET", f"/api/job/{job_id}")
        if isinstance(job, dict) and job.get("status") in ("completed", "failed"):
            return job
        time.sleep(1.0)
    job["status"] = "timeout"
    return job


# ----------------------------------------------------------------------
#  field specifications, derived from reading frontend/static/js/app.js
# ----------------------------------------------------------------------
# every job type goes through handleCompleted() -> renderProperties()
COMMON = {
    "energy_hartree": (int, float),
    "energy_ev": (int, float),
    "homo_ev": (int, float),
    "lumo_ev": (int, float),
    "gap_ev": (int, float),
    "nelec": int,
    "nbf": int,
    "converged": bool,
    "scf_seconds": (int, float),
    "functional": str,
    "basis": str,
    "charge": int,
    "multiplicity": int,
    "warnings": list,
    "orbital_ladder": list,
    "mulliken_charges": list,
    "dipole": dict,
    "geometry": dict,
    "molecule": dict,
}

# renderProperties() reads these nested keys
DIPOLE_KEYS = ["x", "y", "z", "magnitude"]
CHARGE_KEYS = ["atom", "symbol", "charge"]
LADDER_KEYS = ["index", "energy_ha", "energy_ev", "occupation", "label"]
MOLECULE_KEYS = ["name", "formula", "atoms", "bonds", "natoms", "charge", "multiplicity"]

# geometry_optimization adds these
OPT_EXTRA = {"optimized_xyz": str, "opt_backend": str,
             "opt_converged": bool, "opt_grms": (int, float),
             "opt_gmax": (int, float), "opt_criteria": dict,
             "n_steps": int}
# excited_states adds these
EXC_EXTRA = {"excited_states": list}
STATE_KEYS = ["state", "energy_ev", "wavelength_nm", "oscillator_strength"]
SCAN_POINT_KEYS = ["index", "r", "energy_hartree", "energy_rel_kcal", "converged"]


def check_present(label: str, obj: dict, keys: list[str]) -> None:
    """Presence-only check, for keys whose type is not worth asserting."""
    for key in keys:
        if key not in obj:
            FAILURES.append(f"{label}: missing '{key}'")


def check_fields(label: str, obj: dict, spec: dict) -> None:
    for key, typ in spec.items():
        if key not in obj:
            FAILURES.append(f"{label}: missing '{key}'")
            continue
        val = obj[key]
        # bool is a subclass of int; do not let it satisfy a numeric check
        if typ is not bool and isinstance(val, bool):
            FAILURES.append(f"{label}: '{key}' is a bool, expected {typ}")
            continue
        if typ is int and isinstance(val, float):
            continue  # json turns ints into floats sometimes; harmless
        if not isinstance(val, typ):
            FAILURES.append(f"{label}: '{key}' is {type(val).__name__}, expected {typ}")


def check_list_of(label: str, rows: list, keys: list[str], expect_rows: bool = True) -> None:
    if expect_rows and not rows:
        FAILURES.append(f"{label}: expected at least one row, got none")
        return
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            FAILURES.append(f"{label}[{i}]: not an object")
            continue
        for k in keys:
            if k not in row:
                FAILURES.append(f"{label}[{i}]: missing '{k}'")


def run(kind: str, molecule: str, *, extra_spec: dict | None = None,
        require_rows: dict | None = None, **kw) -> dict | None:
    body = {"molecule": molecule, "kind": kind, "functional": "b3lyp",
            "basis": "6-31g*", **kw}
    code, resp = _req("POST", "/api/job", body)
    if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
        FAILURES.append(f"{kind}/{molecule}: submit failed HTTP {code} {str(resp)[:80]}")
        return None
    job = _wait(resp["job_id"])
    if job.get("status") != "completed":
        FAILURES.append(f"{kind}/{molecule}: job {job.get('status')} "
                        f"{(job.get('error') or '')[:80]}")
        return None

    result = job.get("result") or {}
    label = f"{kind}/{molecule}"

    check_fields(f"{label}.result", result, COMMON)
    if extra_spec:
        check_fields(f"{label}.result", result, extra_spec)

    check_fields(f"{label}.dipole", result.get("dipole") or {}, {k: (int, float) for k in DIPOLE_KEYS})
    check_list_of(f"{label}.mulliken_charges", result.get("mulliken_charges") or [], CHARGE_KEYS)
    check_list_of(f"{label}.orbital_ladder", result.get("orbital_ladder") or [], LADDER_KEYS)
    check_present(f"{label}.molecule", result.get("molecule") or {}, MOLECULE_KEYS)

    geo = result.get("geometry") or {}
    if "bonds" not in geo:
        FAILURES.append(f"{label}.geometry: missing 'bonds'")

    # job-level fields the UI reads
    for key in ("kind", "status", "explanation"):
        if not job.get(key):
            FAILURES.append(f"{label}.job: missing/empty '{key}'")

    # the orbital ladder must mark exactly one HOMO and one LUMO
    ladder = result.get("orbital_ladder") or []
    nh = sum(1 for o in ladder if o.get("label") == "HOMO")
    nl = sum(1 for o in ladder if o.get("label") == "LUMO")
    if nh != 1 or nl != 1:
        FAILURES.append(f"{label}: ladder labels HOMO={nh}, LUMO={nl} (want 1 and 1)")

    # The ladder is one spin channel, not both.  An unrestricted calculation
    # has two sets of orbital energies and the diagram shows the alpha one, so
    # without this field the figure cannot be reproduced from what it says.
    if not result.get("ladder_channel"):
        FAILURES.append(
            f"{label}: the orbital ladder does not say which spin channel it "
            f"plots; an unrestricted calculation has two with different "
            f"energies and the diagram shows one, so the figure cannot be "
            f"reproduced from what it says")

    # A//B.  B3LYP/6-31G* on a force-field conformer and on a DFT minimum are
    # different calculations with different gaps and dipoles, and a result
    # that does not say which one it is cannot be compared with anything.
    geom = result.get("geometry_source")
    if not geom:
        FAILURES.append(
            f"{label}: the result does not say what geometry the numbers were "
            f"computed on, so it cannot be written as A//B or reproduced")
    elif kind == "geometry_optimization" and "optimis" not in str(geom).lower():
        FAILURES.append(
            f"{label}: the job optimised the geometry but reports it as "
            f"{geom!r}; the source has to say the geometry is a DFT minimum "
            f"at this level, or the result reads as a force-field conformer")
    elif kind != "geometry_optimization" and "optimis" in str(geom).lower():
        FAILURES.append(
            f"{label}: the job never optimised but reports the geometry as "
            f"{geom!r}; that claims a minimum this calculation did not find")

    # population analysis must actually be produced, and each scheme must sum
    # to the molecular charge -- a partition that does not is meaningless.
    # Loewdin charges were empty for *every* job (PySCF's mulliken_pop takes no
    # 'xctype' argument, and the TypeError was swallowed), so the UI advertised
    # a scheme it never received.
    total_q = result.get("charge")
    for scheme in ("mulliken_charges", "lowdin_charges"):
        rows = result.get(scheme)
        if not isinstance(rows, list) or not rows:
            FAILURES.append(
                f"{label}.{scheme}: empty -- the population analysis did not run")
            continue
        if isinstance(total_q, int):
            s = sum(float(r.get("charge", 0.0)) for r in rows)
            if abs(s - total_q) > 1e-3:
                FAILURES.append(
                    f"{label}.{scheme}: sums to {s:+.4f}, expected {total_q:+d}")

    if require_rows:
        for name, keys in require_rows.items():
            rows = result.get(name) or []
            check_list_of(f"{label}.{name}", rows, keys)

    nbad = len([f for f in FAILURES if f.startswith(label)])
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] {label}  "
          f"({len(result)} result fields)", flush=True)
    return result


def uv_scale_sections(fail: list) -> None:
    """The absolute absorption scale, the shape of the curve, and the window.

    Split out of ``main`` so it can be run on its own: these checks need no
    server and no SCF, and being able to execute them directly is what makes
    the mutation tests for them cheap enough to actually run.
    """
    # Two defects, both invisible in the old figure because it was normalised
    # to its own maximum:
    #
    #   * the curve carried a |dE/dlambda| Jacobian.  That factor conserves the
    #     *area* of the plot, which is not what an absorbance spectrum is, and
    #     it varies by (lambda_max/lambda_min)^2 -- 19.75 across a 180-800 nm
    #     window.  Two bands with equal f and equal width were therefore drawn
    #     at 8.9:1.  The first assertion below is exactly that pair.
    #   * the Gaussians had amplitude f instead of unit area, so the ordinate
    #     was on no absolute scale at all.  Fixing it makes the area under the
    #     curve over wavenumber an exact multiple of sum(f), which is what the
    #     remaining assertions check.
    from backend.engine import analysis as _analysis
    # These two are written as literals ON PURPOSE.  Reading them back out of
    # the module would make the assertions below unfalsifiable: scaling the
    # constant in analysis.py would scale the "expected" value with it and the
    # check would still pass.  (That is not hypothetical -- the first version
    # of this section did exactly that, and a factor-of-ten mutation of
    # EPS_INTEGRAL_PER_F fired nothing.)
    #   2.3154e8 = 1/4.319e-9, from e^2/(4 eps0 me c) and
    #              eps = N_A sigma / (1000 ln 10)
    #   8065.5439 = HARTREE2CM / HARTREE2EV, 1 eV in cm^-1
    EPS_PER_F = 2.3154e8
    EV2CM = 8065.5439
    if abs(_analysis.EPS_INTEGRAL_PER_F / EPS_PER_F - 1.0) > 1e-9 \
            or abs(_analysis.EV2CM / EV2CM - 1.0) > 1e-6:
        fail.append(
            f"uvscale: the module now carries EPS_INTEGRAL_PER_F = "
            f"{_analysis.EPS_INTEGRAL_PER_F} and EV2CM = {_analysis.EV2CM}; "
            f"the derived values are {EPS_PER_F} and {EV2CM}. Changing either "
            f"rescales every absolute ordinate in the app.")
    nbad = 0
    EV2NM_ = 1239.8419843320026
    FWHM = 0.40
    sigma_ev = FWHM / 2.3548200450309493
    sigma_cm = sigma_ev * EV2CM
    eps_analytic = EPS_PER_F / (sigma_cm * math.sqrt(2 * math.pi))

    pair = [
        {"state": 1, "energy_ev": EV2NM_ / 250.0, "wavelength_nm": 250.0,
         "oscillator_strength": 1.0},
        {"state": 2, "energy_ev": EV2NM_ / 500.0, "wavelength_nm": 500.0,
         "oscillator_strength": 1.0},
    ]
    uv = _analysis.uv_curve(pair, fwhm_ev=FWHM)
    xs, ys = uv.get("x") or [], uv.get("y") or []
    if len(xs) != len(ys) or not xs:
        fail.append("uvscale: the synthetic pair produced no curve")
        nbad += 1
    else:
        def _at(target):
            i = min(range(len(xs)), key=lambda k: abs(xs[k] - target))
            return float(ys[i]), float(xs[i])
        h1, l1 = _at(250.0)
        h2, l2 = _at(500.0)
        if abs(l1 - 250.0) > 2.0 or abs(l2 - 500.0) > 2.0:
            fail.append(
                f"uvscale: the equal-f pair at 250/500 nm was sampled at "
                f"{l1:.1f}/{l2:.1f} nm")
            nbad += 1
        elif h1 / max(h2, 1e-9) < 0.8 or h1 / max(h2, 1e-9) > 1.25:
            fail.append(
                f"uvscale: two bands with equal f and equal width came out at "
                f"{h1:.2f} vs {h2:.2f} ({h1 / max(h2, 1e-9):.2f}:1) at 250 and "
                f"500 nm. Equal f must mean equal height; a factor that "
                f"depends on wavelength is a Jacobian, not a spectrum")
            nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] equal oscillator strength "
          f"means equal band height", flush=True)

    nbad = 0
    eps_max = uv.get("epsilon_max")
    if eps_max is None:
        fail.append(
            "uvscale: the curve has no absolute ordinate. A paper quotes "
            "eps_max in L mol-1 cm-1, and a curve normalised to its own "
            "maximum cannot supply it.")
        nbad += 1
    elif abs(float(eps_max) / eps_analytic - 1.0) > 0.01:
        fail.append(
            f"uvscale: a single band with f = 1 and FWHM {FWHM} eV peaks at "
            f"eps = {float(eps_max):.1f}, but the eps<->f conversion "
            f"(2.3154e8 = 1/4.319e-9) requires {eps_analytic:.1f}. A wrong "
            f"constant here is invisible in a normalised plot.")
        nbad += 1
    # the peaks list must carry the same number, and it must be the analytic
    # band maximum rather than whatever the grid happened to sample
    pk = (uv.get("peaks") or [{}])[0]
    if pk.get("epsilon_max_l_mol_cm") is None:
        fail.append("uvscale: the peak table has no eps_max column")
        nbad += 1
    elif abs(float(pk["epsilon_max_l_mol_cm"]) / eps_analytic - 1.0) > 1e-3:
        fail.append(
            f"uvscale: peak eps_max {pk['epsilon_max_l_mol_cm']} disagrees "
            f"with the analytic {eps_analytic:.1f}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] the absolute eps scale is the "
          f"derived one", flush=True)

    nbad = 0
    # The sum rule, on a band with a window wide enough that the band is whole.
    # `measured` must be the trapezoid of the epsilon array the payload ships
    # (so a stale or hand-set number cannot pass) and it must equal the
    # analytic integral of unit-area Gaussians.
    solo = _analysis.uv_curve(
        [{"state": 1, "energy_ev": EV2NM_ / 400.0, "wavelength_nm": 400.0,
          "oscillator_strength": 1.0}], fwhm_ev=FWHM,
        wmin=200.0, wmax=900.0, n_electrons=10)
    sr = solo.get("sum_rule") or {}
    eps_arr = np.asarray(solo.get("epsilon") or [], dtype=float)
    nu_arr = EV2NM_ / np.asarray(solo.get("x") or [], dtype=float) * EV2CM
    if eps_arr.size == 0 or eps_arr.size != nu_arr.size:
        fail.append("uvsumrule: no epsilon array to integrate")
        nbad += 1
    else:
        order = np.argsort(nu_arr)
        recomputed = float(np.trapezoid(eps_arr[order], nu_arr[order]))
        # 0.1%, not machine precision: the payload rounds the wavelength grid
        # to two decimals, so re-integrating it cannot reproduce the internal
        # value exactly -- but a stale or mis-scaled integral is off by far more
        if abs(recomputed / max(float(sr.get("measured_l_mol_cm2") or 0), 1e-30)
               - 1.0) > 1e-3:
            fail.append(
                f"uvsumrule: the reported integral "
                f"{sr.get('measured_l_mol_cm2')} is not the integral of the "
                f"reported curve ({recomputed})")
            nbad += 1
        if abs(float(sr.get("covered_fraction") or 0.0) - 1.0) > 1e-6:
            fail.append(
                f"uvsumrule: a whole band in a 200-900 nm window should be "
                f"fully covered, got {sr.get('covered_fraction')}")
            nbad += 1
        if abs(float(sr.get("expected_l_mol_cm2") or 0)
               / EPS_PER_F - 1.0) > 1e-6:
            fail.append(
                f"uvsumrule: for one unit of f the expected integral is "
                f"{sr.get('expected_l_mol_cm2')}, not "
                f"{EPS_PER_F}")
            nbad += 1
        if abs(float(sr.get("residual_pct", 99.0))) > 0.05:
            fail.append(
                f"uvsumrule: integral(eps dnu) is off by "
                f"{sr.get('residual_pct')}% from 2.3154e8 * sum(f). A missing "
                f"1/(sigma*sqrt(2pi)), a stray Jacobian or the wrong constant "
                f"all land here.")
            nbad += 1
    if sr.get("sum_f_over_n_electrons") is None:
        fail.append(
            "uvsumrule: the payload does not report sum(f)/N_electrons, so a "
            "reader cannot tell how much of the spectrum these roots cover")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] integral(eps dnu) equals "
          f"2.3154e8 * sum(f)", flush=True)

    nbad = 0
    # A fixed window cannot serve a small molecule: ethylene's first six roots
    # all lie below 146 nm, so a 180-800 nm default draws nothing but Gaussian
    # tails at 100%.  The window has to come from the states.
    deep = _analysis.uv_curve(
        [{"state": i + 1, "energy_ev": EV2NM_ / lam, "wavelength_nm": lam,
          "oscillator_strength": f_} for i, (lam, f_) in enumerate(
            [(120.0, 0.50), (140.0, 0.20), (300.0, 0.01)])],
        n_electrons=16)
    win = deep.get("window_nm") or []
    if len(win) != 2:
        fail.append("uvwindow: the curve does not report its window")
        nbad += 1
    else:
        for lam in (120.0, 140.0, 300.0):
            if not (win[0] <= lam <= win[1]):
                fail.append(
                    f"uvwindow: the auto window {win} excludes the {lam:.0f} nm "
                    f"band, so the figure cannot show it")
                nbad += 1
        if deep.get("window_source") != "auto":
            fail.append(
                f"uvwindow: the window came from '{deep.get('window_source')}' "
                f"rather than from the states")
            nbad += 1
        if deep.get("notes"):
            fail.append(
                f"uvwindow: an auto window that contains every band should "
                f"raise no warning, got {deep['notes']}")
            nbad += 1
    # and when the caller *does* pin a window that excludes a band, the payload
    # has to say so rather than quietly drawing a truncated spectrum
    pinned = _analysis.uv_curve(
        [{"state": 1, "energy_ev": EV2NM_ / 120.0, "wavelength_nm": 120.0,
          "oscillator_strength": 0.50}], wmin=180.0, wmax=800.0)
    if not pinned.get("notes"):
        fail.append(
            "uvwindow: a pinned 180-800 nm window excludes the only band "
            "(120 nm) but the payload raises no warning")
        nbad += 1
    elif pinned.get("strongest", {}).get("in_window") is not False:
        fail.append(
            "uvwindow: the payload does not flag the strongest transition as "
            "outside the plotted window")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] the plotted window contains "
          f"the bands it is meant to show", flush=True)


def raman_sections(fail: list) -> None:
    """The Raman activity convention, the depolarization ceiling, the naming
    rule that decides when a polarizability is worth reporting at all.

    Split out of ``main`` for the same reason as ``uv_scale_sections``: none of
    this needs a server or an SCF, so the mutation harness can run the real
    assertions directly instead of reimplementing them.
    """
    from backend.engine import analysis as _a
    from backend.engine.dft import basis_has_diffuse

    # ---- the activity formula, on tensors whose answer is arithmetic ----
    # diag(1,2,3): a_bar = 2, gamma^2 = 1/2[(1-2)^2+(2-3)^2+(3-1)^2] = 3,
    # so activity = 45*4 + 7*3 = 201 and rho = 9/(180+12) = 0.046875.
    # Written as literals on purpose: deriving them from raman_activity would
    # make the check unfalsifiable.
    nbad = 0
    iso, g2, act = _a.raman_activity(np.diag([1.0, 2.0, 3.0]))
    if abs(iso - 2.0) > 1e-12 or abs(g2 - 3.0) > 1e-12 or abs(act - 201.0) > 1e-9:
        fail.append(
            f"ramanact: diag(1,2,3) must give a_bar=2, gamma^2=3, "
            f"activity=201; got {iso}, {g2}, {act}")
        nbad += 1
    if abs(_a.depolarization_ratio(iso, g2) - 0.046875) > 1e-12:
        fail.append(
            f"ramanact: rho for diag(1,2,3) must be 9/192 = 0.046875; got "
            f"{_a.depolarization_ratio(iso, g2)}")
        nbad += 1

    # A purely off-diagonal derivative: a_bar = 0, gamma^2 = 3.  This is the
    # case that pins the ceiling -- rho = 3*3/(0 + 4*3) = 0.75 exactly, and the
    # activity is 7*3 = 21, not 45*something.
    off = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    iso2, g22, act2 = _a.raman_activity(off)
    if abs(iso2) > 1e-12 or abs(g22 - 3.0) > 1e-12 or abs(act2 - 21.0) > 1e-9:
        fail.append(
            f"ramanact: the purely off-diagonal tensor must give a_bar=0, "
            f"gamma^2=3, activity=21; got {iso2}, {g22}, {act2}")
        nbad += 1
    if abs(_a.depolarization_ratio(iso2, g22) - 0.75) > 1e-12:
        fail.append(
            f"ramanact: a mode with no isotropic derivative must sit exactly "
            f"at rho = 0.75; got {_a.depolarization_ratio(iso2, g22)}")
        nbad += 1

    # ---- the invariants have to be rotationally invariant ---------------
    # a_bar and gamma^2 are invariants of a rank-2 tensor, so rotating the
    # tensor must not move them.  A formula with a transposed index, a missing
    # factor of 6, or the wrong off-diagonal pair (xy, xz, yz instead of
    # xy, yz, zx) is not invariant under rotation and this catches it -- the
    # single-tensor checks above would not, because a diagonal tensor has no
    # off-diagonal terms to get wrong.
    rng = np.random.default_rng(20260914)
    worst = 0.0
    worst_rho = 0.0
    for _ in range(200):
        m = rng.normal(size=(3, 3))
        q, _r = np.linalg.qr(m)
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1.0
        t = rng.normal(size=(3, 3))
        t = 0.5 * (t + t.T)
        i0, g0, a0 = _a.raman_activity(t)
        i1, g1, a1 = _a.raman_activity(q @ t @ q.T)
        scale = max(abs(a0), 1e-12)
        worst = max(worst, abs(i1 - i0) / max(abs(i0), 1e-9),
                    abs(g1 - g0) / max(abs(g0), 1e-9), abs(a1 - a0) / scale)
        worst_rho = max(worst_rho, _a.depolarization_ratio(i1, g1))
    if worst > 1e-9:
        fail.append(
            f"ramaninv: rotating the tensor moved the Placzek invariants by "
            f"{worst:.3e} relative, but a_bar and gamma^2 are invariants")
        nbad += 1
    # Over 200 random tensors, one of which is essentially pure off-diagonal,
    # the ceiling must not be crossed by any of them.
    if worst_rho > 0.7500001:
        fail.append(
            f"ramaninv: rho reached {worst_rho} over random tensors, above "
            f"the 3/4 ceiling that the formula guarantees")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] the Raman activity and the "
          f"depolarization ceiling follow from the tensor", flush=True)

    # ---- the Frobenius identity, on random symmetric tensors ------------
    # sum_ij a_ij^2 = 3 a_bar^2 + (2/3) gamma^2 is what lets the sum rule (a
    # Frobenius statement) say anything about the activities.  Checked on
    # random tensors rather than on the one used above, because for a diagonal
    # tensor the off-diagonal half of gamma^2 never enters.
    nbad = 0
    worst = 0.0
    for _ in range(200):
        t = rng.normal(size=(3, 3))
        t = 0.5 * (t + t.T)
        i0, g0, _a0 = _a.raman_activity(t)
        frob = float(np.sum(t * t))
        worst = max(worst, abs(frob - (3.0 * i0 ** 2 + (2.0 / 3.0) * g0))
                    / max(frob, 1e-12))
    if worst > 1e-9:
        fail.append(
            f"ramanfrob: ||a||_F^2 and 3 a_bar^2 + (2/3) gamma^2 differ by "
            f"{worst:.3e} relative; they are the same number written two ways, "
            f"and the sum rule relies on that")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] the Frobenius norm and the "
          f"Placzek invariants agree", flush=True)

    # ---- the naming rule that decides whether alpha is worth reporting ---
    # The basis matters by a factor of two for the polarizability, so which
    # bases carry a warning is not cosmetic.  These are literals from the
    # published naming convention: aug- prefixes and + signs mean diffuse,
    # and nothing in the app's basis list is diffuse without saying so.
    nbad = 0
    expect = {
        "6-31g*": False, "6-31g": False, "def2-svp": False, "cc-pvdz": False,
        "def2-tzvp": False, "6-311g*": False,
        "6-31+g*": True, "6-31++g**": True, "aug-cc-pvdz": True,
        "aug-cc-pvtz": True,
    }
    wrong = {b: basis_has_diffuse(b) for b, want in expect.items()
             if basis_has_diffuse(b) != want}
    if wrong:
        fail.append(
            f"ramandiffuse: the diffuse-basis rule mislabels {wrong}; a "
            f"polarizability from a basis without diffuse functions is wrong "
            f"by tens of per cent and has to be flagged")
        nbad += 1
    # And the flag has to actually appear on a result computed without them,
    # or the warning is decoration.  This is the wiring, not the rule.
    from backend.engine.dft import BASIS_SETS

    for b in ("6-31+g*", "6-31++g**", "aug-cc-pvdz", "aug-cc-pvtz"):
        if b not in BASIS_SETS:
            fail.append(
                f"ramandiffuse: {b} is not offered by the app, so a Raman "
                f"figure cannot be computed at a basis that supports it")
            nbad += 1
    for b in ("6-31+g*", "6-31++g**", "aug-cc-pvdz", "aug-cc-pvtz"):
        if BASIS_SETS.get(b, {}).get("diffuse") is not True:
            fail.append(
                f"ramandiffuse: {b} is in the basis list but not marked "
                f"diffuse, so the UI cannot group it or hint at it")
            nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] the diffuse-basis rule and the "
          f"basis list agree", flush=True)


def nmr_sections(fail: list) -> None:
    """The NMR conventions, none of which can be guessed from the docs.

    Split out of ``main`` for the same reason as ``raman_sections``: none of it
    needs a server or an SCF, so the mutation harness runs the real assertions
    instead of a copy of them.
    """
    from backend.engine import nmr as _n

    # ---- the scale, against its closed form ------------------------------
    # sigma[ppm] = sigma[au] * alpha^2 * 1e6 with alpha = 1/137.035999084.
    # Written as a literal because deriving it from PPM would make the check
    # unfalsifiable; an earlier round had a sign flip bolted on here to paper
    # over a perturbation-convention error, and this is what would catch it.
    want = (1.0 / 137.035999084) ** 2 * 1e6
    if abs(_n.PPM - want) > 1e-9 or abs(_n.PPM - 53.251355) > 1e-5:
        fail.append(
            f"nmrscale: PPM must be alpha^2 * 1e6 = 53.251355; got {_n.PPM}")
    if _n.PPM < 0:
        fail.append(f"nmrscale: PPM is negative ({_n.PPM}) -- a sign flip")

    # ---- symmetrisation: trace-preserving, and it does remove the asymmetry
    rng = np.random.default_rng(20260914)
    for _ in range(50):
        m = rng.normal(size=(3, 3)) * 30.0
        sym, asym = _n.symmetrise(m[None, :, :])
        if abs(np.trace(sym[0]) - np.trace(m)) > 1e-9:
            fail.append(
                "nmrsym: symmetrising moved the trace, so the isotropic "
                "shielding and every chemical shift would move with it")
            break
        if np.abs(sym[0] - sym[0].T).max() > 1e-9:
            fail.append("nmrsym: the symmetrised tensor is still asymmetric")
            break
        # the returned residual must be the discarded part, in ppm
        got = float(asym[0])
        exp = float(np.abs(m - m.T).max() * _n.PPM)
        if abs(got - exp) > 1e-6 * max(1.0, exp):
            fail.append(
                f"nmrsym: the reported asymmetry {got} is not "
                f"max|M - M^T| * PPM = {exp}")
            break

    # ---- principal values and the Haeberlen invariants -------------------
    p = _n.principal_values(np.diag([3.0, 1.0, 2.0]))
    if not (abs(p[0] - 1.0) < 1e-12 and abs(p[1] - 2.0) < 1e-12
            and abs(p[2] - 3.0) < 1e-12):
        fail.append(f"nmrtensor: principal values must be ascending 1,2,3; "
                    f"got {list(p)}")
    inv = _n.tensor_invariants(np.diag([1.0, 2.0, 3.0]))
    if (abs(inv["sigma_iso_ppm"] - 2.0) > 1e-12
            or abs(inv["span_ppm"] - 2.0) > 1e-12
            or abs(inv["skew"]) > 1e-12
            or abs(inv["anisotropy_ppm"] - 1.5) > 1e-12):
        fail.append(
            "nmrtensor: diag(1,2,3) must give iso 2, span 2, skew 0, "
            f"d-sigma 1.5; got {inv}")
    # The extreme skew: sigma = diag(0,0,3) has iso 1, span 3 and
    # skew = 3(0-1)/3 = -1 exactly, which is the definition's lower bound.
    inv2 = _n.tensor_invariants(np.diag([0.0, 0.0, 3.0]))
    if abs(inv2["skew"] + 1.0) > 1e-12 or abs(inv2["anisotropy_ppm"] - 3.0) > 1e-12:
        fail.append(
            f"nmrtensor: diag(0,0,3) must give skew -1 and d-sigma 3; got {inv2}")

    # ---- grouping and the spectrum --------------------------------------
    entries = [
        {"index": 0, "symbol": "C", "isotope": "13C", "delta_ppm": 128.0,
         "sigma_iso_ppm": 70.0},
        {"index": 1, "symbol": "C", "isotope": "13C", "delta_ppm": 128.02,
         "sigma_iso_ppm": 70.0},
        {"index": 2, "symbol": "C", "isotope": "13C", "delta_ppm": 126.0,
         "sigma_iso_ppm": 72.0},
        {"index": 3, "symbol": "H", "isotope": "1H", "delta_ppm": 7.26,
         "sigma_iso_ppm": 25.0},
        {"index": 4, "symbol": "H", "isotope": "1H", "delta_ppm": 7.27,
         "sigma_iso_ppm": 25.0},
    ]
    g = _n.group_equivalent(entries, tol_ppm=0.05)
    counts = sorted(x["count"] for x in g)
    if counts != [1, 2, 2]:
        fail.append(
            f"nmrgroup: two carbons 0.02 ppm apart must merge and the third "
            f"must not; got counts {counts}")
    if [x["delta_ppm"] for x in g] != sorted(
            [x["delta_ppm"] for x in g], reverse=True):
        fail.append("nmrgroup: groups must come out high shift first")

    # ---- the grouping must not depend on the order the atoms are in ------
    # The first version compared each entry against the group's running mean,
    # in input order, so the same multiset of shifts could come out as (1,3)
    # or as (2,2) depending on how the file happened to be written.  Measured:
    # 18 of the 24 orderings of 0.00/0.03/0.06/0.09 gave (1,3).
    import itertools
    chain = [0.0, 0.03, 0.06, 0.09]
    seen_counts = set()
    worst_span = 0.0
    for perm in itertools.permutations(range(len(chain))):
        es = [{"index": i, "symbol": "C", "isotope": "13C",
               "delta_ppm": chain[i], "sigma_iso_ppm": 170.0 - chain[i]}
              for i in perm]
        gg = _n.group_equivalent(es, tol_ppm=0.05)
        seen_counts.add(tuple(sorted(x["count"] for x in gg)))
        for grp in gg:
            vals = [chain[i] for i in grp["indices"]]
            worst_span = max(worst_span, max(vals) - min(vals))
    if len(seen_counts) != 1:
        fail.append(
            "nmrgroup: the same four shifts give different groupings "
            f"depending on the atom order: {sorted(seen_counts)}. The "
            "integration a paper prints would depend on how the input file "
            "was written")
    # The anchor rule: never merge two nuclei further apart than the
    # tolerance.  Chaining against a running mean cannot promise this -- on
    # 0.00/0.03/0.06/0.09 it walks 0.09 -> 0.06 -> 0.03 and reports one group
    # spanning 0.06 ppm, which is wider than the tolerance it was given.
    if worst_span > 0.05 + 1e-12:
        fail.append(
            f"nmrgroup: a group spans {worst_span:.4f} ppm, more than the "
            "0.05 ppm tolerance, so nuclei that are not equivalent were "
            "merged by chaining")
    # the same, spelled out on one fixed ordering
    wide = [0.0, 0.04, 0.09]
    gw = _n.group_equivalent(
        [{"index": i, "symbol": "C", "isotope": "13C", "delta_ppm": s,
          "sigma_iso_ppm": 170.0} for i, s in enumerate(wide)], tol_ppm=0.05)
    for grp in gw:
        vals = [wide[i] for i in grp["indices"]]
        if max(vals) - min(vals) > 0.05 + 1e-12:
            fail.append(
                f"nmrgroup: group {grp['indices']} spans "
                f"{max(vals) - min(vals):.3f} ppm, more than the 0.05 ppm "
                "tolerance, so chaining merged nuclei that are not equivalent")
            break
    # and the fraction the figure's integral column reports
    if abs(sum(x["fraction"] for x in g) - 1.0) > 1e-9:
        fail.append(
            f"nmrgroup: the group fractions sum to "
            f"{sum(x['fraction'] for x in g)}, not 1")

    spec = _n.nmr_spectrum(g, linewidth_hz=1.0, spectrometer_mhz=400.0)
    xs = spec["shift_ppm"]
    if len(xs) < 100 or xs[0] >= xs[-1]:
        fail.append(f"nmrspectrum: the ppm grid must be ascending; got "
                    f"{len(xs)} points from {xs[0] if xs else None}")
    if abs(max(spec["intensity"]) - 1.0) > 1e-9:
        fail.append(
            f"nmrspectrum: the curve must be normalised to 1; peak is "
            f"{max(spec['intensity'])}")
    # A single group's Lorentzian peaks exactly at its shift.
    one = _n.nmr_spectrum([{"element": "H", "isotope": "1H", "delta_ppm": 3.0,
                            "count": 2, "sigma_iso_ppm": 30.0}],
                          linewidth_hz=0.4, spectrometer_mhz=400.0)
    ix = int(np.argmax(one["intensity"]))
    if abs(one["shift_ppm"][ix] - 3.0) > 0.02:
        fail.append(
            f"nmrspectrum: the peak sits at {one['shift_ppm'][ix]} ppm, not "
            "at the stick's 3.0")
    hwhm = 0.4 / 400.0

    def at(d):
        """The curve's value at exactly 3.0 + d ppm, linearly interpolated.

        The nearest-grid-point version this used to be cannot be right any
        more: the grid is now fine enough that the nearest point to 3.0+hwhm
        is up to half a step away, so a correct Lorentzian reads anywhere in
        0.45-0.55 there.  Interpolating tests the curve rather than the grid.
        """
        x = np.asarray(one["shift_ppm"], dtype=float)
        y = np.asarray(one["intensity"], dtype=float)
        return float(np.interp(3.0 + d, x, y))

    # The analytic Lorentzian: hwhm^2 / (d^2 + hwhm^2) = 0.5, 0.2, 0.1 at
    # d = 1, 2, 3 half-widths.  These are the shape, not a fit to it.
    for mult, want in ((1.0, 0.5), (2.0, 0.2), (3.0, 0.1)):
        got = at(mult * hwhm)
        if abs(got - want) > 0.01:
            fail.append(
                f"nmrspectrum: at {mult:g} half-width the Lorentzian must be "
                f"at {want}; got {got:.5f}")
            break
    if not at(3 * hwhm) < at(hwhm) < at(0.0):
        fail.append("nmrspectrum: the line is not falling off with distance")

    # ---- the grid has to resolve the line -------------------------------
    # This is the defect probe_nmr24/25 measured: with a fixed 2000-point grid
    # the step at a 12 ppm window is 2.79 half-widths, so each line is sampled
    # somewhere on its own flank and the normalisation turns that into a wrong
    # relative intensity.  Two lines equal by construction came out with a
    # height ratio between 1.01 and 1.72 depending only on where they fell.
    hwhm_ppm = 1.0 / 400.0
    for span, note in ((12.0, "a wide 1H window"), (2.0, "a narrow one")):
        sp = _n.nmr_spectrum(
            [{"element": "H", "isotope": "1H", "delta_ppm": 0.0, "count": 1,
              "sigma_iso_ppm": 30.0},
             {"element": "H", "isotope": "1H", "delta_ppm": span, "count": 1,
              "sigma_iso_ppm": 30.0}],
            linewidth_hz=1.0, spectrometer_mhz=400.0, wmin=0.0,
            wmax=max(span, 0.001))
        step = sp["grid_step_ppm"]
        if step > hwhm_ppm / 4.0:
            fail.append(
                f"nmrspectrum: {note} (span {span} ppm) is sampled every "
                f"{step:.5f} ppm, more than a quarter of the {hwhm_ppm:.5f} ppm "
                "half-width, so the drawn line is not the Lorentzian the "
                "caption names")
            break
        # two nuclei, one line each: the drawn peaks must be equal
        sx = np.asarray(sp["shift_ppm"], dtype=float)
        sy = np.asarray(sp["intensity"], dtype=float)
        peaks = [float(sy[i]) for i in range(1, len(sy) - 1)
                 if sy[i] > sy[i - 1] and sy[i] > sy[i + 1]]
        if len(peaks) != 2:
            fail.append(
                f"nmrspectrum: two signals {span} ppm apart produced "
                f"{len(peaks)} local maxima, so the figure does not show the "
                "two lines the table lists")
            break
        ratio = max(peaks) / max(1e-12, min(peaks))
        if abs(ratio - 1.0) > 0.02:
            fail.append(
                f"nmrspectrum: two equal signals {span} ppm apart are drawn "
                f"with a height ratio of {ratio:.4f}; both are one nucleus, so "
                "the ratio is a grid artefact, not a result")
            break

    # ---- the Larmor frequency is per nucleus ----------------------------
    # A spectrometer is named by its 1H frequency.  Using that number for a
    # 13C line makes it four times too narrow, and the linewidth is what sets
    # the grid, so it also makes the figure need the wrong number of points.
    lc = _n.larmor_mhz("C", 400.0)
    want_c = 400.0 * 10.7084 / 42.5775
    if abs(lc - want_c) > 1e-6:
        fail.append(
            f"nmrspectrum: 13C on a 400 MHz (1H) instrument must precess at "
            f"{want_c:.4f} MHz; got {lc:.4f}")
    ratio_ch = _n.larmor_mhz("C", 400.0) / _n.larmor_mhz("H", 400.0)
    if abs(ratio_ch - 10.7084 / 42.5775) > 1e-6:
        fail.append(
            f"nmrspectrum: the 13C/1H Larmor ratio is {ratio_ch:.6f}; it must "
            "be the gyromagnetic ratio 10.7084/42.5775, or the ppm width of a "
            "line would be wrong for one of them")
    # and the ppm width of the same 1 Hz line must follow it
    a_h = _n.nmr_spectrum([{"element": "H", "isotope": "1H", "delta_ppm": 3.0,
                            "count": 1, "sigma_iso_ppm": 30.0}],
                          linewidth_hz=1.0, spectrometer_mhz=400.0,
                          isotope="1H", larmor_mhz_=_n.larmor_mhz("H", 400.0))
    a_c = _n.nmr_spectrum([{"element": "C", "isotope": "13C", "delta_ppm": 30.0,
                            "count": 1, "sigma_iso_ppm": 170.0}],
                          linewidth_hz=1.0, spectrometer_mhz=400.0,
                          isotope="13C", larmor_mhz_=_n.larmor_mhz("C", 400.0))
    if not (a_c["linewidth_ppm"] > 3.5 * a_h["linewidth_ppm"]):
        fail.append(
            f"nmrspectrum: a 1 Hz line is {a_h['linewidth_ppm']:.5f} ppm wide "
            f"for 1H but {a_c['linewidth_ppm']:.5f} ppm for 13C; the 13C one "
            "must be about four times wider")

    # ---- the sticks carry the integral ----------------------------------
    for sp in (spec, one):
        st = sp.get("sticks") or []
        if not st:
            fail.append("nmrspectrum: no sticks, so the figure has no signals")
            break
        if not all("rel_intensity" in s for s in st):
            fail.append(
                "nmrspectrum: a stick carries no height, so the chart cannot "
                "draw the integral")
            break
        if abs(max(s["rel_intensity"] for s in st) - 1.0) > 1e-9:
            fail.append(
                "nmrspectrum: the tallest stick must be at 1.0 so the sticks "
                "and the normalised curve share an axis")
            break
    # two nuclei in one group: the stick is at the integral, twice a single one
    two = _n.nmr_spectrum(
        [{"element": "H", "isotope": "1H", "delta_ppm": 3.0, "count": 2,
          "sigma_iso_ppm": 30.0},
         {"element": "H", "isotope": "1H", "delta_ppm": 7.0, "count": 1,
          "sigma_iso_ppm": 25.0}],
        linewidth_hz=1.0, spectrometer_mhz=400.0)
    got = {round(s["delta_ppm"], 3): s["rel_intensity"]
           for s in two["sticks"]}
    if abs(got.get(3.0, 0.0) - 1.0) > 1e-9 or abs(got.get(7.0, 0.0) - 0.5) > 1e-9:
        fail.append(
            f"nmrspectrum: a 2H signal and a 1H signal must have sticks at "
            f"1.0 and 0.5; got {got}")

    # ---- the reference geometries, against their formulas ---------------
    # This is the check that would have caught the Si(CH)4 mistake: a TMS
    # built with one hydrogen per carbon has nao = 33 at STO-3G where
    # Si(CH3)4 needs 41, and its shielding diverged with basis size.
    tms = _n.reference_geometry("tms")
    if len(tms) != 17:
        fail.append(f"nmrref: TMS must be 17 atoms (Si + 4 C + 12 H); "
                    f"got {len(tms)}")
    from collections import Counter
    comp = Counter(s for s, *_ in tms)
    if comp != {"Si": 1, "C": 4, "H": 12}:
        fail.append(f"nmrref: TMS composition is {dict(comp)}, not "
                    "Si1 C4 H12")
    if _n.reference_formula("tms") not in ("CH12Si", "SiC4H12", "C4H12Si"):
        fail.append(f"nmrref: TMS formula came out "
                    f"{_n.reference_formula('tms')!r}")
    si = np.array(tms[0][1:])
    cs = [np.array(a[1:]) for a in tms if a[0] == "C"]
    for c in cs:
        if abs(np.linalg.norm(c - si) - 1.875) > 1e-9:
            fail.append(
                f"nmrref: Si-C is {np.linalg.norm(c - si)}, not 1.875 A")
            break
    # every carbon carries three hydrogens at 1.090 A
    for c in cs:
        hs = [np.array(a[1:]) for a in tms if a[0] == "H"]
        near = [h for h in hs if abs(np.linalg.norm(h - c) - 1.090) < 1e-6]
        if len(near) != 3:
            fail.append(
                f"nmrref: a carbon has {len(near)} hydrogens at 1.090 A, "
                "not 3")
            break
    # no atom may sit on top of another
    pts = [np.array(a[1:]) for a in tms]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < 0.5:
                fail.append(f"nmrref: TMS atoms {i} and {j} are on top of "
                            "each other")
                break

    # ---- the per-direction ERI builder must equal the reference expression
    # This is the regression test for the memory fix.  The naive form built
    # the full (3, nao^4) GIAO derivative and contracted it, which needed
    # about 11 GB at nao = 98 and thrashed a 15 GB machine; the fix builds one
    # direction at a time.  The two must be numerically identical, and this is
    # cheap because it runs on a two-atom molecule.
    from pyscf import gto as _gto
    mol = _gto.M(atom=[("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)],
                 basis="sto-3g", verbose=0)
    ops = _n.build_operators(mol)
    if "ig1" not in ops:
        fail.append("nmreri: build_operators no longer returns the raw ig1 "
                    "tensor, so the memory fix cannot be verified")
    else:
        if "O_G" in ops:
            fail.append(
                "nmreri: build_operators still forms the full (3, nao^4) GIAO "
                "ERI derivative -- that is the array that made the TMS "
                "reference thrash")
        b = np.array([3e-4, -1e-4, 5e-5])
        got = _n._giao_eri(ops, b)
        ig1 = ops["ig1"]
        ref = ops["eri0"] - 1j * np.einsum(
            "t,tuvkl->uvkl", b, ig1 + ig1.transpose(0, 3, 4, 1, 2))
        err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
        if err > 1e-12:
            fail.append(
                f"nmreri: the per-direction ERI builder differs from the "
                f"reference expression by {err:.3e} relative")
        zero = _n._giao_eri(ops, [0.0, 0.0, 0.0])
        if not np.allclose(zero, ops["eri0"]):
            fail.append("nmreri: at zero field the ERI must be the "
                        "unperturbed ones, unchanged and real")

    # ---- the memory guard has to be a real number, not a slogan ---------
    # Stated in the unit that was measured, not in the GB that falls out of it.
    # The old form of this check was `4000 < memory_estimate_mb(98) < 6500`,
    # which restates whatever constant the module happens to carry: it could
    # only ever catch a change, never a wrong number, and 56 bytes per nao^4
    # sat inside it happily for rounds while being 23% short of the truth.
    # probe_mem1 sampled the peak resident set of ``shielding_tensor`` alone
    # for nao = 18, 20, 32, 54 and got a slope of 68.8 bytes per nao^4 with an
    # intercept of +0.2 MB (R^2 = 0.99997), so that slope is what is asserted
    # here, with a band wide enough for a different machine's allocator but
    # far too narrow to admit the 56 that array-counting alone suggests.
    need = _n.memory_estimate_mb(98)
    per_nao4 = need * 1e6 / 98 ** 4
    if not (60.0 <= per_nao4 <= 80.0):
        fail.append(
            f"nmrmem: the estimate implies {per_nao4:.1f} bytes per nao^4; "
            f"the peak was measured at 68.8 (probe_mem1, nao 18..54, "
            f"R^2 0.99997) and anything outside 60-80 is a guess a memory "
            f"guard cannot afford -- too low and it under-refuses, too high "
            f"and it refuses work the machine can do")
    try:
        _n.check_memory(200, "6-31g*")
    except Exception as exc:
        if "budget" not in str(exc) and "GB" not in str(exc):
            fail.append(f"nmrmem: the refusal must carry the number; got {exc}")
    else:
        fail.append(
            "nmrmem: 200 basis functions must be refused, not attempted -- "
            "that is 90 GB of O(nao^4) arrays")

    # Every reference has to fit the *budget*, because the budget is the one
    # refusal that free memory cannot argue with.  Raising the constant from
    # the counted 56 to the measured 69 moved TMS from 5165 MB to 6364 MB and
    # silently across the old 6000 MB budget -- so on a cold cache, which is
    # what a fresh install has, every 1H, 13C and 29Si shift would have become
    # unavailable on any machine, and no gate would have said so: they all run
    # with the cache warm and reference_shieldings reads the cache before it
    # reaches check_memory.  This is the check that closes that door.  It asks
    # the budget question only, not the free-memory one, so a busy machine
    # cannot turn it red.
    for key in sorted(_n.REFERENCES):
        try:
            atoms = _n.reference_geometry(key)
            mol = _n._build_mol(_n._xyz(atoms), "6-31g*", 0, 1)
            nao = int(mol.nao_nr())
        except Exception as exc:                         # noqa: BLE001
            fail.append(f"nmrmem: the {key} reference could not be sized: "
                        f"{exc}")
            continue
        need = _n.memory_estimate_mb(nao)
        if need > _n.MEMORY_BUDGET_MB:
            fail.append(
                f"nmrmem: the {key} reference is {nao} basis functions, "
                f"estimated at {need / 1024:.1f} GB, and the budget is "
                f"{_n.MEMORY_BUDGET_MB / 1024:.1f} GB -- so with a cold cache "
                f"every shift quoted against it is unavailable on every "
                f"machine, whatever the RAM")

    # ---- the nucleus table and the reference map ------------------------
    for sym, info in _n.NUCLEI.items():
        for k in ("isotope", "spin", "abundance", "gamma"):
            if k not in info:
                fail.append(f"nmrnuclei: {sym} has no '{k}'")
        spin = info.get("spin", 0)
        if spin <= 0 or abs(spin * 2 - round(spin * 2)) > 1e-9:
            fail.append(f"nmrnuclei: {sym} spin {spin} is not a positive "
                        "half-integer")
        mass = "".join(c for c in info.get("isotope", "") if c.isdigit())
        if not mass:
            fail.append(f"nmrnuclei: {sym} isotope {info.get('isotope')!r} "
                        "carries no mass number")
    for sym, key in _n.REFERENCE_FOR.items():
        if sym not in _n.NUCLEI:
            fail.append(f"nmrref: {sym} is referenced but is not an NMR "
                        "nucleus in the table")
        if key not in _n.REFERENCES:
            fail.append(f"nmrref: {sym} points at unknown reference {key!r}")
        # This one belongs here, over the nuclei that *have* a reference.  It
        # spent a revision inside the reasons loop below, where it fired on
        # every element that has no reference -- which is the exact
        # opposite of what it asks.  The baseline caught it immediately, which
        # is the only reason it is worth writing down: an assertion in the
        # wrong loop is not a weaker assertion, it is a different one.
        if sym not in _n.IUPAC_PRIMARY:
            fail.append(
                f"nmrref: {sym} is referenced but has no IUPAC primary "
                "reference recorded, so nothing says whether the scale this "
                "build prints is the one the literature uses")
    # Every NMR-active element without a reference has to say why, and the set
    # has to be exactly that -- no more, no fewer.  The list used to be just a
    # list -- the count is not written down here, because it has rotted twice
    # already (77Se left this table when it acquired a reference) and the list
    # itself is the count.  A list with no reasons is how a gap
    # becomes folklore: nobody can tell an omission that was decided from one
    # that was forgotten.  Tying the two sets together means adding a
    # reference forces its reason to be deleted, and deleting a reason forces
    # the reference to be added -- either way the two cannot drift apart.
    active = {sym for sym, info in _n.NUCLEI.items()
              if info.get("spin", 0) > 0}
    no_ref = {sym for sym in active if sym not in _n.REFERENCE_FOR}
    why = set(_n.NO_REFERENCE_WHY)
    if no_ref != why:
        fail.append(
            f"nmrref: the unreferenced nuclei are {sorted(no_ref)} but the "
            f"reasons cover {sorted(why)}; "
            f"missing reasons {sorted(no_ref - why)}, "
            f"stale reasons {sorted(why - no_ref)}")
    for sym, reason in sorted(_n.NO_REFERENCE_WHY.items()):
        if not str(reason).strip():
            fail.append(f"nmrref: {sym} has no reason recorded for having no "
                        "reference")

    # A nucleus the NMR module knows about but the molecule builder cannot
    # construct is a different problem from one that merely has no reference
    # compound, and the two were conflated for deuterium.  NO_REFERENCE_WHY
    # said "no reference is provided yet; TMS-d12 would be computable here" --
    # a to-do note about a compound, for a nucleus no job can contain, because
    # the element table has no deuterium at all.
    #
    # Asserted here rather than described, so the two modules cannot drift:
    # the set of unreachable nuclei is exactly what is recorded as such, each
    # record carries a reason, each is named to the user, and the user-facing
    # reason and the structural one are the same string rather than two
    # stories about the same nucleus.
    from backend.engine import elements as _elements

    buildable = set()
    for sym in _n.NUCLEI:
        try:
            _elements.get(sym)
            buildable.add(sym)
        except Exception:
            pass
    unreachable = set(_n.NUCLEI) - buildable
    recorded = set(_n.UNBUILDABLE_NUCLEI)
    if unreachable != recorded:
        fail.append(
            f"nmrref: the NMR module knows {sorted(unreachable)} as NMR-active "
            f"but the molecule builder cannot construct it, and the record of "
            f"that is {sorted(recorded)}; unrecorded {sorted(unreachable - recorded)}, "
            f"stale {sorted(recorded - unreachable)}.  A nucleus that can never "
            "reach a calculation looks identical to one that simply has no "
            "reference compound, and they need different work")
    for sym, why in sorted(_n.UNBUILDABLE_NUCLEI.items()):
        if not str(why).strip():
            fail.append(f"nmrref: {sym} is recorded as unbuildable with no "
                        "reason, which is indistinguishable from an oversight")
        if sym not in _n.NO_REFERENCE_WHY:
            fail.append(
                f"nmrref: {sym} cannot be built but is not in "
                "NO_REFERENCE_WHY, so a user asking about it is told nothing")
        elif _n.NO_REFERENCE_WHY[sym] != why:
            fail.append(
                f"nmrref: {sym} is unbuildable and the reason recorded for it "
                "differs between UNBUILDABLE_NUCLEI and NO_REFERENCE_WHY, so "
                "the user-facing story and the structural one can drift apart")
    # Every nucleus quoted against something other than the IUPAC primary
    # reference has to say so, and the ones that are on the primary standard
    # have to say nothing -- a note on every nucleus is a note nobody reads.
    # The counts are deliberately not written down here.  This comment said
    # "five" and "seven" until 77Se acquired a reference, which made it six
    # and eight; a number in prose cannot be wrong, so it cannot be trusted.
    for sym in sorted(_n.IUPAC_PRIMARY):
        note = _n.scale_note(sym)
        on_primary = (_n.REFERENCE_FOR.get(sym)
                      == _n.IUPAC_PRIMARY[sym]["key"])
        if on_primary and note is not None:
            fail.append(f"nmrrefscale: {sym} is on the IUPAC primary "
                        f"reference yet carries a note ({note[:60]})")
        if not on_primary:
            if note is None:
                fail.append(
                    f"nmrrefscale: {sym} is quoted against "
                    f"{_n.REFERENCE_FOR.get(sym)!r} instead of the IUPAC "
                    f"primary reference {_n.IUPAC_PRIMARY[sym]['compound']!r} "
                    "and says nothing about it, so a user would read the "
                    "difference between the two scales as an error in the "
                    "calculation")
            else:
                want = _n.IUPAC_PRIMARY[sym]["compound"].split(" (")[0]
                if want not in note:
                    fail.append(
                        f"nmrrefscale: {sym}'s note does not name the IUPAC "
                        f"primary reference {want!r}: {note[:80]}")
    # The two that are off it, and the two numbers that have to be right in
    # them.  380.2 ppm is the ammonia/nitromethane conversion; PH3 has no
    # published offset in this build and has to say so rather than imply zero.
    n_note = _n.scale_note("N") or ""
    if "380.2" not in n_note:
        fail.append(f"nmrrefscale: the 15N note must carry the 380.2 ppm "
                    f"ammonia-to-nitromethane conversion; got {n_note[:90]!r}")
    p_note = _n.scale_note("P") or ""
    if "does not compute" not in p_note:
        fail.append(f"nmrrefscale: the 31P note must admit the offset is not "
                    f"computed here; got {p_note[:90]!r}")
    for key, info in _n.REFERENCES.items():
        try:
            atoms = _n.reference_geometry(key)
        except Exception as exc:
            fail.append(f"nmrref: {key} has no geometry: {exc}")
            continue
        n_elec = sum(_n._Z(s) for s, *_ in atoms)
        if n_elec % 2:
            fail.append(
                f"nmrref: {key} ({info['label']}) has {n_elec} electrons -- "
                "an odd count cannot be run with RHF, and the shielding is "
                "RHF-only")
        if not info.get("geometry"):
            fail.append(f"nmrref: {key} does not say where its geometry came "
                        "from, so the number cannot be reproduced")
    # deuterium has to count as one electron, not raise
    if _n._Z("D") != 1:
        fail.append(f"nmrref: _Z('D') must be 1; got {_n._Z('D')}")

    # ---- the reference is computed at the geometry the product ships -----
    # 17O shielding moves by 539 ppm per angstrom of O-H (probe_nmr28, measured
    # on the three water geometries below), so a reference computed at a
    # geometry the molecule does not have is not a rounding error.  The
    # module carried its own experimental water while the molecule library
    # ships a different one -- 0.9572 A against 0.96857 A, 104.52 deg against
    # 104.00 -- which is 6.11 ppm on every 17O shift in the output, and it
    # made water's own 17O shift 6.11 ppm instead of 0.00.
    #
    # The rule is asserted directly rather than through the shielding: where
    # the library has the compound, the reference geometry has to BE the
    # library geometry, atom for atom.  That is what makes the reference and
    # the molecule the same object, and it is cheap enough to check exactly.
    for key in _n.REFERENCES:
        try:
            atoms = _n.reference_geometry(key)
            src = _n.reference_geometry_source(key)
            libname = _n.reference_library_name(key)
            note = _n.reference_geometry_note(key)
        except Exception as exc:                         # noqa: BLE001
            fail.append(f"nmrrefgeom: {key} could not be resolved: {exc}")
            continue
        if src not in ("library", "builtin"):
            fail.append(f"nmrrefgeom: {key} reports source {src!r}")
        if src == "library":
            if not libname:
                fail.append(f"nmrrefgeom: {key} claims the library but names "
                            "no entry")
            else:
                try:
                    from backend.engine.molecule import resolve
                    got = resolve(libname, kind="name")
                    mine = sorted(
                        (a[0], round(a[1], 6), round(a[2], 6), round(a[3], 6))
                        for a in atoms)
                    theirs = sorted(
                        (a.symbol, round(a.x, 6), round(a.y, 6),
                         round(a.z, 6)) for a in got.atoms)
                    if mine != theirs:
                        fail.append(
                            f"nmrrefgeom: {key} is computed at coordinates "
                            f"that are not the library's '{libname}'.  The "
                            "reference and the molecule a user asks for are "
                            "then different geometries, and 17O is 539 ppm "
                            "per angstrom of O-H")
                except Exception as exc:                 # noqa: BLE001
                    fail.append(
                        f"nmrrefgeom: could not resolve the library entry "
                        f"'{libname}' for {key}: {exc}")
        elif libname is not None:
            fail.append(
                f"nmrrefgeom: {key} reports 'builtin' although the library has "
                f"'{libname}' with the same elements, so the reference is not "
                "the molecule the user gets")
        # the note has to be DERIVED from the coordinates.  A typed string
        # goes on describing the old geometry after the source changes, which
        # is how the library's own water entry came to claim "~104.5 deg" for
        # a geometry that is 104.00.
        want = "the molecule library" if src == "library" \
            else "experimental geometry"
        if want not in note:
            fail.append(
                f"nmrrefgeom: {key}'s note does not say where its geometry "
                f"came from: {note!r}")
        shortest = min(
            (math.dist(a[1:], b[1:]), a[0], b[0])
            for i, a in enumerate(atoms) for b in atoms[i + 1:])
        if f"{shortest[0]:.4f}" not in note:
            fail.append(
                f"nmrrefgeom: {key}'s note does not carry the shortest bond "
                f"length it actually has ({shortest[1]}-{shortest[2]} "
                f"{shortest[0]:.4f} A): {note!r}.  A note that is typed rather "
                "than derived describes whatever the author remembered.")

    # the 19F case is the one where the obvious name is the wrong molecule:
    # the library's chloroform is CHCl3, the reference is CFCl3, and a lookup
    # by name would have put every 19F shift on the CHCl3 scale.
    if _n.reference_geometry_source("chloroform_f") != "builtin":
        fail.append(
            "nmrrefgeom: the 19F reference picked up a library geometry; the "
            "library's chloroform is CHCl3 and the reference is CFCl3")
    _cf = sorted(a[0] for a in _n.reference_geometry("chloroform_f"))
    if _cf != ["C", "Cl", "Cl", "Cl", "F"]:
        fail.append(f"nmrrefgeom: the 19F reference is {_cf}, not CFCl3")

    # ---- the cache key has to depend on the geometry ---------------------
    # The key carried the basis, the tolerance and the field step but not the
    # geometry, so changing a reference geometry would have reused the old
    # shieldings silently -- the same failure as a cache that never writes,
    # one layer down.
    k_a = _n._cache_key("water", "6-31g*", _n.CONV_TOL, _n.FIELD_STEP, "aaaa")
    k_b = _n._cache_key("water", "6-31g*", _n.CONV_TOL, _n.FIELD_STEP, "bbbb")
    k_0 = _n._cache_key("water", "6-31g*", _n.CONV_TOL, _n.FIELD_STEP)
    if k_a == k_b:
        fail.append(
            "nmrcachekey: two different reference geometries produce the same "
            "cache key, so a changed geometry would read the old shieldings")
    if k_a == k_0 or k_b == k_0:
        fail.append(
            "nmrcachekey: the geometry fingerprint does not change the key, so "
            "the geometry is not part of it")
    _w = _n.reference_geometry("water")
    _moved = [("O", 0.0, 0.0, 0.0),
              ("H", 0.0, 0.76, 0.586), ("H", 0.0, -0.76, 0.586)]
    if _n._geometry_signature(_w) == _n._geometry_signature(_moved):
        fail.append(
            "nmrcachekey: the geometry fingerprint ignores the coordinates")

    # ---- the reference cache --------------------------------------------
    # This whole section exists because the cache had NEVER worked.  Five
    # files in data/nmr_cache, every one of them truncated at the same byte
    # ("nao": ) and invalid JSON, because mol.nao_nr() is a numpy.int64 that
    # json.dump cannot serialise -- and the write was wrapped in
    # `except Exception: pass`, so nothing ever said so.  The cost was the
    # full reference on every job: TMS measured at 4829 s.
    import json as _json
    import os as _os
    import tempfile
    try:
        _json.dumps({"nao": np.int64(18)})
        fail.append(
            "nmrcache: json.dumps serialised a numpy.int64 without help, so "
            "the cast that fixes the cache is no longer load-bearing and this "
            "section proves nothing")
    except TypeError:
        pass                                    # the reason the cast is needed
    # the helper has to make it serialisable, and round-trip unchanged.
    # Wrapped, because a contract section that raises instead of recording a
    # failure is a gate that reports nothing -- which is the failure mode this
    # whole round is about.
    payload = {"format": _n.CACHE_FORMAT, "key": "k", "label": "L",
               "basis": "6-31g*", "conv_tol": 1e-13, "field_step_au": 3e-4,
               "natm": 3, "nao": np.int64(18), "geometry": "g",
               "geometry_source": "builtin", "geometry_signature": "0" * 12,
               "method": _n._method_signature(),
               "element_shielding_ppm": {"O": {"sigma_iso_ppm": 329.6,
                                               "n_equivalent": np.int64(1)}}}
    # The fixture above lists its fields by hand and the reader does not, so
    # the two drift apart the moment a field is added to _CACHE_REQUIRED --
    # and the drift does not look like drift.  It looked like this: the
    # fixture wrote a payload the reader is *supposed* to reject, the
    # round-trip below failed, and the failure it printed was "a payload
    # written by _write_cache_atomic is not readable by _read_cache, so the
    # cache can never hit" -- a claim about the product, produced by a stale
    # fixture, and false.  That message is what sent this round looking for a
    # reader bug that did not exist.
    #
    # So the fixture's completeness is asserted here, first.  Adding a field
    # to _CACHE_REQUIRED now fails with a sentence about the fixture.
    _absent = [f for f in _n._CACHE_REQUIRED if f not in payload]
    if _absent:
        fail.append(
            f"nmrcache: the round-trip fixture does not carry {_absent}, "
            "which _CACHE_REQUIRED demands.  Every assertion below it is then "
            "testing a shape the reader no longer accepts, and the first "
            "failure it reports blames the product")
    try:
        blob = _json.dumps(payload, default=_n._json_default)
        if _json.loads(blob)["nao"] != 18:
            fail.append(
                "nmrcache: _json_default did not turn numpy.int64 into int")
    except TypeError as exc:
        fail.append(
            f"nmrcache: a payload containing numpy scalars cannot be "
            f"serialised at all ({exc}), which is the bug that made every "
            "reference file truncated at `\"nao\": `")
        blob = None
    with tempfile.TemporaryDirectory() as td:
        def put(name, obj):
            """Write obj as JSON, returning False (and recording why) if the
            serialiser refuses it.  Every one of these is a mutation target,
            and a contract section that raises reports nothing at all."""
            try:
                with open(_os.path.join(td, name), "w",
                          encoding="utf-8") as fh:
                    _json.dump(obj, fh, default=_n._json_default)
                return True
            except TypeError as exc:
                fail.append(
                    f"nmrcache: could not serialise a {name} entry ({exc})")
                return False

        p = _os.path.join(td, "entry.json")
        err = _n._write_cache_atomic(p, payload)
        if err is not None:
            fail.append(f"nmrcache: the atomic writer failed on a good "
                        f"payload: {err}")
        else:
            back = _n._read_cache(p)
            if back is None or back["nao"] != 18:
                fail.append(
                    "nmrcache: a payload written by _write_cache_atomic is "
                    "not readable by _read_cache, so the cache can never hit")
            # no temporary left behind
            leftovers = [f for f in _os.listdir(td) if f != "entry.json"]
            if leftovers:
                fail.append(
                    f"nmrcache: the atomic writer left {leftovers} behind")
        # a truncated file must be rejected, not half-read
        if blob is not None:
            bad = _os.path.join(td, "trunc.json")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write(blob[:len(blob) // 2])
            if _n._read_cache(bad) is not None:
                fail.append(
                    "nmrcache: a truncated entry was accepted; this is exactly "
                    "the state all five shipped cache files were in")
        # an entry from an older layout must be rejected too
        d = dict(payload)
        d["format"] = 1
        if put("old.json", d) and _n._read_cache(
                _os.path.join(td, "old.json")) is not None:
            fail.append(
                "nmrcache: an entry from format 1 was accepted by a reader "
                f"that expects format {_n.CACHE_FORMAT}")
        # and one missing a field the caller needs
        d = dict(payload)
        del d["element_shielding_ppm"]
        if put("short.json", d) and _n._read_cache(
                _os.path.join(td, "short.json")) is not None:
            fail.append(
                "nmrcache: an entry with no element_shielding_ppm was "
                "accepted, so a job would report shifts of None")
        # an entry that does not say which geometry it was computed at is an
        # anonymous number.  Every 17O shift in the output is a difference
        # against this value, and 17O moves 539 ppm/A, so a shielding read
        # without its geometry is a shielding that cannot be checked.
        for field in ("geometry", "geometry_source", "geometry_signature"):
            d = dict(payload)
            del d[field]
            name = f"nogeom_{field}.json"
            if put(name, d) and _n._read_cache(
                    _os.path.join(td, name)) is not None:
                fail.append(
                    f"nmrcache: an entry with no {field} was accepted, so a "
                    "cached shielding can be read without knowing which "
                    "geometry produced it")

        # an entry stamped with a method that is not the current one must be
        # rejected even when it is otherwise complete and well-formed.  This
        # is the same rule check_shift_scales proves against the real files;
        # it is asserted here too because this section costs no SCF and runs
        # every time, while that one is the slowest gate in the suite.
        d = dict(payload)
        d["method"] = "0000000000000000"
        if put("foreign_method.json", d) and _n._read_cache(
                _os.path.join(td, "foreign_method.json")) is not None:
            fail.append(
                "nmrcache: an entry computed by a different method was "
                "accepted, so a change to the shielding code silently "
                "re-uses the old numbers as a constant offset")

    # Every cache file on disk, split by whether the current key can reach it.
    #
    # This used to assert that every .json in the directory was readable,
    # which contradicted the other half of the design: a stale entry is
    # supposed to exist and be *refused*.  The assertion could only be
    # satisfied by deleting files, and deleting them would have removed the
    # evidence that the method guard works.  So it is two claims now, and the
    # second one is the stronger of the two -- a stale entry that becomes
    # readable again means the guard has stopped working, which is exactly
    # the defect this cache key was added to close.
    inv = _n.cache_inventory()
    cache_dir = _n._cache_dir()
    unreadable_live = [p for p in inv["live"]
                       if _n._read_cache(p) is None]
    if unreadable_live:
        fail.append(
            f"nmrcache: {len(unreadable_live)} cache entr"
            f"{'y' if len(unreadable_live) == 1 else 'ies'} at a path the "
            f"current key produces cannot be read: {unreadable_live}. "
            "Regenerate with `python -m backend.gen_nmr_cache`")
    readable_stale = [p for p in inv["stale"]
                      if _n._read_cache(p) is not None]
    if readable_stale:
        fail.append(
            f"nmrcache: {len(readable_stale)} entr"
            f"{'y' if len(readable_stale) == 1 else 'ies'} that the current "
            f"key cannot reach was read back anyway: {readable_stale}. The "
            "reader is not checking what the key says it is checking")
    n_extra = [n for n in sorted(_os.listdir(cache_dir))
               if not n.endswith(".json")]
    if n_extra:
        fail.append(f"nmrcache: {n_extra} in the cache directory is not a "
                    ".json entry")

    # ---- the level of theory is stated, not implied ---------------------
    src = open(__file__.replace("contract.py", "engine/nmr.py"),
               encoding="utf-8").read()
    if "Hartree-Fock" not in src:
        fail.append(
            "nmrlevel: engine/nmr.py never states that the shielding is "
            "computed at Hartree-Fock, which is a measured hard limit of "
            "this build rather than a choice")


def _reason_shape(tag: str, el: str, entry: dict, fail: list) -> None:
    """The shape a reason sentence has to have, for either table.

    Written once because both tables hold the same kind of claim -- "here is
    how well this method reproduces this nucleus, and here is what the verdict
    rests on" -- and they did *not* have the same amount of checking.  Every
    assertion below was applied to ``SCALE_UNMEASURED`` and none of it to
    ``SCALE_CHECK``, so the three measured entries that have something to say
    were checked for "the rendered text carries the number" and not for "the
    number is not also typed into the source".

    A mutation is how that was found.  Typing ``-101`` back into the selenium
    sentence -- replacing its ``{computed:.0f}`` placeholder with the literal
    the placeholder would have produced -- changed nothing: zero failures.  A
    check that cannot object to the defect it exists for is not a check.

    Three claims, and each of them is about something a reader would otherwise
    have to take on trust:

    * the sentence must not restate a number the table already holds, because
      a number written twice is a number that can disagree with itself;
    * ``reason_kind`` must say whether the sentence is a measurement or an
      argument, because "this cites a number" and "this is a decision" are
      different claims and an unset field is neither;
    * ``why_quotes`` must name the fields the sentence quotes, and each named
      field must actually have a placeholder -- otherwise a placeholder can be
      deleted and the claim silently becomes a sentence with no number in it.
    """
    why = str(entry.get("why") or "")
    if not why:
        return
    computed = entry.get("computed_ppm")
    if computed is None:
        return
    val = float(computed)
    rng = float(entry.get("range_ppm", 0.0))

    # The literal must not be in the *prose*.  Placeholders are stripped first,
    # because a format spec contains digits of its own: "{computed:.0f}" holds
    # a "0", and a naive substring test would call that a typed-in number.
    prose = _re.sub(r"\{[^}]*\}", "", why)
    pairs = [("computed_ppm", val)]
    if rng > 0.0:
        # Only when there is a range to quote.  format(0.0, ".0f") is "0", and
        # "0" is inside "3000" -- so an absent range would make this fire on
        # the word "77Se" and on every other string containing a zero.
        pairs.append(("range_ppm", rng))
    if entry.get("experimental_ppm") is not None:
        pairs.append(("experimental_ppm", float(entry["experimental_ppm"])))
    for field, value in pairs:
        # A set, because two of the four specs produce the same string for a
        # negative number -- format(-101.38, ".0f") and format(-101.38, "+.0f")
        # are both "-101" -- and reporting the same defect twice reads as two
        # defects.
        for literal in sorted({format(value, s)
                               for s in (".0f", ".1f", "+.0f", "+.1f")}):
            if literal in prose:
                fail.append(
                    f"{tag} restates {field} as the literal {literal!r} inside "
                    "its own reason; the sentence has to interpolate the value "
                    "instead, or the table and the prose can drift apart "
                    "without either being wrong on its own")

    kind = entry.get("reason_kind")
    quotes = entry.get("why_quotes")
    if kind not in ("measurement", "argument"):
        fail.append(f"{tag} declares its reason as {kind!r}; it has to be "
                    "'measurement' or 'argument', because 'this reason "
                    "quotes a number' and 'this reason does not' are "
                    "different claims and an unset field is neither")
    if quotes is None:
        fail.append(f"{tag} does not say which of its numbers its reason "
                    "quotes, so an omission and a decision are the same entry")
        quotes = ()
    if kind == "measurement" and "computed" not in quotes:
        fail.append(
            f"{tag} calls its reason a measurement but does not quote its "
            "computed value, so the text a user reads states a verdict "
            "without the number the verdict was reached from")
    if kind == "argument" and quotes:
        fail.append(
            f"{tag} calls its reason an argument yet quotes {list(quotes)} -- "
            "a reason that cites a number is a measurement, and the two have "
            "different work behind them")
    allowed = ("computed", "range", "experimental", "geometry_spread",
               "conformer_spread")
    for name in quotes:
        if name not in allowed:
            fail.append(f"{tag} says its reason quotes {name!r}, which is not "
                        f"one of its fields {list(allowed)}")
        elif "{" + name not in why:
            fail.append(
                f"{tag} says its reason quotes {name} but the text has no "
                f"{{{name}}} placeholder, so the number a user needs to read "
                "the claim is not in it")


def nmr_scale_sections(fail: list) -> None:
    """The accuracy ledger: does the method reproduce the scale it prints?

    A reference compound makes a shift *definable*.  It does not make it
    *correct*.  Until this round the only accuracy statement in the build was
    the twelve 1H/13C shifts in the module docstring of engine/nmr.py --
    attributed to probe_nmr20, recomputed by nothing, and silent about 14N,
    17O, 19F, 29Si, 31P and 77Se even though the product prints shifts for all
    of them.

    Measured, the ledger reads: 1H and 13C reproduced, 14N 24% high, 19F 22%
    low, and 77Se not reproduced at all.  ``SCALE_CHECK`` records each
    measurement and ``backend.check_shift_scales`` recomputes every one of
    them.  What is asserted *here* is what a recomputation cannot establish:

      * the ledger is complete -- every referenced nucleus is either measured
        or recorded as unmeasured, with the two sets exactly partitioning
        REFERENCE_FOR, so adding a reference forces one of them to be filled
        in rather than leaving a nucleus nobody has ever checked;
      * the arithmetic is self-consistent -- a ``usable`` entry is inside its
        own tolerance and an unusable one is not, so falsifying a recorded
        number without re-deciding the verdict fails;
      * and each tolerance is *falsifiable*: within a small multiple of the
        error it was measured at, so it is a claim something can fail rather
        than a band wide enough to be vacuous;
      * the verdict is tied to the reference it is about, so a ledger entry
        cannot describe a different compound from the one the product uses;
      * and the note a user reads carries the numbers the verdict was reached
        from, so "the 19F scale is 22% low" is never a claim without a
        measurement behind it.
    """
    from backend.engine import nmr as _n

    check = _n.SCALE_CHECK
    unmeasured = _n.SCALE_UNMEASURED
    covered = set(check) | set(unmeasured)
    referenced = set(_n.REFERENCE_FOR)
    if covered != referenced:
        fail.append(
            f"nmrscale: the nuclei with a reference are {sorted(referenced)} "
            f"but the accuracy ledger covers {sorted(covered)}; "
            f"unmeasured and unrecorded {sorted(referenced - covered)}, "
            f"stale {sorted(covered - referenced)}.  A nucleus that prints a "
            "shift and appears in neither table is one whose accuracy nobody "
            "has ever looked at")
    for el, entry in sorted(unmeasured.items()):
        tag = f"nmrscale: {el}"
        if not str(entry.get("why") or "").strip():
            fail.append(f"{tag} is listed as unmeasured with no reason, which "
                        "is indistinguishable from an oversight")
        # The pair has to be the product's own, for the same reason the
        # measured entries do: an unmeasured gap is only actionable if it names
        # the comparison it is about.
        key = _n.REFERENCE_FOR.get(el)
        label = str(_n.REFERENCES.get(key, {}).get("label", ""))
        if str(entry.get("reference", "")) not in label:
            fail.append(
                f"{tag} records its reference as {entry.get('reference')!r} but "
                f"the product references {el} against {label!r}, so the gap "
                "describes a different comparison")
        if entry.get("reference_geometry") != key:
            fail.append(
                f"{tag} names geometry {entry.get('reference_geometry')!r} for "
                f"its reference while the product uses {key!r}")
        # A computed value is what separates "we set the comparison up and the
        # number is missing" from "we never computed it".  Those need different
        # work, and before this round the table could not tell them apart.
        val = entry.get("computed_ppm")
        if val is None:
            fail.append(
                f"{tag} is recorded as an unmeasured pair with no computed "
                "value, so 'not measured' and 'not computed' are the same "
                "entry -- and they need different work")
            continue
        rng = float(entry.get("range_ppm", 0.0))
        if rng <= 0.0:
            fail.append(f"{tag} has a computed value and no shift range to "
                        "place it in")
        elif abs(float(val)) > rng:
            fail.append(
                f"{tag} records {float(val):.1f} ppm against a {rng:.0f} ppm "
                f"range for the nucleus, which is not a shift on that scale")

        # The shape of the sentence.  Shared with SCALE_CHECK -- these
        # assertions were written here first and applied to this table only,
        # which is how the measured entries came to be unchecked.  See
        # _reason_shape for how that was found.
        _reason_shape(tag, el, entry, fail)
        try:
            note = _n.gap_why(el)
        except Exception as exc:                          # noqa: BLE001
            fail.append(f"{tag} cannot render its reason: "
                        f"{type(exc).__name__}: {exc}")
            note = ""
        for name in (entry.get("why_quotes") or ()):
            if name == "computed":
                value = float(val)
            elif name == "range":
                value = rng
            else:
                continue                     # _reason_shape checked the shape
            if not any(format(value, s) in note for s in (".0f", ".1f", "+.1f")):
                fail.append(
                    f"{tag} quotes {name} in its reason but the rendered text "
                    f"does not carry {value}; the placeholder drops the number "
                    "instead of showing it")
        # Directional claims are claims, so they are shaped like ones: a
        # geometry that resolves and a sign.  The magnitude is the gate's
        # business -- see TREND_FLOOR_PPM in check_shift_scales.
        for i, item in enumerate(entry.get("trend") or []):
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                fail.append(f"{tag} trend[{i}] is {item!r}, not a "
                            "(geometry, sign) pair")
                continue
            geom, sign = item
            if sign not in (+1, -1):
                fail.append(f"{tag} trend[{i}] has sign {sign!r}; a direction "
                            "is +1 or -1, and anything else makes the claim "
                            "unfalsifiable")
            try:
                _n.scale_geometry(str(geom))
            except Exception as exc:                      # noqa: BLE001
                fail.append(f"{tag} trend[{i}] names geometry {geom!r}, which "
                            f"this build cannot resolve ({type(exc).__name__})")

    for el, entry in sorted(check.items()):
        tag = f"nmrscale: {el}"
        key = _n.REFERENCE_FOR.get(el)
        if key is None:
            fail.append(f"{tag} has an accuracy measurement but no reference "
                        "in REFERENCE_FOR, so the measurement is about a "
                        "scale the product does not use")
            continue
        label = str(_n.REFERENCES.get(key, {}).get("label", ""))
        if str(entry.get("reference", "")) not in label:
            fail.append(
                f"{tag} records its reference as {entry.get('reference')!r} "
                f"but the product references {el} against {label!r} -- the "
                "ledger would be a measurement of a different compound")
        if entry.get("reference_geometry") != key:
            fail.append(
                f"{tag} measures the scale at geometry "
                f"{entry.get('reference_geometry')!r} while the product uses "
                f"{key!r}")
        if not str(entry.get("source", "")).strip():
            fail.append(f"{tag} records a number with no source, so nobody can "
                        "tell where it came from")
        computed = entry.get("computed_ppm")
        if computed is None:
            fail.append(f"{tag} has no computed_ppm")
            continue
        exp = entry.get("experimental_ppm")
        tol = float(entry.get("tolerance_ppm", 0.0))
        usable = bool(entry.get("usable"))
        if usable:
            if exp is None:
                fail.append(
                    f"{tag} is marked usable with no experimental value to be "
                    "usable against -- 'it works' with nothing measured is the "
                    "claim this table exists to stop")
                continue
            err = abs(float(computed) - float(exp))
            if err > tol:
                fail.append(
                    f"{tag} is marked usable, but the computed shift "
                    f"{float(computed):.2f} is {err:.2f} ppm from the "
                    f"experimental {float(exp):.2f} and the recorded "
                    f"tolerance is {tol:.2f}")
            elif err > 0.0 and tol > 3.0 * err:
                # A tolerance is a claim about accuracy, so it has to be a
                # claim with teeth.  Below the demonstrated error it is simply
                # false and the branch above catches it.  Above it, it has to
                # stay within a small multiple of the error it was measured
                # at, or it stops being falsifiable: the ledger carried a
                # 2.5 ppm tolerance for 1H, which is a fifth of the entire 1H
                # shift range, and no degradation of this build could fail it.
                # That number was inherited from the module docstring's
                # twelve-shift table, where a single band covered both 1H and
                # 13C and the 1H half therefore passed by construction.
                #
                # The 3x factor is deliberately crude and it will eventually
                # be too strict: a pair measured to 0.02 ppm would not be
                # allowed a 0.1 ppm tolerance, though 0.1 ppm is a tight
                # claim for any nucleus.  That is the right way round -- the
                # gate asks the author to tighten the number or to widen this
                # rule on purpose, and both are better than a band nothing
                # can fail.  The measured ratios today are 1.2x (13C), 1.3x
                # (14N), 1.3x (19F) and 2.3x (1H).
                fail.append(
                    f"{tag} claims a {tol:.2f} ppm tolerance while the error "
                    f"it was measured at is {err:.2f} ppm, {tol / err:.1f}x "
                    "smaller.  A tolerance that far above the measurement is "
                    "not a claim but a formality -- nothing this ledger is "
                    "meant to catch could ever fail it")
        else:
            # An unusable scale has to be unusable for a reason with a number
            # in it, not merely labelled.  For 77Se the reason is that the
            # method's whole response is a fraction of the nucleus's shift
            # range and smaller than the spread the reference geometry alone
            # produces.
            rng = float(entry.get("range_ppm", 0.0))
            spread = float(entry.get("reference_geometry_spread_ppm", 0.0))
            if rng <= 0.0:
                fail.append(f"{tag} is marked unusable with no shift range "
                            "recorded, so 'the method is not good enough' has "
                            "nothing to be not good enough against")
            elif abs(float(computed)) > rng / 10.0:
                fail.append(
                    f"{tag} is marked unusable but its computed response "
                    f"{abs(float(computed)):.1f} ppm is not small against the "
                    f"{rng:.0f} ppm range of the nucleus -- if the method "
                    "covers a tenth of the range the verdict is wrong and the "
                    "shift should be reported")
            if spread <= abs(float(computed)) / 2.0:
                fail.append(
                    f"{tag} is marked unusable, but the reference geometry "
                    f"spread ({spread:.1f} ppm) is not comparable to the "
                    f"computed response ({abs(float(computed)):.1f} ppm); the "
                    "verdict rests on the geometry mattering as much as the "
                    "chemistry and that is not what the numbers say")
        # The shape of the sentence, from the same helper the gap table uses.
        # Without this line the measured entries were checked only for "the
        # rendered text carries the number" and never for "the number is not
        # also typed into the source" -- which is how three of them came to
        # hold a second copy of their own fields.
        _reason_shape(tag, el, entry, fail)
        # What the user actually reads is rendered from the entry, so this
        # assertion is about the rendered sentence rather than about a string
        # stored beside the table.  It used to be the other way round: each
        # entry carried a hardcoded ``note`` and this section *required* the
        # numbers to appear in it -- which enforced a second copy of every
        # value a few lines above, and three entries had drifted into exactly
        # that state.  Asking for the number in the rendered text is the same
        # guarantee with one copy of the number instead of two.
        note = _n.scale_why(el) or ""
        if not usable and not note:
            fail.append(f"{tag} is unusable and says nothing about it, so the "
                        "shift would simply be missing from the spectrum")
        if note:
            for token in (f"{abs(float(computed)):.0f}",):
                if token not in note:
                    fail.append(
                        f"{tag}'s reason does not carry the computed number "
                        f"{token}: {note[:80]!r}")
        if exp is not None and note and usable:
            if f"{abs(float(exp)):.0f}" not in note:
                fail.append(
                    f"{tag}'s reason does not carry the experimental number "
                    f"{abs(float(exp)):.0f}: {note[:80]!r}")

    # The verdict on selenium, checked through the same helper compute() calls
    # rather than by reading the table again.
    #
    # The numbers in the message are read out of the entry, not typed here.  The
    # previous version of this block *said* that in its comment while its
    # message hardcoded "101", "3000" and "72" -- a third copy of numbers that
    # already lived in the entry and in the sentence, and one that would not
    # have moved when the other two did.  A comment describing a property the
    # code does not have is worse than no comment: it is a claim, and this one
    # was false.  It is exactly the failure this whole round is about, written
    # by the assertion that exists to catch it.
    se = _n.SCALE_CHECK["Se"]
    if _n.scale_usable("Se"):
        fail.append(
            "nmrscale: scale_usable('Se') is True, so a 77Se shift would be "
            "printed -- but the method's whole Me2Se -> H2Se response is "
            f"{abs(float(se['computed_ppm'])):.0f} ppm against a "
            f"{float(se['range_ppm']):.0f} ppm range and the reference's own "
            f"shielding moves "
            f"{float(se['reference_geometry_spread_ppm']):.0f} ppm between two "
            "defensible geometries of (CH3)2Se")
    if _n.scale_usable("H") is not True or _n.scale_usable("C") is not True:
        fail.append("nmrscale: the validated nuclei 1H and 13C must be usable")
    # A reason is what a user reads, so it has to carry the numbers the verdict
    # was reached from -- every one of them, for every entry that has one.
    # Rendered from the entry rather than stored beside it, so a new entry is
    # covered without anyone remembering to add it, and so each number exists
    # in one place instead of two.
    #
    # ``conformer_spread_ppm`` is deliberately not required: it is a supporting
    # number the reason does not quote, and requiring it would be worse than
    # useless, because the token for 8.37 is "8" and the reason says "77Se" --
    # a token check that passes on a coincidence is a check that cannot fail.
    for el, entry in sorted(check.items()):
        note = _n.scale_why(el) or ""
        if not note:
            continue
        want = []
        if entry.get("computed_ppm") is not None:
            want.append(f"{abs(float(entry['computed_ppm'])):.0f}")
        if entry.get("experimental_ppm") is not None:
            want.append(f"{abs(float(entry['experimental_ppm'])):.0f}")
        for field in ("range_ppm", "reference_geometry_spread_ppm"):
            if entry.get(field):
                want.append(f"{float(entry[field]):.0f}")
        missing = [t for t in want if t not in note]
        if missing:
            fail.append(
                f"nmrscale: the accuracy note for {el} does not carry "
                f"{missing}, so the user is told the scale is off without "
                f"being told by how much: {note[:90]!r}")
    for el in ("H", "C"):
        if _n.accuracy_note(el) is not None:
            fail.append(
                f"nmrscale: {el} is measured and reproduced yet carries an "
                f"accuracy note ({_n.accuracy_note(el)[:60]}); a note on every "
                "nucleus is a note nobody reads")
    # Selenium has a reference now, so it must not still be in the list of
    # elements with no reference -- the two are different causes and the
    # unreferenced table is asserted to be exactly the elements that have no
    # reference at all.
    if "Se" in _n.NO_REFERENCE_WHY:
        fail.append(
            "nmrscale: Se is still in NO_REFERENCE_WHY, but this build does "
            "define a 77Se reference -- the reason there is no 77Se shift is "
            "the method, not the availability of a compound, and the two are "
            "recorded in different tables on purpose")
    # ---- the twelve-shift rows, and the tolerances that rest on them -----
    # A per-nucleus tolerance measured on one molecule is not a claim about the
    # nucleus.  That is not a hypothesis: the 13C tolerance was set from methane
    # alone (2.09 ppm out) and methanol had always falsified it at 3.04.  So the
    # usable entries are partitioned -- those whose tolerance is checked against
    # a multi-row ledger, and those named in TOLERANCE_UNBACKED with the reason
    # and the work it needs.  The partition is asserted, so adding a usable
    # entry without a ledger fails here instead of passing unnoticed.
    ledger = _n.SHIFT_LEDGER
    unbacked = _n.TOLERANCE_UNBACKED
    usable = {el for el, e in check.items() if e.get("usable")}
    backed = set(ledger)
    if backed - usable:
        fail.append(
            f"nmrscale: SHIFT_LEDGER covers {sorted(backed - usable)}, which "
            "has no usable SCALE_CHECK entry -- the rows would be checked "
            "against a tolerance that does not exist")
    if usable != backed | set(unbacked):
        fail.append(
            f"nmrscale: the usable nuclei are {sorted(usable)} but the ledgers "
            f"cover {sorted(backed)} and TOLERANCE_UNBACKED names "
            f"{sorted(unbacked)}; unaccounted {sorted(usable - backed - set(unbacked))}, "
            f"stale {sorted((backed | set(unbacked)) - usable)}.  A nucleus "
            "whose tolerance is checked against nothing is a claim about the "
            "nucleus that nothing tests")
    for el, why in sorted(unbacked.items()):
        if len(str(why).strip()) < 60:
            fail.append(
                f"nmrscale: {el} is recorded as having a one-molecule "
                f"tolerance with no usable reason ({str(why)[:60]!r}); "
                "'not done yet' and 'not recorded' look identical from outside")
    for el, entry in sorted(ledger.items()):
        tag = f"nmrscale: shift ledger {el}"
        rows = entry.get("rows") or []
        # One row would reproduce the defect the ledger exists to fix.
        if len(rows) < 2:
            fail.append(
                f"{tag} has {len(rows)} row(s); a tolerance checked against one "
                "molecule is the thing this table was added to stop")
        if float(entry.get("band_ppm", 0.0)) <= 0.0:
            fail.append(f"{tag} has no band to count its rows against")
        key = _n.REFERENCE_FOR.get(el)
        label = str(_n.REFERENCES.get(key, {}).get("label", ""))
        if str(entry.get("reference", "")) not in label:
            fail.append(
                f"{tag} records its reference as {entry.get('reference')!r} but "
                f"the product references {el} against {label!r}, so the rows "
                "describe a different comparison")
        if entry.get("reference_geometry") != key:
            fail.append(
                f"{tag} names geometry {entry.get('reference_geometry')!r} for "
                f"its reference while the product uses {key!r}")
        names = [str(r[0]) for r in rows if isinstance(r, (list, tuple))]
        if len(names) != len(rows):
            fail.append(f"{tag} has a row that is not a sequence")
        for name in sorted(entry.get("exceptions") or {}):
            if name not in names:
                fail.append(
                    f"{tag} names {name!r} as an exception, but no row has that "
                    "name -- the exception would excuse nothing while reading "
                    "as if it excused something")
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                fail.append(f"{tag} has a row that is not (name, geometry, "
                            "computed, experimental)")
                continue
            name, geom, computed, exp = row
            if computed is None or exp is None:
                fail.append(
                    f"{tag} row {name!r} has a null number, so there is nothing "
                    "for the recomputation to be checked against")
                continue
            # The exception has to be a decision with a number behind it, not a
            # label.  Whether it *is* over the tolerance is checked by the gate,
            # which has the recomputed values; what can be checked here is that
            # an exception is not also the row the tolerance was set from.
            if str(name) in (entry.get("exceptions") or {}) and \
                    abs(float(computed) - float(exp)) == 0.0:
                fail.append(
                    f"{tag} row {name!r} is excepted and has zero error, which "
                    "means the exception excuses a perfect row")
        # The tolerance rule, against the numbers the ledger records.  The gate
        # runs the same rule against the numbers it recomputes, so a tolerance
        # that is wrong relative to the recorded rows and a ledger that has
        # drifted away from the method are two different failures.
        fail += _n.tolerance_holds(el, _n.ledger_errors(el))
    print(f"[{'PASS' if not any('nmrscale' in f for f in fail) else 'FAIL'}] "
          f"the accuracy ledger covers every referenced nucleus "
          f"({len(check)} measured, {len(unmeasured)} unmeasured); "
          f"{len(backed)} tolerance(s) rest on a multi-row ledger "
          f"({sum(len(ledger[e]['rows']) for e in ledger)} rows) and "
          f"{len(unbacked)} are named as one-molecule",
          flush=True)


def nmr_payload_sections(nres: dict, fail: list) -> int:
    """Assertions on a finished NMR payload.  Returns the failure count."""
    nbad = 0
    nuclei = nres.get("nuclei") or []
    if not nuclei:
        fail.append("nmr/water: the result carries no nuclei")
        return 1

    lvl = nres.get("level") or {}
    if lvl.get("scf") != "RHF":
        fail.append(
            f"nmr/water: the payload says the SCF level is "
            f"{lvl.get('scf')!r}; the shielding is RHF-only in this build and "
            "the payload has to say so")
        nbad += 1
    if lvl.get("functional_requested") and \
            lvl.get("functional_used_for_shielding") is not None:
        fail.append(
            "nmr/water: the payload claims a functional was used for the "
            "shielding; it cannot be in this build")
        nbad += 1
    # The warning list is what the user actually reads, so the limitation has
    # to be in it -- not only in a field nobody renders.
    if not any("HARTREE-FOCK" in str(w) for w in (nres.get("warnings") or [])):
        fail.append(
            "nmr/water: no warning states that the shielding is at "
            "Hartree-Fock, so a user who asked for B3LYP will misread every "
            "absolute shielding")
        nbad += 1

    # the consistency check that catches a mixed perturbation convention
    dg = nres.get("diagnostics") or {}
    dE = dg.get("dE_dB") or []
    if len(dE) != 3:
        fail.append(f"nmr/water: expected 3 dE/dB values, got {len(dE)}")
        nbad += 1
    elif max(abs(float(v)) for v in dE) > 1e-6:
        fail.append(
            f"nmr/water: dE/dB reached {max(abs(float(v)) for v in dE):.3e}; "
            "a closed shell has no linear Zeeman term, so this is a "
            "convention error and every shielding is suspect")
        nbad += 1
    if not all(r.get("converged") for r in (dg.get("scf_runs") or [])):
        fail.append("nmr/water: a field-perturbed SCF did not converge")
        nbad += 1

    syms = [n["symbol"] for n in nuclei]
    if sorted(syms) != ["H", "H", "O"]:
        fail.append(f"nmr/water: nuclei are {syms}, not O + 2 H")
        nbad += 1
    for n in nuclei:
        t = n.get("tensor_ppm")
        if not (isinstance(t, list) and len(t) == 3
                and all(len(r) == 3 for r in t)):
            fail.append(
                f"nmr/water/{n.get('symbol')}: the tensor is not 3x3, so the "
                "front end cannot draw principal axes")
            nbad += 1
            continue
        m = np.array(t, dtype=float)
        if np.abs(m - m.T).max() > 1e-9:
            fail.append(
                f"nmr/water/{n.get('symbol')}: the reported tensor is "
                "asymmetric; the exact tensor cannot be, and the panel would "
                "show a number the physics forbids")
            nbad += 1
        ev = n.get("eigenvalues_ppm") or []
        if len(ev) != 3 or not (ev[0] <= ev[1] + 1e-9 <= ev[2] + 2e-9):
            fail.append(
                f"nmr/water/{n.get('symbol')}: principal values {ev} are not "
                "ascending")
            nbad += 1
        if not isinstance(n.get("delta_ppm"), (int, float)):
            fail.append(
                f"nmr/water/{n.get('symbol')}: no chemical shift, so the "
                "table would be blank for a nucleus that has a reference")
            nbad += 1
        if not n.get("reference"):
            fail.append(
                f"nmr/water/{n.get('symbol')}: a shift is reported with no "
                "reference compound named, so the number cannot be "
                "reproduced")
            nbad += 1
        if not n.get("nmr_active"):
            fail.append(
                f"nmr/water/{n.get('symbol')}: not marked NMR active, but "
                "both O and H are")
            nbad += 1

    # the water 17O shielding is the number the whole 17O scale rests on
    iso_o = [n["sigma_iso_ppm"] for n in nuclei if n["symbol"] == "O"]
    if iso_o and not (318.0 < iso_o[0] < 330.0):
        fail.append(
            f"nmr/water: sigma(17O) = {iso_o[0]:.2f} ppm. The converged value "
            "at the library's water geometry is 323.87 (probe_nmr28); 345.0 "
            "is the under-converged artifact that a conv_tol of 1e-11 "
            "produces at small field steps, and 329.98 is what the module's "
            "own experimental geometry gives -- a geometry the product no "
            "longer uses")
        nbad += 1
    iso_h = [n["sigma_iso_ppm"] for n in nuclei if n["symbol"] == "H"]
    if iso_h and not (31.0 < float(np.mean(iso_h)) < 32.5):
        fail.append(
            f"nmr/water: sigma(1H) = {float(np.mean(iso_h)):.3f} ppm, outside "
            "31.0-32.5")
        nbad += 1

    groups = nres.get("groups") or []
    if not groups:
        fail.append("nmr/water: no shift groups, so the spectrum has no "
                    "sticks")
        nbad += 1
    spec = nres.get("spectrum") or {}
    xs = spec.get("shift_ppm") or []
    ys = spec.get("intensity") or []
    if len(xs) < 100 or len(ys) != len(xs):
        fail.append(f"nmr/water: the curve has {len(xs)} points and "
                    f"{len(ys)} intensities")
        nbad += 1
    elif xs[0] >= xs[-1]:
        fail.append(
            "nmr/water: the ppm grid runs high-to-low; the reversal belongs "
            "to the chart, and a pre-reversed array would be reversed twice")
        nbad += 1
    if ys and abs(max(ys) - 1.0) > 1e-9:
        fail.append(f"nmr/water: the curve peaks at {max(ys)}, not 1")
        nbad += 1

    # ---- one spectrum per isotope ---------------------------------------
    # A spectrum is acquired for one nucleus.  The payload used to carry a
    # single axis holding every group, with one linewidth in Hz at the 1H
    # Larmor frequency -- so a 13C line came out four times too narrow and the
    # two nuclei shared a window neither of them fits in.
    spectra = nres.get("spectra") or {}
    if not spectra:
        fail.append("nmr/water: the payload carries no per-isotope spectra")
        nbad += 1
    else:
        want_iso = {"1H", "17O"}
        if set(spectra) != want_iso:
            fail.append(
                f"nmr/water: spectra are for {sorted(spectra)}, but water has "
                f"{sorted(want_iso)}")
            nbad += 1
        for iso, sp in sorted(spectra.items()):
            if sp.get("isotope") != iso:
                fail.append(
                    f"nmr/water: the spectrum filed under {iso} says it is "
                    f"{sp.get('isotope')!r}")
                nbad += 1
            if not sp.get("shift_ppm"):
                fail.append(f"nmr/water: the {iso} spectrum is empty")
                nbad += 1
                continue
            lo, hi = min(sp["shift_ppm"]), max(sp["shift_ppm"])
            for grp in (nres.get("groups_by_isotope") or {}).get(iso, []):
                if not lo <= grp["delta_ppm"] <= hi:
                    fail.append(
                        f"nmr/water: the {iso} signal at "
                        f"{grp['delta_ppm']:.3f} ppm is outside its own "
                        f"window {lo:.3f}..{hi:.3f}")
                    nbad += 1
            # the Larmor frequency, and therefore the ppm width of the line,
            # is a property of the nucleus
            from backend.engine import nmr as _nn
            want_l = _nn.larmor_mhz("H" if iso == "1H" else "O",
                                    sp["spectrometer_mhz"])
            if abs(sp["larmor_mhz"] - want_l) > 1e-6:
                fail.append(
                    f"nmr/water: the {iso} spectrum precesses at "
                    f"{sp['larmor_mhz']:.4f} MHz; it must be {want_l:.4f}")
                nbad += 1
            if sp.get("grid_step_ppm") is None:
                fail.append(
                    f"nmr/water: the {iso} spectrum does not report its grid "
                    "step, so the figure's resolution cannot be checked")
                nbad += 1
            if not sp.get("resolved", True):
                fail.append(
                    f"nmr/water: the {iso} spectrum says it was drawn with a "
                    "widened line; at a 6-31G* water window that must not "
                    "happen")
                nbad += 1
            st = sp.get("sticks") or []
            if len(st) != len((nres.get("groups_by_isotope") or {}).get(iso, [])):
                fail.append(
                    f"nmr/water: the {iso} spectrum has {len(st)} sticks for "
                    f"{len((nres.get('groups_by_isotope') or {}).get(iso, []))}"
                    " groups")
                nbad += 1
            if st and not all("rel_intensity" in s for s in st):
                fail.append(
                    f"nmr/water: a {iso} stick carries no height, so the "
                    "chart cannot draw the integral")
                nbad += 1
        # the default spectrum has to be one of them, and 1H when there is one
        if "1H" in spectra and nres.get("spectrum_isotope") != "1H":
            fail.append(
                "nmr/water: the panel opens on "
                f"{nres.get('spectrum_isotope')!r}; with a 1H signal present "
                "it must open on 1H")
            nbad += 1
        if nres.get("spectrum_isotope") not in spectra:
            fail.append(
                "nmr/water: the panel opens on "
                f"{nres.get('spectrum_isotope')!r}, which is not one of the "
                f"spectra present ({sorted(spectra)})")
            nbad += 1
        if nres.get("spectrum") != spectra.get(nres.get("spectrum_isotope")):
            fail.append(
                "nmr/water: the single-isotope `spectrum` field is not the "
                "default isotope's spectrum, so an older caller would draw the "
                "wrong axis")
            nbad += 1
        if sorted(nres.get("spectrum_isotopes") or []) != sorted(spectra):
            fail.append(
                f"nmr/water: spectrum_isotopes is "
                f"{nres.get('spectrum_isotopes')}, spectra is "
                f"{sorted(spectra)}")
            nbad += 1
    # an element with no reference must be named, not silently dropped
    if "unreferenced_elements" not in nres:
        fail.append(
            "nmr/water: the payload has no unreferenced_elements field, so a "
            "nucleus whose reference could not be computed would simply be "
            "missing from the spectrum with nothing to say why")
        nbad += 1
    else:
        # it has to be exactly the active elements that have no reference --
        # not always empty, and not the whole molecule
        want_missing = sorted(set(nres.get("active_elements") or [])
                              - set(nres.get("references") or {}))
        if sorted(nres["unreferenced_elements"]) != want_missing:
            fail.append(
                f"nmr/water: unreferenced_elements is "
                f"{sorted(nres['unreferenced_elements'])}, but the active "
                f"elements without a reference are {want_missing}")
            nbad += 1
        elif want_missing:
            fail.append(
                f"nmr/water: {want_missing} has no reference, so those nuclei "
                "have no chemical shift and are absent from the spectrum. Both "
                "references are small enough for any machine; regenerate with "
                "`python -m backend.gen_nmr_cache`")
            nbad += 1

    refs = nres.get("references") or {}
    if "H" not in refs or "O" not in refs:
        fail.append(
            f"nmr/water: references computed for {sorted(refs)}, but water "
            "contains H and O")
        nbad += 1
    else:
        if "TMS" not in str(refs["H"].get("label", "")):
            fail.append(
                f"nmr/water: 1H is referenced to "
                f"{refs['H'].get('label')!r}; the standard is TMS")
            nbad += 1
        # The scale has to travel with the reference, in the same object the
        # panel reads the reference compound's name out of.  Water's H and O
        # are both on their IUPAC primary reference, so the value here is
        # None -- but the *field* has to be present, or the panel cannot tell
        # "on the primary scale" from "the payload predates the idea of a
        # scale", and the note would be invisible for exactly the two nuclei
        # that need it.
        for el in ("H", "O"):
            if "scale_note" not in refs[el]:
                fail.append(
                    f"nmr/water: the {el} reference carries no scale_note "
                    "field, so the panel cannot show which scale the shift "
                    "is on")
                nbad += 1
        # water is its own 17O reference, so its shift must be zero
        o_shift = [n["delta_ppm"] for n in nuclei if n["symbol"] == "O"]
        if o_shift and abs(o_shift[0]) > 0.01:
            fail.append(
                f"nmr/water: water is its own 17O reference, so its shift "
                f"must be 0.00; got {o_shift[0]:.3f}")
            nbad += 1
    return nbad


def nmr_unreferenced_sections(nres: dict, fail: list) -> int:
    """Assertions on a molecule that has an NMR-active element with no
    reference at all.

    This needs its own dataset.  Water has nothing unreferenced once TMS is
    cached, so ``unreferenced_elements: []`` is the *correct* value there and
    a mutation of it fires nothing -- which is exactly how the first version
    of that mutation was caught (N18, "FIRED NOTHING").

    Thiophene is the right molecule: C and H are referenced to TMS and S is
    not referenced at all.  Thirteen NMR-active elements have no reference in
    this build (Al, B, Br, Cl, D, Hg, I, Li, Na, Pb, Pt, S, Sn), so this is
    the common case, not an exotic one.  (Selenium was the fourteenth until it
    acquired a reference compound; it is now in the accuracy ledger instead,
    which is a different table because it is a different problem.)

    The rule is that the payload has to NAME them.  A nucleus whose reference
    is missing used to be dropped from the table without a word, so a ³³S
    measurement would be read as if the molecule had no sulfur.
    """
    nbad = 0
    if "unreferenced_elements" not in nres:
        fail.append(
            "nmrmissing: the payload has no unreferenced_elements field, so a "
            "nucleus with no reference is simply absent from the table with "
            "nothing to say why")
        nbad += 1
    else:
        want = sorted(set(nres.get("active_elements") or [])
                      - set(nres.get("references") or {}))
        got = sorted(nres["unreferenced_elements"])
        if got != want:
            fail.append(
                f"nmrmissing: unreferenced_elements is {got}, but the active "
                f"elements without a reference are {want}.  This molecule has "
                "sulfur, no 33S reference is computed, and S is the element "
                "that has to be named")
            nbad += 1
        if "S" not in got:
            fail.append(
                "nmrmissing: sulfur is absent from unreferenced_elements, so "
                "the payload claims the molecule has no NMR-active sulfur")
            nbad += 1

    # and the ones that DO have a reference must still be there -- naming the
    # missing element must not take the others down with it
    shifts = {n.get("symbol") for n in (nres.get("nuclei") or [])
              if isinstance(n.get("delta_ppm"), (int, float))}
    for el in ("C", "H"):
        if el not in shifts:
            fail.append(
                f"nmrmissing: {el} has a reference but no chemical shift, so "
                "naming the unreferenced element broke the referenced ones")
            nbad += 1
    # a nucleus with no reference must not carry a shift
    for n in (nres.get("nuclei") or []):
        if n.get("symbol") == "S" and n.get("delta_ppm") is not None:
            fail.append(
                f"nmrmissing: sulfur has no reference yet carries a shift of "
                f"{n['delta_ppm']}")
            nbad += 1
    # and the user has to be told
    warns = " ".join(str(w) for w in (nres.get("warnings") or []))
    if "unreferenced_elements" in nres and nres["unreferenced_elements"]:
        if "S" not in warns:
            fail.append(
                "nmrmissing: sulfur has no reference and no warning names it, "
                "so its absence from the spectrum reads as absence of sulfur")
            nbad += 1
    # the 1H spectrum must exist even though S does not
    spectra = nres.get("spectra") or {}
    if "1H" not in spectra:
        fail.append(
            f"nmrmissing: no 1H spectrum ({sorted(spectra)}); the protons "
            "have a reference and must still be plotted")
        nbad += 1
    return nbad


def raman_payload_sections(vres: dict, fail: list) -> int:
    """Assertions on the Raman fields of a finished vibrations payload.

    Extracted from ``main`` so the mutation harness can run the real
    assertions against a payload built directly by
    ``analysis.vibrations`` -- without a server, and without paying for
    the API round trip on every mutation.  Returns the failure count.
    """
    nbad = 0
    freq = vres.get("frequencies_cm1") or []

    # ---- Raman, from the same job --------------------------------
    # The two spectra share the Hessian and the displaced geometries, so
    # this costs nothing extra here -- and the point of asserting them
    # together is that the interesting fact about water is the *contrast*:
    # the symmetric stretch is the weakest IR band and the strongest Raman
    # line.  An implementation that reused the IR intensities for the Raman
    # activities, or that mis-projected the tensor onto the modes, gets that
    # exactly backwards, and no shape check on either curve alone would
    # notice.
    intens = vres.get("ir_intensities_km_mol") or []
    ram = vres.get("raman") or {}
    if ram.get("available") is False:
        fail.append(
            f"raman/water: not computed -- {ram.get('reason')}")
        nbad += 1
    elif not ram.get("activities_a4_amu"):
        fail.append(
            "raman/water: the vibrational analysis returned no Raman "
            "activities, so there is no Raman figure")
        nbad += 1
    else:
        act = [float(v) for v in ram["activities_a4_amu"]]
        rho = [float(v) for v in (ram.get("depolarization") or [])]
        if len(act) != len(freq) or len(rho) != len(freq):
            fail.append(
                f"raman/water: {len(act)} activities and {len(rho)} "
                f"depolarization ratios for {len(freq)} modes")
            nbad += 1
        # The ceiling is a theorem, so any violation is a bug, not a
        # numerical wobble.
        if rho and max(rho) > 0.7501:
            fail.append(
                f"raman/water: rho reached {max(rho)}, above the 3/4 "
                f"ceiling that the formula guarantees")
            nbad += 1
        if act and intens and len(act) == len(intens):
            strong_raman = max(range(len(act)), key=lambda i: act[i])
            weak_ir = min(range(len(intens)), key=lambda i: intens[i])
            if strong_raman != weak_ir:
                fail.append(
                    f"raman/water: the strongest Raman mode is {strong_raman} "
                    f"but the weakest IR mode is {weak_ir}; for water both "
                    f"are the symmetric O-H stretch, and an implementation "
                    f"that reuses the IR intensities or mis-projects the "
                    f"tensor gets this backwards")
                nbad += 1
            # And it has to be the *symmetric* stretch, not merely the same
            # index: the mode is identified by frequency, so a spectrum
            # whose bands were all shifted or mislabelled cannot satisfy
            # this by having both columns agree on the wrong mode.
            if freq[strong_raman] < 3400.0:
                fail.append(
                    f"raman/water: the strongest Raman line is at "
                    f"{freq[strong_raman]} cm-1, which is not an O-H "
                    f"stretch")
                nbad += 1
            # A B2 mode in C2v has no isotropic polarizability derivative,
            # so its rho is exactly 3/4.  This is symmetry, not a fit: it
            # tests the projection and the tensor at once.
            degen = [i for i in range(len(rho))
                     if abs(rho[i] - 0.75) < 5e-3]
            if not degen:
                fail.append(
                    f"raman/water: no mode sits at rho = 0.750; the "
                    f"asymmetric O-H stretch is B2 and its isotropic "
                    f"derivative vanishes by symmetry. Got {rho}")
                nbad += 1
        sr2 = ram.get("sum_rule") or {}
        resid2 = sr2.get("residual_pct")
        if resid2 is None:
            fail.append(
                "raman/water: no Raman sum rule, so the activities are not "
                "shown to be on an absolute scale")
            nbad += 1
        elif abs(float(resid2)) > 0.5:
            fail.append(
                f"raman/water: the Raman sum rule is off by {resid2}%, so "
                f"the activities are not on an absolute scale")
            nbad += 1
        if sr2.get("translational_frob2_per_amu") not in (0, 0.0):
            fail.append(
                f"raman/water: the translational term of the Raman sum "
                f"rule is "
                f"{sr2.get('translational_frob2_per_amu')}, but alpha is "
                f"invariant under translation so it is exactly zero")
            nbad += 1
        fvi = ram.get("frobenius_vs_invariants_pct")
        if fvi is None or abs(float(fvi)) > 0.01:
            fail.append(
                f"raman/water: the Frobenius norm and 3 a_bar^2 + "
                f"(2/3) gamma^2 differ by {fvi}%, but they are the same "
                f"number written two ways")
            nbad += 1
        # 6-31G* has no diffuse functions, so the polarizability is
        # expected to be badly low -- and the result has to SAY so rather
        # than letting a normalised curve hide it.
        if ram.get("alpha_reliable") is not False:
            fail.append(
                f"raman/water: alpha_reliable is "
                f"{ram.get('alpha_reliable')!r} at 6-31G*, which has no "
                f"diffuse functions")
            nbad += 1
        if not ram.get("basis_warning"):
            fail.append(
                "raman/water: no basis warning at 6-31G*, though the "
                "polarizability there is ~48% below the measured value")
            nbad += 1
        sp = vres.get("raman_spectrum") or {}
        if len(sp.get("x") or []) < 100 or not sp.get("peaks"):
            fail.append(
                f"raman/water: Raman curve has {len(sp.get('x') or [])} "
                f"points and {len(sp.get('peaks') or [])} peaks")
            nbad += 1
        elif sp["x"][0] < sp["x"][-1]:
            fail.append(
                "raman/water: the Raman axis runs low-to-high; the "
                "convention is high wavenumber on the left")
            nbad += 1
        else:
            # The peak table has to carry rho alongside the activity, or
            # the figure cannot be captioned with the number that says
            # whether each band is polarised.
            for k in ("activity_a4_amu", "depolarization", "scaled_cm1"):
                if not all(k in (p or {}) for p in sp["peaks"]):
                    fail.append(
                        f"raman/water: the peak table has no '{k}' column")
                    nbad += 1
                    break
    return nbad


# No minimum-energy structure in the library has a bond angle sharper than
# this.  Cyclopropane does -- 60 degrees -- and is deliberately not in the
# library; the stored methanol used to close 62 degrees, which is how this
# check came to exist.
MIN_BOND_ANGLE = 70.0

# The provenance strings a library entry is allowed to carry.  Every one of
# them is defined in backend/engine/molecule.py and is the answer to a
# question the code can actually answer, so a hand-typed string -- however
# plausible -- is not accepted.
_GEOMETRY_SOURCES = frozenset({
    "MMFF94 (RDKit ETKDGv3)",
    "UFF (RDKit ETKDGv3; MMFF94 has no parameters for this molecule)",
    "MMFF94 (RDKit ETKDGv3; the relaxation did not converge)",
    "B3LYP / 6-31G* stationary point (built-in library geometry)",
})


def library_sections(fail: list) -> int:
    """Assertions on the molecule library itself, needing no server.

    The library is the product's single source of truth for a molecular
    geometry, and until now nothing checked it.  Two things went wrong because
    of that, and both are cheap to catch:

    * **A description that states a geometry the entry does not have.**  The
      water entry said "~104.5 deg H-O-H angle" for a geometry that is
      104.00.  That is not cosmetic: the same geometry is the 17O NMR
      reference, 17O shielding moves 539 ppm per angstrom of O-H, and a
      reader who trusted the description would be quoting a number from a
      molecule that does not exist.
    * **An entry that cannot be used at all** -- an XYZ whose atom count
      disagrees with its header, a charge or multiplicity that is missing, a
      formula that is not the formula of its own atoms.

    Extracted from ``main`` so the mutation harness can run it directly.
    """
    nbad = 0
    from backend.engine.elements import bond_cutoff as _bond_cutoff
    from backend.engine.molecule import (
        _entry_desc,
        _entry_name,
        from_name as _from_name,
        from_smiles as _from_smiles,
    )
    import backend.verify_library_geometry as _vlg

    # RDKit is what turns a SMILES into the structure an entry is checked
    # against, so it is imported once here and its absence is a failure, not a
    # skip: without it no library geometry can be built or verified at all.
    _Chem = None
    _AllChem = None
    try:
        from rdkit import Chem as _Chem, RDLogger as _RDLogger
        from rdkit.Chem import AllChem as _AllChem

        _RDLogger.DisableLog("rdApp.*")
    except ImportError as exc:
        fail.append(f"library: RDKit is unavailable ({exc}); no library "
                    "geometry can be built or checked without it")
        nbad += 1

    lib_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             "..", "data", "molecules", "library.json")
    try:
        with open(lib_path, encoding="utf-8") as fh:
            lib = _json.load(fh)
    except Exception as exc:                             # noqa: BLE001
        fail.append(f"library: cannot be read ({exc})")
        return 1

    if len(lib) < 20:
        fail.append(f"library: only {len(lib)} entries; the UI advertises far "
                    "more than that")
        nbad += 1

    # Every nucleus that has a reference compound must be reachable from the
    # library by name.  Nothing stated this rule, which is how the library
    # reached 62 entries without a single one containing phosphorus: 31P had a
    # reference, a scale note, a place in the isotope list and a spectrum
    # channel, and no molecule a user could ask for by name.  The feature
    # existed only for people who already knew a SMILES.  The rule is cheap to
    # check -- it reads the declared formulas and the reference table, no SCF
    # -- and it is the assertion whose absence let that sit there unnoticed.
    try:
        from backend.engine import nmr as _nuc
    except Exception as exc:                              # noqa: BLE001
        fail.append(f"library: cannot import the nucleus table ({exc})")
        _nuc = None
    if _nuc is not None:
        _sym = _re.compile(r"([A-Z][a-z]?)")
        carried = set()
        for entry in lib.values():
            if isinstance(entry, dict):
                carried |= {s for s in _sym.findall(entry.get("formula", ""))
                            if s in _nuc.NUCLEI}
        unreachable = sorted(set(_nuc.REFERENCE_FOR) - carried)
        if unreachable:
            fail.append(
                f"library: {', '.join(unreachable)} can be shifted -- a "
                "reference compound is defined for each -- but no library "
                "entry contains them, so they are reachable only by typing a "
                "SMILES")
            nbad += 1

    for name, entry in sorted(lib.items()):
        if not isinstance(entry, dict):
            fail.append(f"library/{name}: entry is {type(entry).__name__}")
            nbad += 1
            continue
        for field in ("formula", "xyz", "charge", "multiplicity"):
            if field not in entry:
                fail.append(f"library/{name}: no {field}")
                nbad += 1
        # The name and the description, read the way the module reads them.
        # library.json was written by two scripts with two schemas -- the
        # original entries carry display_name/description and the seventeen
        # that extend_library appended carried name/desc -- and every reader
        # looked for the first pair, so those seventeen reached the UI with
        # their lowercase key as their name and a blank description while the
        # real strings sat in the same dict.  What is asserted is the outcome
        # a user sees, not which key holds it.
        shown = _entry_name(entry, name)
        if not shown or shown == name:
            fail.append(
                f"library/{name}: shows as {shown!r}, which is the key itself "
                "-- the display name is missing or under a key nothing reads")
            nbad += 1
        if not _entry_desc(entry):
            fail.append(
                f"library/{name}: shows no description.  The molecule panel "
                "renders this string, so a missing one is a blank row")
            nbad += 1
        lines = (entry.get("xyz") or "").splitlines()
        if len(lines) < 2:
            fail.append(f"library/{name}: the XYZ block is empty")
            nbad += 1
            continue
        try:
            natm = int(lines[0].split()[0])
        except (ValueError, IndexError):
            fail.append(f"library/{name}: the XYZ header is not an atom count: "
                        f"{lines[0]!r}")
            nbad += 1
            continue
        atoms = []
        for line in lines[2:2 + natm]:
            p = line.split()
            if len(p) < 4:
                fail.append(f"library/{name}: XYZ line {line!r} is not "
                            "'symbol x y z'")
                nbad += 1
                atoms = []
                break
            try:
                atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
            except ValueError:
                fail.append(f"library/{name}: XYZ line {line!r} has a "
                            "non-numeric coordinate")
                nbad += 1
                atoms = []
                break
        if len(atoms) != natm:
            if atoms:
                fail.append(f"library/{name}: header says {natm} atoms and "
                            f"{len(atoms)} lines follow")
                nbad += 1
            continue
        if len(lines) > 2 + natm:
            fail.append(f"library/{name}: {len(lines) - 2 - natm} trailing "
                        "line(s) after the atoms")
            nbad += 1

        # ---- one schema, so the next reader is not the one that breaks ----
        # The tolerant accessors above exist so that a legacy entry is never
        # *invisible*; this exists so that the legacy schema is never written
        # again.  They are different jobs, and the seventeen entries that
        # arrived under name/desc are the reason both are needed.
        legacy = [f for f in ("name", "desc", "key") if f in entry]
        if legacy:
            fail.append(
                f"library/{name}: carries the legacy field(s) {legacy}.  The "
                "canonical pair is display_name/description and the dict key "
                "is the key; two schemas in one file is how seventeen entries "
                "came to reach the UI blank")
            nbad += 1

        # ---- where these coordinates came from, and whether that is true ----
        # Every entry was stamped "MMFF94 (RDKit ETKDGv3)" whatever its
        # coordinates actually were, and re-embedding the library from its own
        # SMILES showed the claim was false for 37 of the 62 entries.  Two of
        # those were not molecules at all.  A wrong provenance string is worse
        # than none: it tells a reader the geometry is reproducible when
        # nothing here can reproduce it, and no result built on it can be
        # written down as A//B.
        src = str(entry.get("geometry_source") or "")
        if src not in _GEOMETRY_SOURCES:
            fail.append(
                f"library/{name}: records its geometry source as {src[:60]!r}, "
                "which is not a string this program can stand behind.  An "
                "entry either says where its coordinates came from or admits "
                "that the library does not record it")
            nbad += 1
        else:
            verdict = _vlg.force_field_verdict(entry, name)
            claims_ff = src.startswith(("MMFF94 (RDKit", "UFF (RDKit"))
            if claims_ff and not verdict["ok"]:
                fail.append(
                    f"library/{name}: claims {src!r}, but re-embedding it from "
                    f"its own SMILES does not reproduce it: {verdict['reason']}")
                nbad += 1
            elif claims_ff and verdict["source"] != src:
                fail.append(
                    f"library/{name}: claims {src!r}, but the pipeline that "
                    f"reproduces its coordinates is {verdict['source']!r}")
                nbad += 1
            elif src.startswith("B3LYP /") and verdict["ok"]:
                fail.append(
                    f"library/{name}: claims {src!r}, but its coordinates are "
                    f"reproduced by {verdict['source']!r} -- a force-field "
                    "relaxation, not an optimisation")
                nbad += 1

        # the formula has to be the formula of the atoms
        counts: dict = {}
        for a in atoms:
            counts[a[0]] = counts.get(a[0], 0) + 1
        order = sorted(counts, key=lambda s: (0 if s == "C" else
                                              1 if s == "H" else 2, s))
        want = "".join(s + (str(counts[s]) if counts[s] > 1 else "")
                       for s in order)
        got = str(entry.get("formula", "")).replace(" ", "")
        if _canon_formula(got) != _canon_formula(want):
            fail.append(f"library/{name}: formula {got!r} is not the formula of "
                        f"its own atoms ({want})")
            nbad += 1

        # ---- is it a molecule at all? ----
        # This does not depend on provenance, and it is the one check that
        # cannot wait for anyone to ask where the coordinates came from.  The
        # stored methanol put two of its methyl hydrogens 1.0938 A apart --
        # an H-C-H angle of 62 degrees -- and methanethiol carried the same
        # impossible numbers.  Every energy, spectrum and shielding computed
        # from such a geometry describes nothing at all.
        nb: dict = {i: [] for i in range(len(atoms))}
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                if math.dist(atoms[i][1:], atoms[j][1:]) <= _bond_cutoff(
                        atoms[i][0], atoms[j][0]):
                    nb[i].append(j)
                    nb[j].append(i)

        # The hydrogens have to be on the atoms the SMILES puts them on.  The
        # angle test below only sees bonds the perception above found, so an
        # atom displaced far enough stops being counted as bonded at all and
        # the angle it destroyed is no longer measured -- which is exactly
        # what a first attempt at this section missed.
        smiles = str(entry.get("smiles") or "")
        if _Chem is not None and smiles:
            ref = _Chem.AddHs(_Chem.MolFromSmiles(smiles))
            if ref is not None:
                want_h = sorted(
                    (a.GetSymbol(),
                     sum(1 for x in a.GetNeighbors() if x.GetSymbol() == "H"))
                    for a in ref.GetAtoms() if a.GetSymbol() != "H")
                have_h = sorted(
                    (atoms[i][0], sum(1 for j in nb[i] if atoms[j][0] == "H"))
                    for i in range(len(atoms)) if atoms[i][0] != "H")
                if have_h != want_h:
                    fail.append(
                        f"library/{name}: its coordinates give "
                        f"{have_h}, but the SMILES {smiles!r} names a molecule "
                        f"with {want_h}.  Whatever is stored here is not that "
                        "molecule, so nothing computed from it is either")
                    nbad += 1

                # One piece or several, as the SMILES says.  Water_dimer is
                # two -- that is what a hydrogen bond is -- but dihydrogen is
                # one, and it took fixing the H-H bond cutoff to see it: the
                # radii gave 0.713 A against a bond of 0.743 A, so the program
                # reported the H2 entry as two unconnected atoms.
                want_frags = len(_Chem.GetMolFrags(ref))
                seen, nfrags = {0}, 1
                stack = [0]
                while stack:
                    i = stack.pop()
                    for j in nb[i]:
                        if j not in seen:
                            seen.add(j)
                            stack.append(j)
                for start in range(len(atoms)):
                    if start in seen:
                        continue
                    nfrags += 1
                    seen.add(start)
                    stack = [start]
                    while stack:
                        i = stack.pop()
                        for j in nb[i]:
                            if j not in seen:
                                seen.add(j)
                                stack.append(j)
                if nfrags != want_frags:
                    fail.append(
                        f"library/{name}: its coordinates fall into {nfrags} "
                        f"piece(s) but the SMILES {smiles!r} names "
                        f"{want_frags}.  A structure that is not connected the "
                        "way its own name says is not that molecule")
                    nbad += 1

        if len(atoms) > 2:
            tightest, where = 180.0, ""
            for i, lst in nb.items():
                if len(lst) < 2:
                    continue
                for a in range(len(lst)):
                    for b in range(a + 1, len(lst)):
                        v1 = np.array(atoms[lst[a]][1:]) - np.array(atoms[i][1:])
                        v2 = np.array(atoms[lst[b]][1:]) - np.array(atoms[i][1:])
                        cosang = float(np.dot(v1, v2)
                                       / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                        ang = math.degrees(
                            math.acos(max(-1.0, min(1.0, cosang))))
                        if ang < tightest:
                            tightest, where = ang, f"{atoms[i][0]}{i}"
            if tightest < MIN_BOND_ANGLE:
                fail.append(
                    f"library/{name}: its tightest bond angle is "
                    f"{tightest:.1f} deg at {where}, sharper than any stable "
                    f"structure in this library ({MIN_BOND_ANGLE:g} deg).  "
                    "Coordinates like these are not a molecule and every "
                    "number computed from them describes nothing")
                nbad += 1

        # ---- the description must not claim a geometry it does not have ----
        desc = _entry_desc(entry)
        if not desc:
            continue

        angles = [(float(x), len(x.split(".")[1]) if "." in x else 0)
                  for x in _re.findall(
                      r"(\d+\.?\d*)\s*(?:deg|degree|°)", desc)]
        dists = [(float(x), len(x.split(".")[1]))
                 for x in _re.findall(
                     r"(\d+\.\d+)\s*(?:A\b|angstrom|Å)", desc)]
        if not angles and not dists:
            continue

        # The tolerance is the precision the description quotes, not a fixed
        # window: a number written to one decimal has to be right to one
        # decimal.  A flat 0.6 deg window was tried first and it passed the
        # very defect this section exists for -- water's "~104.5 deg" against
        # a geometry that is 104.00 -- because 0.5 < 0.6.  An approximate
        # number is not exempt either: "~104.5" for 104.00 is wrong by ten
        # times the precision it was written to.
        def _tol(decimals: int) -> float:
            return 0.5 * 10.0 ** (-decimals) + 1e-4

        have_ang = []
        n = len(atoms)
        for a in range(n):
            nb = [k for k in range(n) if k != a
                  and 0.4 < math.dist(atoms[a][1:], atoms[k][1:]) <= 2.0]
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    v1 = np.array(atoms[nb[i]][1:]) - np.array(atoms[a][1:])
                    v2 = np.array(atoms[nb[j]][1:]) - np.array(atoms[a][1:])
                    cosang = float(np.dot(v1, v2)
                                   / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                    have_ang.append(math.degrees(
                        math.acos(max(-1.0, min(1.0, cosang)))))
        have_dist = [math.dist(atoms[i][1:], atoms[j][1:])
                     for i in range(n) for j in range(i + 1, n)]

        for claim, dp in angles:
            if not any(abs(claim - h) <= _tol(dp) for h in have_ang):
                near = min((abs(claim - h), h) for h in have_ang)
                fail.append(
                    f"library/{name}: the description claims {claim:g} deg to "
                    f"{dp} decimal(s), which is not any angle in its geometry "
                    f"(nearest {near[1]:.4f} deg, off by {near[0]:.4f})")
                nbad += 1
        for claim, dp in dists:
            if not any(abs(claim - h) <= _tol(dp) for h in have_dist):
                near = min((abs(claim - h), h) for h in have_dist)
                fail.append(
                    f"library/{name}: the description claims {claim:g} A to "
                    f"{dp} decimal(s), which is not any interatomic distance "
                    f"in its geometry (nearest {near[1]:.4f} A, off by "
                    f"{near[0]:.4f})")
                nbad += 1

    # ---- loading an entry must report what the entry records ----
    # from_name used to stamp one string on all 62 entries, so "what geometry
    # were these numbers computed on?" had the same answer for a force-field
    # conformer and for a structure relaxed here at B3LYP/6-31G*.
    for name in sorted(lib):
        try:
            mol = _from_name(name)
        except Exception as exc:                             # noqa: BLE001
            fail.append(f"library/{name}: from_name raises "
                        f"{type(exc).__name__}: {exc}")
            nbad += 1
            continue
        recorded = str(lib[name].get("geometry_source") or "")
        if mol.geometry_source != recorded:
            fail.append(
                f"library/{name}: the entry records {recorded[:50]!r} but "
                f"loading it reports {mol.geometry_source[:50]!r}.  A user is "
                "told one thing and the file says another")
            nbad += 1

    # ---- a force field that was never applied must not be credited ----
    # MMFFOptimizeMolecule reports "no parameters for this molecule" by
    # returning -1, which a try/except cannot see, so boron hydrides and the
    # diatomics went out labelled MMFF94 when no MMFF94 relaxation had
    # happened -- the coordinates were the raw embedding.  The test asks RDKit
    # itself, independently of what the module decided to say.
    if _Chem is not None:
        for smi, label in (("B", "borane"), ("[HH]", "dihydrogen"),
                           ("O=O", "dioxygen"), ("CCO", "ethanol")):
            built = _Chem.AddHs(_Chem.MolFromSmiles(smi))
            has_mmff = (built is not None
                        and _AllChem.MMFFGetMoleculeProperties(built) is not None)
            got = _from_smiles(smi).geometry_source
            if not has_mmff and got.startswith("MMFF94"):
                fail.append(
                    f"from_smiles({label!r}) reports {got!r}, but MMFF94 has "
                    "no parameters for this molecule, so no MMFF94 relaxation "
                    "took place -- the coordinates come from somewhere else "
                    "and the label hides it")
                nbad += 1
            elif has_mmff and not got.startswith("MMFF94"):
                fail.append(
                    f"from_smiles({label!r}) reports {got!r} although MMFF94 "
                    "has parameters for this molecule and was applied")
                nbad += 1

    return nbad


def _canon_formula(text: str) -> str:
    """A formula reduced to its element counts, order-independent."""
    counts: dict = {}
    for sym, num in _re.findall(r"([A-Z][a-z]?)(\d*)", text):
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
    return "".join(f"{k}{counts[k]}" for k in sorted(counts))


def _pinned_sections(src: str) -> set:
    """The contract sections a harness source refers to.

    A call, not a mention: ``contract.raman_sections(fail)`` counts, and the
    prose in ``mutate_raman``'s docstring that names the same function without
    parentheses does not.  Split out from ``harness_sections`` so it can be
    falsified directly rather than only through the files on disk.
    """
    return set(_re.findall(r"\bcontract\.([a-z_]+_sections)\(", src))


def harness_sections(fail: list) -> int:
    """Every assertion a mutation harness pins must also run in this gate.

    A mutation harness proves that an assertion *can* fail.  It does not prove
    that the assertion runs anywhere in production.  When a harness calls a
    section that ``main`` never calls, the harness is pinning a check that
    nothing else performs -- and the coverage it reports is coverage of a check
    that does not happen.

    This was real in the NMR harness, one layer out: two mutations reported
    FIRED NOTHING because the assertions guarding them lived in
    ``check_shift_scales`` and the harness had never called it.  Wiring it in
    fixed those two.  This assertion is what keeps the next one from drifting
    in silently, and it costs no SCF.

    Read from source rather than from a hand-kept list of names: a list is one
    more thing that can be correct in the file and wrong in fact, which is the
    failure this whole round is about.
    """
    here = _os.path.dirname(_os.path.abspath(__file__))
    try:
        with open(_os.path.join(here, "contract.py"), encoding="utf-8") as fh:
            gate_src = fh.read()
    except OSError as exc:
        fail.append(f"harness: cannot read contract.py ({exc}), so nothing "
                    "below was compared against this gate")
        return 0

    m = _re.search(r"^def main\(\).*?(?=^def |\Z)", gate_src, _re.S | _re.M)
    gated = set(_re.findall(r"\b([a-z_]+_sections)\(",
                            m.group(0) if m else ""))

    pinned_total = 0
    for harness in ("mutate_nmr", "mutate_raman", "mutate_uv",
                    "mutate_library", "mutate_planner"):
        path = _os.path.join(here, harness + ".py")
        if not _os.path.exists(path):
            fail.append(f"harness: {harness}.py is named here but does not "
                        "exist, so whatever it was pinning is unchecked")
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
        except OSError as exc:
            fail.append(f"harness: cannot read {harness}.py ({exc})")
            continue
        # The whole file, not a function named run_checks.  Assuming that name
        # was this check's first version, and two of the five harnesses do not
        # use it -- so it reported them as broken when they were fine.  That is
        # the same mistake as the one this round keeps finding, made by the
        # check that was written to catch it.
        pinned = _pinned_sections(src)
        pinned_total += len(pinned)
        for name in sorted(pinned - gated):
            fail.append(
                f"harness: {harness} pins contract.{name}, which main() never "
                "calls -- so that harness is testing a check no gate performs")

    # Three ways this check could be vacuous, each with a known answer.  A
    # check that always passes and a check that never runs print the same
    # thing, so the ability to fail is asserted rather than assumed.
    if _pinned_sections("contract.uv_scale_sections(fail)") != \
            {"uv_scale_sections"}:
        fail.append("harness: the extractor does not see a call it is looking "
                    "straight at, so every comparison above is vacuous")
    if _pinned_sections("    # contract.uv_scale_sections is named here"):
        fail.append("harness: the extractor matches a mention in prose, so a "
                    "comment about a section reads as a call to it")
    if "nmr_sections" not in gated:
        fail.append("harness: no sections were found in main() at all, so this "
                    "check would accept anything -- the gate-side extraction "
                    "is broken, not the harnesses")

    print(f"  {pinned_total} section(s) are pinned by a mutation harness and "
          f"every one of them also runs in this gate")
    return pinned_total


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    BASE = f"http://127.0.0.1:{args.port}"

    print(f"=== ChatDFT result-contract audit -> {BASE} ===\n")

    # ------------------------------------------------------------------
    #  the molecule library, before anything is computed from it
    # ------------------------------------------------------------------
    # It needs no server and it is what every other section is built on: a
    # library entry with a description that contradicts its own coordinates
    # puts a wrong number in the user's hands before any calculation runs.
    nbad = library_sections(FAILURES)
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] every library entry is usable "
          f"and its description matches its own geometry", flush=True)

    # closed shell
    run("single_point", "water")
    # open shell (different code path in _harvest)
    run("single_point", "methyl_radical")
    # charged
    run("single_point", "hydroxide")
    # triplet
    o2 = run("single_point", "oxygen", multiplicity=3)
    # larger molecule
    run("single_point", "naphthalene")

    # ------------------------------------------------------------------
    #  spin contamination
    # ------------------------------------------------------------------
    # A single unrestricted determinant is not an eigenfunction of S^2, and
    # the SCF can also land on the wrong state.  <S^2> is the number a paper
    # quotes to show an open-shell calculation *is* the spin state it claims,
    # and it used to appear nowhere in the payload: a clean triplet and a
    # broken-symmetry mess produced the same result.
    nbad = 0
    sp = (o2 or {}).get("spin") or {}
    for k in ("s_squared", "s_squared_expected", "contamination",
              "contamination_pct", "multiplicity"):
        if not isinstance(sp.get(k), (int, float)):
            FAILURES.append(f"spin/O2: {k} is {sp.get(k)!r}, expected a number")
            nbad += 1
    if isinstance(sp.get("s_squared"), (int, float)):
        # O2 is a textbook triplet: <S^2> = S(S+1) = 2 minus a small
        # contamination.  A wrong spin state, or a closed-shell determinant
        # mislabelled as a triplet, does not land anywhere near 2.
        if abs(sp["s_squared"] - 2.0) > 0.25:
            FAILURES.append(
                f"spin/O2: <S^2> = {sp['s_squared']}, expected ~2.0 for a "
                f"triplet (S(S+1) with S = 1)")
            nbad += 1
        if sp.get("contamination_pct") is not None \
                and sp["contamination_pct"] < 0.0:
            FAILURES.append(
                f"spin/O2: contamination is {sp['contamination_pct']}%; a "
                f"single determinant can only push <S^2> above the exact "
                f"value, never below it")
            nbad += 1
    if (o2 or {}).get("spin") is None:
        FAILURES.append("spin/O2: no spin block for an open-shell calculation")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] an open-shell result reports "
          f"<S^2> so its spin state can be checked", flush=True)

    # The direction that gives the block above its meaning: a wavefunction
    # that is *not* the spin state it claims has to be called out.  A triplet
    # CH4 stretched to r(C-H) = 2.08 A is contaminated by 30% (<S^2> = 2.60
    # against 2.0) because the alpha and beta orbitals localise differently
    # once the bonds are pulled apart -- the mechanism behind every
    # unrestricted-determinant failure, and cheap enough to reproduce here.
    nbad = 0
    try:
        from backend.engine.dft import DFTEngine

        stretched = ("5\nCH4\nC 0 0 0\nH 1.200 1.200 1.200\n"
                     "H -1.200 -1.200 1.200\nH -1.200 1.200 -1.200\n"
                     "H 1.200 -1.200 -1.200\n")
        bad = DFTEngine(atom_xyz=stretched, charge=0, multiplicity=3,
                        functional="b3lyp", basis="6-31g*").run_scf()
        bsp = bad.get("spin") or {}
        if not isinstance(bsp.get("contamination_pct"), (int, float)) \
                or bsp["contamination_pct"] < 10.0:
            FAILURES.append(
                f"spin/contaminated: the stretched triplet reports "
                f"{bsp.get('contamination_pct')}% contamination, expected "
                f"well past 10% (this case measures 30%); the check has no "
                f"contaminated case to check")
            nbad += 1
        if not any("spin state" in str(w).lower()
                   for w in (bad.get("warnings") or [])):
            FAILURES.append(
                "spin/contaminated: a 30%-contaminated wavefunction carries "
                "no warning, so its energies and spin density will be read as "
                "those of a pure triplet")
            nbad += 1
    except Exception as exc:                             # noqa: BLE001
        FAILURES.append(f"spin/contaminated: the check could not run: {exc}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] a wavefunction that is not the "
          f"spin state it claims says so", flush=True)

    # ------------------------------------------------------------------
    #  solvation: labelled, and actually applied
    # ------------------------------------------------------------------
    # A solvated number and a gas-phase number are different quantities with
    # the same name -- water's dipole goes from 2.07 D to 2.27 D and its
    # energy drops 4.6 kcal/mol at eps = 78.4 -- and nothing in the result
    # said which one it was holding.
    nbad = 0
    gas = run("single_point", "water")
    code, resp = _req("POST", "/api/job", {"molecule": "water",
                                           "kind": "single_point",
                                           "functional": "b3lyp",
                                           "basis": "6-31g*",
                                           "solvation": "78.4"})
    wet = None
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        wet = (_wait(resp["job_id"]).get("result") or {})
    else:
        FAILURES.append(f"solvation: job did not run (HTTP {code})")
        nbad += 1
    if wet:
        if not wet.get("solvation"):
            FAILURES.append(
                "solvation/water: a solvated calculation reports no solvent "
                "in its result, so the number cannot be reproduced")
            nbad += 1
        if not wet.get("solvation_model"):
            FAILURES.append(
                "solvation/water: the result gives a dielectric but not the "
                "continuum model; 'eps = 78.4' alone is not a method")
            nbad += 1
        if gas:
            # the direction that proves the model was really switched on
            d_gas = ((gas.get("dipole") or {}).get("magnitude") or 0.0)
            d_wet = ((wet.get("dipole") or {}).get("magnitude") or 0.0)
            if not d_gas or not d_wet or d_wet <= d_gas * 1.02:
                FAILURES.append(
                    f"solvation/water: the dipole is {d_wet:.4f} D solvated "
                    f"against {d_gas:.4f} D in the gas phase; a polarisable "
                    f"continuum must enhance it (measured +9.5%), so the "
                    f"solvent was labelled but not applied")
                nbad += 1
            de = abs(float(wet.get("energy_hartree") or 0.0)
                     - float(gas.get("energy_hartree") or 0.0))
            if de < 1e-4:
                FAILURES.append(
                    f"solvation/water: the solvated and gas-phase energies "
                    f"differ by {de:.2e} Ha; the continuum contributes "
                    f"several kcal/mol and is not switched on")
                nbad += 1
            # and the gas-phase side must not claim a solvent it never had
            if gas.get("solvation") or gas.get("solvation_model"):
                FAILURES.append(
                    f"solvation/water: the gas-phase run reports solvation="
                    f"{gas.get('solvation')!r}, model="
                    f"{gas.get('solvation_model')!r}; it had none")
                nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] a solvated result names its "
          f"continuum model, and a gas-phase one claims no solvent", flush=True)

    # geometry optimisation: must supply optimized_xyz
    opt_job = run("geometry_optimization", "water", extra_spec=OPT_EXTRA,
                  max_steps=20)
    # "Converged" has to mean the gradient criterion was met, not merely that
    # the optimiser ran.  PySCF's optimize() returns kernel(...)[1] and throws
    # its own convergence flag away, so for a long time the app logged
    # "converged" unconditionally and passed an unrelaxed structure downstream
    # into frequencies, surfaces and barriers.  The flag, the numbers and the
    # criteria all have to be present and agree.
    nbad = 0
    if not isinstance(opt_job, dict):
        FAILURES.append("opt: the geometry optimisation returned nothing")
        nbad += 1
    else:
        crit = opt_job.get("opt_criteria") or {}
        conv = opt_job.get("opt_converged")
        grms, gmax = opt_job.get("opt_grms"), opt_job.get("opt_gmax")
        if conv is not True:
            FAILURES.append(
                f"opt/water: a 20-step optimisation of water reports "
                f"converged={conv}; it is three atoms from its minimum")
            nbad += 1
        for k in ("convergence_grms", "convergence_gmax"):
            if not isinstance(crit.get(k), (int, float)):
                FAILURES.append(f"opt/water: no {k} in the reported criteria")
                nbad += 1
        if grms is not None and isinstance(crit.get("convergence_grms"), float) \
                and grms > crit["convergence_grms"] and conv:
            FAILURES.append(
                f"opt/water: |g|rms {grms:.2e} is above the criterion "
                f"{crit['convergence_grms']:.0e} yet the job claims to have "
                f"converged")
            nbad += 1
        if gmax is not None and grms is not None and gmax < grms:
            FAILURES.append(
                f"opt/water: |g|max {gmax:.2e} is below |g|rms {grms:.2e}, "
                f"which is impossible")
            nbad += 1
        if not opt_job.get("n_steps"):
            FAILURES.append(
                "opt/water: the optimisation reports no step count; a figure "
                "built on a relaxation has to say how many steps it took")
            nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] an optimisation reports whether "
          f"it actually reached the gradient criterion", flush=True)

    # The opposite direction, which is the only thing that makes the check
    # above mean anything: a relaxation that was not allowed to finish has to
    # say so.  Pass on this and a flag that is simply always True would
    # satisfy every assertion in the block above.
    nbad = 0
    try:
        from backend.engine.dft import DFTEngine

        starved_xyz = "3\nwater\nO 0 0 0\nH 1.26 0 0.48\nH -1.26 0 0.48\n"
        eng = DFTEngine(atom_xyz=starved_xyz, charge=0, multiplicity=1,
                        functional="b3lyp", basis="6-31g*")
        starved = eng.optimize(max_steps=2)
        if starved.get("opt_converged") is not False:
            FAILURES.append(
                f"opt/starved: a relaxation stopped after 2 steps, far from "
                f"the minimum (|g|rms {starved.get('opt_grms')}), reports "
                f"converged={starved.get('opt_converged')} -- the flag is not "
                f"reading the gradient")
            nbad += 1
        if not (starved.get("warnings") or []):
            FAILURES.append(
                "opt/starved: an unconverged optimisation carries no warning, "
                "so the structure will be used as if it were a minimum")
            nbad += 1
        # and it must still be a number, not a blank
        if not isinstance(starved.get("opt_grms"), (int, float)):
            FAILURES.append("opt/starved: no residual gradient reported")
            nbad += 1
    except Exception as exc:                             # noqa: BLE001
        FAILURES.append(f"opt/starved: the check could not run: {exc}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] a relaxation that was cut short "
          f"says so instead of calling itself converged", flush=True)

    # TD-DFT: must supply excited_states with the spectrum fields
    run("excited_states", "formaldehyde", extra_spec=EXC_EXTRA, nstates=5,
        require_rows={"excited_states": STATE_KEYS})

    # bond scan: the curve must exist and be internally consistent.  This job
    # type was advertised for a long time with no implementation behind it, so
    # the gate has to assert more than "a result came back".
    code, resp = _req("POST", "/api/job", {"molecule": "water", "kind": "scan",
                                           "functional": "b3lyp", "basis": "6-31g*"})
    nbad = 0
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"])
        sc = (job.get("result") or {}).get("scan")
        if not isinstance(sc, dict):
            FAILURES.append("scan: missing 'scan' block")
            nbad += 1
        else:
            pts = sc.get("points") or []
            if not pts:
                FAILURES.append("scan: no points in the curve")
                nbad += 1
            else:
                check_list_of("scan.points", pts, SCAN_POINT_KEYS)
                kmin = sc.get("minimum_index")
                if not isinstance(kmin, int) or not (0 <= kmin < len(pts)):
                    FAILURES.append(f"scan: minimum_index {kmin} is out of range")
                    nbad += 1
                else:
                    if pts[kmin]["energy_hartree"] != min(
                            p["energy_hartree"] for p in pts):
                        FAILURES.append("scan: minimum_index is not the lowest point")
                        nbad += 1
                    if abs(sc.get("minimum", {}).get("r", -1.0)
                           - pts[kmin]["r"]) > 1e-3:
                        FAILURES.append(
                            "scan: minimum.r disagrees with points[minimum_index].r")
                        nbad += 1
            # A scan is rigid: every other nucleus is frozen while this bond
            # is stretched.  Saying so is what keeps the minimum from being
            # quoted as an equilibrium bond length.
            if sc.get("rigid") is not True:
                FAILURES.append(
                    f"scan: 'rigid' is {sc.get('rigid')!r}; the curve is only "
                    f"interpretable if the payload says the rest of the "
                    f"molecule was frozen")
                nbad += 1
        sres = job.get("result") or {}
        if not isinstance(sres.get("scan_seconds"), (int, float)):
            FAILURES.append(
                f"scan: scan_seconds is {sres.get('scan_seconds')!r}; a "
                f"reported duration has to be measured, and it was "
                f"hardcoded to None")
            nbad += 1
    else:
        FAILURES.append(f"scan: submit HTTP {code}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] scan/water curve", flush=True)

    # ------------------------------------------------------------------
    #  where the well actually bottoms out
    # ------------------------------------------------------------------
    # The minimum of a grid is a grid point, so its error is set by where the
    # grid happens to sit: refining the grid does not steadily improve it
    # (against a fine reference on water's O-H well it is off by +0.0115 A at
    # 8 points, +0.0040 at 12, +0.0005 at 16, -0.0091 at 25, -0.0054 at 40 --
    # worse at 25 than at 16).  Quoting it to three decimals, as the narration
    # used to, claims a precision the scan cannot support.
    #
    # The assertion is therefore *not* "the fit beats the grid here": when the
    # grid lands on the bottom it can win, and an earlier version of this
    # check failed for exactly that reason.  The claim is about stability --
    # slide the grid by half a step and the sampled minimum jumps (measured
    # 0.0264 A at 12 points, 0.0121 at 25) while the parabola vertex does not
    # (0.0003 and 0.0005 A).  That is what makes the fitted value quotable.
    nbad = 0
    try:
        from backend.engine.dft import scan_bond
        from backend.engine import molecule as molmod

        xyz = molmod.from_name("water").to_xyz()

        def _scan(steps, lo=0.85, hi=1.45):
            return scan_bond(xyz, 0, 1, functional="b3lyp", basis="6-31g*",
                             steps=steps, lo_scale=lo, hi_scale=hi)["scan"]

        fine = _scan(51, 0.95, 1.05)
        r_ref = ((fine.get("minimum_fitted") or {}).get("r")
                 or fine["minimum"]["r"])
        for steps in (12, 25):
            gs, fs = [], []
            for shift in (0.0, 0.5):
                d = (1.45 - 0.85) / (steps - 1) * shift
                sc = _scan(steps, 0.85 + d, 1.45 + d)
                grid_r = sc["minimum"]["r"]
                fit = sc.get("minimum_fitted") or {}
                if not isinstance(fit.get("r"), (int, float)):
                    FAILURES.append(
                        f"scan/fit({steps}): no fitted minimum in the payload, "
                        f"so the curve offers a grid point as the bottom of "
                        f"the well")
                    nbad += 1
                    continue
                if fit.get("energy_hartree") is not None \
                        and fit["energy_hartree"] \
                        > sc["minimum"]["energy_hartree"]:
                    FAILURES.append(
                        f"scan/fit({steps}): the fitted minimum sits above the "
                        f"lowest sampled point, which no parabola through "
                        f"three of them can do")
                    nbad += 1
                if abs(fit["r"] - grid_r) > (sc.get("grid_step") or 0.0) + 1e-9:
                    FAILURES.append(
                        f"scan/fit({steps}): the vertex ({fit['r']} A) is "
                        f"further from the lowest sampled point ({grid_r} A) "
                        f"than one grid step")
                    nbad += 1
                if abs(fit["r"] - r_ref) > 0.01:
                    FAILURES.append(
                        f"scan/fit({steps}): the fitted minimum is "
                        f"{fit['r']} A against a reference of {r_ref} A, off "
                        f"by {abs(fit['r'] - r_ref):.4f} A (tolerance 0.01)")
                    nbad += 1
                gs.append(grid_r)
                fs.append(fit["r"])
            if len(gs) == 2:
                spread_g = abs(gs[1] - gs[0])
                spread_f = abs(fs[1] - fs[0])
                if spread_f > 0.1 * spread_g + 1e-6:
                    FAILURES.append(
                        f"scan/fit({steps}): sliding the grid by half a step "
                        f"moves the fitted minimum by {spread_f:.4f} A and the "
                        f"sampled minimum by {spread_g:.4f} A -- the fit is "
                        f"supposed to be the estimate that does not depend on "
                        f"where the grid sits")
                    nbad += 1
        # The other way to get a meaningless minimum: scan a window that never
        # brackets the bottom at all.  Every point compressed means the lowest
        # one is the last one, and reporting its r as a minimum is wrong
        # however precisely it is quoted.
        narrow = scan_bond(xyz, 0, 1, functional="b3lyp", basis="6-31g*",
                           steps=9, lo_scale=0.85, hi_scale=0.95)
        npts = len(narrow["scan"]["points"])
        if narrow["scan"]["minimum_index"] != npts - 1:
            FAILURES.append(
                f"scan/edge: the compressed window's lowest point is index "
                f"{narrow['scan']['minimum_index']}, expected {npts - 1} -- "
                f"the check has no unbracketed case to check")
            nbad += 1
        if not any("no minimum was actually bracketed" in str(w)
                   for w in (narrow.get("warnings") or [])):
            FAILURES.append(
                "scan/edge: a scan whose lowest point is its endpoint carries "
                "no warning, so an endpoint will be read as a minimum")
            nbad += 1
    except Exception as exc:                             # noqa: BLE001
        FAILURES.append(f"scan/fit: the check could not run: {exc}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] a scan reports where the well "
          f"bottoms out, not just its lowest grid point", flush=True)

    # comparison job: result is {a: ..., b: ...}
    code, resp = _req("POST", "/api/chat", {"message": "Compare water and ammonia"})
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"])
        if job.get("status") == "completed":
            res = job.get("result") or {}
            for side in ("a", "b"):
                if side not in res:
                    FAILURES.append(f"compare: missing side '{side}'")
                else:
                    check_fields(f"compare.{side}", res[side], COMMON)
            if not job.get("explanation"):
                FAILURES.append("compare.job: missing 'explanation'")
            # the comparison must actually be narrated: the analysis used to be
            # a one-line "Comparison complete" placeholder while the real
            # explain_compare() sat unreachable in the planner.
            text = job.get("explanation") or ""
            for phrase, why in (
                ("HOMO-LUMO gap", "the gap comparison is missing"),
                ("Water", "the first molecule is not named"),
                ("Ammonia", "the second molecule is not named"),
            ):
                if phrase not in text:
                    FAILURES.append(f"compare narration: {why}")
            if len(text) < 200:
                FAILURES.append(
                    f"compare narration is a stub ({len(text)} chars)")
            nbad = len([f for f in FAILURES if f.startswith("compare")])
            print(f"[{'PASS' if nbad == 0 else 'FAIL'}] compare/water+ammonia", flush=True)
        else:
            FAILURES.append(f"compare: job {job.get('status')}")
    else:
        FAILURES.append(f"compare: chat submit HTTP {code}")

    # the narrative the UI shows must be a real sentence, not a stub
    _, jobs = _req("GET", "/api/jobs")
    if isinstance(jobs, dict) and jobs.get("jobs"):
        short = [j["id"] for j in jobs["jobs"] if j.get("status") == "completed"]
        if short:
            _, full = _req("GET", f"/api/job/{short[0]}")
            if isinstance(full, dict):
                text = full.get("explanation") or ""
                if len(text) < 80:
                    FAILURES.append(f"narration too short ({len(text)} chars)")
                print(f"[{'PASS' if len(text) >= 80 else 'FAIL'}] "
                      f"narrative length = {len(text)} chars", flush=True)

    # the interpretation must stay scientifically honest: a Kohn-Sham gap is
    # not an optical gap, and a large gap does not imply a saturated molecule
    # (benzene and pyridine both have gaps above 6 eV).
    code, resp = _req("POST", "/api/job", {"molecule": "benzene", "kind": "single_point",
                                           "functional": "b3lyp", "basis": "6-31g*"})
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"])
        text = (job.get("explanation") or "").lower()
        for phrase, why in [
            ("not an optical gap", "the KS-gap caveat is missing"),
            ("does not identify the bonding", "the gap is over-interpreted"),
        ]:
            if phrase not in text:
                FAILURES.append(f"narration: {why} ('{phrase}' absent)")
        bad = [p for p in ("saturated molecule -", "small conjugated system")
               if p in text]
        if bad:
            FAILURES.append(f"narration: contains misleading claim(s) {bad}")
        nbad = len([f for f in FAILURES if f.startswith("narration")])
        print(f"[{'PASS' if nbad == 0 else 'FAIL'}] narration scientific honesty",
              flush=True)

    # the narration must name the atom that really carries the most HOMO
    # density.  ``frontier_orbitals`` is ordered by atom index, so taking the
    # first entry is wrong: naphthalene once reported C1 (7.5%) as the reactive
    # site when C3/C5/C8/C10 each carried 17.5% -- a different position, and
    # silent, because the sentence still read plausibly.
    def homo_narration(molecule):
        code, resp = _req("POST", "/api/job", {"molecule": molecule,
                                               "kind": "single_point",
                                               "functional": "b3lyp",
                                               "basis": "6-31g*"})
        if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
            return None, None
        job = _wait(resp["job_id"])
        comp = ((job.get("result") or {}).get("frontier_orbitals") or {}).get("homo") or []
        return job.get("explanation") or "", comp

    text, comp = homo_narration("naphthalene")
    nbad = 0
    if text is None:
        FAILURES.append("narration: naphthalene HOMO job did not run")
        nbad += 1
    else:
        ranked = sorted(comp, key=lambda c: c["percent"], reverse=True)
        if not ranked:
            FAILURES.append("narration: naphthalene HOMO composition is empty")
            nbad += 1
        else:
            label = f"{ranked[0]['symbol']}{ranked[0]['atom']}"
            if label not in text:
                FAILURES.append(
                    f"narration: largest HOMO contributor {label} "
                    f"({ranked[0]['percent']}%) is not named")
                nbad += 1
            if "delocalised" not in text.lower():
                FAILURES.append(
                    "narration: a delocalised HOMO is described as localised")
                nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] narration names the true HOMO site",
          flush=True)

    # a lone-pair HOMO really is localised, so the opposite wording is required
    text, comp = homo_narration("water")
    nbad = 0
    if text is None:
        FAILURES.append("narration: water HOMO job did not run")
        nbad += 1
    elif "concentrated on O1" not in text:
        FAILURES.append("narration: a lone-pair HOMO is not described as localised")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] narration localises a lone-pair HOMO",
          flush=True)

    # Population analysis: the narration must name the atom the numbers point
    # at, and must not describe a charge *separation* as a concentration.  The
    # old wording listed the three largest-magnitude atoms under "concentrates
    # charge on", so water read "concentrates charge on O1 (-0.36), H2 (+0.18),
    # H3 (+0.18)" -- the two ends of the molecular dipole called one site.
    def charge_narration(molecule):
        code, resp = _req("POST", "/api/job", {"molecule": molecule,
                                               "kind": "single_point",
                                               "functional": "b3lyp",
                                               "basis": "6-31g*"})
        if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
            return None, []
        job = _wait(resp["job_id"])
        rows = (job.get("result") or {}).get("mulliken_charges") or []
        return job.get("explanation") or "", rows

    nbad = 0
    for molecule in ("water", "acetone"):
        text, rows = charge_narration(molecule)
        tag = f"charge narration/{molecule}"
        if text is None:
            FAILURES.append(f"{tag}: job did not run")
            nbad += 1
            continue
        if not rows:
            FAILURES.append(f"{tag}: no Mulliken charges returned")
            nbad += 1
            continue
        ext = max(rows, key=lambda r: abs(r["charge"]))
        label = f"{ext['symbol']}{ext['atom']}"
        if label not in text:
            FAILURES.append(
                f"{tag}: largest-magnitude atom {label} "
                f"({ext['charge']:+.3f} e) is not named")
            nbad += 1
        rich = min(r["charge"] for r in rows)
        poor = max(r["charge"] for r in rows)
        if rich <= -0.15 and poor >= 0.15 and "separates charge" not in text:
            FAILURES.append(
                f"{tag}: both ends are charged ({rich:+.3f}/{poor:+.3f}) but "
                f"the text does not call it a separation")
            nbad += 1
        if "concentrates charge" in text:
            FAILURES.append(f"{tag}: still says 'concentrates charge'")
            nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] "
          f"charge narration names the true extrema", flush=True)

    # Frontier-orbital composition must be a real partition of the orbital.
    # It used to be the naive sum of |c|^2 over each atom's AOs, which drops
    # every overlap cross term -- MOs are normalised as c^T S c = 1, not
    # c^T c = 1.  That agreed well enough on HOMOs, so nothing looked wrong,
    # but it inverted water's LUMO: it reported O1 46% / H2 27% / H3 27% for
    # an O-H antibonding orbital whose weight really sits on the hydrogens
    # (H 36% / H 36% / O 27% by Loewdin, or H 53% each with O *negative* by
    # Mulliken, the cross terms being negative in an antibonding orbital).
    def frontier(molecule):
        code, resp = _req("POST", "/api/job", {"molecule": molecule,
                                               "kind": "single_point",
                                               "functional": "b3lyp",
                                               "basis": "6-31g*"})
        if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
            return None
        job = _wait(resp["job_id"])
        return (job.get("result") or {}).get("frontier_orbitals")

    nbad = 0
    for molecule in ("water", "naphthalene", "pyridine"):
        fo = frontier(molecule)
        tag = f"frontier/{molecule}"
        if not fo:
            FAILURES.append(f"{tag}: no frontier composition returned")
            nbad += 1
            continue
        for key in ("homo", "lumo"):
            rows = fo.get(key) or []
            if not rows:
                FAILURES.append(f"{tag}.{key}: empty")
                nbad += 1
                continue
            bad = [r for r in rows
                   if not (0.0 <= float(r.get("percent", -1.0)) <= 100.0)]
            if bad:
                FAILURES.append(
                    f"{tag}.{key}: {len(bad)} contribution(s) outside 0-100% "
                    f"(min {min(float(r['percent']) for r in bad):.2f})")
                nbad += 1
            total = sum(float(r["percent"]) for r in rows)
            if total > 100.5:
                FAILURES.append(
                    f"{tag}.{key}: contributions sum to {total:.1f}% -- "
                    f"not a partition of one orbital")
                nbad += 1
    # The targeted physics check: water's LUMO is O-H antibonding, so its
    # weight belongs on the hydrogens.  The naive scheme named oxygen.
    fo = frontier("water")
    if fo and fo.get("lumo"):
        top = max(fo["lumo"], key=lambda r: r["percent"])
        if top["symbol"] != "H":
            FAILURES.append(
                f"frontier/water.lumo: dominated by {top['symbol']}"
                f"{top['atom']} ({top['percent']:.1f}%), but the LUMO is "
                f"O-H antibonding and its weight lies on the hydrogens")
            nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] "
          f"frontier composition is a real partition", flush=True)

    # the TD-DFT narration must call the *strongest* transition strongest.
    # ``excited_states`` is ordered by energy, so the first bright state is not
    # it: naphthalene's S1 has f = 0.094 while S4 has f = 1.99.
    code, resp = _req("POST", "/api/job", {"molecule": "naphthalene",
                                           "kind": "excited_states",
                                           "functional": "b3lyp",
                                           "basis": "6-31g*", "nstates": 6})
    nbad = 0
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"])
        states = (job.get("result") or {}).get("excited_states") or []
        text = job.get("explanation") or ""
        if not states:
            FAILURES.append("tddft: no excited states returned")
            nbad += 1
        else:
            top = max(states, key=lambda s: s.get("oscillator_strength") or 0.0)
            if f"strongest transition is S{top['state']}" not in text:
                FAILURES.append(
                    f"tddft narration: the strongest transition is S{top['state']} "
                    f"(f = {top.get('oscillator_strength'):.4f}) but the text does "
                    f"not say so")
                nbad += 1
    else:
        FAILURES.append("tddft: excited-state job did not run")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] narration names the strongest transition",
          flush=True)

    # ------------------------------------------------------------------
    #  the UV-Vis figure
    # ------------------------------------------------------------------
    # A stick list of oscillator strengths is not a spectrum.  The server has
    # always built the broadened curve ("the state list alone is not a
    # figure") and for a long time nothing drew it, while the pane promised
    # "the absorption curve".  These assertions are what would have caught
    # that, plus the two ways a spectrum can be quietly wrong: a wavelength
    # that does not match its own energy, and a band shifted by broadening in
    # wavelength space instead of energy space.
    nbad = 0
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        res = job.get("result") or {}
        if not states:
            FAILURES.append("uvvis: no excited states to build a curve from")
            nbad += 1
        for s in states:
            e, lam = s.get("energy_ev"), s.get("wavelength_nm")
            if e and lam and abs(e * lam - 1239.84198) > 0.5:
                FAILURES.append(
                    f"uvvis: S{s.get('state')} is {e:.4f} eV at {lam:.1f} nm, "
                    f"but E*lambda is {e * lam:.1f} and must be 1239.84")
                nbad += 1
            fosc = s.get("oscillator_strength")
            if fosc is not None and fosc < -1e-9:
                FAILURES.append(
                    f"uvvis: S{s.get('state')} has a negative oscillator "
                    f"strength ({fosc:.4g}); a transition can be dark but not "
                    f"negative")
                nbad += 1
        uv = res.get("uv_spectrum") or {}
        xs, ys = uv.get("x") or [], uv.get("y") or []
        if not xs or not ys or len(xs) != len(ys):
            FAILURES.append(
                "uvvis: no broadened absorption curve in the result. A paper "
                "plots a spectrum, not a stick list, so the figure cannot be "
                "drawn from this payload.")
            nbad += 1
        else:
            if abs(max(ys) - 100.0) > 1e-6 or min(ys) < -1e-9:
                FAILURES.append(
                    f"uvvis: the curve spans {min(ys):.3f}..{max(ys):.3f}; a "
                    f"normalised absorbance must run from 0 to 100")
                nbad += 1
            # and the band has to sit where the strongest state is: broadened
            # in wavelength space instead of energy space it slides
            peak = xs[max(range(len(ys)), key=lambda i: ys[i])] if states else 0
            want = 1239.84198 / top["energy_ev"] if states else 0
            if states and (top.get("oscillator_strength") or 0) > 0.05 \
                    and abs(peak - want) > 6.0:
                FAILURES.append(
                    f"uvvis: the curve peaks at {peak:.1f} nm but the strongest "
                    f"state (S{top['state']}, f="
                    f"{top['oscillator_strength']:.3f}) is at {want:.1f} nm -- "
                    f"a broadening artefact, not the spectrum")
                nbad += 1
            # the absolute ordinate and the sum rule have to survive the trip
            # through the API, not just exist inside uv_curve
            eps = uv.get("epsilon") or []
            if len(eps) != len(xs):
                FAILURES.append(
                    f"uvvis: the payload ships {len(xs)} wavelengths but "
                    f"{len(eps)} absolute eps values; the paper-ready ordinate "
                    f"is missing")
                nbad += 1
            srule = uv.get("sum_rule") or {}
            if srule.get("residual_pct") is None:
                FAILURES.append(
                    "uvvis: the payload carries no sum-rule residual, so "
                    "nothing states whether the absolute scale is right")
                nbad += 1
            if not uv.get("window_nm"):
                FAILURES.append(
                    "uvvis: the payload does not report the plotted window, so "
                    "a figure captioned from it cannot be reproduced")
                nbad += 1
            elif uv.get("window_source") == "auto":
                win = uv["window_nm"]
                for s in states:
                    lam = s.get("wavelength_nm")
                    if lam and (s.get("oscillator_strength") or 0) > 0.05 \
                            and not (win[0] <= lam <= win[1]):
                        FAILURES.append(
                            f"uvvis: the auto window {win} excludes bright "
                            f"S{s['state']} at {lam:.1f} nm")
                        nbad += 1
        if not res.get("excited_method"):
            FAILURES.append(
                "uvvis: the result does not say whether it ran TDA or full "
                "TD-DFT, so the figure cannot be reproduced from it")
            nbad += 1
    else:
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] the UV-Vis curve is computed, "
          f"labelled and not shifted by its broadening", flush=True)

    uv_scale_sections(FAILURES)
    raman_sections(FAILURES)
    nmr_sections(FAILURES)
    nmr_scale_sections(FAILURES)

    # ------------------------------------------------------------------
    #  the harnesses that are supposed to pin all of the above
    # ------------------------------------------------------------------
    # Zero SCF, and it is a claim about this file as much as about them: a
    # harness that pins a section nothing runs reports coverage of a check
    # that does not happen, and that reads exactly like coverage.
    harness_sections(FAILURES)

    # ------------------------------------------------------------------
    #  vibrational analysis
    # ------------------------------------------------------------------
    # Three independent physics assertions, each of which failed before:
    #   * the entropy once came out as 388 J/mol/K (a dead `max(i, 1e-40)`
    #     floor on the moments of inertia, which are ~1e-47 kg m^2);
    #   * `hessian.RHF(mf)` on an RKS object silently drops the XC second
    #     derivative and gives 2151/2952/3271 cm-1;
    #   * the symmetry number was never passed in, so S_rot was off by R ln2.
    nbad = 0
    code, resp = _req("POST", "/api/job", {"molecule": "water",
                                           "kind": "vibrations",
                                           "functional": "b3lyp",
                                           "basis": "6-31g*",
                                           "optimize_first": True})
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"])
        vres = job.get("result") or {}
        # Bound to its own name as well: sections far below read this, and
        # a later job assigned to `vres` would silently swap the molecule out
        # from under them.  That is not hypothetical -- the diffuse-basis
        # polarizability section below did exactly that on its first run.
        vib_water = vres
        freq = vres.get("frequencies_cm1") or []
        therm = vres.get("thermochemistry") or {}
        ir = vres.get("ir_spectrum") or {}
        if len(freq) != 3:
            FAILURES.append(
                f"vibrations/water: {len(freq)} modes, expected 3 "
                f"(3N-6 for N = 3)")
            nbad += 1
        if vres.get("n_imaginary"):
            FAILURES.append(
                f"vibrations/water: {vres['n_imaginary']} imaginary mode(s) "
                f"at a verified minimum")
            nbad += 1
        # harmonic B3LYP/6-31G* water: 1711 / 3722 / 3846 cm-1 (unscaled).
        # A 1000 cm-1 window catches a missing XC second derivative, which
        # shifts the bend up by ~440 cm-1 and the stretches down by ~770.
        for f in freq:
            if not (1300.0 <= f <= 4300.0):
                FAILURES.append(
                    f"vibrations/water: frequency {f} cm-1 is outside the "
                    f"plausible window (1300-4300) for this molecule")
                nbad += 1
        if not any(f > 3000.0 for f in freq):
            FAILURES.append("vibrations/water: no O-H stretch above 3000 cm-1")
            nbad += 1
        s = therm.get("entropy_j_mol_k")
        if s is None:
            FAILURES.append("vibrations/water: no entropy returned")
            nbad += 1
        elif abs(s - 188.8) > 3.0:
            FAILURES.append(
                f"vibrations/water: S0 = {s} J/mol/K, measured value is "
                f"188.8 (off by {abs(s - 188.8):.1f})")
            nbad += 1
        if therm.get("symmetry_number") != 2:
            FAILURES.append(
                f"vibrations/water: symmetry number "
                f"{therm.get('symmetry_number')}, expected 2 (C2v)")
            nbad += 1
        # Frequencies are only frequencies at a stationary point, and the
        # pre-optimisation is internal to this job.  If it did not reach one,
        # 3N-6 numbers still arrive and still look like a spectrum, so the
        # result has to carry the verdict itself.
        if vres.get("opt_converged") is not True:
            FAILURES.append(
                f"vibrations/water: the pre-optimisation is reported as "
                f"opt_converged={vres.get('opt_converged')!r}; frequencies "
                f"are only meaningful at a stationary point")
            nbad += 1
        if not isinstance(vres.get("opt_grms"), (int, float)):
            FAILURES.append(
                "vibrations/water: no residual gradient (opt_grms) from the "
                "pre-optimisation, so the convergence claim cannot be judged")
            nbad += 1
        # An IR spectrum is a figure, and a figure has to say what produced
        # it.  This result used to carry frequencies and nothing else -- no
        # functional, no basis, no energy -- so the spectrum could not be
        # captioned from its own payload and the Properties pane went blank
        # beside it.
        for k in ("functional", "basis", "functional_label", "basis_label"):
            if not vres.get(k):
                FAILURES.append(
                    f"vibrations/water: no '{k}' in the result, so the "
                    f"spectrum cannot be captioned with the level of theory "
                    f"that produced it")
                nbad += 1
        if not isinstance(vres.get("energy_hartree"), (int, float)):
            FAILURES.append(
                "vibrations/water: no total energy in the result; the "
                "Hessian was evaluated somewhere, and a figure has to say "
                "where")
            nbad += 1
        # explain_vibrations() builds its "at the X / Y level" from these
        # fields.  They were absent, so every IR narration read "at the  /
        # level" -- a figure with a blank caption -- and nothing failed.
        expl = str(job.get("explanation") or "")
        if vres.get("functional_label") and vres["functional_label"] not in expl:
            FAILURES.append(
                f"vibrations/water: the narration never names "
                f"{vres['functional_label']!r}; it claims a level of theory "
                f"built from fields the result did not carry")
            nbad += 1
        if len(ir.get("x") or []) < 100 or not ir.get("peaks"):
            FAILURES.append(
                f"vibrations/water: IR curve has "
                f"{len(ir.get('x') or [])} points and "
                f"{len(ir.get('peaks') or [])} peaks")
            nbad += 1
        else:
            xs = ir["x"]
            if xs[0] < xs[-1]:
                FAILURES.append(
                    "vibrations/water: IR axis runs low-to-high; the "
                    "convention is high wavenumber on the left")
                nbad += 1

        nbad += raman_payload_sections(vres, FAILURES)
    else:
        FAILURES.append("vibrations: job did not run")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] vibrational analysis is physical",
          flush=True)

    # ------------------------------------------------------------------
    #  the NMR shielding tensor
    # ------------------------------------------------------------------
    # Water is the right test molecule because it is its own 17O reference:
    # if the reference machinery is wired up correctly, water's 17O shift has
    # to come out at exactly 0.00, and no amount of wrongness in the
    # intermediate numbers can fake that.  It also pins the absolute 17O
    # shielding, which is the number the whole 17O scale rests on -- and the
    # one an under-converged SCF gets wrong by 15 ppm (345.0 instead of
    # 329.6) while still reporting `converged = True`.
    nbad = 0
    code, resp = _req("POST", "/api/job", {"molecule": "water",
                                           "kind": "nmr",
                                           "basis": "6-31g*",
                                           "nmr_linewidth_hz": 1.0,
                                           "nmr_spectrometer_mhz": 400.0})
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"], timeout=900.0)
        nres = job.get("result") or {}
        if job.get("status") != "completed":
            FAILURES.append(
                f"nmr/water: job ended {job.get('status')} -- "
                f"{str(job.get('error'))[:200]}")
            nbad += 1
        else:
            nbad += nmr_payload_sections(nres, FAILURES)
            expl = str(job.get("explanation") or "")
            # The narration has to name the level of theory it actually used,
            # because the shielding is RHF whatever functional was requested.
            if "Hartree-Fock" not in expl:
                FAILURES.append(
                    "nmr/water: the narration never says the shielding is at "
                    "Hartree-Fock, so a reader would assume the requested "
                    "functional was used")
                nbad += 1
    else:
        FAILURES.append(f"nmr: job did not run ({code}) {str(resp)[:120]}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] the NMR shielding tensor is "
          f"physical and its level of theory is stated", flush=True)

    # ------------------------------------------------------------------
    #  a nucleus with no reference at all
    # ------------------------------------------------------------------
    # Water has nothing unreferenced, so every assertion above is blind to the
    # common case: thirteen NMR-active elements have no reference in this
    # build (Al, B, Br, Cl, D, Hg, I, Li, Na, Pb, Pt, S, Sn).  Thiophene
    # carries one of them.
    #
    # The defect this exists for was found while looking for a dataset that
    # could make the "unreferenced_elements" assertion fail: that list was
    # built as
    #     {s for s in syms if s in NUCLEI and s in REFERENCE_FOR} - set(refs)
    # -- and the `s in REFERENCE_FOR` term excludes exactly the elements that
    # have no reference *defined*, which is the whole point of the field.  So
    # thiophene's sulfur was dropped without a word, and the list came back
    # empty on the one molecule where it should not have been.
    #
    # The carrier is thiophene, and thiophene is nao = 82 at 6-31G*, which the
    # memory guard prices at about 3.0 GB against the free physical memory *at
    # the moment the gate runs*.  So on a busy machine this check went red for
    # a reason that had nothing to do with the contract -- which is how a gate
    # teaches its reader to ignore it.  The list below is tried in order and
    # the substitution is printed.
    nbad = 0
    refused = []
    tres = None
    carrier = None
    for mol in UNREFERENCED_CARRIERS:
        code, resp = _req("POST", "/api/job", {"molecule": mol,
                                               "kind": "nmr",
                                               "basis": "6-31g*",
                                               "nmr_linewidth_hz": 1.0,
                                               "nmr_spectrometer_mhz": 400.0})
        if code != 200 or not (isinstance(resp, dict) and resp.get("job_id")):
            FAILURES.append(f"nmrmissing: the {mol} job did not run ({code}) "
                            f"{str(resp)[:120]}")
            nbad += 1
            break
        tjob = _wait(resp["job_id"], timeout=900.0)
        if tjob.get("status") == "completed":
            tres = tjob.get("result") or {}
            carrier = mol
            break
        err = str(tjob.get("error") or "")
        # A refusal for cost is not a contract violation, so it moves on to the
        # next carrier.  Any *other* failure is a violation and stops here --
        # otherwise a real defect would be relabelled as a memory limit and
        # then quietly retried on a molecule that does not exhibit it.
        if "physical memory is free" not in err and "budget is" not in err:
            FAILURES.append(
                f"nmrmissing: the {mol} job ended {tjob.get('status')} -- "
                f"{err[:200]}")
            nbad += 1
            break
        refused.append(f"{mol}: {err[:150]}")
    if tres is not None:
        nbad += nmr_unreferenced_sections(tres, FAILURES)
    elif nbad == 0:
        # Every carrier refused for cost.  Reported as a failure rather than
        # skipped, and the message says which of the two it is: a check that
        # quietly stops running is worse than one that says why it could not.
        FAILURES.append(
            "nmrmissing: every carrier for the unreferenced-nucleus check was "
            "refused for memory on this machine -- " + "; ".join(refused)
            + ".  This is an environment limit rather than a contract "
            "violation, but it is reported as a failure because an assertion "
            "that silently stops running is indistinguishable from one that "
            "passes")
        nbad += 1
    # Which molecule carried the check is part of the verdict, not a side
    # note.  Printing it on its own line was tried and did not work: the gate
    # runner elides every line that is not a verdict, so the substitution
    # happened invisibly and a reader of the summary would not know whether
    # this check ran on thiophene or on a substitute.
    used = carrier or "no carrier"
    if carrier and carrier != UNREFERENCED_CARRIERS[0]:
        used += f", substituted for {UNREFERENCED_CARRIERS[0]} refused for memory"
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] a nucleus with no reference is "
          f"named rather than dropped (carrier: {used})", flush=True)

    # ------------------------------------------------------------------
    #  the absolute polarizability, at a basis that can carry one
    # ------------------------------------------------------------------
    # Everything asserted about Raman so far is a *ratio* -- band heights
    # relative to each other, rho, the sum rule.  None of it can see a
    # polarizability that is uniformly too small, which is exactly the failure
    # mode of a basis without diffuse functions: water comes out at 5.13 a.u.
    # at 6-31G* and 9.89 at aug-cc-pVTZ, a factor of two, with every ratio
    # inside the molecule unchanged.
    #
    # So this runs the same molecule at a diffuse basis and compares against
    # the measured value.  alpha_iso of water is 9.6-9.9 a.u. from experiment
    # (1.42-1.47 A^3); B3LYP/aug-cc-pVDZ is known to come in slightly low at
    # 9.47, which is why the window below is 9.0-10.5 rather than the
    # experimental range itself.
    nbad = 0
    code, resp = _req("POST", "/api/job", {"molecule": "water",
                                           "kind": "vibrations",
                                           "functional": "b3lyp",
                                           "basis": "aug-cc-pvdz",
                                           "with_raman": True})
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"], timeout=900.0)
        avres = job.get("result") or {}
        ram = avres.get("raman") or {}
        iso = ram.get("alpha_iso_au")
        if iso is None:
            FAILURES.append(
                "raman/abs: no alpha_iso at aug-cc-pVDZ, so the absolute "
                "polarizability is never checked")
            nbad += 1
        elif not (9.0 <= float(iso) <= 10.5):
            FAILURES.append(
                f"raman/abs: alpha_iso of water at B3LYP/aug-cc-pVDZ is "
                f"{iso} a.u.; the measured value is 9.6-9.9 and the "
                f"converged B3LYP value 9.5-9.9, so a number outside 9.0-10.5 "
                f"means the polarizability kernel is wrong, not the basis")
            nbad += 1
        # The tensor has to be symmetric and positive definite: alpha is a
        # response function, so a negative eigenvalue would mean the molecule
        # is destabilised by a field, which is unphysical for a closed-shell
        # ground state.
        ten = ram.get("alpha_tensor_au")
        if ten:
            a = np.asarray(ten, dtype=float)
            if np.abs(a - a.T).max() > 1e-6:
                FAILURES.append(
                    f"raman/abs: the polarizability tensor is not symmetric "
                    f"(max asymmetry {np.abs(a - a.T).max():.2e})")
                nbad += 1
            ev = np.linalg.eigvalsh(0.5 * (a + a.T))
            if ev.min() <= 0.0:
                FAILURES.append(
                    f"raman/abs: the polarizability tensor has a "
                    f"non-positive eigenvalue ({ev.min():.4f}), which would "
                    f"mean a field destabilises the molecule")
                nbad += 1
            # Water is not spherical: the out-of-plane component is the
            # smallest, and by a few per cent.  A tensor that came back
            # isotropic would mean the three directions were never
            # distinguished, which a symmetrisation bug does.
            diag = np.diag(a)
            if not (diag.max() / max(diag.min(), 1e-9) > 1.02):
                FAILURES.append(
                    f"raman/abs: the water tensor came back isotropic "
                    f"({diag}); the out-of-plane component is measurably the "
                    f"smallest")
                nbad += 1
        if ram.get("alpha_reliable") is not True:
            FAILURES.append(
                f"raman/abs: alpha_reliable is {ram.get('alpha_reliable')!r} "
                f"at aug-cc-pVDZ, which does have diffuse functions")
            nbad += 1
        if ram.get("basis_warning"):
            FAILURES.append(
                "raman/abs: a basis warning was raised at aug-cc-pVDZ, which "
                "does have diffuse functions -- the warning would then be "
                "noise rather than a signal")
            nbad += 1
        # And the spectrum has to be drawn from the diffuse basis too.
        sp = avres.get("raman_spectrum") or {}
        if len(sp.get("peaks") or []) != len(avres.get("frequencies_cm1") or []):
            FAILURES.append(
                f"raman/abs: {len(sp.get('peaks') or [])} Raman peaks for "
                f"{len(avres.get('frequencies_cm1') or [])} modes")
            nbad += 1
        # rho at the diffuse basis should be near the measured gas-phase
        # values: the symmetric stretch strongly polarised (measured ~0.06)
        # and the asymmetric stretch at the 3/4 limit.
        act = [float(v) for v in (ram.get("activities_a4_amu") or [])]
        rho = [float(v) for v in (ram.get("depolarization") or [])]
        freq = avres.get("frequencies_cm1") or []
        if act and rho and len(act) == len(rho) == len(freq):
            top = max(range(len(act)), key=lambda i: act[i])
            if freq[top] < 3400.0:
                FAILURES.append(
                    f"raman/abs: the strongest Raman line is at {freq[top]} "
                    f"cm-1, not an O-H stretch")
                nbad += 1
            if rho[top] > 0.35:
                FAILURES.append(
                    f"raman/abs: the strongest Raman line has rho = "
                    f"{rho[top]:.3f}; water's symmetric stretch is strongly "
                    f"polarised (measured ~0.06, so a value near 0.75 would "
                    f"mean the isotropic and anisotropic parts were swapped)")
                nbad += 1
            if not any(abs(r - 0.75) < 5e-3 for r in rho):
                FAILURES.append(
                    f"raman/abs: no mode at rho = 0.750; the B2 asymmetric "
                    f"stretch has no isotropic derivative. Got {rho}")
                nbad += 1
    else:
        FAILURES.append("raman/abs: the diffuse-basis job did not run")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] the absolute polarizability "
          f"matches experiment at a diffuse basis", flush=True)

    # ------------------------------------------------------------------
    #  a Raman figure that is not drawn has to say why
    # ------------------------------------------------------------------
    # Everything above tests the case where Raman works.  The case where it
    # does not is the one that silently produces a blank panel: the response
    # solve is skipped for a big molecule or an open-shell one, and unless the
    # result carries the reason, the user sees a vibrational analysis with an
    # IR spectrum and no explanation for the missing Raman one.
    #
    # Driven through ``raman_max_atoms`` rather than by building a molecule
    # with 17 atoms, because the branch taken is identical and this runs in
    # seconds.  sto-3g keeps the Hessian cheap; nothing here depends on the
    # frequencies.
    nbad = 0
    try:
        from backend.engine.analysis import (attach_vibrational_spectra,
                                             vibrations as _vib)
        from backend.engine.dft import DFTEngine

        res = _vib(
            DFTEngine(atom_xyz="3\nwater\nO 0.0 0.0 0.119\n"
                               "H 0.0 0.763 -0.477\nH 0.0 -0.763 -0.477\n",
                      charge=0, multiplicity=1, functional="b3lyp",
                      basis="sto-3g"),
            optimize_first=False, with_raman=True, raman_max_atoms=1,
        )
        # The curves are built by this call on the server side; doing it here
        # too keeps the assertion about what the API actually returns rather
        # than about an intermediate.
        attach_vibrational_spectra(res)
        ram = res.get("raman")
        if not isinstance(ram, dict):
            FAILURES.append(
                "raman/skip: the vibrational analysis dropped the Raman key "
                "entirely when it skipped the response solve, so the UI has "
                "nothing to explain the empty panel with")
            nbad += 1
        elif ram.get("available") is not False:
            FAILURES.append(
                f"raman/skip: with raman_max_atoms=1 on water the Raman "
                f"section reports {ram!r} instead of available=False")
            nbad += 1
        elif not str(ram.get("reason") or "").strip():
            FAILURES.append(
                "raman/skip: the Raman section is marked unavailable but "
                "carries no reason")
            nbad += 1
        elif "atom" not in str(ram["reason"]).lower():
            FAILURES.append(
                f"raman/skip: the stated reason does not mention the atom "
                f"count it was skipped for: {ram['reason']!r}")
            nbad += 1
        # And the IR spectrum must still be there: skipping Raman is not
        # allowed to take the vibrational analysis down with it.
        if not (res.get("ir_spectrum") or {}).get("peaks"):
            FAILURES.append(
                "raman/skip: skipping the Raman response also lost the IR "
                "spectrum")
            nbad += 1
        if res.get("raman_spectrum"):
            FAILURES.append(
                "raman/skip: a Raman curve was produced even though the "
                "section is marked unavailable")
            nbad += 1
    except Exception as exc:                                  # noqa: BLE001
        FAILURES.append(f"raman/skip: the check itself raised {exc!r}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] a skipped Raman figure says "
          f"why it is missing", flush=True)

    # ------------------------------------------------------------------
    #  vibrational analysis on a geometry that was never relaxed
    # ------------------------------------------------------------------
    # The block above asserts `opt_converged is True`, which a flag that is
    # simply always True would satisfy.  This is the opposite direction: run
    # the same code path with a pre-optimisation that is stopped two steps
    # from a deliberately strained start, and check the non-convergence
    # reaches the frequency result instead of dying on a discarded engine.
    nbad = 0
    try:
        from backend.engine.analysis import vibrations
        from backend.engine.dft import DFTEngine

        class _Starved(DFTEngine):
            """The real optimiser, only not allowed to finish."""

            def optimize(self, max_steps: int = 40, progress=None) -> dict:
                return super().optimize(max_steps=2, progress=progress)

        starved_xyz = "3\nwater\nO 0 0 0\nH 1.26 0 0.48\nH -1.26 0 0.48\n"
        sv = vibrations(
            _Starved(atom_xyz=starved_xyz, charge=0, multiplicity=1,
                     functional="b3lyp", basis="6-31g*"),
            optimize_first=True,
        )
        if sv.get("opt_converged") is not False:
            FAILURES.append(
                f"vibrations/starved: frequencies on a geometry whose "
                f"pre-optimisation was cut short after 2 steps report "
                f"opt_converged={sv.get('opt_converged')!r}; the verdict did "
                f"not travel from the optimisation to the frequencies")
            nbad += 1
        if not any("did not converge" in str(w)
                   for w in (sv.get("warnings") or [])):
            FAILURES.append(
                "vibrations/starved: the unconverged pre-optimisation left no "
                "warning in the result, so the frequencies will be read as a "
                "minimum's")
            nbad += 1
        gr = sv.get("opt_grms")
        if not isinstance(gr, (int, float)) or gr <= 0.0:
            FAILURES.append(f"vibrations/starved: opt_grms is {gr!r}")
            nbad += 1
    except Exception as exc:                             # noqa: BLE001
        FAILURES.append(f"vibrations/starved: the check could not run: {exc}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] frequencies on an unrelaxed "
          f"geometry say so instead of passing for a minimum", flush=True)

    # ------------------------------------------------------------------
    #  the frequency scale factor belongs to the method
    # ------------------------------------------------------------------
    # Harmonic frequencies are scaled before being compared with a measured
    # band, and the factor is fitted per functional *and* basis.  0.961 is
    # B3LYP/6-31G* and nothing else; it used to be applied to every level.
    # At HF/6-31G* the published factor is 0.899, so water's bands came out
    # 113 / 251 / 259 cm-1 too high -- and the narration called it "the
    # 0.961 factor standard for this level of theory".
    nbad = 0
    try:
        from backend.engine.analysis import freq_scale

        for func, basis, want in (("hf", "6-31g*", 0.899),
                                  ("b3lyp", "6-31g*", 0.960),
                                  ("blyp", "6-31g*", 0.992)):
            got, src = freq_scale(func, basis)
            if abs(got - want) > 1e-9:
                FAILURES.append(
                    f"scale/{func}-{basis}: {got}, expected the CCCBDB value "
                    f"{want}")
                nbad += 1
            if "CCCBDB" not in str(src):
                FAILURES.append(
                    f"scale/{func}-{basis}: the source is {src!r}; a borrowed "
                    f"scale factor has to say where it came from")
                nbad += 1
        # and a level with no published factor gets none, plus a reason
        got, src = freq_scale("m06l", "def2-svp")
        if abs(got - 1.0) > 1e-9:
            FAILURES.append(
                f"scale/m06l-def2-svp: {got}, but no factor is published for "
                f"that level -- applying one borrowed from another method is "
                f"the bug this replaced")
            nbad += 1
        if "no frequency scale factor" not in str(src).lower():
            FAILURES.append(
                f"scale/m06l-def2-svp: applying no factor but not saying why "
                f"({src!r}); the reader cannot tell 'unscaled on purpose' "
                f"from 'forgot to scale'")
            nbad += 1
    except Exception as exc:                             # noqa: BLE001
        FAILURES.append(f"scale: the check could not run: {exc}")
        nbad += 1

    # ... and the number has to reach the curve, not just sit in the payload:
    # the whole IR spectrum is shifted by it.
    code, resp = _req("POST", "/api/job", {"molecule": "water",
                                           "kind": "vibrations",
                                           "functional": "hf",
                                           "basis": "6-31g*"})
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        hf_job = _wait(resp["job_id"])
        hres = hf_job.get("result") or {}
        hir = hres.get("ir_spectrum") or {}
        if abs(float(hres.get("freq_scale") or 0) - 0.899) > 1e-9:
            FAILURES.append(
                f"scale/hf: an HF/6-31G* frequency run reports freq_scale="
                f"{hres.get('freq_scale')}, expected 0.899")
            nbad += 1
        if abs(float(hir.get("scale_factor") or 0) - 0.899) > 1e-9:
            FAILURES.append(
                f"scale/hf: the IR curve was built with scale_factor="
                f"{hir.get('scale_factor')}, so every band is shifted by the "
                f"B3LYP factor instead of HF's")
            nbad += 1
        for pk in (hir.get("peaks") or [])[:1]:
            want = pk["frequency_cm1"] * 0.899
            if abs(pk["scaled_cm1"] - want) > 1.0:
                FAILURES.append(
                    f"scale/hf: band {pk['frequency_cm1']} cm-1 is reported "
                    f"scaled to {pk['scaled_cm1']}, expected {want:.0f}")
                nbad += 1
    else:
        FAILURES.append(f"scale/hf: job did not run (HTTP {code})")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] the frequency scale factor is "
          f"the one published for the level that ran", flush=True)

    # ------------------------------------------------------------------
    #  IR intensities on an absolute scale
    # ------------------------------------------------------------------
    # A wrong absolute scale is invisible: the plotted curve is normalised to
    # 100 before it is drawn, so a spectrum whose every band is 4x too intense
    # looks exactly like a correct one.  The old projection divided PySCF's
    # norm_mode by sqrt(m) a SECOND time -- norm_mode is already mass-weighted
    # -- which reweights each atom's contribution by 1/sqrt(m_a): oxygen by
    # 1/4 next to hydrogen.  Mass-dependent, so no single constant repairs it.
    #
    # The guard is the intensity sum rule, which needs no published value and
    # no unit convention: in mass-weighted coordinates the 3N-6 modes plus 3
    # translations plus 3 rotations span the space and an orthogonal change of
    # basis preserves the sum of squares, so
    #   sum_i |dmu/dQ_i|^2 == APT trace - (rotation) - (translation).
    # Water's rotational term is FOUR TIMES the vibrational one, so a sum rule
    # that omits it reports a 50% error for a perfect projection -- hence the
    # separate assertion that it is present and non-zero.
    nbad = 0
    for label, res in (("ir/water", vib_water),):
        sr = res.get("ir_sum_rule") or {}
        if not sr:
            FAILURES.append(
                f"{label}: no `ir_sum_rule` in the result, so nothing says "
                f"whether the intensities are on an absolute scale at all")
            nbad += 1
            continue
        resid = sr.get("residual_pct")
        if resid is None:
            FAILURES.append(f"{label}: ir_sum_rule carries no residual ({sr})")
            nbad += 1
        elif abs(float(resid)) > 0.5:
            FAILURES.append(
                f"{label}: the intensity sum rule is off by {resid}% -- the "
                f"projection onto the normal modes does not conserve "
                f"sum|dmu/dQ|^2 (got {sr.get('vibrational_d2_amu_a2')}, "
                f"expected {sr.get('expected_d2_amu_a2')} D^2/amu/A^2), so "
                f"the km/mol values are not on an absolute scale")
            nbad += 1
        if float(sr.get("rotational_d2_amu_a2") or 0) <= 0.0:
            FAILURES.append(
                f"{label}: the rotational term of the sum rule is zero; for "
                f"water it is four times the vibrational term, so the test "
                f"is not really being applied")
            nbad += 1
        # Per-mode regression.  The sum rule above is a physical law and is
        # what catches a systematic misweighting; this is here for the error it
        # cannot see -- a redistribution between modes that leaves the total
        # intact.  These are the values the sum-rule-verified projection gives
        # at B3LYP/6-31G*, and the band is deliberately loose (10%): the point
        # is to catch a coding error, not to police the method.
        refs = (("bend", 79.6), ("sym stretch", 1.8), ("asym stretch", 20.1))
        got = res.get("ir_intensities_km_mol") or []
        if len(got) == len(refs):
            for (name, ref), g in zip(refs, got):
                g = float(g)
                if abs(g - ref) > max(0.10 * ref, 0.15):
                    FAILURES.append(
                        f"{label}: the {name} intensity is {g} km/mol, "
                        f"expected {ref} at B3LYP/6-31G* (+-10%)")
                    nbad += 1
        units = str(res.get("ir_intensity_units") or "")
        if "km/mol" not in units:
            FAILURES.append(
                f"{label}: `ir_intensity_units` is {units!r}; a figure caption "
                f"cannot state whether these are napierian or decadic without "
                f"naming the unit")
            nbad += 1

    # A second molecule whose heavy atom is three times oxygen's mass: the
    # mass-dependence of the old bug does not show up in water alone.
    try:
        from backend.engine.dft import DFTEngine
        from backend.engine.analysis import vibrations as _vib
        h2s = _vib(DFTEngine(atom_xyz="3\nh2s\n"
                                      "S 0.000000  0.000000  0.102000\n"
                                      "H 0.000000  0.958400 -0.816000\n"
                                      "H 0.000000 -0.958400 -0.816000\n",
                             charge=0, multiplicity=1,
                             functional="b3lyp", basis="6-31g*"))
        sr = h2s.get("ir_sum_rule") or {}
        resid = sr.get("residual_pct")
        if resid is None:
            FAILURES.append(f"ir/h2s: no sum-rule residual ({sr})")
            nbad += 1
        elif abs(float(resid)) > 0.5:
            FAILURES.append(
                f"ir/h2s: the intensity sum rule is off by {resid}% "
                f"(got {sr.get('vibrational_d2_amu_a2')}, expected "
                f"{sr.get('expected_d2_amu_a2')}).  How far off a 1/sqrt(m_a) "
                f"reweighting lands depends on the mass distribution -- water "
                f"shows -9.2%, H2S only -1.3% -- so one molecule is not "
                f"enough to detect it")
            nbad += 1
    except Exception as exc:                                 # noqa: BLE001
        FAILURES.append(f"ir/h2s: the check could not run: {exc}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] IR intensities are on an "
          f"absolute scale (sum rule closes to <0.5%)", flush=True)

    # ------------------------------------------------------------------
    #  bonding: Mayer bond orders and Hirshfeld charges
    # ------------------------------------------------------------------
    # Two new tables, each with a property that can be checked without a
    # published number:
    #   * Hirshfeld charges must sum to the molecular charge -- exactly, by
    #     construction, up to grid quadrature.
    #   * Chemically equivalent atoms must come out equivalent.  This is what
    #     fails if the free-atom density is not spherically averaged: an
    #     unaveraged carbon or oxygen is orientation-dependent, and water's two
    #     hydrogens then get -0.291/+0.158/+0.133 instead of two equal +0.159.
    #   * Hirshfeld must move far less than Mulliken when the basis grows --
    #     the whole reason to prefer it.
    # Plus the published Mayer bond orders as regression values.
    WATER_XYZ = ("3\nwater\nO 0.0 0.0 0.1173\n"
                 "H 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n")
    CH4_XYZ = ("5\nmethane\nC 0.0 0.0 0.0\n"
               "H 0.6291 0.6291 0.6291\nH -0.6291 -0.6291 0.6291\n"
               "H -0.6291 0.6291 -0.6291\nH 0.6291 -0.6291 -0.6291\n")
    METHYL_XYZ = ("4\nmethyl\nC 0.0 0.0 0.0\nH 0.6291 0.6291 0.6291\n"
                  "H -0.6291 -0.6291 0.6291\nH -0.6291 0.6291 -0.6291\n")
    NH4_XYZ = ("5\nammonium\nN 0.0 0.0 0.0\nH 0.6 0.6 0.6\n"
               "H -0.6 -0.6 0.6\nH -0.6 0.6 -0.6\nH 0.6 -0.6 -0.6\n")

    def _bond(xyz, basis="6-31g*", charge=0, mult=1):
        from backend.engine.dft import DFTEngine

        eng = DFTEngine(atom_xyz=xyz, charge=charge, multiplicity=mult,
                        functional="b3lyp", basis=basis)
        return eng, eng.run_scf()

    nbad = 0
    try:
        from backend.engine.bonding import bonding_analysis, mayer_bond_orders

        for name, xyz, ref, tol in (
                ("N2", "2\nn2\nN 0.0 0.0 0.0\nN 0.0 0.0 1.0977\n", 2.83, 0.12),
                ("CO", "2\nco\nC 0.0 0.0 0.0\nO 0.0 0.0 1.1283\n", 2.40, 0.12),
                ("water", WATER_XYZ, 0.80, 0.10),
                ("methane", CH4_XYZ, 0.97, 0.10)):
            e, _ = _bond(xyz)
            m = mayer_bond_orders(e.mol, e.mf)
            if not m.get("bonds"):
                FAILURES.append(f"bonding/mayer-{name}: no bonds above threshold")
                nbad += 1
                continue
            got = float(m["bonds"][0]["order"])
            if abs(got - ref) > tol:
                FAILURES.append(
                    f"bonding/mayer-{name}: the strongest bond order is {got}, "
                    f"the published value at this level is {ref} (+-{tol}).  "
                    f"The spin-resolved convention would give {got/2:.3f}.")
                nbad += 1
        # equivalent bonds must be exactly equal, not merely close
        e, _ = _bond(WATER_XYZ)
        m = mayer_bond_orders(e.mol, e.mf)
        if len(m["bonds"]) != 2 or abs(m["bonds"][0]["order"]
                                       - m["bonds"][1]["order"]) > 1e-9:
            FAILURES.append(
                f"bonding/mayer-water: the two O-H bonds must be equal by "
                f"symmetry, got {[b['order'] for b in m['bonds']]}")
            nbad += 1
        e, _ = _bond(METHYL_XYZ, mult=2)
        m = mayer_bond_orders(e.mol, e.mf)
        orders = [b["order"] for b in m["bonds"]]
        if len(orders) != 3 or max(orders) - min(orders) > 1e-9:
            FAILURES.append(
                f"bonding/mayer-methyl: the three C-H bonds of a methyl "
                f"radical must be equal, got {orders}")
            nbad += 1
        cval = next((v["valence"] for v in m["valence"]
                     if v["symbol"] == "C"), None)
        if cval is None or not (2.6 <= cval <= 3.4):
            FAILURES.append(
                f"bonding/mayer-methyl: carbon's Mayer valence is {cval}, "
                f"expected ~3 for CH3")
            nbad += 1
    except Exception as exc:                                 # noqa: BLE001
        FAILURES.append(f"bonding/mayer: the check could not run: {exc}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] Mayer bond orders match the "
          f"published values, and equivalent bonds are exactly equal",
          flush=True)

    nbad = 0
    try:
        from backend.engine.bonding import hirshfeld_charges

        for name, xyz, q, mult in (("water", WATER_XYZ, 0, 1),
                                   ("ammonium", NH4_XYZ, 1, 1)):
            e, _ = _bond(xyz, charge=q, mult=mult)
            charges, diag = hirshfeld_charges(e.mol, e.mf, "b3lyp")
            err = abs(float(diag["charge_sum"]) - q)
            if err > 1e-3:
                FAILURES.append(
                    f"bonding/hirshfeld-{name}: the charges sum to "
                    f"{diag['charge_sum']}, but the molecule carries {q:+d}; "
                    f"the partition has lost or invented charge")
                nbad += 1
            wdev = float(diag.get("weight_sum_max_deviation", 1.0))
            if wdev > 1e-9:
                FAILURES.append(
                    f"bonding/hirshfeld-{name}: the promolecular weights miss "
                    f"unity by up to {wdev:.2e} at a grid point that carries "
                    f"density; they are not a partition of unity, so the "
                    f"charges are not a partition of the electron count")
                nbad += 1
            if abs(float(diag["grid_integral_error_pct"])) > 0.01:
                FAILURES.append(
                    f"bonding/hirshfeld-{name}: the density integrates to "
                    f"{diag['grid_integral_electrons']} electrons instead of "
                    f"{diag['n_electrons']} "
                    f"({diag['grid_integral_error_pct']}%); the density was "
                    f"evaluated wrongly and every charge below it is void")
                nbad += 1
        # equivalent atoms, and the spherical average that makes them so
        e, _ = _bond(WATER_XYZ)
        charges, _ = hirshfeld_charges(e.mol, e.mf, "b3lyp")
        hs = [c["charge"] for c in charges if c["symbol"] == "H"]
        if len(hs) != 2 or abs(hs[0] - hs[1]) > 1e-6:
            FAILURES.append(
                f"bonding/hirshfeld-water: the two hydrogens must be "
                f"equivalent, got {hs}.  Unequal hydrogens mean the free-atom "
                f"density was not spherically averaged.")
            nbad += 1
        if not charges or charges[0]["symbol"] != "O" or charges[0]["charge"] >= 0:
            FAILURES.append(
                f"bonding/hirshfeld-water: oxygen should carry the negative "
                f"charge, got {charges[0]['charge'] if charges else None}")
            nbad += 1
    except Exception as exc:                                 # noqa: BLE001
        FAILURES.append(f"bonding/hirshfeld: the check could not run: {exc}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] Hirshfeld charges sum to the "
          f"molecular charge and equivalent atoms come out equivalent",
          flush=True)

    nbad = 0
    try:
        from backend.engine.bonding import hirshfeld_charges

        sets = {}
        for basis in ("6-31g*", "6-31g**"):
            e, res = _bond(WATER_XYZ, basis=basis)
            charges, _ = hirshfeld_charges(e.mol, e.mf, "b3lyp")
            sets[basis] = (
                [c["charge"] for c in charges],
                [q["charge"] for q in (res.get("mulliken_charges") or [])],
            )
        if len(sets["6-31g*"][0]) == len(sets["6-31g**"][0]) > 0:
            dh = max(abs(a - b) for a, b in zip(sets["6-31g*"][0],
                                                sets["6-31g**"][0]))
            dmk = max(abs(a - b) for a, b in zip(sets["6-31g*"][1],
                                                 sets["6-31g**"][1]))
            if not (dh < dmk):
                FAILURES.append(
                    f"bonding/robustness: adding polarisation functions moved "
                    f"Hirshfeld by {dh:.4f} and Mulliken by {dmk:.4f}; "
                    f"Hirshfeld is supposed to be the basis-set-robust "
                    f"partition, so these two look swapped")
                nbad += 1
    except Exception as exc:                                 # noqa: BLE001
        FAILURES.append(f"bonding/robustness: the check could not run: {exc}")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] Hirshfeld is the basis-set-"
          f"robust partition (it moves less than Mulliken)", flush=True)

    # The engine can compute all of this and the API still drop it -- the
    # Properties pane reads the *job payload*, not the engine's return value.
    # And a table nobody can read is not a result: the narration has to name
    # the numbers, or the figure caption has to be written from scratch.
    nbad = 0
    code, resp = _req("POST", "/api/job", {"molecule": "water",
                                           "kind": "single_point",
                                           "functional": "b3lyp",
                                           "basis": "6-31g*"})
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"])
        bd = (job.get("result") or {}).get("bonding") or {}
        if bd.get("mayer_error") or bd.get("hirshfeld_error"):
            FAILURES.append(
                f"bonding/api: the analysis failed on a plain water single "
                f"point: {bd.get('mayer_error') or bd.get('hirshfeld_error')}")
            nbad += 1
        bonds = (bd.get("mayer") or {}).get("bonds") or []
        if len(bonds) != 2:
            FAILURES.append(
                f"bonding/api: a single point on water returns {len(bonds)} "
                f"bond(s), expected the two O-H bonds; the bond table on the "
                f"Properties pane would render empty")
            nbad += 1
        charges = (bd.get("hirshfeld") or {}).get("charges") or []
        if len(charges) != 3:
            FAILURES.append(
                f"bonding/api: a single point on water returns {len(charges)} "
                f"Hirshfeld charge(s), expected 3")
            nbad += 1
        text = str(job.get("explanation") or "")
        for want, why in (("Mayer bond orders", "the bond orders"),
                          ("Hirshfeld", "the Hirshfeld charges")):
            if want not in text:
                FAILURES.append(
                    f"bonding/narration: the write-up never mentions "
                    f"{why} ({want!r} absent); the numbers exist but a reader "
                    f"has to go looking for them")
                nbad += 1
    else:
        FAILURES.append(f"bonding/api: job did not run (HTTP {code})")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] bond orders and Hirshfeld "
          f"charges reach the API and the write-up", flush=True)

    # ------------------------------------------------------------------
    #  density of states
    # ------------------------------------------------------------------
    # The projections must add up to the total.  They do by construction,
    # because the weights are Loewdin populations -- using |c|^2 instead
    # (the bug that inverted water's LUMO) breaks the partition.
    nbad = 0
    code, resp = _req("POST", "/api/job", {"molecule": "pyridine",
                                           "kind": "dos",
                                           "functional": "b3lyp",
                                           "basis": "6-31g*"})
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"])
        d = (job.get("result") or {}).get("dos") or {}
        total = d.get("total") or []
        proj = d.get("projected") or {}
        if not total or not proj:
            FAILURES.append("dos/pyridine: empty total or projected curve")
            nbad += 1
        else:
            if len(proj) < 2:
                FAILURES.append(
                    f"dos/pyridine: {len(proj)} projection(s), pyridine has "
                    f"three elements")
                nbad += 1
            acc = [0.0] * len(total)
            for curve in proj.values():
                for i, v in enumerate(curve):
                    acc[i] += v
            peak = max(abs(v) for v in total) or 1.0
            worst = max(abs(a - b) for a, b in zip(acc, total)) / peak
            if worst > 0.005:
                FAILURES.append(
                    f"dos/pyridine: projections miss the total by "
                    f"{worst * 100:.2f}% of the peak -- not a partition")
                nbad += 1
            if not (d.get("homo_ev") < d.get("lumo_ev", float("inf"))):
                FAILURES.append("dos/pyridine: HOMO above LUMO")
                nbad += 1
    else:
        FAILURES.append("dos: job did not run")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] DOS projections partition the total",
          flush=True)

    # ------------------------------------------------------------------
    #  the DOS is a density, so its area has to be a number of states
    # ------------------------------------------------------------------
    # Being a partition is not enough.  A curve whose area depends on how much
    # it was smoothed is not a density of states: it is a drawing whose height
    # is set by a parameter nobody reports.  Unnormalised Gaussians gave water
    # an area of 5.4 at FWHM 0.3 eV and 19.0 at 1.0 eV -- the same molecule,
    # 3.3x apart -- and the occupied region integrated to 2.5 states instead
    # of 5.  Two integrals pin it down: the whole curve must hold every
    # orbital, and everything below the HOMO must hold the occupied ones
    # (within one state, because the HOMO's own Gaussian straddles the line
    # and a degenerate HOMO straddles it twice).
    nbad = 0
    for mol_name in ("water", "pyridine"):
        code, resp = _req("POST", "/api/job", {"molecule": mol_name,
                                               "kind": "dos",
                                               "functional": "b3lyp",
                                               "basis": "6-31g*"})
        if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
            FAILURES.append(f"dos/{mol_name}: job did not run")
            nbad += 1
            continue
        d = ((_wait(resp["job_id"]).get("result") or {}).get("dos")) or {}
        xs, ys = d.get("x_energy_ev") or [], d.get("total") or []
        nmo, nocc = d.get("n_orbitals"), d.get("n_occupied")
        if not xs or not ys or len(xs) != len(ys):
            FAILURES.append(f"dos/{mol_name}: no DOS curve returned")
            nbad += 1
            continue
        if nmo is None or nocc is None:
            FAILURES.append(
                f"dos/{mol_name}: the payload does not say how many orbitals "
                f"it broadened, so its area cannot be checked")
            nbad += 1
            continue
        area = 0.0
        occ_area = 0.0
        homo = d.get("homo_ev")
        for i in range(len(xs) - 1):
            dx = xs[i + 1] - xs[i]
            trap = 0.5 * (ys[i] + ys[i + 1]) * dx
            area += trap
            if homo is not None and xs[i + 1] <= homo:
                occ_area += trap
        if abs(area - nmo) > 0.02 * nmo:
            FAILURES.append(
                f"dos/{mol_name}: the whole curve integrates to {area:.2f} but "
                f"there are {nmo} orbitals -- the broadening is not "
                f"normalised, so the y axis is arbitrary")
            nbad += 1
        # The lower bound is loose because a degenerate HOMO straddles the
        # line once per partner (water 4.5 of 5, benzene's twofold HOMO 20 of
        # 21); the unnormalised curve gave 2.5 of 5, so the window still
        # catches it by a wide margin.
        if not (nocc - 2.05 <= occ_area <= nocc + 0.05):
            FAILURES.append(
                f"dos/{mol_name}: the occupied part integrates to "
                f"{occ_area:.2f} states, but {nocc} orbitals are occupied")
            nbad += 1
        if "states/eV" not in (d.get("ylabel") or ""):
            FAILURES.append(
                f"dos/{mol_name}: the y axis is labelled "
                f"'{d.get('ylabel')}' with no unit; a density of states "
                f"counts states per eV")
            nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] the DOS is a normalised density: "
          f"its area is a number of states", flush=True)

    # ------------------------------------------------------------------
    #  reactivity descriptors
    # ------------------------------------------------------------------
    # Condensed Fukui functions are differences of *population*, so every
    # value must be non-negative and the set must sum to one.  Computing
    # them from charges (Z - population) flips every sign and used to put
    # formaldehyde's most electrophilic site on a hydrogen.
    nbad = 0
    code, resp = _req("POST", "/api/job", {"molecule": "formaldehyde",
                                           "kind": "reactivity",
                                           "functional": "b3lyp",
                                           "basis": "6-31g*"})
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"])
        rres = job.get("result") or {}
        fk = rres.get("fukui") or {}
        rows = fk.get("rows") or []
        if not rows:
            FAILURES.append("reactivity/formaldehyde: no Fukui rows")
            nbad += 1
        else:
            for key in ("f_plus", "f_minus"):
                worst = min(r.get(key, 0.0) for r in rows)
                if worst < -0.05:
                    FAILURES.append(
                        f"reactivity/formaldehyde: {key} reaches {worst:.3f}; "
                        f"condensed Fukui values cannot be negative (signs "
                        f"are inverted if charges were used)")
                    nbad += 1
                tot = sum(r.get(key, 0.0) for r in rows)
                if abs(tot - 1.0) > 0.05:
                    FAILURES.append(
                        f"reactivity/formaldehyde: {key} sums to {tot:.4f}, "
                        f"must sum to 1")
                    nbad += 1
            # the chemical pay-off: a carbonyl is attacked by nucleophiles at
            # carbon and by electrophiles at oxygen.
            site = fk.get("most_electrophilic_site")
            if site != "O2":
                FAILURES.append(
                    f"reactivity/formaldehyde: most electrophilic site is "
                    f"{site}, expected O2 (the carbonyl oxygen)")
                nbad += 1
        if not ((rres.get("gap_ev") or 0) > 0):
            FAILURES.append("reactivity/formaldehyde: non-positive gap")
            nbad += 1
    else:
        FAILURES.append("reactivity: job did not run")
        nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] Fukui functions sum to one and "
          f"point at the right atom", flush=True)

    # ------------------------------------------------------------------
    #  surface grids
    # ------------------------------------------------------------------
    # The map has to survive three independent checks.  The grid itself is
    # audited like any cube file (header vs payload).  The physics is audited
    # against the one molecule whose electrostatic potential is not up for
    # debate: water is negative at oxygen and positive at hydrogen, and
    # formaldehyde is negative at the carbonyl oxygen and positive at the
    # carbonyl carbon -- the same answer the Fukui functions give by a
    # completely different route.  And the ions audit the *sign*: an anion's
    # potential is negative everywhere in the shell and a cation's positive
    # everywhere, so a narration that reports "the most positive region" for
    # hydroxide is describing a number that is still negative.
    nbad = 0
    for molecule, rich_sym, poor_sym in (("water", "O", "H"),
                                         ("formaldehyde", "O", "C")):
        code, resp = _req("POST", "/api/job", {"molecule": molecule,
                                               "kind": "surfaces",
                                               "functional": "b3lyp",
                                               "basis": "6-31g*"})
        if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
            FAILURES.append(f"surfaces/{molecule}: job did not start")
            nbad += 1
            continue
        job = _wait(resp["job_id"])
        sres = job.get("result") or {}
        s = sres.get("surfaces") or {}
        items = s.get("surfaces") or []
        if not items:
            FAILURES.append(f"surfaces/{molecule}: no grids produced")
            nbad += 1
            continue

        mep = next((i for i in items if i.get("kind") == "mep"), None)
        if not mep:
            FAILURES.append(f"surfaces/{molecule}: no MEP grid")
            nbad += 1
        else:
            rng = mep.get("range") or []
            if len(rng) != 2 or abs(rng[0] + rng[1]) > 1e-9 or rng[1] <= 0:
                FAILURES.append(
                    f"surfaces/{molecule}: MEP range {rng} is not symmetric "
                    f"about zero")
                nbad += 1
            neg = mep.get("most_negative") or {}
            pos = mep.get("most_positive") or {}
            site_n = neg.get("site") or ""
            val_n = neg.get("value_hartree")
            site_p = pos.get("site") or ""
            val_p = pos.get("value_hartree")
            # The most negative site is not marginal: the carbonyl oxygen's
            # shell reaches -0.055 Ha where nothing near the carbon goes below
            # -0.012, so this one is worth asserting directly.
            if not site_n.startswith(rich_sym):
                FAILURES.append(
                    f"surfaces/{molecule}: most negative site is {site_n!r}, "
                    f"expected an atom starting with {rich_sym!r}")
                nbad += 1
            if val_n is None or val_n >= 0:
                FAILURES.append(
                    f"surfaces/{molecule}: most negative site {site_n} has "
                    f"potential {val_n} Ha/e, which is not negative")
                nbad += 1
            # The most positive site is a different matter, and pretending
            # otherwise is how a test ends up checking the grid.  For
            # formaldehyde the carbon and the hydrogens sit 0.7 kcal/mol apart
            # on this shell -- the relaxed geometry put the maximum on H, the
            # geometry before it put it on C, and nothing about the chemistry
            # moved.  What is not marginal is the contrast between the two
            # elements, so that is what gets asserted, per atom, with a margin
            # two orders of magnitude smaller than the effect.
            if val_p is None or val_p <= 0:
                FAILURES.append(
                    f"surfaces/{molecule}: most positive site {site_p} has "
                    f"potential {val_p} Ha/e, which is not positive")
                nbad += 1
            if site_p.startswith(rich_sym):
                FAILURES.append(
                    f"surfaces/{molecule}: the most positive point on the "
                    f"shell sits on {site_p}, an electron-rich "
                    f"{rich_sym} atom")
                nbad += 1
            atoms = mep.get("atom_extremes") or []
            if not atoms:
                FAILURES.append(
                    f"surfaces/{molecule}: the MEP reports no per-atom shell "
                    f"extremes, so the only checkable claim about where the "
                    f"positive region sits is a global argmax that moves with "
                    f"the grid")
                nbad += 1
            else:
                rich = [a for a in atoms
                        if str(a.get("element", "")).startswith(rich_sym)]
                poor = [a for a in atoms
                        if str(a.get("element", "")).startswith(poor_sym)]
                if not rich or not poor:
                    FAILURES.append(
                        f"surfaces/{molecule}: the per-atom extremes name no "
                        f"{rich_sym} or no {poor_sym} atom")
                    nbad += 1
                else:
                    rich_lo = min(float(a["min_hartree"]) for a in rich)
                    poor_lo = min(float(a["min_hartree"]) for a in poor)
                    rich_hi = max(float(a["max_hartree"]) for a in rich)
                    poor_hi = max(float(a["max_hartree"]) for a in poor)
                    # 0.02 Ha = 12 kcal/mol against an effect of 43
                    # (formaldehyde) and 80 (water) kcal/mol.
                    if rich_lo > poor_lo - 0.02:
                        FAILURES.append(
                            f"surfaces/{molecule}: the {rich_sym} shell "
                            f"reaches {rich_lo:+.4f} Ha and the {poor_sym} "
                            f"shell {poor_lo:+.4f} Ha, so the electron-rich "
                            f"atom is not the most negative place on the "
                            f"molecule")
                        nbad += 1
                    # 0.005 Ha = 3 kcal/mol against 31 (formaldehyde) and 5.5
                    # (water) kcal/mol.
                    if poor_hi < rich_hi + 0.005:
                        FAILURES.append(
                            f"surfaces/{molecule}: the {poor_sym} shell peaks "
                            f"at {poor_hi:+.4f} Ha and the {rich_sym} shell at "
                            f"{rich_hi:+.4f} Ha, so the electron-poor atom has "
                            f"no more positive region than the rich one")
                        nbad += 1
            if not mep.get("potential_changes_sign"):
                FAILURES.append(
                    f"surfaces/{molecule}: a neutral polar molecule must "
                    f"show both signs of potential")
                nbad += 1

        # The legend and the prose must describe the same numbers.  An
        # earlier sampling scheme read percentiles over the whole grid box
        # and reported extremes from the same grid, and the two disagreed by
        # a factor of six (naphthalene: range +/-15 kcal/mol, extreme 89).
        if mep:
            rng = mep.get("range") or []
            for role in ("most_negative", "most_positive"):
                val = (mep.get(role) or {}).get("value_hartree")
                if val is None or len(rng) != 2:
                    continue
                if not (rng[0] - 1e-9 <= val <= rng[1] + 1e-9):
                    FAILURES.append(
                        f"surfaces/{molecule}: {role} = {val} Ha/e lies "
                        f"outside the displayed colour range {rng} -- the "
                        f"legend and the text disagree")
                    nbad += 1

        if not any(i.get("kind") == "orbital" for i in items):
            FAILURES.append(f"surfaces/{molecule}: no orbital grids")
            nbad += 1

        # the grids must actually be served, and be well-formed cubes
        base = s.get("url_base") or ""
        if not base.startswith("/api/cubes/"):
            FAILURES.append(f"surfaces/{molecule}: bad url_base {base!r}")
            nbad += 1
        else:
            import urllib.request as _url
            for it in items:
                try:
                    with _url.urlopen(BASE + base + "/" + it["file"],
                                      timeout=120) as fh:
                        text = fh.read().decode("utf-8", "replace")
                except Exception as exc:
                    FAILURES.append(
                        f"surfaces/{molecule}: {it['file']} -> {exc}")
                    nbad += 1
                    continue
                lines = text.splitlines()
                if len(lines) < 8:
                    FAILURES.append(
                        f"surfaces/{molecule}: {it['file']} has {len(lines)} "
                        f"lines, not a cube file")
                    nbad += 1
                    continue
                try:
                    natoms = int(lines[2].split()[0])
                    dims = [abs(int(lines[3 + a].split()[0])) for a in range(3)]
                except (ValueError, IndexError):
                    FAILURES.append(
                        f"surfaces/{molecule}: {it['file']} unreadable header")
                    nbad += 1
                    continue
                if natoms <= 0:
                    # 3Dmol only applies the bohr -> angstrom factor when the
                    # atom count is positive; a negative one would render the
                    # grid 1.89x too large.
                    FAILURES.append(
                        f"surfaces/{molecule}: {it['file']} declares natoms "
                        f"{natoms}; must be positive (bohr convention)")
                    nbad += 1
                vals = " ".join(lines[6 + natoms:]).split()
                if len(vals) != dims[0] * dims[1] * dims[2]:
                    FAILURES.append(
                        f"surfaces/{molecule}: {it['file']} header says "
                        f"{dims[0]}x{dims[1]}x{dims[2]} = "
                        f"{dims[0] * dims[1] * dims[2]} values but "
                        f"{len(vals)} were read")
                    nbad += 1
    # Ions: the potential cannot change sign, and the narration must not
    # call the least-negative point of an anion "the most positive region".
    for molecule, want_sign in (("hydroxide", -1), ("ammonium", +1)):
        code, resp = _req("POST", "/api/job", {"molecule": molecule,
                                               "kind": "surfaces",
                                               "functional": "b3lyp",
                                               "basis": "6-31g*"})
        if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
            FAILURES.append(f"surfaces/{molecule}: job did not start")
            nbad += 1
            continue
        job = _wait(resp["job_id"])
        s = (job.get("result") or {}).get("surfaces") or {}
        mep = next((i for i in (s.get("surfaces") or [])
                    if i.get("kind") == "mep"), None)
        if not mep:
            FAILURES.append(f"surfaces/{molecule}: no MEP grid")
            nbad += 1
            continue
        if mep.get("potential_changes_sign"):
            FAILURES.append(
                f"surfaces/{molecule}: an ion's potential must keep one sign "
                f"on the accessible surface, but it was reported as changing")
            nbad += 1
        neg = (mep.get("most_negative") or {}).get("value_hartree")
        pos = (mep.get("most_positive") or {}).get("value_hartree")
        for role, val in (("most_negative", neg), ("most_positive", pos)):
            if val is None:
                FAILURES.append(f"surfaces/{molecule}: {role} missing")
                nbad += 1
            elif (val * want_sign) <= 0:
                FAILURES.append(
                    f"surfaces/{molecule}: {role} = {val} Ha/e, but a "
                    f"{'anion' if want_sign < 0 else 'cation'} must have "
                    f"that sign everywhere on its surface")
                nbad += 1
        # Assert the claim that is actually correct, rather than merely
        # banning a phrase: for an ion the text has to say the potential
        # keeps one sign and draw the consequence.
        text = job.get("explanation") or ""
        want_word = "negative" if want_sign < 0 else "positive"
        # note where the emphasis markers sit in the source sentence:
        #   "The potential stays **negative everywhere** on the ..."
        if f"stays **{want_word} everywhere**" not in text:
            FAILURES.append(
                f"surfaces/{molecule}: the narration does not say the "
                f"potential stays {want_word} everywhere -- it must, because "
                f"no part of an ion's accessible surface has the other sign")
            nbad += 1
        if "no electron-rich face" not in text:
            FAILURES.append(
                f"surfaces/{molecule}: the narration implies the ion "
                f"presents both an electron-rich and an electron-poor face")
            nbad += 1
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] surface grids are well-formed "
          f"and put the charge where chemistry says", flush=True)

    # ------------------------------------------------------------------
    # NCI: the reduced density gradient has to find a hydrogen bond in a
    # water dimer and must not invent one in methane.
    #
    # This is the whole point of the analysis, so the gate asserts the
    # physical claim rather than the shape of the payload: an interaction
    # appears as a cluster of grid points where the density is an order of
    # magnitude below a covalent bond yet the gradient nearly vanishes.
    # A monomer has no such place, so if both systems "find" a spike the
    # descriptor is broken -- and so is the figure drawn from it.
    # ------------------------------------------------------------------
    nbad = 0
    nci_seen = {}
    for molecule, want in (("water_dimer", True), ("methane", False)):
        code, resp = _req("POST", "/api/job", {"molecule": molecule,
                                               "kind": "nci",
                                               "functional": "b3lyp",
                                               "basis": "6-31g*"})
        if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
            FAILURES.append(f"nci/{molecule}: job did not start")
            nbad += 1
            continue
        job = _wait(resp["job_id"])
        d = (job.get("result") or {}).get("nci") or {}
        if not d:
            FAILURES.append(f"nci/{molecule}: no nci block in the result")
            nbad += 1
            continue
        nci_seen[molecule] = d

        sc = d.get("scatter") or {}
        series = {s.get("key"): s for s in (sc.get("series") or [])}
        if set(series) != {"attractive", "vdw", "repulsive"}:
            FAILURES.append(
                f"nci/{molecule}: scatter series are {sorted(series)}, "
                f"expected attractive/vdw/repulsive")
            nbad += 1

        plotted = 0
        for key, s in series.items():
            pts = s.get("points") or []
            plotted += len(pts)
            if not pts:
                FAILURES.append(f"nci/{molecule}: series {key} is empty")
                nbad += 1
                continue
            for pt in pts:
                x, y = pt[0], pt[1]
                if not (isinstance(x, (int, float))
                        and isinstance(y, (int, float))
                        and math.isfinite(x) and math.isfinite(y)):
                    FAILURES.append(
                        f"nci/{molecule}: series {key} has a non-finite "
                        f"point {pt!r}")
                    nbad += 1
                    break
                if y < -1e-9:
                    FAILURES.append(
                        f"nci/{molecule}: RDG {y} is negative -- the reduced "
                        f"density gradient is a magnitude and cannot be")
                    nbad += 1
                    break
                if abs(x) > 0.0500001:
                    FAILURES.append(
                        f"nci/{molecule}: sign(l2)rho {x} lies outside the "
                        f"plotted range +/-0.05")
                    nbad += 1
                    break
        if plotted < 30:
            FAILURES.append(
                f"nci/{molecule}: only {plotted} points plotted, too few for "
                f"the figure to show anything")
            nbad += 1

        got = bool(d.get("has_weak_interaction"))
        if got != want:
            FAILURES.append(
                f"nci/{molecule}: has_weak_interaction={got}, expected "
                f"{want} ({d.get('n_spike_points')} spike points, lowest RDG "
                f"{d.get('min_rdg')})")
            nbad += 1

        if want:
            # Where the spike sits is not free: a hydrogen bond lands at a
            # density far below a covalent bond, and on the attractive side.
            at = d.get("sign_l2_rho_at_spike")
            rdg = d.get("min_rdg")
            if at is None or not (-0.05 < at < -0.005):
                FAILURES.append(
                    f"nci/{molecule}: the spike sits at sign(l2)rho={at}, "
                    f"which is not a weakly attractive density "
                    f"(expected between -0.05 and -0.005)")
                nbad += 1
            if rdg is None or not (0.0 < rdg < 0.35):
                FAILURES.append(
                    f"nci/{molecule}: lowest RDG is {rdg}, too high for a "
                    f"non-covalent contact (expected below 0.35)")
                nbad += 1
            text = job.get("explanation") or ""
            if "weak-interaction spike" not in text:
                FAILURES.append(
                    f"nci/{molecule}: the narration does not report the "
                    f"spike that the data contains")
                nbad += 1
        else:
            text = job.get("explanation") or ""
            if "No weak-interaction spike was found" not in text:
                FAILURES.append(
                    f"nci/{molecule}: the narration must say plainly that no "
                    f"spike was found instead of hinting at one")
                nbad += 1

        # grids served and well-formed, same rules as any other cube
        base = d.get("url_base") or ""
        if not base.startswith("/api/cubes/"):
            FAILURES.append(f"nci/{molecule}: bad url_base {base!r}")
            nbad += 1
        else:
            import urllib.request as _url
            for it in (d.get("surfaces") or []):
                try:
                    with _url.urlopen(BASE + base + "/" + it["file"],
                                      timeout=180) as fh:
                        text = fh.read().decode("utf-8", "replace")
                except Exception as exc:
                    FAILURES.append(f"nci/{molecule}: {it['file']} -> {exc}")
                    nbad += 1
                    continue
                lines = text.splitlines()
                try:
                    natoms = int(lines[2].split()[0])
                    dims = [abs(int(lines[3 + a].split()[0])) for a in range(3)]
                except (ValueError, IndexError):
                    FAILURES.append(
                        f"nci/{molecule}: {it['file']} unreadable header")
                    nbad += 1
                    continue
                if natoms <= 0:
                    FAILURES.append(
                        f"nci/{molecule}: {it['file']} declares natoms "
                        f"{natoms}; must be positive (bohr convention)")
                    nbad += 1
                vals = " ".join(lines[6 + natoms:]).split()
                if len(vals) != dims[0] * dims[1] * dims[2]:
                    FAILURES.append(
                        f"nci/{molecule}: {it['file']} header says "
                        f"{dims[0] * dims[1] * dims[2]} values but "
                        f"{len(vals)} were read")
                    nbad += 1

    # The two cases must actually differ, not merely each pass its own test:
    # a descriptor that reports the same number everywhere would satisfy
    # both assertions above only if the thresholds were slack.
    if "water_dimer" in nci_seen and "methane" in nci_seen:
        a = nci_seen["water_dimer"].get("min_rdg")
        b = nci_seen["methane"].get("min_rdg")
        if a is not None and b is not None and not (b > 2.0 * a):
            FAILURES.append(
                f"nci: the dimer ({a:.3f}) and methane ({b:.3f}) reach "
                f"similar lowest RDG -- the descriptor does not separate a "
                f"hydrogen bond from no interaction at all")
            nbad += 1

    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] NCI finds the hydrogen bond "
          f"in a water dimer and none in methane", flush=True)

    # ------------------------------------------------------------------
    #  automatic design
    # ------------------------------------------------------------------
    # `renderDesign` and `designCard` read every one of these fields.  A
    # missing one does not raise -- it renders as `undefined` in the panel,
    # which is exactly the class of bug this audit exists to catch.
    #
    # The brief demands a substructure, so the run also proves the promise
    # the panel makes: every returned molecule really carries what was
    # asked for.  "design 4 amines" used to come back with molecules that
    # had none, because the pool-filling loop stopped as soon as any twelve
    # candidates had survived and a demanded group is rare in a uniform
    # fragment draw.
    before = len(FAILURES)
    code, resp = _req("POST", "/api/design", {
        "text": "design 3 molecules containing an amine, logP below 3",
        "n_candidates": 3,
        "dft_top": 1,
        "use_model": False,
    }, timeout=60.0)
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"], timeout=900.0)
        d = job.get("result") or {}
        if job.get("status") != "completed":
            FAILURES.append(f"design: job ended as {job.get('status')}")

        check_present("design", d, [
            "id", "brief", "candidates", "generated", "rejected", "dft_ran",
            "dft_cached", "origins", "model_notes", "seconds", "targets",
            "qualified", "requirements", "unmet_requirements",
            "rejection_reasons", "quality_checks",
        ])

        cands = d.get("candidates") or []
        if not cands:
            FAILURES.append("design: no candidates returned")
        for cand in cands:
            check_present("design.candidate", cand,
                          ["smiles", "name", "origin", "descriptors", "scoring",
                           "qualified"])
            check_present("design.candidate.descriptors",
                          cand.get("descriptors") or {},
                          ["formula", "atoms", "heavy_atoms", "values",
                           "alerts", "alert_count"])
            check_present("design.candidate.scoring", cand.get("scoring") or {},
                          ["score", "targets", "blocking", "measured"])
            for key in ("sa", "qed"):
                if ((cand.get("descriptors") or {}).get("values")
                        or {}).get(key) is None:
                    FAILURES.append(
                        f"design.candidate/{cand.get('smiles')}: no `{key}` "
                        "descriptor -- the card would render an empty cell")
        
        # the requirement really was enforced on the returned molecules
        try:
            from rdkit import Chem

            pattern = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(N=*)]")
            offenders = [c["smiles"] for c in cands
                         if not Chem.MolFromSmiles(c["smiles"])
                         .HasSubstructMatch(pattern)]
            if offenders:
                FAILURES.append(
                    "design: molecules returned without the demanded amine: "
                    + ", ".join(offenders[:3]))
            if d.get("qualified", 0) < len(cands):
                FAILURES.append(
                    f"design: qualified={d.get('qualified')} but "
                    f"{len(cands)} candidates were listed")
        except ImportError:
            pass

        # the measured lead must be reported, with its geometry provenance:
        # the level of theory alone is not reproducible
        measured = [c for c in cands if c.get("dft")]
        if not measured:
            FAILURES.append(
                "design: dft_top=1 but no candidate carries a `dft` block -- "
                "the measured number was paid for and then hidden")
        for cand in measured:
            check_present("design.candidate.dft", cand["dft"],
                          ["gap_ev", "homo_ev", "lumo_ev", "dipole",
                           "functional_label", "basis_label",
                           "geometry_source", "optimized", "cached"])
            if not cand["dft"].get("geometry_source"):
                FAILURES.append(
                    "design: a measured candidate does not say where its "
                    "geometry came from")

        # a zero score must be explained, not just displayed as 0 -- either
        # by naming the criteria that were missed, or by saying that nothing
        # could be judged yet (a gap-only brief has no score before the SCF)
        for cand in cands:
            sc = cand.get("scoring") or {}
            if sc.get("scored") is False:
                if not sc.get("unscored_reason"):
                    FAILURES.append(
                        f"design.candidate/{cand.get('smiles')}: scored=false "
                        "with no `unscored_reason` -- the panel would show a "
                        "bare zero")
            elif sc.get("score") == 0.0 and not sc.get("blocking"):
                FAILURES.append(
                    f"design.candidate/{cand.get('smiles')}: score 0.0 with "
                    "no `blocking` entry -- the panel would show a zero with "
                    "no reason")
    else:
        FAILURES.append(f"design: job did not run ({code})")
    # `check_present` appends to FAILURES without returning, so the section's
    # failure count is whatever it added
    nbad = len(FAILURES) - before
    print(f"[{'PASS' if nbad == 0 else 'FAIL'}] design payload is complete and "
          f"its requirement is enforced", flush=True)

    print()
    if FAILURES:
        print(f"=== FAIL: {len(FAILURES)} contract problem(s) ===")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("=== PASS: every field the front end reads is present ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
