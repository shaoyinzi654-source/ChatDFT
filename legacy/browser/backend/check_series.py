"""Series audit: a set of molecules must come back as a set, not a subset.

The defect this gate exists to prevent is quiet and easy to reintroduce. For a
long time "compare the gaps of benzene, pyridine, furan and pyrrole" returned
a two-molecule comparison: the first two were kept, the last two vanished with
no error, no warning and no mention in the reply. The user was shown a chart
of a four-molecule request that contained two molecules.

So this gate asserts more than "a result came back":

  * every requested molecule appears in ``points``, in the order asked for;
  * the electron counts and formulas are the ones those names actually mean
    (a silently substituted structure would still produce a gap);
  * a molecule that cannot be calculated is kept as a row carrying its
    reason, because a dropped failure looks exactly like a real gap in a
    trend;
  * the relative energies are non-negative with a zero at the lowest one;
  * the chat router sends a multi-molecule request to the series job rather
    than to the two-molecule comparison.

Runs against a live server.  Usage:  python -m backend.check_series
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
FAILURES: list[str] = []


def _req(method: str, path: str, payload: dict | None = None,
         timeout: float = 60.0):
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


def _wait(job_id: str, timeout: float = 900.0) -> dict:
    deadline = time.time() + timeout
    job: dict = {}
    while time.time() < deadline:
        _, job = _req("GET", f"/api/job/{job_id}")
        if isinstance(job, dict) and job.get("status") in ("completed", "failed"):
            return job
        time.sleep(1.0)
    return {"status": "timeout"}


def _bad(msg: str) -> None:
    FAILURES.append(msg)
    print("       !", msg, flush=True)


# The set from the bug report, plus one name that is not a molecule. Every
# entry carries the electron count and formula the name actually means, so a
# structure that resolved to the wrong thing still fails the gate.
SET = {
    "benzene":  ("C6H6", 42),
    "pyridine": ("C5H5N", 42),
    "furan":    ("C4H4O", 36),
    "pyrrole":  ("C4H5N", 36),
}
NOT_A_MOLECULE = "hexafluoroxylophone"


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    BASE = f"http://127.0.0.1:{args.port}"

    print("=== ChatDFT series audit ===\n", flush=True)

    specs = list(SET.keys()) + [NOT_A_MOLECULE]

    # ---- the endpoint rejects a set of one ------------------------------
    code, resp = _req("POST", "/api/series",
                      {"molecules": ["benzene"], "functional": "b3lyp",
                       "basis": "6-31g*"})
    if code != 400:
        _bad(f"/api/series accepted a single-molecule set (HTTP {code}); "
             f"a one-point trend is not a series")
    else:
        print("[PASS] /api/series rejects a set of one", flush=True)

    code, resp = _req("POST", "/api/series",
                      {"molecules": ["benzene"] * 25, "functional": "b3lyp",
                       "basis": "6-31g*"})
    if code != 400:
        _bad(f"/api/series accepted 25 molecules (HTTP {code}); the documented "
             f"ceiling is 24 and above that a run never finishes")
    else:
        print("[PASS] /api/series rejects more than 24 molecules", flush=True)

    # ---- the run itself --------------------------------------------------
    code, resp = _req("POST", "/api/series",
                      {"molecules": specs, "functional": "b3lyp",
                       "basis": "6-31g*"}, timeout=60.0)
    if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
        _bad(f"/api/series would not start: HTTP {code} {resp}")
        print("\n=== FAIL: series could not be started ===")
        return 1

    print(f"       running {len(specs)} molecules at b3lyp/6-31G* ...",
          flush=True)
    started = time.time()
    job = _wait(resp["job_id"])
    if job.get("status") != "completed":
        _bad(f"series job did not complete (status={job.get('status')}, "
             f"{job.get('error') or job.get('message')})")
        print("\n=== FAIL: series job never completed ===")
        return 1

    result = job.get("result") or {}
    points = result.get("points") or []
    elapsed = time.time() - started

    nbad = 0
    before = len(FAILURES)

    # ---- 1. nobody was dropped ------------------------------------------
    if len(points) != len(specs):
        _bad(f"{len(specs)} molecules were requested but {len(points)} came "
             f"back: the set was silently truncated")
    else:
        got = [p.get("spec") for p in points]
        if got != specs:
            _bad(f"points came back in the wrong order/content: {got}")
        else:
            print(f"[PASS] all {len(specs)} requested molecules appear, in "
                  f"order ({elapsed:.0f} s)", flush=True)

    by_spec = {p.get("spec"): p for p in points}

    # ---- 2. each one is the molecule it claims to be --------------------
    for spec, (formula, nelec) in SET.items():
        p = by_spec.get(spec)
        if p is None:
            continue
        if p.get("formula") != formula:
            _bad(f"{spec}: formula is {p.get('formula')!r}, expected {formula}")
        if p.get("nelectrons") != nelec:
            _bad(f"{spec}: {p.get('nelectrons')} electrons, expected {nelec}")
        gap = p.get("gap_ev")
        if gap is None:
            _bad(f"{spec}: no HOMO-LUMO gap in the result")
            continue
        if not (2.0 < gap < 14.0):
            _bad(f"{spec}: gap {gap:.3f} eV is outside 2-14 eV; a gap that "
                 f"size at B3LYP/6-31G* means the wrong structure was built")
        homo, lumo = p.get("homo_ev"), p.get("lumo_ev")
        if homo is None or lumo is None:
            _bad(f"{spec}: missing HOMO or LUMO")
        elif lumo <= homo:
            _bad(f"{spec}: LUMO ({lumo:.3f}) is not above HOMO ({homo:.3f})")
        dip = p.get("dipole")
        if dip is None:
            _bad(f"{spec}: no dipole moment")
        elif dip < 0:
            _bad(f"{spec}: negative dipole magnitude {dip}")
        if not p.get("converged"):
            _bad(f"{spec}: SCF did not converge")
    if len(FAILURES) == before:
        print("[PASS] every molecule resolved to the right structure and a "
              "converged gap", flush=True)

    # ---- 2b. the geometry every number sits on -------------------------
    # A series is a comparison, and a comparison is only fair if every member
    # was treated the same way -- which includes where the coordinates came
    # from.  These are single points on force-field conformers unless
    # something optimised them, and a gap or a dipole relaxed at this level of
    # theory is a different number.  Unsaid, the figure cannot be reproduced
    # or compared with anyone else's.
    before = len(FAILURES)
    for spec in SET:
        p = by_spec.get(spec)
        if p is None:
            continue
        if not p.get("geometry_source"):
            _bad(f"{spec}: the point does not say where its geometry came "
                 f"from, so its gap and dipole cannot be quoted as A//B")
    if not result.get("geometry_source"):
        _bad("the series does not report a geometry source at series level")
    elif "dft" in str(result["geometry_source"]).lower():
        _bad(f"a plain series of single points reports its geometry as "
             f"{result['geometry_source']!r}; nothing here was optimised, so "
             f"claiming a DFT geometry is worse than saying nothing")
    if len(FAILURES) == before:
        print(f"[PASS] every point names the geometry it was computed on "
              f"({result.get('geometry_source')})", flush=True)
    nbad = len(FAILURES) - before

    # ---- 3. the numbers actually differ ---------------------------------
    gaps = [p["gap_ev"] for p in points
            if p.get("spec") in SET and p.get("gap_ev") is not None]
    if len(gaps) >= 2:
        spread = max(gaps) - min(gaps)
        if spread < 0.05:
            _bad(f"all four gaps sit within {spread:.3f} eV; a descriptor "
                 f"that reports nearly the same value for benzene, pyridine, "
                 f"furan and pyrrole is not resolving the differences")
        else:
            print(f"[PASS] the set is not electronically flat: gaps span "
                  f"{min(gaps):.3f}-{max(gaps):.3f} eV ({spread:.3f} eV)",
                  flush=True)
        # benzene is the most aromatic and the most stable of this set, so its
        # gap is the widest; this is the ordering a paper would report
        ben = by_spec.get("benzene", {}).get("gap_ev")
        fur = by_spec.get("furan", {}).get("gap_ev")
        if ben is not None and fur is not None and not ben > fur:
            _bad(f"benzene gap ({ben:.3f} eV) is not wider than furan "
                 f"({fur:.3f} eV); the trend contradicts the known ordering "
                 f"of this series")
        else:
            print(f"[PASS] benzene ({ben:.3f} eV) is wider than furan "
                  f"({fur:.3f} eV), the expected ordering", flush=True)

    # ---- 4. the failure is a row, not a hole -----------------------------
    p = by_spec.get(NOT_A_MOLECULE)
    if p is None:
        _bad(f"{NOT_A_MOLECULE} was dropped entirely instead of being kept "
             f"as a failed row")
    else:
        if not p.get("error"):
            _bad(f"{NOT_A_MOLECULE} has no error text on its row")
        if p.get("gap_ev") is not None:
            _bad(f"{NOT_A_MOLECULE} reports a gap; a name that is not a "
                 f"molecule must not yield numbers")
        if len(FAILURES) == before + nbad:
            print(f"[PASS] the unresolvable name is kept as a row carrying "
                  f"its reason: {str(p.get('error'))[:60]}", flush=True)
    if result.get("n_failed") != 1:
        _bad(f"n_failed is {result.get('n_failed')}, expected 1")
    if result.get("n_ok") != len(SET):
        _bad(f"n_ok is {result.get('n_ok')}, expected {len(SET)}")

    # ---- 5. relative energies -------------------------------------------
    # This set spans C6H6, C5H5N, C4H4O and C4H5N, so the difference of two
    # total energies is not a physical quantity.  A column that reported
    # "benzene is 23915 kcal/mol above furan" would be a number with nothing
    # behind it, so the gate asserts the column is *absent* here and present
    # for a set of isomers.
    rel = [p.get("relative_kcal") for p in points
           if p.get("relative_kcal") is not None]
    if rel:
        _bad(f"relative energies were reported across different formulas "
             f"(max {max(rel):.1f} kcal/mol); that difference is not a "
             f"physical quantity")
    elif result.get("relative_meaningful") is not False:
        _bad("relative_meaningful should be False for a mixed-formula set")
    elif not result.get("relative_note"):
        _bad("no explanation was given for the missing relative column")
    else:
        print("[PASS] no relative-energy column across different formulas; "
              "the reason is stated", flush=True)

    code2, resp2 = _req("POST", "/api/series",
                        {"molecules": ["butane", "isobutane"],
                         "functional": "b3lyp", "basis": "6-31g*"})
    if code2 == 200 and isinstance(resp2, dict) and resp2.get("job_id"):
        iso = _wait(resp2["job_id"])
        r2 = iso.get("result") or {}
        pts2 = r2.get("points") or []
        rel2 = [p.get("relative_kcal") for p in pts2
                if p.get("relative_kcal") is not None]
        if not r2.get("relative_meaningful"):
            _bad("butane/isobutane share a formula but no relative energies "
                 "were produced")
        elif len(rel2) != 2:
            _bad(f"expected 2 relative energies for the isomer pair, got "
                 f"{len(rel2)}")
        elif abs(min(rel2)) > 1e-6 or min(rel2) < -1e-6:
            _bad(f"the reference of the isomer pair sits at {min(rel2)}, "
                 f"not 0")
        else:
            print(f"[PASS] isomers do get a relative-energy column "
                  f"(0 and {max(rel2):.2f} kcal/mol)", flush=True)
    else:
        _bad(f"could not start the isomer pair: HTTP {code2} {resp2}")

    # ---- 6. the contract the front end reads -----------------------------
    for key in ("job_type", "functional_label", "basis_label", "points",
                "n_ok", "n_failed", "properties"):
        if key not in result:
            _bad(f"result is missing '{key}', which the Series panel reads")
    props = {p.get("key") for p in (result.get("properties") or [])}
    for key in ("gap_ev", "homo_ev", "lumo_ev", "dipole", "relative_kcal"):
        if key not in props:
            _bad(f"properties list has no '{key}' column")
    if result.get("job_type") != "series":
        _bad(f"job_type is {result.get('job_type')!r}, expected 'series'")
    if len(FAILURES) == before + nbad:
        print("[PASS] the result carries every field the Series panel reads",
              flush=True)

    # ---- 7. the chat router sends it here --------------------------------
    # auto_run=False: the point is to see where the request is routed, not to
    # pay for another five single points.
    for text, want in (
        ("compare the HOMO-LUMO gaps of benzene, pyridine, furan and pyrrole",
         4),
        ("calculate the dipole of water, ammonia, methane and hydrogen sulfide",
         4),
    ):
        code, resp = _req("POST", "/api/chat",
                          {"message": text, "auto_run": False})
        if code != 200 or not isinstance(resp, dict):
            _bad(f"chat failed on {text!r}: HTTP {code} {resp}")
            continue
        intents = resp.get("intent") or {}
        mols = intents.get("molecules") or []
        if intents.get("job_type") != "series":
            _bad(f"{text!r} was planned as "
                 f"{intents.get('job_type')!r}, expected 'series'")
        elif len(mols) != want:
            _bad(f"{text!r} kept {len(mols)} molecules ({mols}), expected "
                 f"{want}")
        else:
            print(f"[PASS] chat plans a {want}-molecule request as a series "
                  f"({', '.join(str(m) for m in mols)})", flush=True)

    # auto_run=False must not start a job: the user asked to see the plan
    code, resp = _req("POST", "/api/chat",
                      {"message": "compare water, ammonia and methane",
                       "auto_run": False})
    if isinstance(resp, dict) and resp.get("action") != "confirm":
        _bad(f"auto_run=False still started work: action="
             f"{resp.get('action')!r}, expected 'confirm'")
    elif isinstance(resp, dict):
        print("[PASS] auto_run=False only plans a series, does not start one",
              flush=True)

    # two molecules must still be a comparison, not a series
    code, resp = _req("POST", "/api/chat",
                      {"message": "compare water and ammonia",
                       "auto_run": False})
    if isinstance(resp, dict) and (resp.get("intent") or {}).get(
            "job_type") == "series":
        _bad("a two-molecule request was planned as a series; it should stay "
             "a side-by-side comparison")
    elif isinstance(resp, dict):
        print(f"[PASS] two molecules still plan as "
              f"{(resp.get('intent') or {}).get('job_type')!r}, not series",
              flush=True)

    print()
    if FAILURES:
        print(f"=== FAIL: {len(FAILURES)} series problem(s) ===")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("=== PASS: a set of molecules comes back as that set ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
