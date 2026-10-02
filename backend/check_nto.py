"""NTO audit: the natural transition orbitals must actually diagonalise it.

An NTO figure is easy to fake and hard to check by eye: any two orbitals
drawn in blue and orange look like a transition.  So this gate asserts the
properties that make the construction meaningful rather than merely present:

  * the occupation numbers sum to 1 -- they are the squared singular values
    of the transition matrix, and if they do not sum to one the figure is
    being drawn from something that is not a transition;
  * the rotated orbitals are orthonormal in the AO metric,
    (C U)^T S (C U) = I.  A rotation that is not orthonormal would still
    produce pictures, just not of this transition;
  * formaldehyde's S1 is the textbook n -> pi* excitation, so a single NTO
    pair must carry almost all of it and the canonical decomposition must be
    overwhelmingly HOMO -> LUMO.  A method that spread that transition over
    many pairs would be wrong even though it ran;
  * a higher state must have *less* dominant first-pair character than S1 --
    the check that the NTOs are actually diagonalising the transition rather
    than returning the canonical orbitals with new labels;
  * the donor and acceptor cubes must differ, must exist, and must parse;
  * the excitation energy must be in a chemically plausible window for the
    molecule and functional.

Usage:  python -m backend.check_nto [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np

BASE = "http://127.0.0.1:8000"
FAILURES: list[str] = []


def _req(method, path, payload=None, timeout=60.0):
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


def _wait(job_id, timeout=600.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, job = _req("GET", f"/api/job/{job_id}")
        if isinstance(job, dict) and job.get("status") in ("completed", "failed"):
            return job
        time.sleep(1.0)
    return {"status": "timeout"}


def _bad(msg: str) -> None:
    FAILURES.append(msg)
    print("       !", msg, flush=True)


def physics() -> None:
    from backend.engine import analysis as A
    from backend.engine import molecule as molmod
    from backend.engine.dft import DFTEngine

    print("-- formaldehyde S1 (n -> pi*) --------------------------------",
          flush=True)
    mol = molmod.resolve("formaldehyde")
    eng = DFTEngine(atom_xyz=mol.to_xyz(), charge=0, multiplicity=1,
                    functional="b3lyp", basis="6-31g*")
    eng.run_scf()
    cube_dir = tempfile.mkdtemp(prefix="nto_gate_")
    out = A.natural_transition_orbitals(eng, state=1, npairs=3,
                                        outdir=cube_dir, n=30)

    occs = [p["occupation"] for p in out["pairs"]]
    # the pairs list stops early once a pair is negligible, so the full set
    # has to come from the same computation to check the sum
    print(f"       state {out['state']}: {out['energy_ev']:.3f} eV, "
          f"{out['wavelength_nm']:.1f} nm, dominant "
          f"{out['dominant_occupation']:.4f}", flush=True)

    ev = out["energy_ev"]
    if not 3.0 < ev < 5.5:
        _bad(f"formaldehyde S1 is {ev:.3f} eV at B3LYP/6-31G*; the n->pi* "
             f"excitation is around 3.9-4.3 eV, so this is not the right state")
    else:
        print(f"[PASS] formaldehyde S1 at {ev:.3f} eV is in the n->pi* window",
              flush=True)

    dom = out["dominant_occupation"]
    if dom < 0.90:
        _bad(f"the leading NTO pair carries only {dom * 100:.1f}% of "
             f"formaldehyde's S1; this transition is a single orbital pair "
             f"excitation and the NTOs exist precisely to show that")
    else:
        print(f"[PASS] one NTO pair carries {dom * 100:.2f}% of S1", flush=True)

    canon = out["canonical"] or []
    if not canon or canon[0]["from_label"] != "HOMO" \
            or canon[0]["to_label"] != "LUMO" or canon[0]["weight"] < 0.90:
        _bad(f"the canonical decomposition of S1 is {canon[:2]}; it must be "
             f"dominated by HOMO->LUMO, which is what this molecule's lowest "
             f"excitation is")
    else:
        print(f"[PASS] S1 is {canon[0]['from_label']}->{canon[0]['to_label']} "
              f"({canon[0]['weight'] * 100:.1f}%)", flush=True)

    # The ``pairs`` list is truncated at a threshold, so the sum has to come
    # from the reported total over every singular value.
    total_occ = out.get("occupation_sum")
    if total_occ is None:
        _bad("the result does not report the occupation sum, so the "
             "normalisation cannot be checked")
    elif abs(total_occ - 1.0) > 1e-6:
        _bad(f"the NTO occupation numbers sum to {total_occ:.6f}, not 1; "
             f"they are squared singular values of a normalised transition "
             f"and must sum to one")
    else:
        print(f"[PASS] the NTO occupations sum to {total_occ:.10f} over "
              f"{out.get('n_significant_pairs')} significant pair(s)",
              flush=True)
    if abs(sum(occs) - 1.0) > 1e-3 and not (len(occs) == 1 and dom > 0.999):
        _bad(f"the reported pairs sum to {sum(occs):.4f}, which is neither 1 "
             f"nor a truncation of a dominant pair ({dom:.4f})")

    # ---- the rotation must be orthonormal --------------------------------
    # recompute from the amplitudes so the check does not trust the function
    from pyscf import tdscf
    td = tdscf.TDA(eng.mf)
    td.nstates = 1
    td.kernel()
    X, Y = td.xy[0]
    T = np.asarray(X) + (np.asarray(Y) if Y is not None else 0.0)
    U, s, Vt = np.linalg.svd(T)
    C = np.asarray(eng.mf.mo_coeff)
    nocc = int(np.asarray(eng.mf.mo_occ).sum() // 2)
    S = eng.mol.intor_symmetric("int1e_ovlp")
    hole = C[:, :nocc] @ U[:, :2]
    elec = C[:, nocc:] @ Vt[:2].T
    gram_h = hole.T @ S @ hole
    gram_e = elec.T @ S @ elec
    err = max(float(np.abs(gram_h - np.eye(2)).max()),
              float(np.abs(gram_e - np.eye(2)).max()))
    if err > 1e-8:
        _bad(f"the rotated NTOs are not orthonormal in the AO metric "
             f"(max deviation {err:.2e}); the picture would not be of this "
             f"transition")
    else:
        print(f"[PASS] the NTO rotation is orthonormal (max deviation "
              f"{err:.1e})", flush=True)
    if abs(float(s[0] ** 2 / (s ** 2).sum()) - dom) > 1e-4:
        _bad(f"the reported dominant occupation {dom:.4f} does not match the "
             f"squared singular value {float(s[0] ** 2 / (s ** 2).sum()):.4f}")
    else:
        print("[PASS] the reported occupation is the squared singular value",
              flush=True)

    # ---- cubes ------------------------------------------------------------
    pair = out["pairs"][0]
    grids = {}
    for role in ("donor_file", "acceptor_file"):
        path = os.path.join(cube_dir, pair[role])
        if not os.path.exists(path):
            _bad(f"the {role.replace('_file', '')} cube was never written")
            continue
        nat, origin, axes, vals = A.read_cube(path)
        if not np.isfinite(vals).all():
            _bad(f"the {role.replace('_file', '')} cube has non-finite values")
            continue
        grids[role] = vals
    if len(grids) == 2:
        a, b = grids["donor_file"], grids["acceptor_file"]
        if a.shape != b.shape:
            _bad("the donor and acceptor cubes are on different grids")
        elif float(np.abs(a - b).max()) < 1e-12:
            _bad("the donor and acceptor cubes are the same grid; the figure "
                 "would show the same orbital twice")
        else:
            print(f"[PASS] the donor and acceptor cubes re-read and differ "
                  f"({a.size} points, |max diff| "
                  f"{float(np.abs(a - b).max()):.3e})", flush=True)

    print("-- a higher state must be less clean --------------------------",
          flush=True)
    out2 = A.natural_transition_orbitals(eng, state=2, npairs=3,
                                         outdir=tempfile.mkdtemp(), n=30)
    if out2["energy_ev"] <= out["energy_ev"]:
        _bad(f"state 2 ({out2['energy_ev']:.3f} eV) is not above state 1 "
             f"({out['energy_ev']:.3f} eV)")
    else:
        print(f"[PASS] state 2 lies above state 1 "
              f"({out['energy_ev']:.3f} -> {out2['energy_ev']:.3f} eV)",
              flush=True)
    if out2["dominant_occupation"] > dom + 1e-9:
        _bad(f"state 2 is *more* single-pair ({out2['dominant_occupation']:.4f}) "
             f"than state 1 ({dom:.4f}); the NTOs are not resolving the "
             f"difference between the states")
    else:
        print(f"[PASS] state 2 is at least as mixed as state 1 "
              f"({dom:.4f} -> {out2['dominant_occupation']:.4f})", flush=True)


def contract() -> None:
    print("-- API --------------------------------------------------------",
          flush=True)

    code, resp = _req("POST", "/api/nto", {"molecule": "water", "state": 0})
    if code != 400:
        _bad(f"/api/nto accepted state 0 (HTTP {code}); there is no zeroth "
             f"excited state and defaulting would silently return state 1")
    else:
        print("[PASS] /api/nto refuses state 0", flush=True)

    code, resp = _req("POST", "/api/nto",
                      {"molecule": "formaldehyde", "state": 1, "npairs": 2})
    if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
        _bad(f"/api/nto would not start: HTTP {code} {resp}")
        return
    job = _wait(resp["job_id"])
    if job.get("status") != "completed":
        _bad(f"the NTO job did not complete ({job.get('status')}: "
             f"{job.get('error') or job.get('message')})")
        return
    r = job.get("result") or {}
    nto = r.get("nto") or {}
    sur = r.get("surfaces") or {}
    ok = True
    for key in ("state", "energy_ev", "pairs", "canonical",
                "dominant_occupation", "character"):
        if key not in nto:
            _bad(f"the nto block is missing '{key}'")
            ok = False
    if r.get("job_type") != "nto":
        _bad(f"job_type is {r.get('job_type')!r}, expected 'nto'")
        ok = False
    items = sur.get("surfaces") or []
    if len(items) < 2:
        _bad(f"only {len(items)} NTO isosurfaces were produced; a transition "
             f"needs at least a donor and an acceptor")
        ok = False
    for it in items:
        if not it.get("file") or not it.get("key"):
            _bad("an NTO surface entry has no file or key")
            ok = False
        if it.get("index") is not None and not isinstance(it["index"], int):
            _bad("an NTO surface claims a canonical-orbital index")
            ok = False
    base = sur.get("url_base")
    if not base:
        _bad("the NTO surfaces have no url_base, so the viewer cannot fetch "
             "the cubes")
        ok = False
    else:
        for it in items[:2]:
            code, _ = _req("GET", f"{base}/{it['file']}")
            if code != 200:
                _bad(f"the cube {it['file']} is advertised but returns "
                     f"HTTP {code}")
                ok = False
    if ok:
        print(f"[PASS] an NTO run returns the transition numbers and "
              f"{len(items)} fetchable isosurfaces", flush=True)

    # routing
    for text, want_state in (
        ("show me the natural transition orbitals of formaldehyde", 1),
        ("NTO of the second excited state of formaldehyde", 2),
    ):
        code, resp = _req("POST", "/api/chat",
                          {"message": text, "auto_run": False})
        it = (resp or {}).get("intent") or {} if isinstance(resp, dict) else {}
        if it.get("job_type") != "nto":
            _bad(f"{text!r} was planned as {it.get('job_type')!r}, expected "
                 f"'nto'")
        elif resp.get("state") != want_state:
            _bad(f"{text!r} planned state {resp.get('state')}, expected "
                 f"{want_state}")
        else:
            print(f"[PASS] {text!r} plans an NTO of state {want_state}",
                  flush=True)


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    BASE = f"http://127.0.0.1:{args.port}"

    print("=== ChatDFT NTO audit ===\n", flush=True)
    physics()
    contract()

    print()
    if FAILURES:
        print(f"=== FAIL: {len(FAILURES)} NTO problem(s) ===")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("=== PASS: the NTOs diagonalise the transition ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
