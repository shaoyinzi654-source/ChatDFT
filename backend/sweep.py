"""End-to-end functional sweep for ChatDFT.

Drives the live HTTP API exactly the way the browser front-end does, covering a
broad spread of scenarios: molecule library entries, SMILES / formula input,
different functionals and basis sets, open-shell and charged species,
comparison jobs, geometry optimisation, TD-DFT, solvation, concept questions
and error paths.

Usage:  python -m backend.sweep               # http://127.0.0.1:8000
        python -m backend.sweep --port 9000
        python -m backend.sweep --fast        # skip optimisations / TD-DFT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
RESULTS: list[tuple[str, bool, str]] = []


# ----------------------------------------------------------------------
#  HTTP helpers
# ----------------------------------------------------------------------
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
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body


def _wait_job(job_id: str, timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    job: dict = {}
    while time.time() < deadline:
        _, job = _req("GET", f"/api/job/{job_id}")
        if not isinstance(job, dict):
            return {"status": "bad-response", "raw": job}
        if job.get("status") in ("completed", "failed", "error"):
            return job
        time.sleep(1.0)
    job["status"] = "timeout"
    return job


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)


# ----------------------------------------------------------------------
#  scenario driver
# ----------------------------------------------------------------------
def scenario(name: str, message: str, *, expect: str = "compute",
             timeout: float = 300.0, require: list[str] | None = None) -> dict | None:
    """Send a chat message and grade the resulting action / job."""
    code, resp = _req("POST", "/api/chat", {"message": message})
    if code != 200 or not isinstance(resp, dict):
        check(name, False, f"chat HTTP {code}")
        return None

    action = resp.get("action")

    if expect == "answer":
        reply = (resp.get("reply") or "").strip()
        ok = action == "answer" and len(reply) > 40
        check(name, ok, f"action={action}, reply={len(reply)} chars")
        return resp

    if expect == "error":
        check(name, action == "error", f"action={action}")
        return resp

    if action != "compute" or not resp.get("job_id"):
        check(name, False, f"expected a compute job, got action={action}")
        return None

    job = _wait_job(resp["job_id"], timeout=timeout)
    status = job.get("status")
    result = job.get("result") or {}
    ok = status == "completed"

    detail = f"status={status}"
    if status == "completed":
        if result.get("energy_hartree") is not None:
            detail += f", E={result['energy_hartree']:.6f} Ha"
        if result.get("gap_ev") is not None:
            detail += f", gap={result['gap_ev']:.3f} eV"
        if result.get("converged") is False:
            detail += ", SCF-not-converged"
        if result.get("optimized_xyz"):
            detail += ", opt-ok"
        if result.get("excited_states"):
            detail += f", {len(result['excited_states'])} states"
        # optional field assertions
        for key in (require or []):
            if not result.get(key):
                ok = False
                detail += f", MISSING {key}"
    else:
        detail += f", error={(job.get('error') or job.get('reply') or '')[:90]}"

    check(name, ok, detail)
    return job


# ----------------------------------------------------------------------
#  main
# ----------------------------------------------------------------------
def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fast", action="store_true", help="skip slow optimisations / TD-DFT")
    args = ap.parse_args()
    BASE = f"http://127.0.0.1:{args.port}"

    print(f"=== ChatDFT functional sweep -> {BASE} ===\n")

    # ---- 0. health & static assets ----------------------------------
    code, health = _req("GET", "/api/health")
    if isinstance(health, dict):
        eng = health.get("engines", {})
        check("health endpoint", code == 200 and eng.get("pyscf"),
              f"pyscf={eng.get('pyscf')}, rdkit={eng.get('rdkit')}, "
              f"library={health.get('library_size')}, "
              f"functionals={len(health.get('functionals', []))}, "
              f"bases={len(health.get('basis_sets', []))}")
    else:
        check("health endpoint", False, f"HTTP {code}")

    for path, minimum in (("/", 5000), ("/static/js/app.js", 10000),
                          ("/static/css/app.css", 5000)):
        code, body = _req("GET", path)
        size = len(body) if isinstance(body, str) else len(json.dumps(body))
        check(f"static asset {path}", code == 200 and size >= minimum, f"{size} bytes")

    # ---- 1. molecule resolution across input syntaxes ---------------
    resolve_cases = [
        ("resolve: library name water", "water", "water"),
        ("resolve: library name benzene", "benzene", "benzene"),
        ("resolve: library name methanol", "methanol", "methanol"),
        ("resolve: library name naphthalene", "naphthalene", "naphthalene"),
        ("resolve: library name aspirin", "aspirin", "aspirin"),
        ("resolve: SMILES ethanol -> named", "CCO", "ethanol"),
        ("resolve: SMILES pyridine -> named", "c1ccncc1", "pyridine"),
        ("resolve: element Co = cobalt", "Co", "cobalt"),
        ("resolve: formula CO = carbon monoxide", "CO", "carbon monoxide"),
        ("resolve: formula NO = nitric oxide", "NO", "nitric oxide"),
        ("resolve: lowercase formula h2o", "h2o", "water"),
        ("resolve: formula C6H6 = benzene", "C6H6", "benzene"),
    ]
    for label, spec, want in resolve_cases:
        code, resp = _req("POST", "/api/molecule/resolve", {"spec": spec})
        got = ""
        if isinstance(resp, dict):
            got = (resp.get("name") or "").lower()
        ok = code == 200 and want in got
        check(label, ok, f"-> {got!r}, {resp.get('natoms') if isinstance(resp, dict) else '?'} atoms")

    code, resp = _req("POST", "/api/molecule/resolve", {"spec": "zzz-not-a-molecule"})
    check("resolve: unknown rejected cleanly", code == 400, f"HTTP {code}")

    # ---- 2. functional / basis matrix -------------------------------
    matrix = [
        ("HF/STO-3G water", "water", "hf", "sto-3g"),
        ("PBE/def2-TZVP water", "water", "pbe", "def2-tzvp"),
        ("PBE0/def2-TZVP water", "water", "pbe0", "def2-tzvp"),
        ("B3LYP/6-311G* water", "water", "b3lyp", "6-311g*"),
        ("B3LYP/6-31G* benzene", "benzene", "b3lyp", "6-31g*"),
        ("wB97X-D/cc-pVDZ water", "water", "wb97x-d", "cc-pvdz"),
        ("TPSS/def2-SVP water", "water", "tpss", "def2-svp"),
    ]
    for label, mol, func, basis in matrix:
        scenario(label,
                 f"Compute the {func.upper()} {basis} single point energy of {mol}",
                 timeout=180)

    # ---- 3. open-shell / charged ------------------------------------
    scenario("open shell: methyl radical",
             "Compute the energy of the methyl radical CH3", timeout=180)
    scenario("charged: hydroxide anion",
             "Compute the energy of the hydroxide anion OH-", timeout=180)
    scenario("diatomic: carbon monoxide",
             "Compute the energy of carbon monoxide at B3LYP/6-31G*", timeout=180)
    scenario("open shell diatomic: nitric oxide",
             "Compute the energy of nitric oxide at B3LYP/6-31G*", timeout=180)

    # ---- 4. geometry optimisation -----------------------------------
    if not args.fast:
        scenario("optimise water (B3LYP/6-31G*)",
                 "Optimise the geometry of water at B3LYP/6-31G*",
                 timeout=300, require=["optimized_xyz"])
        scenario("optimise benzene (PBE/def2-SVP)",
                 "Optimise the geometry of benzene with PBE/def2-SVP",
                 timeout=300, require=["optimized_xyz"])

    # ---- 5. TD-DFT ---------------------------------------------------
    if not args.fast:
        scenario("TD-DFT formaldehyde, 5 states",
                 "Compute the first 5 excited states of formaldehyde with TD-DFT",
                 timeout=300, require=["excited_states"])

    # ---- 5b. bond scan ------------------------------------------------
    # "scan" was an advertised job type with no implementation: the request
    # fell through to a single point while still being labelled a scan and
    # narrated as one.
    code, resp = _req("POST", "/api/chat", {"message": "scan the O-H bond of water"})
    jid = (resp or {}).get("job_id") if isinstance(resp, dict) else None
    if not jid:
        check("scan: submits", False, f"HTTP {code}")
    else:
        job = _wait_job(jid, timeout=300)
        res = job.get("result") or {}
        sc = res.get("scan") or {}
        check("scan: runs as a scan, not a single point",
              job.get("kind") == "scan" and res.get("job_type") == "scan",
              f"kind={job.get('kind')!r} job_type={res.get('job_type')!r}")
        pts = sc.get("points") or []
        check("scan: returns a curve", len(pts) >= 8, f"{len(pts)} points")
        if pts:
            kmin = sc.get("minimum_index")
            check("scan: minimum_index marks the lowest point",
                  kmin is not None
                  and pts[kmin]["energy_hartree"] == min(
                      p["energy_hartree"] for p in pts),
                  f"minimum_index={kmin} at r={sc.get('minimum', {}).get('r')} A")
            # a rigid scan of a real bond must rise on both sides of the well
            check("scan: the curve rises on both sides of the well",
                  pts[0]["energy_rel_kcal"] > 0.5
                  and pts[-1]["energy_rel_kcal"] > 0.5,
                  f"ends at {pts[0]['energy_rel_kcal']:.1f} / "
                  f"{pts[-1]['energy_rel_kcal']:.1f} kcal/mol")
        text = job.get("explanation") or ""
        check("scan: is narrated as a scan",
              "rigid" in text.lower() and "minimum" in text.lower(),
              f"{len(text)} chars")

    # a bond the molecule does not have must be refused, not silently replaced
    code, resp = _req("POST", "/api/chat", {"message": "scan the Fe-Fe bond of water"})
    if isinstance(resp, dict):
        check("scan: an absent bond is reported, not substituted",
              resp.get("action") == "error"
              and "no Fe-Fe bond" in (resp.get("reply") or ""),
              f"action={resp.get('action')!r}: {(resp.get('reply') or '')[:60]!r}")
    else:
        check("scan: an absent bond is reported, not substituted", False,
              f"HTTP {code}")

    # ---- 6. comparison ------------------------------------------------
    scenario("compare water vs ammonia", "Compare water and ammonia", timeout=300)

    # comparing a molecule with itself used to lose the second molecule
    # (_extract_molecules de-duplicates), and the job then ran as a single
    # point while still being labelled "compare" and narrated as a completed
    # comparison.
    code, resp = _req("POST", "/api/chat", {"message": "Compare water and water"})
    jid = (resp or {}).get("job_id") if isinstance(resp, dict) else None
    if not jid:
        check("compare: self-comparison submits", False, f"HTTP {code}")
    else:
        job = _wait_job(jid, timeout=300)
        text = (job or {}).get("explanation") or ""
        check("compare: self-comparison stays a compare job",
              (job or {}).get("kind") == "compare",
              f"kind={(job or {}).get('kind')!r}")
        check("compare: self-comparison is narrated, not stubbed",
              "HOMO-LUMO gap" in text and "rounding" in text,
              f"{len(text)} chars: {text[:60]!r}")

    # ---- 7. solvation -------------------------------------------------
    if not args.fast:
        scenario("water in implicit solvent",
                 "Compute the energy of water in water solvent", timeout=300)

    # ---- 8. concept questions (no compute job) -----------------------
    scenario("concept: what is DFT", "What is density functional theory?", expect="answer")
    scenario("concept: HOMO-LUMO gap", "Explain what the HOMO-LUMO gap means", expect="answer")
    scenario("concept: basis set", "What is a basis set?", expect="answer")

    # ---- 9. error paths ------------------------------------------------
    code, resp = _req("POST", "/api/chat", {"message": ""})
    check("empty message handled without crash", code in (200, 400, 422), f"HTTP {code}")

    # ---- 9b. solvent parsing -------------------------------------------
    # Regression: the solvent scan used to match the *molecule* name, so
    # "the energy of water in dmso" silently solvated in water (eps 78.3553).
    solvent_cases = [
        ("solvent: in dmso", "Compute the energy of water in dmso", "46.826"),
        ("solvent: in chloroform", "Compute the energy of benzene in chloroform", "4.7113"),
        ("solvent: in hexane", "Compute the energy of water in hexane", "1.8819"),
        ("solvent: in acetonitrile", "Compute the energy of water in acetonitrile", "35.688"),
        ("solvent: in toluene", "Compute the energy of acetone in toluene", "2.3741"),
        ("solvent: molecule alone is not a solvent", "Compute the energy of benzene", None),
        ("solvent: 'with benzene' is not solvation", "Compare water with benzene", None),
    ]
    for label, message, want_eps in solvent_cases:
        code, resp = _req("POST", "/api/chat", {"message": message, "auto_run": False})
        got = None
        if isinstance(resp, dict):
            got = (resp.get("intent") or {}).get("solvation")
        check(label, got == want_eps, f"solvation={got!r} (expected {want_eps!r})")

    # ---- 10. physics reference values (structured API) -----------------
    def structured(molecule, **kw) -> dict | None:
        body = {"molecule": molecule, "kind": "single_point",
                "functional": "b3lyp", "basis": "6-31g*"}
        body.update(kw)
        code, resp = _req("POST", "/api/job", body)
        if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
            check(f"reference: {molecule}", False, f"submit HTTP {code}")
            return None
        job = _wait_job(resp["job_id"], timeout=240)
        if not job or job.get("status") != "completed":
            check(f"reference: {molecule}", False,
                  f"job {job.get('status')}: {(job.get('error') or '')[:90]}")
            return None
        return job.get("result") or None

    def ref(label, molecule, key, lo, hi, **kw):
        res = structured(molecule, **kw)
        if not res:
            return
        val = res.get(key)
        if isinstance(val, dict):          # e.g. the dipole block
            val = val.get("magnitude")
        ok = isinstance(val, (int, float)) and lo <= val <= hi
        check(label, ok, f"{key}={val:.4f} (expected {lo}..{hi})")

    ref("ref: water B3LYP/6-31G* energy", "water", "energy_hartree", -76.409, -76.405)
    ref("ref: water dipole ~1.85 D", "water", "dipole", 1.70, 2.30)
    ref("ref: water HOMO-LUMO gap 8-11 eV", "water", "gap_ev", 8.0, 11.0)
    ref("ref: benzene dipole = 0 (D6h)", "benzene", "dipole", -0.001, 0.001)
    ref("ref: CO dipole 0.05-0.20 D", "CO", "dipole", 0.05, 0.20)
    ref("ref: NO dipole 0.05-0.25 D (open shell)", "NO", "dipole", 0.05, 0.25)
    ref("ref: OH radical dipole 1.4-2.0 D", "HO", "dipole", 1.4, 2.0)
    ref("ref: methyl radical dipole = 0 (D3h)", "CH3", "dipole", -0.01, 0.01)
    ref("ref: O2 dipole = 0 (D-inf-h)", "oxygen", "dipole", -0.001, 0.001, multiplicity=3)

    # spin-state regressions: the parity guard must not flatten a genuine triplet
    res = structured("oxygen", multiplicity=3)
    if res:
        check("ref: O2 kept as a triplet (mult=3)", res.get("multiplicity") == 3,
              f"multiplicity={res.get('multiplicity')}, E={res.get('energy_hartree'):.4f} Ha")
    res = structured("oxygen")       # no explicit spin: keep the library ground state
    if res:
        check("ref: O2 default spin = triplet", res.get("multiplicity") == 3,
              f"multiplicity={res.get('multiplicity')}")
    res = structured("CH3", multiplicity=1)      # inconsistent request -> corrected
    if res:
        check("ref: CH3 corrected to a doublet", res.get("multiplicity") == 2,
              f"multiplicity={res.get('multiplicity')} (requested 1, 9 electrons)")

    # ---- 10b. library-geometry regression ------------------------------
    # Several entries once shipped with corrupt coordinates: naphthalene had
    # two hydrogens 0.43 A from a ring carbon and a carbon carrying no
    # hydrogen (which collapsed its HOMO-LUMO gap to 2.29 eV), furan and
    # pyrrole stored a six-membered ring (a different molecule), and
    # chloroform had 1.09 A C-Cl bonds.  The identity and the dipole/gap
    # fingerprints below cannot be reproduced from a bad geometry, so this
    # is the downstream guard for the static integrity gate in
    # backend/check_library.py.
    def identity(molecule, formula, natoms):
        res = structured(molecule, functional="hf", basis="sto-3g")
        if not res:
            return
        mol = res.get("molecule") or {}
        check(
            f"lib: {molecule} resolves to {formula}",
            mol.get("formula") == formula and mol.get("natoms") == natoms,
            f"formula={mol.get('formula')!r} natoms={mol.get('natoms')!r}",
        )

    identity("naphthalene", "C10H8", 18)
    identity("furan", "C4H4O", 9)
    identity("pyrrole", "C4H5N", 10)
    identity("caffeine", "C8H10N4O2", 24)

    ref("lib: naphthalene gap 4.0-5.2 eV (aromatic)", "naphthalene", "gap_ev", 4.0, 5.2)
    ref("lib: naphthalene dipole = 0 (D2h)", "naphthalene", "dipole", -0.001, 0.001)
    ref("lib: furan dipole 0.4-1.1 D", "furan", "dipole", 0.4, 1.1)
    ref("lib: pyrrole dipole 1.6-2.1 D", "pyrrole", "dipole", 1.6, 2.1)
    ref("lib: chloroform dipole 0.9-1.5 D", "chloroform", "dipole", 0.9, 1.5)
    ref("lib: acetone dipole 2.6-3.1 D", "acetone", "dipole", 2.6, 3.1)
    ref("lib: ethanol dipole 1.4-1.9 D", "ethanol", "dipole", 1.4, 1.9)

    # solvation must stabilise a polar solute relative to the gas phase
    gas = structured("water")
    sol = structured("water", solvation="78.3553")
    if gas and sol:
        de_kcal = (sol["energy_hartree"] - gas["energy_hartree"]) * 627.5095
        check("ref: solvation lowers the energy", de_kcal < -1.0,
              f"dE = {de_kcal:.2f} kcal/mol "
              f"({gas['energy_hartree']:.6f} -> {sol['energy_hartree']:.6f} Ha)")

    # ---- summary -------------------------------------------------------
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n=== {passed}/{total} checks passed ===")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  ({detail})")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
