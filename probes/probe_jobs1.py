"""Time the two jobs the contract gate reported as timing out.

Runs a vibrations job (optimise first, then the Hessian) and a surfaces job
against a live server and prints how long each took, so a timeout can be told
apart from a defect.  Changes nothing.

Run:  python probes/probe_jobs1.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

BASE = "http://127.0.0.1:8000"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def wait(jid: str, limit: float = 900.0):
    t0 = time.time()
    while time.time() - t0 < limit:
        r = json.load(urllib.request.urlopen(f"{BASE}/api/job/{jid}", timeout=60))
        # the server says "completed", not "done"
        if r.get("status") in ("completed", "failed", "error"):
            return r, time.time() - t0
        time.sleep(2)
    return {"status": "timeout"}, time.time() - t0


def main() -> int:
    for molecule, kind, extra in (("formaldehyde", "surfaces", {}),
                                  ("water", "surfaces", {})):
        r = post("/api/job", dict(molecule=molecule, kind=kind,
                                  functional="b3lyp", basis="6-31g*", **extra))
        jid = r.get("job_id")
        print(f"{molecule}/{kind}: job {jid}", flush=True)
        res, dt = wait(jid)
        print(f"   status={res.get('status')}  {dt:.1f}s", flush=True)
        r2 = res.get("result") or {}
        if kind == "vibrations":
            print("   modes", len(r2.get("frequencies_cm1") or []),
                  "| opt_converged", r2.get("opt_converged"),
                  "| entropy", (r2.get("thermochemistry") or {}).get("entropy_j_mol_k"))
        else:
            items = (r2.get("surfaces") or {}).get("surfaces") or []
            mep = next((i for i in items if i.get("kind") == "mep"), {})
            print("   atom_extremes:",
                  json.dumps(mep.get("atom_extremes"))[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
