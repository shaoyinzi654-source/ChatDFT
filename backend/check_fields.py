"""Field audit: the ELF, the Laplacian and the spin density must be right.

A contour map is the easiest thing in the program to get *almost* right. It
renders, it looks like the figure in the paper, and nobody can tell whether
the number behind each pixel means what it claims to. So this gate asserts
things that are falsifiable rather than things that merely run:

  * the ELF of H2 at the bond midpoint is 1 -- the two electrons are one
    localised pair, which is the definition the function was built from;
  * the ELF of the same molecule far away is 0;
  * the ELF never leaves [0, 1];
  * the Thomas-Fermi constant is the spin-resolved one, verified by
    rebuilding the closed-shell constant from it: it is the single number
    that, if wrong, shifts every ELF value while looking perfectly plausible;
  * ∇²ρ at the H2 bond midpoint is **negative** -- Bader's criterion for a
    shared (covalent) interaction.  A sign error here would produce a
    beautiful map that classifies every covalent bond as ionic;
  * the spin density of a closed shell is zero to machine precision, and the
    spin density of a doublet integrates to the number of unpaired electrons;
  * the density difference of a dimer is signed -- charge both accumulates
    and is depleted, which is the whole point of the figure.

Plus the API contract: an unknown field must be refused, and a result must
carry the grid, the atoms projected into the plane and the contour levels.

Usage:  python -m backend.check_fields [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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


def _engine(xyz: str, mult: int = 1, functional: str = "b3lyp",
            basis: str = "6-31g*"):
    from backend.engine.dft import DFTEngine
    eng = DFTEngine(atom_xyz=xyz, charge=0, multiplicity=mult,
                    functional=functional, basis=basis)
    eng.run_scf()
    return eng


def physics() -> None:
    from backend.engine import fields as F

    print("-- ELF --------------------------------------------------------",
          flush=True)

    # the Thomas-Fermi constant: two spin channels must rebuild the
    # closed-shell value (3/10)(3 pi^2)^(2/3) = 2.8712
    rho = 0.37
    rebuilt = 2.0 * F.TF_SIGMA * (rho / 2.0) ** (5.0 / 3.0)
    direct = 0.3 * (3.0 * math.pi ** 2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    if abs(rebuilt - direct) > 1e-10 * max(1.0, abs(direct)):
        _bad(f"the ELF reference constant is wrong: two spin channels give "
             f"{rebuilt:.6f} rho^(5/3) but the closed-shell value is "
             f"{direct:.6f} rho^(5/3). A uniform electron gas would not come "
             f"out at ELF = 0.5.")
    else:
        print(f"[PASS] two spin channels rebuild C_F = {direct / rho ** (5/3):.4f} "
              f"(uniform gas -> ELF 0.5)", flush=True)

    eng = _engine("2\nH2\nH 0 0 0\nH 0 0 0.7414\n")
    mol, mf = eng.mol, eng.mf
    bohr = F.BOHR
    mid = np.array([[0.0, 0.0, 0.3707]]) / bohr
    far = np.array([[0.0, 0.0, 9.0]]) / bohr

    chans = F.spin_channels(mf)
    e_mid = float(F.elf(mol, chans, mid)[0])
    e_far = float(F.elf(mol, chans, far)[0])
    if not e_mid > 0.95:
        _bad(f"ELF at the H2 bond midpoint is {e_mid:.4f}; it must approach 1 "
             f"because the two electrons form a single localised pair there")
    else:
        print(f"[PASS] ELF = {e_mid:.4f} at the H2 bond midpoint (one localised "
              f"pair)", flush=True)
    if not e_far < 0.05:
        _bad(f"ELF 9 A away from H2 is {e_far:.4f}; there is no density there, "
             f"so it must be ~0")
    else:
        print(f"[PASS] ELF = {e_far:.4f} in the far field", flush=True)

    print("-- Laplacian --------------------------------------------------",
          flush=True)
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    lap_mid = float(F.laplacian(mol, dm, mid)[0])
    if not lap_mid < 0.0:
        _bad(f"Laplacian at the H2 bond midpoint is {lap_mid:+.4f}; it must be "
             f"negative for a shared (covalent) interaction. A positive value "
             f"would classify this bond as ionic.")
    else:
        print(f"[PASS] Laplacian = {lap_mid:+.4f} at the bond midpoint "
              f"(negative: shared/covalent)", flush=True)
    lap_far = float(F.laplacian(mol, dm, far)[0])
    if abs(lap_far) > 1e-6:
        _bad(f"Laplacian 9 A away is {lap_far:.3e}; it should vanish")
    else:
        print("[PASS] Laplacian vanishes in the far field", flush=True)

    print("-- spin density ----------------------------------------------",
          flush=True)
    w = _engine("3\nwater\nO 0 0 0.119262\nH 0 0.763239 -0.477047\n"
                "H 0 -0.763239 -0.477047\n")
    dmw = np.asarray(w.mf.make_rdm1())
    sd, _ = F.spin_density(w.mol, [dmw], mid)
    if float(np.abs(sd).max()) > 1e-12:
        _bad(f"a closed-shell calculation has non-zero spin density "
             f"({float(np.abs(sd).max()):.3e}); the map must be blank because "
             f"that is the correct answer")
    else:
        print("[PASS] spin density of a closed shell is zero to machine "
              "precision", flush=True)

    r = _engine("4\nmethyl\nC 0 0 0\nH 0 1.079 0\nH 0.934 -0.540 0\n"
                "H -0.934 -0.540 0\n", mult=2)
    dmr = np.asarray(r.mf.make_rdm1())
    if dmr.ndim != 3:
        _bad("the methyl radical came out restricted; the spin density would "
             "be zero and the figure empty")
    else:
        grid, shape = F.box_grid(r.mol, spacing=0.14, margin=3.0)
        sd, tot = F.spin_density(r.mol, [dmr[0], dmr[1]], grid)
        step = 0.14 ** 3
        nspin = float(sd.sum() * step)
        if not 0.90 < nspin < 1.10:
            _bad(f"the spin density integrates to {nspin:.3f}, not 1.0; "
                 f"a doublet has exactly one unpaired electron")
        else:
            print(f"[PASS] spin density of the methyl radical integrates to "
                  f"{nspin:.3f} (one unpaired electron)", flush=True)
        if float(sd.max()) <= 0:
            _bad("the spin density has no positive region at all")

    print("-- density difference ----------------------------------------",
          flush=True)
    dimer = ("6\nwater dimer\n"
             "O -1.43294521 -0.18541674 0.0\n"
             "H -1.94367185  0.63669500 0.0\n"
             "H -0.50231688  0.11190875 0.0\n"
             "O  1.40172699  0.21428286 0.0\n"
             "H  1.55292418 -0.36271733  0.76553811\n"
             "H  1.55292418 -0.36271733 -0.76553811\n")
    d = _engine(dimer)
    plane = F.plane_grid(d.mol, "xy", n=40, spacing=0.12)
    try:
        res = F.density_difference(d, [[0, 1, 2], [3, 4, 5]], plane["coords"])
    except Exception as exc:                          # noqa: BLE001
        _bad(f"the density difference could not be computed: {exc}")
    else:
        v = res["values"]
        if not (v.max() > 1e-5 and v.min() < -1e-5):
            _bad(f"the dimer difference map is one-signed "
                 f"(min {v.min():.3e}, max {v.max():.3e}); binding must move "
                 f"charge both ways, so the map has to have both signs")
        else:
            print(f"[PASS] the dimer difference map is signed: charge both "
                  f"accumulates ({v.max():.2e}) and depletes ({v.min():.2e})",
                  flush=True)
        # sanity: the difference must be small compared with the density itself
        scale = float(np.abs(res["total_density"]).max())
        if scale > 0 and abs(v).max() > 0.5 * scale:
            _bad(f"the difference ({abs(v).max():.3e}) is comparable to the "
                 f"total density ({scale:.3e}); that is not a rearrangement, "
                 f"it is a different molecule")


def contract() -> None:
    print("-- API --------------------------------------------------------",
          flush=True)

    code, resp = _req("POST", "/api/field",
                      {"molecule": "benzene", "kind": "not_a_field"})
    if code != 400:
        _bad(f"/api/field accepted an unknown field (HTTP {code}); the UI "
             f"menu and the engine would then disagree")
    else:
        print("[PASS] /api/field refuses an unknown field", flush=True)

    code, resp = _req("GET", "/api/fields")
    kinds = {}
    if isinstance(resp, dict):
        kinds = {f.get("key") for f in (resp.get("fields") or [])}
    if not {"elf", "laplacian", "spin", "difference"} <= kinds:
        _bad(f"/api/fields lists {sorted(kinds)}; elf, laplacian, spin and "
             f"difference must all be offered")
    else:
        print("[PASS] /api/fields offers every implemented field", flush=True)

    code, resp = _req("POST", "/api/field",
                      {"molecule": "water", "kind": "elf", "plane": "xy",
                       "n": 60})
    if code != 200 or not isinstance(resp, dict) or not resp.get("job_id"):
        _bad(f"/api/field would not start: HTTP {code} {resp}")
        return
    job = _wait(resp["job_id"])
    if job.get("status") != "completed":
        _bad(f"an ELF job did not complete ({job.get('status')}: "
             f"{job.get('error') or job.get('message')})")
        return
    r = job.get("result") or {}
    n = r.get("n")
    vals = r.get("values") or []
    ok = True
    if not n or len(vals) != n or any(len(row) != n for row in vals):
        _bad(f"the grid is not {n}x{n}: got {len(vals)} rows")
        ok = False
    flat = [v for row in vals for v in row]
    if flat and not all(isinstance(v, (int, float)) for v in flat):
        _bad("the grid contains non-numbers")
        ok = False
    if flat and (min(flat) < -1e-9 or max(flat) > 1.0 + 1e-9):
        _bad(f"ELF left [0,1]: {min(flat):.4f} .. {max(flat):.4f}")
        ok = False
    for key in ("levels", "atoms", "u_range", "v_range", "note", "label",
                "units", "functional_label", "basis_label"):
        if key not in r:
            _bad(f"the result is missing '{key}', which the Fields panel reads")
            ok = False
    if r.get("atoms") and not all(
            {"symbol", "u", "v"} <= set(a) for a in r["atoms"]):
        _bad("projected atoms are missing symbol/u/v, so the map cannot show "
             "where the nuclei are")
        ok = False
    if ok:
        print(f"[PASS] an ELF run returns a {n}x{n} grid in [0,1] with atoms, "
              f"levels and units", flush=True)

    # a closed-shell spin request must say so instead of drawing a blank map
    code, resp = _req("POST", "/api/field",
                      {"molecule": "water", "kind": "spin", "n": 40})
    if code == 200 and isinstance(resp, dict) and resp.get("job_id"):
        job = _wait(resp["job_id"])
        r = job.get("result") or {}
        note = (r.get("note") or "") + " "
        flat = [v for row in (r.get("values") or []) for v in row]
        if flat and max(abs(v) for v in flat) > 1e-12:
            _bad("a closed-shell spin density is not zero")
        elif "closed-shell" not in note.lower():
            _bad("the blank closed-shell spin map does not explain that the "
                 "empty picture is the correct answer")
        else:
            print("[PASS] a closed-shell spin map is zero and says why",
                  flush=True)


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    BASE = f"http://127.0.0.1:{args.port}"

    print("=== ChatDFT field audit ===\n", flush=True)
    physics()
    contract()

    print()
    if FAILURES:
        print(f"=== FAIL: {len(FAILURES)} field problem(s) ===")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("=== PASS: every field means what it claims to mean ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
