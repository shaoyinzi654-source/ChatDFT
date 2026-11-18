"""Reaction-path audit: the barrier, and whether the maximum is a real TS.

An energy-profile figure is the one figure in a mechanism paper that carries a
*number* everyone will quote.  It is also the easiest figure to get wrong while
looking perfect: a plot of energy against a coordinate renders identically
whether the barrier is a true saddle or just the highest point of a scan.

So this gate asserts things that can be false:

  * the Hessian machinery returns the right *sign pattern* for structures whose
    answer is known independently -- staggered ethane is a minimum (zero
    imaginary frequencies), eclipsed ethane is the saddle (exactly one), and
    the mode that flips sign is the torsion, not a bond stretch;
  * the same mode that is real at the minimum is the imaginary one at the
    saddle, at a similar magnitude -- that is what "the reaction coordinate is
    this torsion" means;
  * the ethane torsion barrier comes out near the experimental 2.9 kcal/mol,
    from a relaxed scan, with the curve symmetric about the eclipsed maximum;
  * the two endpoints relax to the *same* conformer, so the reaction energy is
    zero -- a scan whose ends disagree by a large number is a broken window;
  * every point on the curve sits on the coordinate that was asked for;
  * the whole figure shares one energy zero, so the scan points and the relaxed
    endpoints can be plotted on the same axis without an invisible offset;
  * the intrinsic reaction coordinate descends monotonically on *both*
    branches, keeps its step size (a preconditioner is doing its job only if
    the line search never has to shrink the step), and the two branches are
    mirror images that leave the saddle in opposite directions;
  * a degenerate reaction -- the ethane torsion, where both endpoints relax to
    the same conformer -- is recognised as degenerate and is *not* given a
    direction it does not have;
  * the API refuses an unreadable coordinate, and a coordinate that names an
    atom the molecule does not have fails with a message that says so.

Usage:  python -m backend.check_reaction [--port 8000]
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
FUNC, BASIS = "b3lyp", "6-31g*"


def _req(method, path, payload=None, timeout=120.0):
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


def _wait(job_id, timeout=1800.0) -> dict:
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


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}", flush=True)


# ======================================================================
#  geometry helpers
# ======================================================================
def _staggered_ethane_xyz() -> str:
    """Ethane from RDKit, so the starting geometry is a real minimum."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("CC"))
    AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE)
    AllChem.MMFFOptimizeMolecule(m)
    conf = m.GetConformer()
    lines = [str(m.GetNumAtoms()), "ethane"]
    for i, a in enumerate(m.GetAtoms()):
        p = conf.GetAtomPosition(i)
        lines.append(f"{a.GetSymbol()} {p.x:.8f} {p.y:.8f} {p.z:.8f}")
    return "\n".join(lines) + "\n"


def _rotate_methyl(coords_bohr: np.ndarray, symbols, axis, deg: float):
    """Rotate the hydrogens on one end of ``axis`` about the axis bond.

    Staggered ethane rotated 60 degrees about its C-C bond is the eclipsed
    conformer exactly.  Building the saddle this way instead of relaxing to it
    means the *only* thing wrong with the structure is the torsion, so a
    Hessian that shows more than one imaginary mode is the Hessian's fault and
    not the geometry's.
    """
    c = np.asarray(coords_bohr, dtype=float).copy()
    i, j = axis
    p0, p1 = c[i], c[j]
    e = (p1 - p0) / float(np.linalg.norm(p1 - p0))
    th = math.radians(deg)
    K = np.array([[0.0, -e[2], e[1]], [e[2], 0.0, -e[0]], [-e[1], e[0], 0.0]])
    rot = np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)
    for a, sym in enumerate(symbols):
        if sym != "H":
            continue
        if float(np.linalg.norm(c[a] - p1)) < 1.3 / 0.52917721092:
            c[a] = p0 + rot @ (c[a] - p0)
    return c


def _freqs(ma) -> list[float]:
    return [float(f) for f in ma["frequencies"]]


# ======================================================================
#  the Hessian must return the right sign pattern
# ======================================================================
def physics() -> dict:
    from backend.engine import reaction as R
    from backend.engine.dft import _split_xyz

    print("-- Hessian machinery ------------------------------------------",
          flush=True)
    xyz = _staggered_ethane_xyz()
    symbols, ang = _split_xyz(xyz)
    coords = np.asarray(ang, dtype=float) / R.BOHR
    n3 = 3 * len(symbols)

    relaxed = R.full_relax(symbols, coords, 0, 1, FUNC, BASIS, max_steps=30)
    grms = float(relaxed["grad_rms"])
    if grms > 5e-4:
        _bad(f"the ethane endpoint did not relax: |g| = {grms:.2e} a.u., so "
             f"the frequency check below is not testing a minimum")
    else:
        _ok(f"ethane relaxed to |g| = {grms:.1e} a.u.")
    x_min = relaxed["coords_bohr"]

    ma_min = R.mode_analysis(symbols, x_min, 0, 1, FUNC, BASIS)
    shape = ma_min["hessian_shape"]
    if list(shape) != [n3, n3]:
        _bad(f"the Hessian is not a ({n3}, {n3}) matrix but {shape}; the "
             f"PySCF tensor comes out as (natm, natm, 3, 3) and reshaping it "
             f"straight to (3N, 3N) interleaves atom and Cartesian indices")
    else:
        _ok(f"Hessian reduced to a ({n3}, {n3}) matrix")

    if ma_min["n_rigid_modes"] != 6:
        _bad(f"a non-linear molecule has 6 rigid-body modes, got "
             f"{ma_min['n_rigid_modes']}")
    else:
        _ok("6 translations/rotations identified and projected out")

    if ma_min["n_vibrations"] != n3 - 6:
        _bad(f"ethane has {n3 - 6} vibrations, got "
             f"{ma_min['n_vibrations']}")
    else:
        _ok(f"{n3 - 6} vibrations reported")

    if ma_min["n_imaginary"] != 0:
        _bad(f"staggered ethane is a minimum but the Hessian reports "
             f"{ma_min['n_imaginary']} imaginary frequencies "
             f"({[round(f, 1) for f in ma_min['imaginary'][:4]]}). The sign "
             f"pattern is what the transition-state verdict rests on, so this "
             f"makes every barrier report meaningless.")
    else:
        _ok("staggered ethane has 0 imaginary frequencies")

    fmin = _freqs(ma_min)
    cc = [f for f in fmin if 850.0 < f < 1150.0]
    if not cc:
        _bad(f"the ethane C-C stretch (~1000 cm^-1) is missing from "
             f"{[round(f, 1) for f in fmin]}")
    else:
        _ok(f"C-C stretch at {cc[0]:.1f} cm^-1")
    ch = [f for f in fmin if 2850.0 < f < 3250.0]
    if len(ch) != 6:
        _bad(f"ethane has 6 C-H stretches, found {len(ch)} in 2850-3250 cm^-1")
    else:
        _ok(f"6 C-H stretches in {ch[0]:.0f}-{ch[-1]:.0f} cm^-1")

    # ---- the saddle, built exactly -------------------------------------
    print("-- the eclipsed conformer -------------------------------------",
          flush=True)
    x_ts = _rotate_methyl(x_min, symbols, (0, 1), 60.0)
    torsion = R.Coordinate("torsion", (2, 0, 1, 5))
    q_min = torsion.value(x_min)
    q_ts = torsion.value(x_ts)
    if not (abs(q_ts - q_min) > 40.0):
        _bad(f"rotating the methyl did not change the torsion: {q_min:.1f} -> "
             f"{q_ts:.1f} deg, so this is not the eclipsed conformer")
    else:
        _ok(f"torsion moved {q_min:.1f} -> {q_ts:.1f} deg (eclipsed)")

    ma_ts = R.mode_analysis(symbols, x_ts, 0, 1, FUNC, BASIS)
    if ma_ts["n_imaginary"] != 1:
        _bad(f"eclipsed ethane is a first-order saddle but the Hessian reports "
             f"{ma_ts['n_imaginary']} imaginary frequencies "
             f"({[round(f, 1) for f in ma_ts['imaginary'][:5]]})")
        return {"min": ma_min, "ts": ma_ts}
    _ok("eclipsed ethane has exactly 1 imaginary frequency")

    nu = float(ma_ts["imaginary"][0])
    if not (-600.0 < nu < -150.0):
        _bad(f"the imaginary frequency of the ethane torsion is {nu:.1f} cm^-1; "
             f"the torsion belongs near -300, and a value beyond -1000 would "
             f"mean the structure has a stretched bond rather than a torsion "
             f"saddle")
    else:
        _ok(f"imaginary mode at {nu:.1f} cm^-1 (torsion-like)")

    align = R._mode_alignment(ma_ts["imaginary_vectors"][0], torsion, x_ts,
                              np.asarray(ma_ts["masses"]))
    if align <= 0.4:
        _bad(f"the imaginary mode overlaps the torsion coordinate by only "
             f"{align:.3f}; it is not the mode the reaction follows")
    else:
        _ok(f"imaginary mode aligns with the torsion ({align:.3f})")

    # the same mode is *real* at the minimum: the torsion is the coordinate
    # that flips sign, which is the physical statement being tested
    lowest_min = min(fmin)
    if abs(lowest_min - abs(nu)) > 0.5 * abs(nu):
        _bad(f"the lowest real mode at the minimum is {lowest_min:.1f} cm^-1 "
             f"but the imaginary mode at the saddle is {nu:.1f}; they should "
             f"be the same vibration, so the saddle does not look like it was "
             f"reached along this coordinate")
    else:
        _ok(f"the same vibration is {lowest_min:.1f} cm^-1 at the minimum and "
            f"{nu:.1f} cm^-1 at the saddle -- it is the one that flips sign")
    return {"min": ma_min, "ts": ma_ts}


# ======================================================================
#  the path
# ======================================================================
def path() -> dict:
    from backend.engine import reaction as R
    from backend.engine.dft import DFTEngine, _split_xyz

    print("-- the ethane torsion path ------------------------------------",
          flush=True)
    xyz = _staggered_ethane_xyz()
    eng = DFTEngine(atom_xyz=xyz, charge=0, multiplicity=1,
                    functional=FUNC, basis=BASIS)
    coord = R.Coordinate("torsion", (2, 0, 1, 5))
    t0 = time.time()
    res = R.reaction_path(eng, coord=coord, npoints=7, q_from=60.0, q_to=150.0,
                          max_steps=20, verify_ts=True)
    print(f"       path took {time.time() - t0:.0f} s", flush=True)

    pts = res["points"]
    if len(pts) != 7:
        _bad(f"asked for 7 points, got {len(pts)}")

    for p in pts:
        if abs(float(p["coord"]) - float(p["target"])) > 0.05:
            _bad(f"a scan point sits at {p['coord']:.3f} deg but was asked for "
                 f"{p['target']:.3f}; geomeTRIC holds a constraint exactly, so "
                 f"a drift means the constraint is not being applied")
            break
    else:
        _ok("every point sits on the coordinate that was requested")

    rel = {round(float(p["coord"])): float(p["relative_kcal"]) for p in pts}
    if min(rel.values()) < -1e-6:
        _bad(f"a relative energy is negative ({min(rel.values()):.4f} "
             f"kcal/mol); the whole figure must share one zero")
    else:
        _ok("no negative relative energies (one energy zero for the figure)")

    if not res["barrier_bracketed"]:
        _bad("the ethane torsion barrier was not bracketed inside 60-150 deg")
        return res
    _ok("barrier bracketed inside the window")

    barrier = float(res["barrier_kcal"])
    if not (2.0 <= barrier <= 3.6):
        _bad(f"the ethane torsion barrier came out at {barrier:.2f} kcal/mol; "
             f"the experimental value is 2.9 and a relaxed B3LYP/6-31G* scan "
             f"belongs in 2.0-3.6")
    else:
        _ok(f"barrier {barrier:.2f} kcal/mol (experiment 2.9)")

    ts = res["ts"] or {}
    if abs(float(ts.get("coord", 0.0)) - 120.0) > 4.0:
        _bad(f"the transition state sits at {ts.get('coord')} deg; the "
             f"eclipsed maximum is at 120")
    else:
        _ok(f"transition state at {float(ts['coord']):.2f} deg (eclipsed 120)")

    # symmetry about the eclipsed maximum: the two flanks must match
    for a, b in ((75, 165), (90, 150), (105, 135)):
        if a in rel and b in rel and abs(rel[a] - rel[b]) > 0.05:
            _bad(f"the profile is not symmetric about 120 deg: {a} deg is "
                 f"{rel[a]:.3f} kcal/mol but {b} deg is {rel[b]:.3f}")
            break
    else:
        _ok("profile symmetric about the eclipsed maximum")

    # both endpoints relax to the same conformer, so the reaction energy is 0
    for side in ("reactant", "product"):
        v = abs(float(res[side]["relative_kcal"]))
        if v > 0.15:
            _bad(f"the {side} endpoint is {v:.3f} kcal/mol from the lowest "
                 f"point on the path; for a torsion scan both ends are the "
                 f"same staggered conformer, so this should be ~0")
        else:
            _ok(f"{side} endpoint at {v:.3f} kcal/mol (same conformer)")

    ver = res["verification"]
    if not ver.get("is_transition_state"):
        _bad(f"the barrier maximum is not a verified transition state: "
             f"n_imaginary = {ver.get('n_imaginary')}, alignment = "
             f"{ver.get('mode_alignment')}, imaginary = "
             f"{ver.get('imaginary_cm')}")
    else:
        _ok(f"transition state verified: 1 imaginary mode at "
            f"{ver['imaginary_cm'][0]:.1f} cm^-1, overlap "
            f"{ver['mode_alignment']:.2f}")

    # ---- the intrinsic reaction coordinate -----------------------------
    print("-- intrinsic reaction coordinate -------------------------------",
          flush=True)
    irc = res.get("irc") or {}
    br = irc.get("branches") or {}
    if not br.get("forward") or not br.get("backward"):
        _bad(f"the IRC did not run even though the saddle is verified: "
             f"{irc}")
        return res

    if not irc.get("preconditioned"):
        _bad("the IRC ran without the Hessian preconditioner; a plain "
             "steepest-descent direction is dominated by the stiff "
             "perpendicular modes and the path stalls")
    else:
        _ok("IRC preconditioned by the saddle's Hessian")

    for name in ("forward", "backward"):
        b = br[name]
        if not b["descends_monotonically"]:
            bad_steps = [
                (i, round(p["energy_hartree"], 8))
                for i, p in enumerate(b["points"])][:4]
            _bad(f"the {name} IRC branch does not descend at every step "
                 f"(first points: {bad_steps}); a descent path cannot go "
                 f"uphill, so it is not an IRC")
        else:
            _ok(f"{name} branch descends monotonically over "
                f"{b['n_steps']} steps")

        drop = b.get("total_drop_kcal")
        if drop is None or abs(drop) < 0.3:
            _bad(f"the {name} branch only drops {drop} kcal/mol from the "
                 f"saddle; it has not actually left the barrier top, so the "
                 f"figure would show two flat stubs")
        else:
            _ok(f"{name} branch drops {abs(drop):.2f} kcal/mol from the saddle")

        mean = b.get("mean_step_size")
        if mean is None or mean < 0.5 * float(irc.get("step_size", 0.1)):
            _bad(f"the {name} branch had to shrink its step to {mean} "
                 f"(from {irc.get('step_size')}); the direction is still "
                 f"being dominated by the stiff modes")
        else:
            _ok(f"{name} branch kept its step at {mean}")

    fwd, bwd = br["forward"], br["backward"]
    if not irc.get("degenerate_reaction"):
        _bad("the ethane torsion is a degenerate reaction -- both endpoints "
             "relax to the same conformer -- and the IRC does not say so")
    elif fwd.get("toward") is not None or bwd.get("toward") is not None:
        _bad(f"a degenerate reaction must not be given a direction, but the "
             f"IRC reports forward->{fwd.get('toward')} and "
             f"backward->{bwd.get('toward')}")
    else:
        _ok("the degenerate torsion is not given a direction")

    # the two branches must be mirror images about the saddle
    q_ts = float(ts.get("coord", 120.0))
    if fwd.get("final_coord") is None or bwd.get("final_coord") is None:
        _bad("an IRC branch has no final coordinate")
    else:
        d_up = abs(float(fwd["final_coord"]) - q_ts)
        d_down = abs(q_ts - float(bwd["final_coord"]))
        if abs(d_up - d_down) > 1.0:
            _bad(f"the IRC branches are not mirror images about the saddle: "
                 f"forward reached {d_up:.2f} deg away, backward "
                 f"{d_down:.2f} deg")
        else:
            _ok(f"branches mirror-symmetric about the saddle "
                f"({d_up:.1f} vs {d_down:.1f} deg)")
        if not (float(fwd["final_coord"]) > q_ts
                > float(bwd["final_coord"])):
            _bad(f"both branches left the saddle the same way: forward "
                 f"{fwd['final_coord']}, backward {bwd['final_coord']}, "
                 f"saddle {q_ts}")
        else:
            _ok("the two branches leave the saddle in opposite directions")

    d_f = abs(float(fwd.get("total_drop_kcal") or 0.0))
    d_b = abs(float(bwd.get("total_drop_kcal") or 0.0))
    if d_f > 1e-9 and abs(d_f - d_b) / max(d_f, d_b) > 0.05:
        _bad(f"the symmetric branches drop different amounts of energy "
             f"({d_f:.3f} vs {d_b:.3f} kcal/mol), which a symmetric torsion "
             f"cannot do")
    else:
        _ok(f"both branches drop the same energy ({d_f:.3f} vs {d_b:.3f})")

    # ------------------------------------------------------------------
    #  the barrier is not one number
    # ------------------------------------------------------------------
    # The scan measures a difference of electronic energies with the nuclei
    # frozen, and for years that was narrated as "a true activation energy".
    # It is not one: the zero-point term alone moves a barrier by 1-3 kcal/mol,
    # which is most of the barrier for a methyl rotation.  So the payload has
    # to carry the corrected barriers, and they have to be right.
    bt = res.get("barrier_thermo") or {}
    if not bt.get("computed"):
        _bad(f"no zero-point or Gibbs correction on the barrier "
             f"({bt.get('reason', 'no reason given')}); a paper cannot quote "
             f"an electronic energy difference as an activation energy")
    else:
        e_only = float(bt["electronic_kcal"])
        e0 = float(bt["zpe_corrected_kcal"])
        dh = float(bt["enthalpy_kcal"])
        dg = float(bt["gibbs_kcal"])
        for name, v in (("electronic", e_only), ("zero-point", e0),
                        ("enthalpy", dh), ("gibbs", dg)):
            if not math.isfinite(v):
                _bad(f"the {name} barrier is {v}, not a number")
        # The saddle is missing the mode that went imaginary, so it is missing
        # that mode's zero-point energy: the corrected barrier has to sit below
        # the electronic one, by roughly half a quantum of the torsion.
        drop = e_only - e0
        if not 0.1 <= drop <= 1.5:
            _bad(f"the zero-point correction moves the barrier by {drop:.3f} "
                 f"kcal/mol; the transition state loses the torsional mode "
                 f"(~300 cm^-1, half a quantum is 0.43), so 0.1-1.5 is the "
                 f"range this can be")
        else:
            _ok(f"the zero-point corrected barrier ({e0:.2f}) sits "
                f"{drop:.2f} kcal/mol below the electronic one ({e_only:.2f}) "
                f"-- the transition state has lost the torsional quantum")
        # 3N-7 against 3N-6: the imaginary mode must be out of the partition
        # function, or the saddle's entropy is computed over a mode that is
        # not a vibration.
        n_r = int((bt.get("reactant") or {}).get("n_real_modes") or 0)
        n_ts = int((bt.get("transition_state") or {}).get("n_real_modes") or 0)
        if n_ts != n_r - 1:
            _bad(f"the transition state counts {n_ts} real modes against the "
                 f"reactant's {n_r}; it must have exactly one fewer, because "
                 f"the imaginary mode is not a vibration")
        else:
            _ok(f"the saddle has one real mode fewer than the minimum "
                f"({n_ts} vs {n_r}): the imaginary mode is excluded from the "
                f"partition function")
        # A broken entropy shows up as tens of kcal/mol; a unimolecular
        # rotation cannot rearrange the partition function by that much.
        if abs(dg - dh) > 3.0:
            _bad(f"the Gibbs barrier ({dg:.2f}) and the enthalpy barrier "
                 f"({dh:.2f}) differ by {abs(dg - dh):.2f} kcal/mol; T*dS for "
                 f"an internal rotation is well under 3 kcal/mol, so the "
                 f"entropy is wrong")
        else:
            _ok(f"activation Gibbs energy {dg:.2f} kcal/mol at "
                f"{bt.get('temperature_K')} K (enthalpy {dh:.2f})")
        if abs(float(bt.get("temperature_K") or 0) - 298.15) > 0.01:
            _bad(f"the corrections are quoted at {bt.get('temperature_K')} K, "
                 f"not the standard 298.15 K")
    return res


# ======================================================================
#  API and narration
# ======================================================================
def contract(res: dict) -> None:
    from backend.agent import planner as P

    print("-- contract ----------------------------------------------------",
          flush=True)

    # The result has to survive the trip through the API.  mode_analysis
    # returns raw numpy eigen-decomposition alongside the rounded numbers the
    # payload keeps, so one stray array here turns into a 500 at the worst
    # possible moment: after a ten-minute calculation.
    import json as _json
    try:
        _json.dumps(res)
    except TypeError as exc:
        _bad(f"the reaction result is not JSON-serialisable: {exc}")
    else:
        _ok("the reaction result serialises cleanly for the API")

    status, _ = _req("POST", "/api/reaction",
                     {"molecule": "ethane", "coordinate": "banana 1 2"})
    if status != 400:
        _bad(f"an unreadable reaction coordinate was accepted (HTTP {status}); "
             f"it must be refused with 400 rather than silently scanning the "
             f"wrong torsion")
    else:
        _ok("an unreadable coordinate is refused with 400")

    status, body = _req("POST", "/api/reaction",
                        {"molecule": "ethane", "coordinate": "torsion 2 1 3 99",
                         "npoints": 3})
    if status != 200:
        _bad(f"submitting a coordinate that names a missing atom returned "
             f"HTTP {status} instead of a job that reports the problem")
    else:
        job = _wait(body["job_id"], timeout=180.0)
        err = str(job.get("error") or "")
        if job.get("status") != "failed":
            _bad(f"a coordinate naming atom 99 of an 8-atom molecule was "
                 f"accepted and ran: status {job.get('status')}")
        elif "99" not in err and "8" not in err:
            _bad(f"the failure does not say which atom was out of range: {err}")
        else:
            _ok(f"a missing atom index fails with a clear message: "
                f"{err.strip()[:80]}")

    # planning: confirm first, compute on request
    status, res = _req("POST", "/api/chat", {
        "message": "what is the energy barrier of the ethane torsion",
        "auto_run": False})
    if status != 200 or not isinstance(res, dict):
        _bad(f"the chat router did not answer a reaction request (HTTP "
             f"{status})")
    elif res.get("action") != "confirm":
        _bad(f"a reaction request with auto_run=False returned action="
             f"{res.get('action')!r}; it must ask for confirmation rather "
             f"than start a job")
    elif res.get("intent", {}).get("job_type") != "reaction":
        _bad(f"the reaction request was routed to "
             f"{res.get('intent', {}).get('job_type')!r}")
    else:
        _ok("a barrier request plans a 'reaction' job and asks to confirm")

    status, res = _req("POST", "/api/chat", {
        "message": "torsion 2 1 3 6 of butane energy profile",
        "auto_run": False})
    if status != 200 or not isinstance(res, dict):
        _bad(f"the chat router did not answer an explicit-coordinate request "
             f"(HTTP {status})")
    elif res.get("intent", {}).get("reaction_coord") != "torsion 2 1 3 6":
        _bad(f"the explicit coordinate was not carried through: "
             f"{res.get('intent', {}).get('reaction_coord')!r}")
    else:
        _ok("an explicit coordinate reaches the intent as written")

    # the two index conventions
    if P.parse_coordinate("torsion 2 1 3 6") != ("torsion", (1, 0, 2, 5)):
        _bad("1-based 'torsion 2 1 3 6' did not become (1, 0, 2, 5)")
    elif P.parse_coordinate("torsion 2 0 1 5") != ("torsion", (2, 0, 1, 5)):
        _bad("0-based 'torsion 2 0 1 5' did not become (2, 0, 1, 5); a spec "
             "containing 0 cannot be 1-based and must be read as 0-based")
    elif P.parse_coordinate("torsion 2 2 2 2") is not None:
        _bad("a coordinate that reuses one atom four times was accepted")
    else:
        _ok("both index conventions parse, repeats are refused")

    pl = P.RuleBasedPlanner()
    routes = {
        "what is the energy barrier of the ethane torsion": "reaction",
        "draw the reaction path for the HCN to HNC isomerisation": "reaction",
        "scan the C-C bond in ethane": "scan",
        "give me the IR spectrum of benzene": "vibrations",
    }
    for text, want in routes.items():
        got = pl.plan(text).job_type
        if got != want:
            _bad(f"{text!r} routed to {got!r}, expected {want!r}")
        else:
            _ok(f"{text[:44]!r} -> {got}")


def narration() -> None:
    from backend.agent import planner as P

    print("-- narration ---------------------------------------------------",
          flush=True)
    sample = {
        "job_type": "reaction",
        "coordinate": {"kind": "torsion", "atoms": [1, 0, 2, 5],
                       "label": "torsion (1, 0, 2, 5)", "unit": "degree"},
        "points": [
            {"coord": 60.0, "relative_kcal": 0.0},
            {"coord": 120.0, "relative_kcal": 2.81},
            {"coord": 150.0, "relative_kcal": 1.34},
        ],
        "barrier_bracketed": True,
        "barrier_kcal": 2.81,
        "reaction_kcal": -0.01,
        "ts": {"coord": 120.0, "relative_kcal": 2.81},
        "reactant": {"relative_kcal": 0.0, "grad_rms": 9e-5},
        "product": {"relative_kcal": 0.0, "grad_rms": 8e-5},
        "verification": {"checked": True, "n_imaginary": 1,
                         "imaginary_cm": [-303.6], "mode_alignment": 0.548,
                         "is_transition_state": True},
        "irc": {
            "steps": 12, "step_size": 0.1, "preconditioned": True,
            "degenerate_reaction": True,
            "branches": {
                "forward": {"total_drop_kcal": -1.53, "final_coord": 152.2,
                            "descends_monotonically": True, "points": []},
                "backward": {"total_drop_kcal": -1.53, "final_coord": 87.7,
                             "descends_monotonically": True, "points": []},
            },
            "note": "The two endpoints relaxed to the same geometry.",
        },
        "n_points": 3, "seconds": 240.0,
        "functional_label": "B3LYP", "basis_label": "6-31G*",
    }
    text = P.explain_reaction(sample, P.JobIntent(job_type="reaction"))
    for needle, why in (
            ("2.81", "the barrier value"),
            ("kcal/mol", "the unit"),
            ("imaginary", "the verification"),
            ("transition state", "the verdict"),
            ("IRC", "the intrinsic reaction coordinate")):
        if needle.lower() not in text.lower():
            _bad(f"the reaction narration never mentions {why} ({needle!r})")
        else:
            _ok(f"narration mentions {why}")

    # an unverified maximum must be described as an upper bound, not as a
    # barrier -- this is the sentence that keeps the figure honest
    bad = dict(sample)
    bad["verification"] = {"checked": True, "n_imaginary": 4,
                           "imaginary_cm": [-1800.0], "mode_alignment": 0.11,
                           "is_transition_state": False}
    bad["irc"] = {"skipped": "the maximum is not a verified transition state, "
                              "so following its lowest mode would not follow "
                              "the reaction"}
    text2 = P.explain_reaction(bad, P.JobIntent(job_type="reaction"))
    if "not a verified transition state" not in text2.lower():
        _bad("an unverified maximum is not flagged as such in the narration")
    elif "upper bound" not in text2.lower():
        _bad("the narration does not say that an unverified barrier is an "
             "upper bound on the activation energy")
    else:
        _ok("an unverified maximum is reported as an upper bound")


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    BASE = f"http://127.0.0.1:{args.port}"

    print("=== ChatDFT reaction-path audit ===\n", flush=True)
    physics()
    print(flush=True)
    path_result = path()
    print(flush=True)
    contract(path_result)
    print(flush=True)
    narration()

    print()
    if FAILURES:
        print(f"=== FAIL: {len(FAILURES)} reaction-path problem(s) ===")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("=== PASS: the barrier is a real barrier, and the TS is a real TS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
