"""Round-15 mutation tests: every Raman assertion has to be able to fail.

The rule this project holds itself to is that an assertion which cannot fail
guards nothing.  So each mutation below breaks one specific thing, the *real*
checks from ``contract`` are run against the result, and the harness reports
whether the message it expected actually fired.

Two groups:

* **Pure functions** -- the activity formula, the depolarization ratio, the
  diffuse-basis naming rule.  These need no SCF, so they are checked through
  ``contract.raman_sections``, which is the same code the gate runs.
* **The projection and the sum rule** -- these only mean anything on a real
  calculation, so the harness runs ``analysis.vibrations`` on water and feeds
  the payload to ``contract.raman_payload_sections``, again the same function
  the gate uses.  Water is pre-optimised once and reused with
  ``optimize_first=False``, which is what keeps this affordable.

Run:  PYTHONPATH=C:/Users/frddx/Desktop/ChatDFT <conda python> -m backend.mutate_raman
"""
from __future__ import annotations

import importlib
import io
import sys
import time

import backend.bootstrap as bootstrap
from backend import mutate_common

bootstrap.setup()

ANALYSIS = "backend/engine/analysis.py"
DFT = "backend/engine/dft.py"

WATER_XYZ = (
    "3\nwater\n"
    "O 0.000000 0.000000 0.119262\n"
    "H 0.000000 0.763239 -0.477047\n"
    "H 0.000000 -0.763239 -0.477047\n"
)

# (label, what it simulates, file, [(old, new), ...], needs_payload)
MUTATIONS = [
    ("R1 activity-weight",
     "use 45 a_bar^2 + 4 gamma^2 instead of +7 (the depolarization weighting)",
     ANALYSIS,
     [("    return iso, gamma2, 45.0 * iso ** 2 + 7.0 * gamma2",
       "    return iso, gamma2, 45.0 * iso ** 2 + 4.0 * gamma2")],
     False),

    ("R2 no-six",
     "drop the factor of 6 on the off-diagonal anisotropy terms",
     ANALYSIS,
     [("    gamma2 = 0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2\n"
       "                    + 6.0 * (xy ** 2 + yz ** 2 + zx ** 2))",
       "    gamma2 = 0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2\n"
       "                    + (xy ** 2 + yz ** 2 + zx ** 2))")],
     False),

    ("R3 wrong-pair",
     "use xy twice and never zx in the anisotropy",
     ANALYSIS,
     [("+ 6.0 * (xy ** 2 + yz ** 2 + zx ** 2))",
       "+ 6.0 * (xy ** 2 + yz ** 2 + xy ** 2))")],
     False),

    ("R4 transposed-index",
     "write (xx - xy)^2 where (xx - yy)^2 belongs",
     ANALYSIS,
     [("    gamma2 = 0.5 * ((xx - yy) ** 2", "    gamma2 = 0.5 * ((xx - xy) ** 2")],
     False),

    ("R5 rho-denominator",
     "put the activity's 7 back into the rho denominator",
     ANALYSIS,
     [("    denom = 45.0 * iso ** 2 + 4.0 * gamma2",
       "    denom = 45.0 * iso ** 2 + 7.0 * gamma2")],
     False),

    ("R6 rho-numerator",
     "drop the 3 in the rho numerator",
     ANALYSIS,
     [("    return 3.0 * gamma2 / denom if abs(denom) > 1e-300 else 0.0",
       "    return gamma2 / denom if abs(denom) > 1e-300 else 0.0")],
     False),

    ("R7 diffuse-always-false",
     "claim no basis has diffuse functions",
     DFT,
     [('    key = (basis or "").strip().lower()',
       '    return False\n    key = (basis or "").strip().lower()')],
     False),

    ("R8 diffuse-always-true",
     "claim every basis has diffuse functions",
     DFT,
     [('    key = (basis or "").strip().lower()',
       '    return True\n    key = (basis or "").strip().lower()')],
     False),

    ("R9 flag-dropped",
     "remove the diffuse flag from aug-cc-pVDZ",
     DFT,
     [('    "aug-cc-pvdz": {"label": "aug-cc-pVDZ (diffuse)", "quality": 5,\n'
       '                    "diffuse": True},',
       '    "aug-cc-pvdz": {"label": "aug-cc-pVDZ (diffuse)", "quality": 5},')],
     False),

    # ---- the ones that need a real calculation ------------------------
    ("R10 no-rotation",
     "leave the rotational term out of the Raman sum rule",
     ANALYSIS,
     [("    expected = total - rot", "    expected = total")],
     True),

    ("R11 translation",
     "make the translational term non-zero, as it is for the dipole",
     ANALYSIS,
     [('        "translational_frob2_per_amu": 0.0,',
       '        "translational_frob2_per_amu": 1.0,')],
     True),

    ("R12 mass-weighting",
     "divide by m instead of 1/m in the sum rule's total",
     ANALYSIS,
     [("    total = float(np.sum(np.sum(dalphadx ** 2, axis=(2, 3)) / m[:, None]))",
       "    total = float(np.sum(np.sum(dalphadx ** 2, axis=(2, 3)) * m[:, None]))")],
     True),

    ("R13 unit-slip",
     "forget the bohr^3 -> A^3 factor on dalpha/dQ but keep it on dalpha/dx",
     ANALYSIS,
     [('        dalpha_dq = np.einsum("kax,axij->kij", modes, dalphadx) * a3',
       '        dalpha_dq = np.einsum("kax,axij->kij", modes, dalphadx)')],
     True),

    ("R14 rot-units",
     "pass alpha0 to the sum rule in a.u. while everything else is in A^3",
     ANALYSIS,
     [("                                  alpha0 * a3, masses, coords)",
       "                                  alpha0, masses, coords)")],
     True),

    ("R15 no-basis-warning",
     "always report alpha as reliable, whatever the basis",
     ANALYSIS,
     [('        if not basis_has_diffuse(basis_used):',
       '        if False:')],
     True),

    ("R16 reuse-IR",
     "make the Raman activities follow the IR intensities",
     ANALYSIS,
     [('        dalpha_dq = np.einsum("kax,axij->kij", modes, dalphadx) * a3',
       '        dalpha_dq = np.einsum("kax,axij->kij", modes, dalphadx) * a3\n'
       '        dalpha_dq = np.eye(3)[None] * np.sqrt(\n'
       '            np.maximum(np.asarray(intensities), 0.0))[:, None, None]')],
     True),
]


def reload_engine():
    for mod in ("backend.engine.dft", "backend.engine.analysis"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
        else:
            importlib.import_module(mod)
    import backend.contract as contract
    importlib.reload(contract)
    return contract


def run_checks(tag: str, payload, contract) -> list:
    fail: list = []
    buf, real = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        contract.raman_sections(fail)
        if payload is not None:
            contract.raman_payload_sections(payload, fail)
    finally:
        sys.stdout = real
    for line in buf.getvalue().split("\n"):
        if line.startswith("["):
            print(f"    {line[:100]}")
    return fail


def make_payload():
    """One real vibrational analysis, reused by every payload mutation."""
    import backend.engine.analysis as A
    from backend.engine.dft import DFTEngine

    eng = DFTEngine(atom_xyz=WATER_XYZ, charge=0, multiplicity=1,
                    functional="b3lyp", basis="6-31g*")
    t0 = time.time()
    res = A.vibrations(eng, optimize_first=False, with_raman=True)
    # The server's job branch calls this to build the two curves; doing it here
    # too means the harness exercises the same lines rather than a copy of them.
    A.attach_vibrational_spectra(res)
    print(f"  (payload built in {time.time() - t0:.1f} s: "
          f"{res['frequencies_cm1']})")
    return res


def _live_server() -> bool:
    """Is something already answering on the gate port?

    Mutating ``analysis.py`` in place is safe only if nothing else is reading
    it.  A server that starts *during* a mutation run imports the mutated
    module and then serves corrupted numbers to anything that talks to it --
    which is not hypothetical: the browser gate was run alongside this harness
    and reported a sum-rule residual of -93.33%, which is this file's own
    "divide by m instead of 1/m" mutation leaking through the server.  Every
    check in that run passed, and every number in it was wrong.

    The harness cannot stop a server being started, but it can refuse to run
    while one is up, which turns a silent contamination into a refusal.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", 8000)) == 0


def main() -> int:
    if _live_server():
        print("REFUSING TO RUN: something is listening on 127.0.0.1:8000.")
        print("This harness rewrites backend/engine/analysis.py in place.  A server")
        print("that starts while it runs will import a mutated module and serve")
        print("corrupted numbers to any browser gate run against it.  Stop the")
        print("server first.")
        return 2

    originals = {p: io.open(p, encoding="utf-8").read() for p in (ANALYSIS, DFT)}

    # Before the baseline: see mutate_common for why a dead mutation must not
    # be allowed to report itself as coverage.
    dead = mutate_common.dead_mutations(MUTATIONS, lambda e: originals[e[2]],
                                        lambda e: e[2])
    if mutate_common.report("Raman", MUTATIONS, dead):
        return 1
    print()

    print("=" * 78)
    print("BASELINE -- unmutated")
    print("=" * 78)
    contract = reload_engine()
    payload = make_payload()
    base = run_checks("baseline", payload, contract)
    if base:
        print(f"  !! baseline already failing: {base}")
        return 1
    print("  baseline: 0 failures, as required\n")

    problems = []
    for label, what, path, subs, needs_payload in MUTATIONS:
        text = originals[path]
        ok = True
        for old, new in subs:
            if old not in text:
                print(f"  !! {label}: mutation not applicable ({old[:60]!r})")
                ok = False
                break
            text = text.replace(old, new, 1)
        if not ok:
            problems.append(f"{label}: mutation not applicable")
            continue

        io.open(path, "w", encoding="utf-8").write(text)
        print("=" * 78)
        print(f"MUTATION {label}: {what}")
        print("=" * 78)
        try:
            contract = reload_engine()
            p = make_payload() if needs_payload else None
            fail = run_checks(label, p, contract)
        finally:
            io.open(path, "w", encoding="utf-8").write(originals[path])
        if not fail:
            problems.append(f"{label}: FIRED NOTHING -- the assertion guards nothing")
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
    back = run_checks("restored", payload, contract)
    if back:
        problems.append(f"restore: {back}")
        print(f"  !! not clean after restore: {back}")
    else:
        print("  clean\n")

    # The mutations rewrite files; confirm the bytes came back.
    for p, orig in originals.items():
        if io.open(p, encoding="utf-8").read() != orig:
            problems.append(f"{p} was not restored byte-for-byte")
            print(f"  !! {p} differs from the original after restore")

    if problems:
        print("MUTATION PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("All mutations fired; source restored cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
