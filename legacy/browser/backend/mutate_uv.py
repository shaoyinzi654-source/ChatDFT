"""Round-14 mutation tests.

Every assertion added this round has to be shown to be capable of failing.
The harness breaks one thing at a time in ``analysis.py``, runs the *real*
checks from ``contract.uv_scale_sections``, and reports which messages fired.
A mutation that fires nothing means the assertion guards nothing.

Run:  PYTHONPATH=C:/Users/frddx/Desktop/ChatDFT <conda python> -m backend.mutate_uv
"""
from __future__ import annotations

import importlib
import io
import sys

import backend.bootstrap as bootstrap
from backend import mutate_common

bootstrap.setup()

SRC = "backend/engine/analysis.py"

# (label, what it simulates, [(old, new), ...])
MUTATIONS = [
    ("M1 jacobian",
     "put the |dE/dlambda| Jacobian back on the ordinate",
     [("    y = eps / eps.max() * 100.0 if eps.max() > 0 else np.zeros_like(eps)",
       "    eps = eps * (EV2NM / (wl ** 2))\n"
       "    y = eps / eps.max() * 100.0 if eps.max() > 0 else np.zeros_like(eps)")]),

    ("M2 no-gauss-norm",
     "drop the unit-area 1/(sigma*sqrt(2pi)) factor",
     [("    g /= sigma * math.sqrt(2.0 * math.pi)",
       "    pass")]),

    ("M3 wrong-constant",
     "use 2.3154e9 instead of 2.3154e8 (a factor of ten)",
     [("EPS_INTEGRAL_PER_F = 2.3154e8", "EPS_INTEGRAL_PER_F = 2.3154e9")]),

    ("M4 fixed-window",
     "go back to the fixed 180-800 nm default window",
     [("    wmin: Optional[float] = None,\n    wmax: Optional[float] = None,",
       "    wmin: Optional[float] = 180.0,\n    wmax: Optional[float] = 800.0,")]),

    ("M5 no-notes",
     "never raise the out-of-window warning",
     [("    notes: List[str] = []", "    notes: List[str] = []\n    _suppress = True"),
      ("    if outside:\n        notes.append(",
       "    if outside and not _suppress:\n        notes.append("),
      ("    if not (wmin <= s_lam <= wmax):\n        notes.append(",
       "    if not (wmin <= s_lam <= wmax) and not _suppress:\n        notes.append(")]),

    ("M6 peak-column",
     "put the relative 0-100 value in the peak table's eps_max column",
     [('"epsilon_max_l_mol_cm": round(\n                EPS_INTEGRAL_PER_F * fi',
       '"epsilon_max_l_mol_cm": round(\n                float(fi) * 100.0 + 0 * EPS_INTEGRAL_PER_F * fi')]),

    ("M7 stale-integral",
     "report a hand-set integral instead of the trapezoid of the curve",
     [("    measured = float(np.trapezoid(eps[order], nu[order]))",
       "    measured = float(expected) * 1.5")]),
]


def run_checks(tag: str) -> list:
    for mod in ("backend.engine.analysis", "backend.contract"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
        else:
            importlib.import_module(mod)
    import backend.contract as contract
    importlib.reload(contract)
    fail: list = []
    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = buf
    try:
        contract.uv_scale_sections(fail)
    finally:
        sys.stdout = real
    lines = [l for l in buf.getvalue().split("\n") if l.startswith("[")]
    print(f"  {tag}")
    for l in lines:
        print(f"    {l[:110]}")
    return fail


def main() -> int:
    original = io.open(SRC, encoding="utf-8").read()

    # Before the baseline: see mutate_common for why a dead mutation must not
    # be allowed to report itself as coverage.
    dead = mutate_common.dead_mutations(MUTATIONS, lambda _e: original,
                                        lambda _e: SRC)
    if mutate_common.report("UV", MUTATIONS, dead):
        return 1
    print()

    print("=" * 78)
    print("BASELINE -- unmutated")
    print("=" * 78)
    base = run_checks("baseline")
    if base:
        print(f"  !! baseline already failing: {base}")
        return 1
    print("  baseline: 0 failures, as required\n")

    problems = []
    for label, what, subs in MUTATIONS:
        text = original
        applied = True
        for old, new in subs:
            if old not in text:
                print(f"  !! {label}: could not apply mutation ({old[:50]!r})")
                applied = False
                break
            text = text.replace(old, new, 1)
        if not applied:
            problems.append(f"{label}: mutation not applicable")
            continue
        io.open(SRC, "w", encoding="utf-8").write(text)
        print("=" * 78)
        print(f"MUTATION {label}: {what}")
        print("=" * 78)
        try:
            fail = run_checks(label)
        finally:
            io.open(SRC, "w", encoding="utf-8").write(original)
        if not fail:
            problems.append(f"{label}: FIRED NOTHING -- the assertion guards nothing")
            print("  !! NO FAILURE RAISED")
        else:
            print(f"  -> {len(fail)} failure(s) raised")
        print()

    print("=" * 78)
    print("RESTORED -- unmutated again")
    print("=" * 78)
    back = run_checks("restored")
    if back:
        problems.append(f"restore: {back}")
        print(f"  !! not clean after restore: {back}")
    else:
        print("  clean\n")

    if problems:
        print("MUTATION PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("All mutations fired; source restored cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
