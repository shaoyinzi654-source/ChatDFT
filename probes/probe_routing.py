"""Which job types can the model planner actually reach?

The planner has two implementations.  `RuleBasedPlanner` routes a sentence
with a keyword table (`_JOB_KEYWORDS`, one regex list per job type).
`LLMPlanner` runs that parser first for the defaults and then asks a model for
a JSON job spec, and the model's `job_type` wins whenever it names a job the
server can dispatch.

That last clause is the whole point of this probe.  The model is told which
job types exist but not what any of them mean -- `JOB_TYPES` is a dict of
name -> description and the prompt joins its **keys** only -- so the types
whose names are self-describing (`nmr`, `nci`, `dos`) are reachable and the
ambiguous ones are guessed at.  `surfaces` is the clear case: the keyword
table sends every one of "MEP map", "electrostatic potential" and "isosurface"
to `surfaces`, and the model sends every one of them somewhere else.  So the
Surfaces pane is reachable through the built-in parser and not through the
model, and nothing says the two disagreed -- `single_point` and `reactivity`
are valid job types, so the cross-check that refuses `nmr_spectrum` accepts
them.

This asks the live server, with `auto_run` off so nothing is calculated.

Start the server first (the model path needs `data/llm.json` or
`CHATDFT_LLM_KEY`, and the default `CHATDFT_PLANNER=auto` to use it):

    python -m backend.server
    python probes/probe_routing.py

Measured 2026-10-02, `sensenova-6.8-flash-lite`:

    series   reached by "Show me a series of ...", "Plot the ... trend for
             ...", "Run a series of ..."; NOT by "Compare the HOMO-LUMO gaps
             of benzene, pyridine, furan and pyrrole" -- which is the sentence
             the Series pane's own empty state recommends.
    surfaces reached by nothing out of seven phrasings.
    nci, nto, dos, field  all reached by their obvious sentence.
"""
from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ.get("CHATDFT_BASE", "http://127.0.0.1:8000")

# The proxy this machine sets would otherwise carry 127.0.0.1 to it too.
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

# sentence, the job type the keyword table would choose for it
CASES = [
    ("Show me a series of benzene, pyridine, furan and pyrrole", "series"),
    ("Plot the HOMO-LUMO gap trend for benzene, pyridine, furan and pyrrole",
     "series"),
    ("Run a series of benzene, pyridine, furan and pyrrole", "series"),
    ("Compare the HOMO-LUMO gaps of benzene, pyridine, furan and pyrrole",
     "series"),
    ("Show me the MEP map of water", "surfaces"),
    ("MEP map of water", "surfaces"),
    ("Render the electrostatic potential of water", "surfaces"),
    ("Plot the LUMO isosurface of water", "surfaces"),
    ("Draw the electrostatic potential surface of water", "surfaces"),
    ("Show me the electron density surface of water", "surfaces"),
    ("Visualise the HOMO of benzene", "surfaces"),
    ("NCI of the water dimer", "nci"),
    ("natural transition orbitals of formaldehyde", "nto"),
    ("DOS of pyridine", "dos"),
    ("ELF of benzene", "field"),
]


def plan(sentence: str) -> dict:
    body = json.dumps({"message": sentence, "auto_run": False}).encode()
    req = urllib.request.Request(
        BASE + "/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as fh:
        return json.loads(fh.read().decode())


def main() -> int:
    reached, missed = 0, 0
    for sentence, want in CASES:
        try:
            r = plan(sentence)
        except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
            print(f"  ERROR  {sentence!r}: {type(exc).__name__}: {exc}")
            continue
        intent = r.get("intent") or {}
        got = intent.get("job_type")
        hit = got == want
        reached += hit
        missed += not hit
        print(f"  {'HIT ' if hit else '    '}want={want:<10} got={str(got):<18}"
              f" planner={r.get('planner')}")
        print(f"       {sentence!r}")
    print()
    print(f"{reached} of {reached + missed} sentences reached the job type the "
          f"keyword table picks for them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
