"""How reproducibly does the model planner route a sentence?

The planner has two implementations.  `RuleBasedPlanner` routes a sentence
with a keyword table (`_JOB_KEYWORDS`, one regex list per job type).
`LLMPlanner` runs that parser first for the defaults and then asks a model for
a JSON job spec, and the model's `job_type` wins whenever it names a job the
server can dispatch.

Two things are worth knowing about that second half, and this probe measures
both.

**The model is not told what the job types mean.**  `JOB_TYPES` is a dict of
name -> description, and the prompt joins its **keys** only
(`", ".join(f'"{k}"' for k in JOB_TYPES)`), so the descriptions have no reader
and the model chooses between eighteen bare identifiers.  That is a fact about
the source and can be read off it.

**The model is not reproducible.**  Ask the same sentence twice and it can
come back with two different job types, which means the same request can draw
two different figures on two different days.  That is NOT readable off the
source, and a probe that asks each sentence once cannot see it -- it reports
whichever answer it happened to get as though it were *the* answer.

This probe was written the wrong way round first.  Its first version asked
each of fifteen sentences once and concluded that `surfaces` was unreachable,
because seven MEP and isosurface phrasings all came back as something else.
The next full run routed "Draw the electrostatic potential surface of water"
to `surfaces` -- the same sentence that had come back `single_point` before.
The conclusion was an artefact of sampling a non-deterministic instrument
once per case.  So this version asks each sentence `REPEATS` times and reports
the *set* of answers.

That first version also divided its summary by the sentences that had
*answered* rather than by the sentences that were *asked*, so a run whose
server died halfway printed "3 of 6" -- a number that reads like a result.  A
sentence that was never asked is not a sentence that agreed, so it is now
named and fails the run.

Start the server first (the model path needs `data/llm.json` or
`CHATDFT_LLM_KEY`, and the default `CHATDFT_PLANNER=auto` to use it):

    python -m backend.server
    python probes/probe_routing.py

Measured 2026-10-02, `sensenova-6.8-flash-lite`, two runs per sentence.  The
pairs are the evidence; the counts are printed by the run, not written here:

    series  'Show me a series of ...'                   series, series
            'Plot the ... trend for ...'                series, series
            'Run a series of ...'                       series, series
            'Compare the HOMO-LUMO gaps of ...'         compare, compare
              (stable -- and this is the sentence the Series pane's own
               empty state recommends, so the panel recommends a sentence
               that does not reach it)
    surfaces
            'Show me the MEP map of water'              single_point, surfaces
            'Draw the electrostatic potential ...'      field, single_point
            'Show me the electron density surface ...'  single_point, field
            'MEP map of water'                          field, field
            'Render the electrostatic potential ...'    field, field
            'Plot the LUMO isosurface of water'         nto, nto
            'Visualise the HOMO of benzene'             single_point, single_point
    nci     'NCI of the water dimer'                    nci, nci
    nto     'natural transition orbitals of ...'        nto, nto
    dos     'DOS of pyridine'                           dos, dos
    field   'ELF of benzene'                            field, field

Not one of the seven surface phrasings matched the keyword table on either
run, and three of them disagreed with themselves.  The four job types whose
names are self-describing were right on both runs.

**Measured again after the prompt was given the descriptions** (round 21, the
fix this probe was written to justify): same fifteen sentences, same two runs
each.

    15 sentences x 2 runs: 15 answered the same job type every time, 0 varied

Every sentence now reaches the job type the keyword table picks for it,
including all seven surface phrasings, and including "Compare the HOMO-LUMO
gaps of benzene, pyridine, furan and pyrrole" -- which is the sentence the
Series pane's own empty state recommends and which used to come back as
`compare`, leaving the pane it recommends showing its empty state.

Both sets of numbers are here on purpose.  The fix is one prompt change and
the effect is the whole table, so the pair is what shows the descriptions were
load-bearing rather than incidental.
"""
from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ.get("CHATDFT_BASE", "http://127.0.0.1:8000")

# The proxy this machine sets would otherwise carry 127.0.0.1 to it too.
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

# One sample per sentence is not a measurement of a non-deterministic
# planner.  Two is the minimum that can show a disagreement; raise it to see
# how wide the spread really is.
REPEATS = int(os.environ.get("PROBE_REPEATS", "2"))

# sentence, the job type the keyword table chooses for it
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
    stable, unstable, failed = 0, 0, []
    for sentence, want in CASES:
        seen: list = []
        for _ in range(REPEATS):
            try:
                r = plan(sentence)
            except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
                failed.append((sentence, f"{type(exc).__name__}: {exc}"))
                print(f"  ERROR  {sentence!r}: {type(exc).__name__}: {exc}")
                break
            seen.append((r.get("intent") or {}).get("job_type"))

        if len(seen) < REPEATS:
            continue

        agrees = len(set(seen)) == 1
        stable += agrees
        unstable += not agrees
        every = all(s == want for s in seen)
        mark = "SAME" if agrees else "VARY"
        print(f"  {mark} want={want:<10} got={seen} "
              f"{'all match' if every else 'NOT the keyword table'}")

    print()
    print(f"{len(CASES)} sentences x {REPEATS} runs: {stable} answered the "
          f"same job type every time, {unstable} varied")
    if failed:
        # A sentence that was never asked is not a sentence that agreed, so it
        # is named and it fails the run.  The first version of this line
        # divided by what had *answered* instead of by what was *asked*, so a
        # run whose server died halfway printed "3 of 6".
        print(f"and {len(failed)} sentence(s) were NEVER ASKED:")
        for sentence, why in failed:
            print(f"  NOT ASKED  {sentence!r}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
