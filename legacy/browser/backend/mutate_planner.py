"""Round-19 mutation tests for the planner.

Every assertion in ``backend/check_planner.py`` has to be shown to be capable
of failing.  The harness breaks one thing at a time in
``backend/agent/planner.py``, runs the *real* gate as a subprocess, and reports
which of its sections fired.

The mutations are the four faults the gate was written for, put back one at a
time:

  P1  the planner resolves the key from the environment itself again -- one
      setting with two readers, so /api/llm and /api/chat can disagree;
  P2  the schema is typed out again, listing seven of the eighteen job types;
  P3  a reply that parsed is applied without checking it against the schema;
  P4  the fallback note stops matching the wording the front end displays;
  P5  create_planner hands back the bare keyword parser when nothing is
      configured, so the object that knows the reason is not the one that
      answers;
  P6  the planner stops asking its client and answers anyway;
  P7  the job vocabulary is named again without saying what any of it
      produces, which is the defect the second half of section 2 exists for;
  P8  a disagreement between the two planners stops being reported;
  P9  job_type_note keeps working and stops being called -- the difference
      between an assertion that is wired and one that merely exists, which is
      what P8 alone cannot show.

P2 earned its wording the hard way.  Its first form broke the schema line's
inline list of job names, and it fired nothing: the vocabulary was being
rendered twice, so the descriptions block below still carried all eighteen
names, the prompt was still correct, and every assertion in section 2 still
passed.  A mutation that fires nothing is an assertion guarding nothing -- and
here the cause was the second rendering, not the mutation.  The duplication is
gone, P2 now breaks the schema block's copy on purpose, and section 2 of the
gate holds the vocabulary to a single rendering so the copy cannot grow back.

The live server must be up: the gate talks to it.  A mutation to planner.py
does not affect the already-imported server, so section 6 (over HTTP) tests
unmutated code in every run -- which is what makes it a control.

Run:  PYTHONPATH=C:/Users/frddx/Desktop/ChatDFT <conda python> -m backend.mutate_planner
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import subprocess
import sys

from backend import mutate_common

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "backend", "agent", "planner.py")
REL = "backend/agent/planner.py"

# (label, what it simulates, [(old, new), ...])
MUTATIONS = [
    ("P1 second-reader",
     "resolve the key from the environment again, ignoring data/llm.json",
     [("        return bool(self.client.available)",
       "        return bool(os.environ.get(\"CHATDFT_LLM_BASE\")\n"
       "                    and os.environ.get(\"CHATDFT_LLM_KEY\"))")]),

    ("P2 schema-drift",
     "type the job vocabulary out in the schema block again, seven of the "
     "eighteen",
     [("            '{\"job_type\": string, one of the job types defined below, '",
       "            '{\"job_type\": one of [\"single_point\",\"geometry_optimization\","
       "\"excited_states\",\"compare\",\"scan\",\"info\",\"library\"], '")]),

    ("P3 trust-the-reply",
     "apply whatever parsed, without checking the job type",
     [("            if value not in JOB_TYPES:\n"
       "                rejected.append(f\"job_type {value!r} is not a job this server runs\")\n"
       "                continue",
       "            if False:\n"
       "                rejected.append(f\"job_type {value!r} is not a job this server runs\")\n"
       "                continue")]),

    ("P4 silent-fallback",
     "stop saying 'unavailable', so the front end shows no warning",
     [("PLANNER_OFF_NOTE = (\"language model unavailable ({reason}); the request was \"\n"
       "                    \"parsed by the built-in keyword parser\")",
       "PLANNER_OFF_NOTE = (\"{reason}\")")]),

    ("P5 bare-parser",
     "hand back the keyword parser itself when nothing is configured",
     [("    if mode == \"local\":\n"
       "        return LLMPlanner(_DisabledClient(),\n"
       "                          reason=\"disabled by CHATDFT_PLANNER=local\"), False\n"
       "    llm = LLMPlanner()\n"
       "    return llm, llm.available",
       "    if mode == \"local\":\n"
       "        return RuleBasedPlanner(), False\n"
       "    llm = LLMPlanner()\n"
       "    return llm if llm.available else RuleBasedPlanner(), llm.available")]),

    ("P6 never-ask",
     "answer without asking the client, and still claim the model planned it",
     [("            payload = self.client.json(\n"
       "                [{\"role\": \"system\", \"content\": self.system_prompt()},\n"
       "                 {\"role\": \"user\", \"content\": text}],\n"
       "                temperature=0.0, max_tokens=1500, timeout=PLAN_TIMEOUT,\n"
       "                validate=lambda p: isinstance(p, dict),\n"
       "            )",
       "            payload = {}")]),

    ("P7 bare-identifiers",
     "name the job types again without saying what any of them produce",
     [("            \"What each job_type produces:\\n\"\n"
       "            + \"\\n\".join(f'  \"{k}\": {v}' for k, v in JOB_TYPES.items()) + \"\\n\"",
       "            \"What each job_type produces:\\n\"")]),

    ("P8 silent-disagreement",
     "report nothing when the parser and the model read one sentence as two "
     "different jobs",
     [("    if parsed == chosen:\n"
       "        return \"\"\n"
       "    return JOB_TYPE_DISAGREEMENT.format(parsed=parsed, chosen=chosen)",
       "    return \"\"")]),

    ("P9 unwired-disagreement",
     "keep job_type_note able to report a disagreement, and never call it",
     [("        disagreement = job_type_note(parsed_job, intent.job_type)\n"
       "        if disagreement:\n"
       "            intent.notes.append(disagreement)",
       "        disagreement = \"\"")]),
]

_FAIL_RE = re.compile(r"=== FAIL: (\d+) planner problem", re.M)
_PASS_RE = re.compile(r"=== PASS: the model plans when it is configured")


def run_gate(tag: str) -> tuple[int, list[str]]:
    """Run the real gate in a subprocess and count its failures."""
    env = dict(os.environ)
    bits = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if ROOT not in bits:
        env["PYTHONPATH"] = os.pathsep.join([ROOT] + bits)
    env["no_proxy"] = "127.0.0.1,localhost"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    proc = subprocess.run([sys.executable, "-W", "ignore", "-m",
                           "backend.check_planner"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, env=env, cwd=ROOT, timeout=1800)
    out = proc.stdout or ""
    failed = _FAIL_RE.findall(out)
    n = int(failed[0]) if failed else 0
    reasons = [l.strip()[2:] for l in out.splitlines()
               if l.strip().startswith("! ")]
    print(f"  {tag}: {n} failure(s)"
          + ("" if n else ("  [clean]" if _PASS_RE.search(out) else
                           "  !! NO VERDICT AT ALL")))
    for r in reasons[:6]:
        print(f"    - {r[:120]}")
    if n and len(reasons) > 6:
        print(f"    ... {len(reasons) - 6} more")
    return n, reasons


def main() -> int:
    original = io.open(SRC, encoding="utf-8").read()
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()

    # Before the baseline: a mutation whose target text has moved reports
    # "not applicable" at the end of the run, which reads like coverage that
    # is merely incomplete.  See mutate_common.
    dead = mutate_common.dead_mutations(MUTATIONS, lambda _e: original,
                                        lambda _e: SRC)
    if mutate_common.report("planner", MUTATIONS, dead):
        return 1
    print()

    print("=" * 78)
    print("BASELINE -- unmutated")
    print("=" * 78)
    n, _ = run_gate("baseline")
    if n:
        print(f"  !! baseline already failing ({n}); fix that first")
        return 1
    print("  baseline: 0 failures, as required\n")

    problems: list[str] = []
    fired: list[tuple[str, int]] = []
    for label, what, subs in MUTATIONS:
        text = original
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
        # Same guard as the NMR harness: never write a file that does not
        # compile.  A broken mutation aborts the run mid-way and the mutations
        # after it are never tested -- which looks exactly like full coverage.
        try:
            compile(text, REL, "exec")
        except SyntaxError as exc:
            problems.append(f"{label}: the substitution does not produce valid "
                            f"Python ({exc.msg} at line {exc.lineno})")
            print(f"  !! INVALID SOURCE, not written: {exc.msg}")
            continue
        print("=" * 78)
        print(f"MUTATION {label}: {what}")
        print("=" * 78)
        io.open(SRC, "w", encoding="utf-8").write(text)
        try:
            n, _ = run_gate(label)
        finally:
            io.open(SRC, "w", encoding="utf-8").write(original)
        if n == 0:
            problems.append(f"{label}: FIRED NOTHING -- the assertion guards "
                            f"nothing")
            print("  !! NO FAILURE RAISED")
        else:
            fired.append((label, n))
        print()

    print("=" * 78)
    print("RESTORED -- unmutated again")
    print("=" * 78)
    back = io.open(SRC, encoding="utf-8").read()
    if hashlib.sha256(back.encode("utf-8")).hexdigest() != digest:
        problems.append("restore: planner.py is not byte-identical to the "
                        "original")
        print("  !! NOT RESTORED BYTE-FOR-BYTE")
    else:
        print("  planner.py restored byte-for-byte")
    n, _ = run_gate("restored")
    if n:
        problems.append(f"restore: {n} failures on unmutated source")
    print()

    for label, n in fired:
        print(f"  {label:18s} fired {n}")
    if problems:
        print("\nMUTATION PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"\nAll {len(MUTATIONS)} mutations fired; source restored cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
