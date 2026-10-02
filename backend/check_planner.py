"""Planner audit: when the model is configured, the model must do the planning.

The defect this gate exists to prevent was live for the whole life of the
project and nobody could see it, because both halves of the contradiction were
individually reasonable.

``GET /api/llm`` said ``available: true``.  It reads ``data/llm.json`` through
``agent.llm.LLMClient``.  ``create_planner()``, which is what the chat box goes
through, resolved ``CHATDFT_LLM_BASE`` / ``CHATDFT_LLM_KEY`` from the
environment only -- a second reader of the same setting.  With a key in the
file and nothing in the environment, the planner answered "no model
configured", returned the keyword parser, and every sentence typed into the
chat box was matched by regular expressions.  The status endpoint said the
opposite, and no response, log line or pixel of the interface disagreed.

Two further faults sat on the same code path, both invisible for the same
reason:

  * the planner read ``choices[0].message.content`` directly, and the models on
    this endpoint are reasoning models that leave ``content`` null and put the
    answer in ``reasoning_content`` -- ``llm.py`` documents and handles that,
    the planner predated it, so even with the environment set it would have
    raised TypeError and fallen back;
  * its schema listed seven job types out of the eighteen the server
    dispatches on, so the model could not have routed a design request, an NTO
    pair or a transition state even if it had been consulted.

So this gate asserts the invariants, not the implementation:

  1. one reader -- the planner's availability *is* the client's, and the
     planner has no configuration logic of its own;
  2. the schema covers every job type the server actually dispatches on,
     scanned out of ``server.py`` rather than copied;
  3. a reply that parses but is off-schema is refused field by field, and an
     on-schema one is applied -- both directions, no network;
  4. the reasoning-field reply shape reaches the planner;
  5. whenever the model does not do the planning, the reply says so, in the
     wording the front end raises its warning box for -- that regex is read
     out of ``app.js``, not copied here;
  6. over HTTP, a chat request planned by the parser must carry that note;
  7. the live model path, asserted in whichever direction it goes: planned by
     the model, or a fallback that names its error.

Runs against a live server.  Usage:  python -m backend.check_planner
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES: list[str] = []


def _req(method: str, path: str, payload: dict | None = None,
         timeout: float = 120.0):
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


def _bad(msg: str) -> None:
    FAILURES.append(msg)
    print("       !", msg, flush=True)


def _section(title: str) -> None:
    print(f"\n-- {title}", flush=True)


# ----------------------------------------------------------------------
# the front end's contract for "the model was not used"
# ----------------------------------------------------------------------
def note_filter() -> str:
    """The regex ``app.js`` uses to decide whether to show a warning box.

    Read out of the front end rather than copied.  Copied, the two can drift
    silently: the planner would keep emitting a sentence nobody displays, and
    this gate would keep passing on a string that no longer reaches the user.
    """
    path = os.path.join(ROOT, "frontend", "static", "js", "app.js")
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    # "filter((n) => /x|y/i" -- the arrow function's own parentheses sit between
    # the call and the literal, so the gap may not be [^)]*: that stops on the
    # ")" of "(n)" and never reaches the slash.
    m = re.search(r"intent\.notes\.filter\([^/]*/([^/\n]+)/i", src)
    if not m:
        _bad("could not find the note filter in app.js, so the 'the model was "
             "not used' contract cannot be checked at all")
        return ""
    return m.group(1)


# ----------------------------------------------------------------------
def main() -> int:
    from backend.agent import planner as P
    from backend.agent.llm import LLMClient, default_client, _message_text

    pattern = note_filter()
    if pattern:
        print(f"the front end shows a note as a warning when it matches "
              f"/{pattern}/i", flush=True)

    # ---- 1. one reader ------------------------------------------------
    _section("one reader for one setting")
    before = len(FAILURES)
    client = default_client()
    plan_ = P.LLMPlanner()
    if plan_.available != client.available:
        _bad(f"GET /api/llm reports available={client.available} (through "
             f"LLMClient) while the chat planner reports "
             f"available={plan_.available} -- one setting, two readers, so the "
             f"status endpoint and the behaviour can disagree")
    if plan_.model != client.model:
        _bad(f"the planner would ask {plan_.model!r} while the client is "
             f"configured for {client.model!r}")
    if plan_.base != client.base:
        _bad(f"the planner would post to {plan_.base!r} while the client is "
             f"configured for {client.base!r}")

    # Delegation, both directions: a client that cannot answer must make the
    # planner unable to answer, and one that can must make it able.  Asserted
    # through the object, so it holds whatever the live configuration is.
    class _Yes:
        base = "http://example.invalid/v1"
        key = "k"
        model = "m"
        available = True

        def json(self, *a, **k):
            return {}

    class _No:
        base = ""
        key = ""
        model = ""
        available = False

    for name, obj, want in (("a client with a key", _Yes(), True),
                            ("a client without one", _No(), False)):
        got = P.LLMPlanner(obj).available
        if got != want:
            _bad(f"LLMPlanner({name}) reports available={got}, expected {want}; "
                 f"the planner is deciding this for itself instead of asking "
                 f"its client")

    # And structurally: a second reader is a second answer.  This is a source
    # check on purpose -- the fault is the presence of the read, and no
    # behaviour can be exhibited when the environment happens to be empty.
    #
    # It looks for a read, not for the name.  The first version of this check
    # failed on the fixed code because plan() *mentions* CHATDFT_LLM_KEY in the
    # sentence telling the user how to configure one -- and a check that cannot
    # tell "reads the variable" from "names the variable in a message" is
    # reporting on prose, not on behaviour.
    src = open(os.path.join(ROOT, "backend", "agent", "planner.py"),
               encoding="utf-8").read()
    start = src.index("class LLMPlanner")
    rest = src[start + 1:]
    end = rest.find("\ndef ")
    body = src[start:start + 1 + end] if end >= 0 else src[start:]
    leaked = sorted(set(re.findall(
        r"""os\.environ(?:\.get\(|\[)\s*['"](CHATDFT_LLM_[A-Z]+)['"]""", body)))
    if leaked:
        _bad(f"LLMPlanner reads {', '.join(leaked)} itself; that is the second "
             f"reader that made /api/llm and /api/chat disagree")
    if len(FAILURES) == before:
        print(f"[PASS] the planner's configuration is the client's "
              f"(available={plan_.available}, model={plan_.model!r}) and it "
              f"reads no environment variable of its own", flush=True)

    # ---- 2. the schema covers what the server dispatches on -----------
    _section("the schema matches the server's vocabulary")
    before = len(FAILURES)
    prompt = P.LLMPlanner.system_prompt()
    missing = [k for k in P.JOB_TYPES if f'"{k}"' not in prompt]
    if missing:
        _bad(f"the prompt does not offer {missing}, so the model cannot route "
             f"those requests and will answer with something it was told about")

    # Scanned out of the dispatcher rather than listed here.  A list written
    # here would be a third copy of the vocabulary, and the whole defect was a
    # second copy that had already drifted.
    server_src = open(os.path.join(ROOT, "backend", "server.py"),
                      encoding="utf-8").read()
    routed = set(re.findall(r'(?:intent\.job_type|job\.kind)\s*==\s*"([a-z_]+)"',
                            server_src))
    unroutable = sorted(k for k in routed if f'"{k}"' not in prompt)
    if unroutable:
        _bad(f"server.py dispatches on {unroutable} but the prompt does not "
             f"mention them")
    if len(FAILURES) == before:
        print(f"[PASS] all {len(P.JOB_TYPES)} job types and all {len(routed)} "
              f"branches server.py dispatches on are in the prompt", flush=True)

    # ---- 3. off-schema replies are refused, on-schema ones applied ----
    _section("a reply is not trusted just because it parsed")
    before = len(FAILURES)

    def fresh():
        return P.RuleBasedPlanner().plan("single point energy of water")

    baseline = fresh()
    i = fresh()
    rejected = P.apply_llm_payload(i, {"job_type": "reaction",
                                       "molecule": "water",
                                       "functional": "PBE0", "charge": "1"})
    if rejected:
        _bad(f"an on-schema reply was refused: {rejected}")
    if i.job_type != "reaction" or i.functional != "pbe0" or i.charge != 1:
        _bad(f"an on-schema reply was not applied: job_type={i.job_type!r} "
             f"functional={i.functional!r} charge={i.charge!r}")

    i = fresh()
    rejected = P.apply_llm_payload(i, {"job_type": "nmr_spectrum",
                                       "molecules": 5,
                                       "functional": "mp2", "charge": "lots"})
    if i.job_type == "nmr_spectrum":
        _bad("a job type the server cannot dispatch was applied to the intent; "
             "the request would fail much later with a missing branch")
    if len(rejected) != 4:
        _bad(f"{len(rejected)} of 4 off-schema fields were reported: {rejected}")
    for field in ("job_type", "molecules", "functional", "charge"):
        if getattr(i, field) != getattr(baseline, field):
            _bad(f"a refused {field} still reached the intent: "
                 f"{getattr(i, field)!r} != {getattr(baseline, field)!r}")

    # the model must not be able to overwrite the channel its own verdict
    # travels in, or the sentence the user reads.  Compared against the
    # baseline rather than against empty: ``raw`` legitimately holds the user's
    # sentence already, so "non-empty" is not evidence of anything.
    i = fresh()
    P.apply_llm_payload(i, {"notes": ["planned by something else"],
                            "raw": "not what the user typed"})
    if i.notes != baseline.notes:
        _bad(f"the model overwrote the notes its own verdict travels in: "
             f"{i.notes!r} != {baseline.notes!r}")
    if i.raw != baseline.raw:
        _bad(f"the model overwrote the user's sentence: {i.raw!r} != "
             f"{baseline.raw!r}")
    if len(FAILURES) == before:
        print("[PASS] on-schema applied, off-schema refused and reported, "
              "notes/raw not writable by the model", flush=True)

    # ---- 4. the reasoning-field reply shape --------------------------
    _section("the reply shape reasoning models actually send")
    before = len(FAILURES)
    body = {"choices": [{"message": {"content": None,
                                     "reasoning_content": '{"job_type": "nmr"}',
                                     "finish_reason": "stop"}}]}
    if "nmr" not in _message_text(body):
        _bad("a reply whose answer is in reasoning_content and whose content "
             "is null reads as empty; the planner would fall back on every "
             "request even with a key configured")
    # and the planner must go through the client's json() rather than its own
    # POST, which is what carries that handling
    class _Recorder:
        base = "http://example.invalid/v1"
        key = "k"
        model = "recorder-model"
        available = True
        calls: list = []

        def json(self, messages, **kw):
            self.calls.append((messages, kw))
            return {"job_type": "nto", "molecule": "benzene"}

    rec = _Recorder()
    i = P.LLMPlanner(rec).plan("show me the natural transition orbitals")
    if not rec.calls:
        _bad("the planner did not ask its client; it is posting on its own")
    elif i.job_type != "nto" or i.molecule != "benzene":
        _bad(f"the client's reply was not applied: job_type={i.job_type!r} "
             f"molecule={i.molecule!r}")
    elif f"planned by {rec.model}" not in " ".join(i.notes):
        _bad(f"a plan produced by the model does not say so: notes={i.notes!r}")
    if len(FAILURES) == before:
        print("[PASS] reasoning_content is read, and the reply goes through "
              "the client and is named in the notes", flush=True)

    # ---- 5. the fallback is never silent -----------------------------
    _section("a fallback that does not say why is a defect")
    before = len(FAILURES)

    class _NoClient(_No):
        pass

    i = P.LLMPlanner(_NoClient(), reason="disabled for this check").plan(
        "compare water and ammonia")
    text = " ".join(i.notes)
    if pattern and not re.search(pattern, text, re.I):
        _bad(f"the keyword parser planned the request and the reply carries no "
             f"note the front end would display (needs /{pattern}/i): "
             f"{i.notes!r}")
    if "disabled for this check" not in text:
        _bad(f"the note does not name the reason it fell back: {i.notes!r}")

    # and with no reason supplied it must still produce an actionable one
    i = P.LLMPlanner(_NoClient()).plan("compare water and ammonia")
    text = " ".join(i.notes)
    if "CHATDFT_LLM_KEY" not in text and "llm.json" not in text:
        _bad(f"with nothing configured the note does not say how to configure "
             f"it: {i.notes!r}")

    # create_planner must never return the bare parser: the object that knows
    # the reason has to be the object that answers
    saved = os.environ.get("CHATDFT_PLANNER")
    try:
        for mode, want_active in (("local", False), ("auto", client.available),
                                  ("llm", client.available)):
            os.environ["CHATDFT_PLANNER"] = mode
            who, active = P.create_planner()
            if active != want_active:
                _bad(f"CHATDFT_PLANNER={mode} reports the model is "
                     f"{'in use' if active else 'not in use'}, expected "
                     f"{'in use' if want_active else 'not in use'}")
            if not hasattr(who, "client"):
                _bad(f"CHATDFT_PLANNER={mode} returned a bare "
                     f"{type(who).__name__}, which cannot report why the model "
                     f"was not used")
            elif not active:
                # Only the fallback branches are exercised here: when the
                # model is live this would be a network call, and section 7
                # already makes exactly one of those.
                note = " ".join(who.plan("single point energy of water").notes)
                if not re.search(pattern or "unavailable", note, re.I):
                    _bad(f"CHATDFT_PLANNER={mode} planned with the parser and "
                         f"said nothing about it: notes={note!r}")
    finally:
        if saved is None:
            os.environ.pop("CHATDFT_PLANNER", None)
        else:
            os.environ["CHATDFT_PLANNER"] = saved
    if len(FAILURES) == before:
        print("[PASS] every route to the keyword parser carries a reason the "
              "front end will display", flush=True)

    # ---- 6. over HTTP -------------------------------------------------
    # The gate server runs with CHATDFT_PLANNER=local, so this is the
    # deterministic half: the reply must say which planner ran and why.  A
    # local mode that says nothing would be indistinguishable from the silent
    # fallback this gate exists for.
    _section("over HTTP: who planned it, and does the reply say so")
    before = len(FAILURES)
    code, resp = _req("POST", "/api/chat",
                      {"message": "Compare water and ammonia",
                       "auto_run": False})
    if code != 200 or not isinstance(resp, dict):
        _bad(f"POST /api/chat failed: HTTP {code} {resp}")
    else:
        who = resp.get("planner")
        if who not in ("llm", "local"):
            _bad(f"the reply does not name the planner at all: planner={who!r}")
        notes = " ".join((resp.get("intent") or {}).get("notes") or [])
        if who == "local" and pattern and not re.search(pattern, notes, re.I):
            _bad(f"the parser planned the request over HTTP and the reply "
                 f"carries no displayable note: {notes!r}")
        if who == "llm" and "planned by" not in notes:
            _bad(f"the reply claims the model planned it but no note names the "
                 f"model: {notes!r}")
        if not FAILURES[before:]:
            print(f"[PASS] /api/chat reports planner={who!r} with a note that "
                  f"explains it", flush=True)

    # the same question to the endpoint that reports the configuration: if it
    # says a model is configured, the chat must not have been parsed by
    # keywords for want of one
    code, info = _req("GET", "/api/llm")
    available = bool(isinstance(info, dict) and info.get("available"))
    if available and resp.get("planner") == "local":
        notes = " ".join((resp.get("intent") or {}).get("notes") or [])
        if "unavailable" not in notes:
            _bad("GET /api/llm says a model is configured while /api/chat "
                 "planned with the keyword parser and did not say why -- this "
                 "is exactly the disagreement the gate exists for")
    if not FAILURES[before:]:
        print(f"[PASS] /api/llm available={available} is consistent with the "
              f"chat planner={resp.get('planner')!r} under "
              f"CHATDFT_PLANNER=local", flush=True)

    # ---- 7. the live model path ---------------------------------------
    # Asserted in whichever direction it goes.  A branch that is skipped when
    # the endpoint is unhappy is a branch that never runs, and "never runs"
    # looks exactly like "passes".
    _section("the live model path")
    before = len(FAILURES)
    if not client.available:
        i = P.LLMPlanner().plan("single point energy of water")
        notes = " ".join(i.notes)
        if not re.search(pattern or "unavailable", notes, re.I):
            _bad(f"no model is configured and the reply does not say so: "
                 f"{i.notes!r}")
        print("[PASS] no model configured: the reply says so and names how to "
              "configure one", flush=True)
    else:
        started = time.time()
        i = P.LLMPlanner().plan("Compute the NMR shifts of ethanol")
        seconds = time.time() - started
        notes = " ".join(i.notes)
        if f"planned by {client.model}" in notes:
            print(f"[PASS] {client.model} planned it in {seconds:.1f}s: "
                  f"job_type={i.job_type!r}", flush=True)
        elif "unavailable" in notes:
            # A degradation, not a defect -- but only if it names its cause,
            # and only if the cause is not "there is no model" while the client
            # says there is one.  That contradiction *is* the defect this gate
            # exists for, and it must not be able to pass as a rate limit.
            m = re.search(r"unavailable \(([^)]*)\)", notes)
            reason = (m.group(1).strip() if m else "")
            if not reason or len(reason) < 6:
                _bad(f"the model was configured but the planner fell back "
                     f"without naming an error: {i.notes!r}")
            elif "no language model configured" in reason:
                _bad("the client reports a model is configured while the "
                     "planner reports none is -- the planner is deciding this "
                     "for itself instead of asking its client")
            else:
                print(f"[PASS] the endpoint did not answer ({reason}); the "
                      f"planner fell back to the keyword parser and said why, "
                      f"in {seconds:.1f}s", flush=True)
        else:
            _bad(f"a model is configured and the planner neither used it nor "
                 f"explained why: notes={i.notes!r}")

    print()
    if FAILURES:
        print(f"=== FAIL: {len(FAILURES)} planner problem(s) ===")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("=== PASS: the model plans when it is configured, and every fallback "
          "names itself ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
