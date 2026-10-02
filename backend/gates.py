"""Run every regression gate against a server this process owns.

    python -m backend.gates                 # all gates
    python -m backend.gates contract sweep  # a subset

Why this exists.  The gates talk to a live HTTP server, and booting one in a
separate shell and hoping it is still alive when the gates start turned out
to be the single most unreliable part of the workflow: the server kept being
reaped between steps, and a missing server looks exactly like a failed gate.
The server therefore runs as a child of this process, which restarts it if it
dies: if the orchestrator is running, it makes sure a server is too.

Each gate runs in its own subprocess as well, so it keeps its own ``__main__``
block and its own exit code, and a gate that hard-crashes cannot take the
suite with it.  This module only aggregates the results.
"""

from __future__ import annotations

import os
import re
import sys
import time

GATES = ["check_selectors", "check_library", "agent.designer", "contract",
         "selftest", "sweep", "check_series", "check_fields", "check_nto",
         "check_reaction", "check_shift_scales", "verify_library_geometry",
         "check_planner"]

# Selftest is only meaningful with --full; the short form skips the slow
# numerical checks that actually catch regressions.  The geometry verifier
# reports by default and only behaves as a gate with --check, which is what
# makes it usable interactively without turning every run into a verdict.
EXTRA_ARGS = {"selftest": ["--full"],
              "verify_library_geometry": ["--check"]}

# A gate that wedges should be reported as wedged, not left to hold the suite
# open.  The slowest gate (check_reaction) finishes in about six minutes.
GATE_TIMEOUT = 1800.0

# Opt-in gates that talk to an external service (the real LLM endpoint) or
# drive a real browser.  Not in the default list because they cost minutes
# and can flake under account-level rate limits.
EXTRA_GATES = ["browser_design", "browser_series", "browser_scan", "browser_uv",
               "browser_raman", "browser_nmr"]

# How many [PASS]/[FAIL] lines each gate emitted, filled in as it runs.
VERDICTS: dict = {}

# Lines that carry a verdict.  Everything else a gate prints is library noise
# (SCF iterations, geomeTRIC's convergence table) and can be elided freely;
# these cannot, because they are the record of what was actually checked.
#
# The gates do not agree on one format -- contract prints "[PASS] ...",
# designer "  PASS  name", selftest "  water  ... ok  (0.4s)" -- so all three
# spellings have to be recognised.  Getting this wrong is not cosmetic: with
# only the bracketed form matched, two of the ten gates reported zero checks
# while passing, and the summary believed them.
_KEEP = re.compile(
    r"\[PASS\]|\[FAIL\]|\bPASS\b|\bFAIL\b|FAILED|ERROR:|^=== "
    r"|^MISSING |^DUPLICATE |^Traceback|\.\.\.\s+(?:ok|FAILED|ERROR)",
    re.MULTILINE)
_GOOD = re.compile(r"\[PASS\]|\bPASS\b|^=== PASS|\.\.\.\s+ok\b", re.MULTILINE)
# The bare word FAILURES is excluded when it is the gate's own tally line
# ("3 FAILURES: a, b, c").  Counting that as a fourth failure made a gate with
# three problems report four, which sends the reader looking for a defect that
# does not exist -- the failure names are already on their own FAIL lines.
_BAD = re.compile(
    r"\[FAIL\]|\bFAIL\b|^=== FAIL|FAILED|\bFAILURES\b(?!\s*:)|ERROR:|"
    r"\.\.\.\s+ERROR",
    re.MULTILINE)


def _wait_until_up(url: str, timeout: float = 60.0) -> bool:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as fh:
                if fh.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH: dict = {}


def _scratch_root() -> str:
    return SCRATCH.get("dir") or ROOT


def _server_env() -> dict:
    env = dict(os.environ)
    bits = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if ROOT not in bits:
        env["PYTHONPATH"] = os.pathsep.join([ROOT] + bits)
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env["PYTHONUNBUFFERED"] = "1"
    # The gates must not depend on a language-model endpoint.  Nearly every
    # gate posts to /api/chat, so with model planning on, a full run would
    # spend minutes of latency in the model and would fail whenever the account
    # is rate limited -- a failure with nothing to do with the code under test,
    # and one that looks exactly like a broken router.  The keyword parser is
    # deterministic, which is what a regression suite wants.
    #
    # This is not a way of not testing the model path: the reply says which
    # planner ran and why, and check_planner asserts the auto rule, the schema
    # and the fallback directly.
    env["CHATDFT_PLANNER"] = "local"
    return env


class Server:
    """The gate server, in its own process.

    It used to run on a thread inside this one.  That was fine until it was
    not: the whole suite started dying after the ninth gate with no traceback
    and no summary, because the interpreter hosting uvicorn crashed and took
    the orchestrator with it -- losing the buffered output that would have
    said which gate ran and what it concluded.  Isolating the gates in
    subprocesses does not help when the process that dies is this one.

    Out here, a server crash is a fact this object can notice and recover
    from, and the server's own PySCF chatter goes to its own log instead of
    being interleaved with the verdicts.
    """

    def __init__(self, port: int = 8000):
        import subprocess

        self.port = port
        self.proc = None
        self.log_path = os.path.join(_scratch_root(), "gate_server.log")
        self._spawn()

    def _spawn(self) -> None:
        import subprocess

        fh = open(self.log_path, "ab", buffering=0)
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "backend.server"],
            stdout=fh, stderr=subprocess.STDOUT,
            env=_server_env(), cwd=_scratch_root(),
        )
        self._fh = fh

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def bound_elsewhere(self) -> str:
        """Did the server we spawned fail to bind the port?

        uvicorn logs the bind error *after* "Application startup complete", so
        the process can be alive for a moment while a different server answers
        the health check.  The log is the only place that says which happened,
        and it is written to a temp directory nobody would otherwise open.
        """
        try:
            with open(self.log_path, "r", encoding="utf-8",
                      errors="replace") as fh:
                text = fh.read()[-4000:]
        except OSError:
            return ""
        for marker in ("error while attempting to bind",
                       "Errno 10048", "address already in use",
                       "Only one usage of each socket address"):
            if marker in text:
                return marker
        return ""

    def ensure(self) -> bool:
        """Restart if it has died; report whether one is answering.

        ``alive`` is not enough on its own.  A process that failed to bind
        exits a moment after it is polled, and the port answers anyway because
        the *previous* listener is still there -- so "the port answers" and
        "our server is the one answering" are two different questions.  The
        caller guards the port before starting (see ``_port_holder``); this
        keeps the invariant honest for the restart path too.
        """
        if not self.alive:
            print("  !! the gate server died -- restarting it", flush=True)
            try:
                self._fh.close()
            except Exception:                            # noqa: BLE001
                pass
            self._spawn()
        if _wait_until_up(f"http://127.0.0.1:{self.port}/api/health",
                          timeout=60.0):
            clash = self.bound_elsewhere()
            if clash or not self.alive:
                reason = clash or f"exited, status {self.proc.poll()}"
                print(f"  !! the gate server could not bind 127.0.0.1:"
                      f"{self.port} ({reason}) -- the port answers, but not "
                      "from a server this suite started.  Refusing to test "
                      "someone else's configuration.", flush=True)
                return False
            return True
        return False

    def stop(self) -> None:
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=10)
        except Exception:                                # noqa: BLE001
            pass
        try:
            self._fh.close()
        except Exception:                                # noqa: BLE001
            pass


def _port_holder(port: int):
    """A short description of whatever is already listening, or None.

    This exists because the suite used to start its own server, watch that
    process fail to bind, and then run every gate against *someone else's*
    server on the same port -- while the comment in ``check_planner`` said "the
    gate server runs with CHATDFT_PLANNER=local".  Observed exactly once, and
    the evidence was a line in a temp directory:

        ERROR: [Errno 10048] error while attempting to bind on address
        ('127.0.0.1', 8000)

    ``ensure()`` could not see it: the spawned process had not exited yet when
    it was polled, and ``_wait_until_up`` succeeded because a *different*
    server was answering.  A gate suite that reports on a configuration it did
    not choose is worse than one that does not run.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) != 0:
            return None
    return f"something is already listening on 127.0.0.1:{port}"


def _run_one(name: str) -> int:
    """Run one gate and report it.

    In a **subprocess**, not in-process.  Running them with ``runpy`` in here
    meant a gate that hard-crashed took the whole suite with it: no traceback,
    no verdict, no summary -- the run simply stopped after the previous gate
    and exited non-zero, which reads exactly like a gate that failed.  That
    happened twice (once after agent.designer, once after check_nto, different
    gates each time, so it is not one bad gate) and both times the remaining
    gates were never run at all.

    A subprocess cannot do that.  It also gets its own address space, which is
    where the crash almost certainly lives -- PySCF and MKL in one interpreter
    for twenty minutes, with a uvicorn thread alongside.
    """
    import subprocess

    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bits = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if root not in bits:
        env["PYTHONPATH"] = os.pathsep.join([root] + bits)

    cmd = [sys.executable, "-u", "-m", f"backend.{name}"] + EXTRA_ARGS.get(name, [])
    # stderr into stdout, NOT captured separately.  Concatenating the two
    # buffers afterwards puts every line a library wrote to fd 2 after every
    # line the gate printed, which buries the verdict list in the middle of
    # the stream -- where _condense() then treats it as noise and elides it.
    # It did exactly that: a gate failed, and the log showed
    # "=== FAIL: 2 contract problem(s) ===" followed by "... 140 lines ...".
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              env=env, cwd=root, timeout=GATE_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(
            exc.stdout, bytes) else str(exc.stdout or "")
        rc = 124
        out += f"\nTIMED OUT after {GATE_TIMEOUT}s\n"
    else:
        out = proc.stdout or ""
        rc = proc.returncode

    # A negative return code on Windows is an access violation / stack
    # overflow: the interpreter died without raising, so there is nothing to
    # print.  Say that instead of leaving an empty section.
    if rc < 0:
        out += (f"\nCRASHED: the interpreter died with status {rc} "
                f"(0x{rc & 0xFFFFFFFF:08X}) and produced no Python "
                f"traceback.\n")

    npass, nfail = _verdicts(out)
    VERDICTS[name] = (npass, nfail)
    print(f"\n{'=' * 66}\n  {name}  ({npass} passed, {nfail} failed)"
          f"\n{'=' * 66}")
    print(_condense(out))
    return rc


def _condense(out: str, head: int = 8, tail: int = 40) -> str:
    """Trim a gate's output for the console without losing any verdict.

    Keeping the head and the tail alone is the obvious rule and it is wrong.
    A gate prints hundreds of lines of library noise and a few dozen verdicts
    scattered through the middle, so head+tail drops the verdicts -- and once
    they are gone, a check that never ran is indistinguishable from one that
    passed.  That is not a hypothetical: it cost a whole round of debugging to
    discover that a new UV-Vis assertion was fine and simply invisible.

    So elide the noise, never the verdicts, and say how much was elided.
    """
    lines = out.splitlines()
    if len(lines) <= head + tail:
        return out
    keep = set(range(head)) | set(range(len(lines) - tail, len(lines)))
    for i, line in enumerate(lines):
        if _KEEP.search(line):
            keep.add(i)
            # A traceback is only readable as a block; keep all of it.
            if line.startswith("Traceback"):
                keep.update(range(i, min(i + 60, len(lines))))
    chunks: list[str] = []
    prev = -1
    for i in sorted(keep):
        if prev >= 0 and i != prev + 1:
            chunks.append(f"  ... {i - prev - 1} lines ...")
        chunks.append(lines[i])
        prev = i
    return "\n".join(chunks)


def _verdicts(out: str) -> tuple[int, int]:
    """(passed, failed).  Counted so an empty gate cannot pass unnoticed.

    A gate that exits 0 without printing a single verdict has not shown that
    anything works -- most likely its checks were skipped.  The count makes
    that visible instead of letting the summary say "PASS".
    """
    return len(_GOOD.findall(out)), len(_BAD.findall(out))


def _scratch_dirs() -> str:
    """Point the app's own data directories at a throwaway tree.

    The gates run the real application, so a design job saves its result and
    prunes the history by unlinking the oldest files, and the surface gates
    write cube files.  Left alone that means a gate run edits and deletes
    things in the user's data directory -- and on a filesystem that audits or
    refuses unlinks, the refusal surfaces as a failed gate with nothing to do
    with the code under test.  Both problems disappear if the whole run has
    its own directory.

    Must be called before the server is imported, since the paths are read at
    import time.
    """
    import tempfile

    scratch = tempfile.mkdtemp(prefix="chatdft-gates-")
    os.environ["CHATDFT_DESIGN_DIR"] = os.path.join(scratch, "designs")
    os.environ["CHATDFT_CACHE_DIR"] = os.path.join(scratch, "cache")
    os.environ["CHATDFT_CUBE_DIR"] = os.path.join(scratch, "cubes")
    SCRATCH["dir"] = scratch
    return scratch


def _source_fingerprint() -> dict:
    """sha256 of every source file the gates exercise.

    Each gate runs in its own subprocess and imports the source fresh, so
    editing a file while a run is in flight means the earlier gates tested one
    version and the later ones tested another -- and the summary reports a
    single verdict for the mixture.  That is not hypothetical: engine/nmr.py was
    edited while a six-gate run was in progress, and the whole run had to be
    thrown away.  Nothing in its output said anything was wrong, and the summary
    said PASS.

    The mutation harness has a guard against a related hazard (it refuses to run
    while a server is listening, because the server would import the mutated
    module).  This is the other direction, and it had no guard at all: the
    source can change underneath a run that is only reading it.

    Only the files a gate can import are tracked.  Reports and logs are written
    during a run on purpose.
    """
    import hashlib

    out = {}
    for root, dirs, files in os.walk(os.path.join(ROOT, "backend")):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache")]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "rb") as fh:
                    out[os.path.relpath(path, ROOT).replace("\\", "/")] = \
                        hashlib.sha256(fh.read()).hexdigest()
            except OSError:
                pass
    for extra in ("frontend/static/js/app.js",):
        path = os.path.join(ROOT, extra)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                out[extra] = hashlib.sha256(fh.read()).hexdigest()
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    only = [a for a in argv if not a.startswith("-")]
    gates = only or GATES
    unknown = [g for g in gates if g not in (set(GATES) | set(EXTRA_GATES))]
    if unknown:
        print(f"unknown gate(s): {', '.join(unknown)}")
        print(f"default: {', '.join(GATES)}")
        print(f"opt-in:  {', '.join(EXTRA_GATES)}")
        return 2

    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    before = _source_fingerprint()
    scratch = _scratch_dirs()
    print(f"scratch: {scratch}")

    # Refuse rather than test the wrong server.  The gates run against
    # 127.0.0.1:8000 and the suite's own server sets CHATDFT_PLANNER=local;
    # if a development server is up, the suite would silently exercise that
    # one instead, with its own configuration and its own (possibly mutated)
    # source.  See _port_holder for the run where this actually happened.
    busy = _port_holder(8000)
    if busy:
        print(f"\nREFUSING TO RUN: {busy}.")
        print("  The gate suite starts its own server with its own")
        print("  environment (CHATDFT_PLANNER=local, a scratch data dir) and")
        print("  runs every gate against 127.0.0.1:8000.  With the port taken,")
        print("  the server it starts cannot bind, and the gates would then be")
        print("  testing whatever is already there -- a different")
        print("  configuration, and possibly a source tree mid-mutation.")
        print("  Stop it first:  python kill_gate_server.py")
        return 2

    server = Server()
    print(f"server log: {server.log_path}", flush=True)
    if not server.ensure():
        print("server did not come up")
        return 1

    results = {}
    try:
        for name in gates:
            # Check before every gate, not just once at the start: the server
            # is the process most likely to die, and a gate that runs against
            # no server reports failures that have nothing to do with it.
            if not server.ensure():
                results[name] = -1
                VERDICTS[name] = (0, 0)
                print(f"\n{'=' * 66}\n  {name}\n{'=' * 66}")
                print("  the gate server could not be reached; not run")
                continue
            results[name] = _run_one(name)
            # Flush: if this process itself dies, whatever is still buffered
            # dies with it, and the record of what passed is exactly what the
            # next person needs.
            sys.stdout.flush()
    finally:
        server.stop()

    print(f"\n{'=' * 66}\n  summary\n{'=' * 66}")
    failed = 0
    for name, rc in results.items():
        npass, nfail = VERDICTS.get(name, (0, 0))
        if rc < 0:
            # The interpreter died without raising.  Naming it matters: this
            # used to end the whole run silently, and "the suite stopped after
            # gate N" reads like a failure of gate N.
            verdict = f"CRASHED (status {rc})"
        elif rc == 124:
            verdict = f"TIMEOUT ({GATE_TIMEOUT:.0f}s)"
        elif rc != 0:
            verdict = f"FAIL (exit {rc})"
        elif npass + nfail == 0:
            # It exited 0 and asserted nothing.  Treat it as broken: a gate
            # that cannot show a single verdict has not shown anything works.
            verdict = "FAIL (no checks ran)"
        else:
            verdict = "PASS"
        print(f"  {name:16s} {verdict:20s} {npass} checks")
        failed += verdict != "PASS"

    # The source must not have moved underneath the run.  Checked last and
    # reported in the summary rather than in a gate section, because it is a
    # statement about the whole run: every per-gate verdict above is a verdict
    # about whatever version of that file was on disk when *that* gate started.
    after = _source_fingerprint()
    moved = sorted(set(before) ^ set(after)) + \
        sorted(k for k in set(before) & set(after) if before[k] != after[k])
    if moved:
        failed += 1
        print("\n  SOURCE CHANGED DURING THE RUN -- every verdict above is a "
              "verdict about a different mixture of versions")
        for k in moved:
            print(f"    {k}")
        print("  re-run with the source frozen; a PASS here means nothing")

    print(f"\n  {len(results) - failed}/{len(results)} gates passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
