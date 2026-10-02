"""Stop whatever is holding the gate port, without stopping this script.

``backend.mutate_nmr`` refuses to run while anything answers on 127.0.0.1:8000,
because it rewrites ``backend/engine/nmr.py`` in place and a server that
imports the mutated module would serve corrupted numbers to anything that
talks to it.  ``backend.gates`` owns a server as a child and restarts it when
it dies, so killing the suite and the server separately leaves one behind.

Two traps this exists to avoid:

* Writing the matcher as ``python -c "... 'backend.server' in cmdline ..."``
  matches the matcher's own command line, and the script terminates itself
  before it reaches anything else.  That is exactly what happened the first
  two times.  As a file, the process command line is just the file name.
* ``backend.gates`` restarts its server, so the suite has to go first and its
  whole child tree with it, or a fresh listener appears seconds later.

Reports what it stopped.  Changes nothing else.

Run:  <conda python> kill_gate_server.py
"""

from __future__ import annotations

import os
import socket
import time

import psutil

PORT = 8000


def listening_pids(port: int) -> list:
    out = []
    for c in psutil.net_connections(kind="inet"):
        if (c.laddr and c.laddr.port == port and c.status == "LISTEN"
                and c.pid):
            out.append(c.pid)
    return sorted(set(out))


def describe(pid: int) -> str:
    try:
        p = psutil.Process(pid)
        return f"{p.name()} :: {' '.join(p.cmdline())[:88]}"
    except psutil.Error:
        return "(gone)"


def main() -> int:
    me = os.getpid()
    print(f"=== freeing 127.0.0.1:{PORT} (this script is pid {me}) ===\n")
    stopped = []
    # The suite first: it restarts its server, so killing the server alone is
    # undone within seconds.
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] == me:
            continue
        try:
            cl = " ".join(p.info["cmdline"] or [])
        except psutil.Error:
            continue
        if "backend.gates" in cl:
            kids = p.children(recursive=True)
            print(f"  gate suite {p.info['pid']}: {cl[:70]}")
            for k in kids:
                try:
                    k.terminate()
                except psutil.Error:
                    pass
            try:
                p.terminate()
            except psutil.Error:
                pass
            psutil.wait_procs(kids + [p], timeout=8)
            stopped.append(p.info["pid"])
    for pid in listening_pids(PORT):
        if pid == me:
            continue
        print(f"  listener  {pid}: {describe(pid)}")
        try:
            q = psutil.Process(pid)
            q.terminate()
            try:
                q.wait(timeout=8)
            except psutil.TimeoutExpired:
                q.kill()
            stopped.append(pid)
        except psutil.Error as exc:
            print(f"    could not stop it: {exc}")
    time.sleep(1.0)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        busy = s.connect_ex(("127.0.0.1", PORT)) == 0
    print(f"\n  stopped {len(stopped)} process(es); "
          f"port {PORT} is {'STILL BUSY' if busy else 'free'}")
    return 1 if busy else 0


if __name__ == "__main__":
    raise SystemExit(main())
