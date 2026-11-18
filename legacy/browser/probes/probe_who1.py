"""Probe -- what is listening on the gate port, and is it still needed?

The NMR mutation harness rewrites backend/engine/nmr.py in place, and it
refuses to start while anything is listening on 127.0.0.1:8000 because a
server that imports a mutated module serves corrupted numbers.  Something is
listening.  Before killing it this probe reports what it is, how long it has
been up, and whether it is a live gate run or a leftover.

Changes nothing.

Run:  python probes/probe_who1.py
"""
import os
import time

import psutil

PORT = 8000


def main() -> int:
    print(f"=== who holds 127.0.0.1:{PORT} ===\n")
    for c in psutil.net_connections(kind="inet"):
        if c.laddr and c.laddr.port == PORT and c.status == "LISTEN":
            pid = c.pid
            try:
                p = psutil.Process(pid)
            except psutil.Error:
                print(f"pid {pid}: gone")
                continue
            age = time.time() - p.create_time()
            print(f"pid      {pid}")
            print(f"name     {p.name()}")
            print(f"cmdline  {' '.join(p.cmdline())[:160]}")
            print(f"cwd      {p.cwd()}")
            print(f"started  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.create_time()))}"
                  f"  ({age/60:.1f} min ago)")
            print(f"rss      {p.memory_info().rss / 2**20:.0f} MB")
            kids = p.children(recursive=True)
            print(f"children {len(kids)}")
            for k in kids[:5]:
                print(f"   {k.pid:7d}  {k.name()}  {' '.join(k.cmdline())[:100]}")
            par = p.parent()
            print(f"parent   {par.pid if par else None}  "
                  f"{par.name() if par else ''}")
            if par:
                print(f"   cmdline {' '.join(par.cmdline())[:140]}")
        else:
            continue
    print()
    print(f"this process: pid {os.getpid()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
