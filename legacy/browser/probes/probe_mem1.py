"""Probe -- is the NMR memory guard's estimate anywhere near the truth?

``nmr.check_memory`` refuses a calculation whose estimate exceeds 1.25x the
free physical memory, and it is right to: the first TMS reference was measured
at 5165 MB against a 6000 MB budget on a machine with 1 GB free, took 4829 s
instead of 285, and a refusal that names the shortfall beats an hour of
thrashing.

But the estimate itself is a constant times nao^4, and nobody checked the
constant against a real run.  On this machine that matters: thiophene is
refused as "about 2.5 GB" when 1.7 GB is free, so a gate that has passed for
rounds now fails on an environmental shortfall -- and if the constant is
pessimistic, the refusal is refusing work the machine could do.

This probe measures the peak resident memory of the shielding machinery for
molecules small enough to run, and puts it next to the estimate.  Changes
nothing.

Run:  python probes/probe_mem1.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import nmr
from backend.engine.molecule import _load_library

PEAK = {"mb": 0.0}
STOP = threading.Event()


def _sample() -> None:
    import psutil

    proc = psutil.Process()
    while not STOP.is_set():
        try:
            PEAK["mb"] = max(PEAK["mb"], proc.memory_info().rss / 2 ** 20)
        except Exception:
            pass
        time.sleep(0.02)


def measure(key: str, basis: str = "6-31g*") -> None:
    entry = _load_library().get(key)
    if entry is None:
        print(f"{key}: not in the library")
        return
    mol = nmr._build_mol(entry["xyz"], basis, int(entry.get("charge", 0)),
                         int(entry.get("multiplicity", 1)))
    nao = int(mol.nao_nr())
    need = nmr.memory_estimate_mb(nao)
    PEAK["mb"] = 0.0
    STOP.clear()
    base = None
    import psutil

    base = psutil.Process().memory_info().rss / 2 ** 20
    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    t0 = time.time()
    try:
        ops = nmr.build_operators(mol)
        base = psutil.Process().memory_info().rss / 2 ** 20
        PEAK["mb"] = base
        nmr.shielding_tensor(mol, ops)
    except Exception as exc:
        STOP.set()
        t.join(timeout=1)
        print(f"{key}: failed ({type(exc).__name__}: {str(exc)[:70]})")
        return
    finally:
        STOP.set()
        t.join(timeout=1)
    dt = time.time() - t0
    grew = PEAK["mb"] - base
    print(f"{key:12s} nao={nao:4d}   estimate {need/1024:7.2f} GB   "
          f"measured peak +{grew:7.1f} MB   ({dt:.1f}s)   "
          f"ratio {need / max(grew, 1e-6):5.2f}x")


def main() -> int:
    print("=== NMR: estimated vs measured peak memory ===\n")
    for key in ("methane", "ammonia", "water", "ethanol", "formaldehyde"):
        measure(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
