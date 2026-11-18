"""Probe -- is the methane outlier in probe_mem1 a warm-up artefact?

probe_mem1 measured estimate/measured ratios of 0.31, 1.04, 0.92, 0.82, 0.81
for methane, ammonia, water, ethanol, formaldehyde.  Four of those say the
constant is right to within 20%; the fifth says it is three times too small.
Methane was the *first* molecule measured, which is exactly when a Python
process pays for its imports, its BLAS thread pools and PySCF's lazy
initialisation -- none of which scale with nao.

If that is the explanation, the outlier must follow the *order*, not the
molecule: measuring methane last should bring it into line, and measuring it
twice should show the second run cheaper than the first.  If the outlier
follows the molecule instead, the estimate really is wrong at small nao and
the guard is under-refusing small jobs (harmless now, but it would mean the
constant is not a constant).

Changes nothing.

Run:  python probes/probe_mem2.py
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
    import psutil

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
    nmr.build_operators(mol)
    base = psutil.Process().memory_info().rss / 2 ** 20
    PEAK["mb"] = base
    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    t0 = time.time()
    try:
        nmr.shielding_tensor(mol, nmr.build_operators(mol))
    except Exception as exc:
        print(f"{key}: failed ({type(exc).__name__}: {str(exc)[:70]})")
        return
    finally:
        STOP.set()
        t.join(timeout=1)
    dt = time.time() - t0
    grew = PEAK["mb"] - base
    per = grew * 1e6 / (nao ** 4)
    print(f"{key:12s} nao={nao:4d}  est {need:8.1f} MB  measured {grew:8.1f} MB  "
          f"({dt:5.1f}s)  ratio {need / max(grew, 1e-6):5.2f}x  "
          f"bytes/nao^4 {per:5.1f}")


def main() -> int:
    print("=== same molecules, methane moved to the end and run twice ===\n")
    for key in ("ammonia", "water", "ethanol", "methane", "methane"):
        measure(key)
    print("\n(if the methane ratio joins the others, the outlier was warm-up)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
