"""probe_nmr21.py -- cost breakdown for the TMS reference after the memory fix.

The first attempt at the TMS reference (17 atoms, nao = 98 at 6-31G*) thrashed
a 15 GB machine for over half an hour and never finished, because
`build_operators` formed the full (3, nao, nao, nao, nao) GIAO ERI derivative
and then contracted it -- four arrays of 2.26 GB live at once, about 11 GB
peak, with 1 GB free.

`_giao_eri` now builds one field direction at a time.  This probe times the
pieces separately so the fix is measured rather than assumed, and reports the
peak working set.

Run:  python -u probes/probe_nmr21.py
"""
import time

import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from backend.engine import nmr


def rss_mb():
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc),
            pmc.cb)
        return pmc.WorkingSetSize / 1e6, pmc.PeakWorkingSetSize / 1e6
    except Exception:
        return float("nan"), float("nan")


def main():
    key = "tms"
    atoms = nmr.reference_geometry(key)
    print(f"TMS geometry: {len(atoms)} atoms")
    t0 = time.time()
    mol = nmr._build_mol(nmr._xyz(atoms), "6-31g*", 0, 1)
    nao = mol.nao_nr()
    print(f"  nao = {nao}   build mol: {time.time() - t0:.1f}s")
    print(f"  memory estimate: {nmr.memory_estimate_mb(nao):.0f} MB "
          f"(budget {nmr.MEMORY_BUDGET_MB} MB)")
    cur, peak = rss_mb()
    print(f"  working set now {cur:.0f} MB, peak {peak:.0f} MB")

    t0 = time.time()
    ops = nmr.build_operators(mol)
    print(f"  build_operators: {time.time() - t0:.1f}s")
    cur, peak = rss_mb()
    print(f"  working set now {cur:.0f} MB, peak {peak:.0f} MB")
    print(f"  ig1 shape {ops['ig1'].shape}  "
          f"({ops['ig1'].nbytes / 1e9:.2f} GB)")

    t0 = time.time()
    m0 = nmr._scf_at(mol, ops, [0.0] * 3, None)
    print(f"  unperturbed SCF: {time.time() - t0:.1f}s  "
          f"converged={m0.converged}  E={m0.e_tot.real:.6f}")
    cur, peak = rss_mb()
    print(f"  working set now {cur:.0f} MB, peak {peak:.0f} MB")

    dm0 = m0.make_rdm1()
    t0 = time.time()
    mp_ = nmr._scf_at(mol, ops, [3e-4, 0.0, 0.0], dm0)
    print(f"  one perturbed SCF: {time.time() - t0:.1f}s  "
          f"converged={mp_.converged}")
    cur, peak = rss_mb()
    print(f"  working set now {cur:.0f} MB, peak {peak:.0f} MB")

    total = 7 * (time.time() - t0)
    print(f"\n  estimated full reference (7 SCF): "
          f"~{(time.time() - t0) * 7 / 60:.1f} min after the integrals are up")


if __name__ == "__main__":
    main()
