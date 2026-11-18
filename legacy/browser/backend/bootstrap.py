"""
Runtime environment bootstrap for the scientific stack.

ChatDFT links against libraries that live outside the normal Python import
path, and on Windows a DLL's dependencies are *not* resolved relative to the
DLL itself unless the search path says so.  Two things therefore have to be
arranged before ``pyscf`` is imported:

* the conda environment's ``Library/bin`` directory (which holds MKL and
  ``libiomp5md``) must be reachable, and
* ``pyscf/lib`` (which holds libcint, libxc and the PySCF C libraries) must be
  reachable.

Importing this module performs that setup once and is safe to call repeatedly.
``backend.engine`` imports it automatically, so application code never has to
think about it.  It is also harmless on platforms where none of this applies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_APPLIED = False


def _candidate_dll_dirs() -> list[Path]:
    """Directories that may hold shared libraries the stack needs."""
    dirs: list[Path] = []
    prefix = Path(sys.prefix)

    # conda / venv layout
    dirs.append(prefix / "Library" / "bin")        # MKL, libiomp5md, runtime
    dirs.append(prefix / "Library" / "mingw-w64" / "bin")
    dirs.append(prefix / "Scripts")
    dirs.append(prefix / "bin")                    # POSIX venv

    # the project's own toolchain (only present on the machine that built PySCF)
    here = Path(__file__).resolve().parent.parent
    dirs.append(here / "tools" / "mingw64" / "bin")

    # a PySCF installed anywhere on sys.path
    for entry in sys.path:
        try:
            p = Path(entry or ".") / "pyscf" / "lib"
        except (TypeError, ValueError):
            continue
        dirs.append(p)

    return [d for d in dirs if d.is_dir()]


def setup() -> dict:
    """Make the scientific stack's native libraries discoverable.

    Returns a small report so callers (and the health endpoint) can show what
    was configured.  Calling this more than once is a no-op.
    """
    global _APPLIED
    if _APPLIED:
        return {"applied": False, "reason": "already configured"}

    dirs = _candidate_dll_dirs()
    added: list[str] = []

    # os.add_dll_directory() is the modern, safe way to extend the DLL search
    # path on Windows.  It is unavailable (or unnecessary) elsewhere.
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        for d in dirs:
            try:
                add_dll_directory(str(d))
                added.append(str(d))
            except OSError:
                pass

    # Some libraries (notably MKL's threading layer) are still resolved through
    # PATH, so extend that too -- this is exactly what `conda activate` does.
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    lowered = {p.lower() for p in parts}
    prepend = [str(d) for d in dirs if str(d).lower() not in lowered]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + parts)

    _APPLIED = True
    return {
        "applied": True,
        "dll_directories": added,
        "path_prepended": prepend,
    }


def report() -> str:
    """One-line human-readable summary, for logs."""
    info = setup()
    if not info.get("applied"):
        return "environment already configured"
    return (f"added {len(info.get('dll_directories', []))} DLL directories, "
            f"{len(info.get('path_prepended', []))} PATH entries")
