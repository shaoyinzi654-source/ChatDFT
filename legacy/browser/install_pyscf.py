"""
Install the locally compiled PySCF into the active Python environment.

PySCF is pure Python plus a set of C shared libraries loaded through
``ctypes``; there are no CPython extension modules.  That means the whole
"install" reduces to

    1. collect every ``*.dll`` the CMake build produced,
    2. drop it into ``pyscf/lib/`` (the directory PySCF's loader looks in),
    3. copy the ``pyscf`` package tree into ``site-packages``.

Step 2 has to include the MinGW runtime (``libgcc_s_seh-1``, ``libgomp-1``,
``libwinpthread-1``) and MKL, because the DLLs we built link against them and
Windows resolves a DLL's dependencies next to the DLL itself.

Run with the interpreter you want PySCF installed into::

    python install_pyscf.py

Optional environment variables
------------------------------
PYSCF_SRC      path to the unpacked PySCF source tree
PYSCF_BUILD    path to the CMake build directory
MINGW_BIN      path to the MinGW-w64 ``bin`` directory (runtime DLLs)
MKL_BIN        path to the directory holding ``mkl_rt.2.dll``
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PYSCF_SRC = Path(os.environ.get(
    "PYSCF_SRC", r"C:/Users/frddx/AppData/Local/Temp/pcdl/pyscf-2.6.2"))
PYSCF_BUILD = Path(os.environ.get(
    "PYSCF_BUILD", str(ROOT / "build" / "gcc_new")))
MINGW_BIN = Path(os.environ.get(
    "MINGW_BIN", str(ROOT / "tools" / "mingw64" / "bin")))
MKL_BIN = Path(os.environ.get(
    "MKL_BIN", str(Path(sys.prefix) / "Library" / "bin")))

# Runtime DLLs the compiled PySCF libraries depend on.
#
# MKL is deliberately *not* copied here.  A conda MKL installation is a set of
# mutually dependent DLLs (mkl_rt.2 + mkl_core.2 + mkl_intel_thread.2 + ...)
# that must all live in the same directory, and mkl_rt.2 delay-loads the
# others relative to itself.  Copying just mkl_rt.2.dll next to PySCF makes
# MKL fail to start.  Instead the application prepends the environment's
# Library/bin to PATH at start-up (see backend/bootstrap.py), exactly as
# `conda activate` does.
RUNTIME_DLLS = [
    "libgcc_s_seh-1.dll",
    "libgomp-1.dll",
    "libwinpthread-1.dll",
]


def log(msg: str) -> None:
    print(f"[install_pyscf] {msg}")


def collect_build_dlls() -> list[Path]:
    """Every shared library the CMake build produced."""
    found: list[Path] = []
    if not PYSCF_BUILD.is_dir():
        raise SystemExit(f"build directory not found: {PYSCF_BUILD}")

    for path in PYSCF_BUILD.rglob("*.dll"):
        # Skip CMake's own scratch directories and the throw-away test binaries.
        parts = {p.lower() for p in path.parts}
        if "cmakescratch" in parts:
            continue
        found.append(path)

    if not found:
        raise SystemExit(f"no DLLs found under {PYSCF_BUILD}")
    return found


def copy_runtime(dest: Path) -> int:
    """Copy MinGW/MKL runtime DLLs next to the PySCF libraries."""
    copied = 0
    for name in RUNTIME_DLLS:
        for base in (MINGW_BIN, MKL_BIN, Path(sys.prefix) / "Library" / "bin"):
            candidate = base / name
            if candidate.is_file():
                shutil.copy2(candidate, dest / name)
                log(f"runtime  {name}")
                copied += 1
                break
        else:
            log(f"WARNING  runtime DLL not found, skipping: {name}")
    return copied


def install_package(lib_dir: Path) -> Path:
    """Copy the ``pyscf`` package tree into site-packages."""
    src_pkg = PYSCF_SRC / "pyscf"
    if not src_pkg.is_dir():
        raise SystemExit(f"pyscf package not found at {src_pkg}")

    import site

    candidates = [Path(p) for p in site.getsitepackages()]
    target_root = next((p for p in candidates if p.is_dir()), None)
    if target_root is None:
        raise SystemExit("could not locate site-packages")

    dest_pkg = target_root / "pyscf"
    if dest_pkg.exists():
        log(f"removing previous install at {dest_pkg}")
        shutil.rmtree(dest_pkg)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n in ("__pycache__", "build", ".git")}

    shutil.copytree(src_pkg, dest_pkg, ignore=ignore)
    log(f"installed package -> {dest_pkg}")
    return dest_pkg


def main() -> int:
    log(f"source : {PYSCF_SRC}")
    log(f"build  : {PYSCF_BUILD}")
    log(f"python : {sys.executable}")

    src_lib = PYSCF_SRC / "pyscf" / "lib"
    if not src_lib.is_dir():
        raise SystemExit(f"pyscf/lib not found at {src_lib}")

    dlls = collect_build_dlls()
    log(f"found {len(dlls)} compiled libraries")

    # 1. compiled libraries -> pyscf/lib/
    for dll in dlls:
        shutil.copy2(dll, src_lib / dll.name)
        log(f"library  {dll.name}")

    # 2. runtime dependencies -> pyscf/lib/
    copy_runtime(src_lib)

    # 3. package tree -> site-packages (now containing the libraries)
    dest = install_package(src_lib.parent)
    log(f"libraries present in {dest / 'lib'}: "
        f"{len(list((dest / 'lib').glob('*.dll')))}")

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
