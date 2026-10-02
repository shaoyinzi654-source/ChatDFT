"""Rebuild corrupt library geometries from their SMILES strings.

Several hand-written XYZ blocks in ``data/molecules/library.json`` were
wrong: two hydrogens parked 0.43 A from a ring carbon in naphthalene, an
all-1.09 A chloroform, and -- worse -- furan and pyrrole whose coordinates
described a six-membered ring (i.e. a different molecule entirely).

The fix is not to hand-patch numbers but to regenerate the geometry from
the entry's own SMILES with the same RDKit pipeline ``from_smiles`` uses,
so that ``smiles``, ``formula`` and ``xyz`` are guaranteed consistent.

Only the entries that fail ``check_library`` are touched; the file is
rewritten in place with its original key order and formatting.

Usage:
    python -m backend.rebuild_library                 # fix every failing entry
    python -m backend.rebuild_library naphthalene ... # fix specific entries
    python -m backend.rebuild_library --dry-run       # report, change nothing
"""

from __future__ import annotations

import json
import os
import re
import sys

from backend.engine import molecule as molmod

_LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "molecules", "library.json",
)


def _parse_formula(f: str) -> dict:
    return {el: int(c or 1) for el, c in re.findall(r"([A-Z][a-z]?)(\d*)", f) if el}


def _load() -> dict:
    with open(_LIB_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _save(data: dict, trailing_newline: bool) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if trailing_newline:
        text += "\n"
    with open(_LIB_PATH, "w", encoding="utf-8") as fh:
        fh.write(text)


def failing_keys(data: dict) -> list[str]:
    from backend import check_library

    return [k for k in data if check_library.check_entry(k)]


def rebuild(key: str, entry: dict) -> tuple[bool, str]:
    """Regenerate one entry's xyz from its SMILES. Returns (ok, message)."""
    smiles = entry.get("smiles", "").strip()
    if not smiles:
        return False, "no SMILES to rebuild from"

    try:
        mol = molmod.from_smiles(smiles, name=molmod._entry_name(entry, key))
    except Exception as exc:  # noqa: BLE001
        return False, f"RDKit could not embed: {exc}"

    got = mol.compute_formula()
    want = entry.get("formula", got)
    if _parse_formula(got) != _parse_formula(want):
        return False, f"SMILES gives {got} but the entry declares {want}"

    if mol.charge != entry.get("charge", 0):
        return False, (
            f"SMILES formal charge {mol.charge} != entry charge {entry.get('charge', 0)}"
        )

    entry["xyz"] = mol.to_xyz().rstrip("\n")
    return True, f"rebuilt {mol.natoms()} atoms, formula {got}"


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    explicit = [a for a in argv if not a.startswith("-")]

    data = _load()
    with open(_LIB_PATH, encoding="utf-8") as fh:
        trailing_newline = fh.read().endswith("\n")

    targets = explicit or failing_keys(data)
    if not targets:
        print("nothing to do: every library geometry already passes")
        return 0

    print(f"{'would rebuild' if dry else 'rebuilding'} {len(targets)} entry(ies)\n")
    changed, failed = [], []
    for key in targets:
        if key not in data:
            print(f"[SKIP] {key}: not in the library")
            failed.append(key)
            continue
        ok, msg = rebuild(key, data[key])
        if ok:
            print(f"[ OK ] {key}: {msg}")
            changed.append(key)
        else:
            print(f"[FAIL] {key}: {msg}")
            failed.append(key)

    if not dry and changed:
        _save(data, trailing_newline)
        print(f"\nwrote {_LIB_PATH}")

    print(f"\n{len(changed)} rebuilt, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
