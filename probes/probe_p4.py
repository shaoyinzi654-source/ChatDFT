"""Probe -- does the new coverage assertion actually fire?

An assertion nobody has seen fail is a guess.  This runs the library section
twice: once against the real library, and once against a copy with the
phosphorus entries removed -- which is exactly the state the library was in
before this round, and the state the assertion exists to make impossible.

Changes nothing on disk; the doctored library is passed in memory.

Run:  python probes/probe_p4.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.contract as contract


def main() -> int:
    print("=== does the coverage assertion see the gap? ===\n")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real = os.path.join(root, "data", "molecules", "library.json")
    with open(real, encoding="utf-8") as fh:
        text = fh.read()

    fail: list = []
    contract.library_sections(fail)
    hits = [f for f in fail if "reachable only by typing a SMILES" in f]
    print(f"  real library      : {len(fail)} failure(s), "
          f"{len(hits)} about coverage")
    if hits:
        print(f"    {hits[0][:100]}")
        print("    !! the assertion fires on the real library")

    # Now hide the phosphorus, the way it was before.
    import json
    lib = json.loads(text)
    for k in ("methylphosphine", "dimethylphosphine"):
        lib.pop(k, None)
    stripped = json.dumps(lib, indent=2, ensure_ascii=False)

    tmp = real + ".probe_p4"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(stripped)
        # Point the section at the doctored file by swapping the path it reads.
        # The suffix has to be computed *before* the patch: `c._os` and `os`
        # are the same module object, so a call to os.path.join inside the
        # replacement goes straight back into the replacement.  The first
        # version of this probe did exactly that and died of recursion.
        import backend.contract as c
        real_join = c._os.path.join
        suffix = real_join("molecules", "library.json")

        class _Patched:
            def __call__(self, *a):
                out = real_join(*a)
                return tmp if out.endswith(suffix) else out

        c._os.path.join = _Patched()
        try:
            fail2: list = []
            c.library_sections(fail2)
        finally:
            c._os.path.join = real_join
        hits2 = [f for f in fail2 if "reachable only by typing a SMILES" in f]
        print(f"\n  without phosphorus: {len(hits2)} coverage failure(s)")
        for h in hits2:
            print(f"    {h[:110]}")
        ok = bool(hits2) and not hits
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    print()
    print("  PASS  the assertion fires on the gap and not on the real library"
          if ok else
          "  FAIL  the assertion does not distinguish the two")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
