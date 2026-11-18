"""Probe -- why does mutation L20 report 'not applicable'?

The harness reports that the string it wants to replace is not in
backend/engine/elements.py, but grepping for that string finds it.  One of
those two statements is wrong, and a mutation harness that silently cannot
apply a mutation is worse than one that fails loudly: it reports "nothing was
caught" when the truth is "nothing was run" -- the same shape as the crash in
the NMR harness, one layer down.

This prints the two strings character by character until they differ.

Changes nothing.

Run:  python probes/probe_p5.py
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import mutate_library as M


def main() -> int:
    src = io.open(M.ELE, encoding="utf-8").read()
    print(f"=== L20 against {M.ELE} ({len(src)} chars) ===\n")
    for label, what, subs in M.CODE_MUTATIONS:
        if label != "L20":
            continue
        for old, new in subs:
            print(f"  old  = {old!r}")
            print(f"  new  = {new!r}")
            print(f"  `old in src` -> {old in src}")
            # The line it means, found by its distinctive part.
            target = None
            for i, line in enumerate(src.splitlines(), 1):
                if "frozenset" in line and '"H", "H"' in line:
                    target = (i, line)
            if target is None:
                print("  no line in the file matches 'frozenset' + H,H")
                continue
            i, line = target
            print(f"  line {i} = {line!r}")
            if line == old:
                print("  the line equals `old` exactly -- the replacement "
                      "should have applied")
            else:
                print(f"  they differ: file {len(line)} chars, "
                      f"pattern {len(old)} chars")
                for j, (a, b) in enumerate(zip(line, old)):
                    if a != b:
                        print(f"    first difference at index {j}: "
                              f"file {a!r} vs pattern {b!r}")
                        break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
