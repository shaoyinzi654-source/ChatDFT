"""Probe -- does the new startup guard actually catch the L20 defect?

An assertion nobody has seen fail is a guess.  This puts MUTATION_FILE back to
the key it had ("L20", while the loop looks up the full label), calls the
guard, and checks that it refuses -- and separately checks that a mutation
whose pattern is genuinely absent from the file it names is also caught, since
that is the other half of the same failure.

Nothing on disk changes; the dict is patched in memory and restored.

Run:  python probes/probe_p6.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import mutate_library as M


def main() -> int:
    print("=== does the guard catch the L20 defect? ===\n")
    ok = True

    real = dict(M.MUTATION_FILE)
    try:
        M.MUTATION_FILE.clear()
        M.MUTATION_FILE["L20"] = M.ELE          # the key as it used to be
        hits = M._check_mutation_files()
    finally:
        M.MUTATION_FILE.clear()
        M.MUTATION_FILE.update(real)

    print(f"  with the old key      : {len(hits)} problem(s)")
    for h in hits[:3]:
        print(f"    {h[:110]}")
    caught = any("not a mutation label" in h for h in hits)
    print(f"  {'ok  ' if caught else 'FAIL'} the short key is rejected")
    ok &= caught

    # And the other half: a pattern that is not in the file it names.
    old_subs = M.CODE_MUTATIONS[0][2]
    patched = list(M.CODE_MUTATIONS)
    patched[0] = (patched[0][0], patched[0][1],
                  [("this string is nowhere in the file", "x")])
    saved = M.CODE_MUTATIONS[:]
    try:
        M.CODE_MUTATIONS[:] = patched
        hits2 = M._check_mutation_files()
    finally:
        M.CODE_MUTATIONS[:] = saved

    print(f"\n  with a missing pattern: {len(hits2)} problem(s)")
    for h in hits2[:3]:
        print(f"    {h[:110]}")
    caught2 = any("cannot run" in h for h in hits2)
    print(f"  {'ok  ' if caught2 else 'FAIL'} the missing pattern is rejected")
    ok &= caught2

    print()
    print("  PASS  both halves of the failure are caught up front"
          if ok else "  FAIL  the guard does not catch it")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
