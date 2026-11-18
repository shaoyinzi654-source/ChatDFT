"""One preflight, shared by every mutation harness in this package.

A mutation is a ``(old, new)`` substitution applied to a source file, and the
assertion it is supposed to break is only *tested* if ``old`` is still in the
file.  When it is not, the harness reports "mutation not applicable" -- which
reads exactly like "the assertion was not exercised", i.e. like coverage that
is merely incomplete, when the truth is that the mutation is dead and the
assertion has had nothing pointed at it for however long the target text has
been gone.

That has happened twice in this project, both times for the same reason: a
mutation that quotes the code it is trying to break stops matching the moment
that code changes.  N37 quoted the ledger's entry fields and stopped matching
when the entries were restructured; N34 quoted the 13C tolerance of 2.5 ppm and
stopped matching when the tolerance was corrected to 3.2.  Both times the
harness ran its baseline and walked the whole list before mentioning it, which
is minutes of SCF for an answer that is available in milliseconds.

So: check every substitution against the source *first*.  This module walks
exactly the application the harnesses perform -- ``text.replace(old, new, 1)``,
in order, on one text -- so a mutation that passes the preflight cannot later
report "not applicable".
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence


def substitutions(entry: Sequence) -> List[tuple]:
    """The ``(old, new)`` pairs of a mutation entry, whatever its arity.

    The harnesses grew independently and their entries are 3, 4 or 5 wide:
    ``(label, what, subs)``, ``(label, what, subs, needs_payload)`` and
    ``(label, what, path, subs, needs_payload)``.  The pairs are always the
    only element that is a list of 2-tuples, so finding them by shape is more
    robust than by position -- a harness that grows another field does not
    silently start checking the wrong one.
    """
    for item in entry:
        if (isinstance(item, list) and item
                and all(isinstance(p, tuple) and len(p) == 2 for p in item)):
            return list(item)
    raise ValueError(f"mutation entry has no substitution list: {entry[0]!r}")


def apply_to(text: str, subs: Iterable[tuple]):
    """Apply the substitutions the way the harnesses do.

    Returns ``(text, missing)``: ``missing`` is the first ``old`` that was not
    found, or ``""`` when every one applied.  Stops at the first miss, because
    the remaining substitutions were never reached in a real run either.
    """
    for old, new in subs:
        if old not in text:
            return text, old
        text = text.replace(old, new, 1)
    return text, ""


def dead_mutations(mutations, source_of: Callable, path_of: Callable) -> List[str]:
    """Every mutation that no longer has anything to break.

    ``source_of(entry)`` returns the text that entry's substitutions are
    applied to; ``path_of(entry)`` returns the file it lives in, which is what
    a syntax error has to be reported against.
    """
    dead: List[str] = []
    for entry in mutations:
        label = entry[0]
        text, missing = apply_to(source_of(entry), substitutions(entry))
        if missing:
            dead.append(f"{label}: {missing[:70]!r} is not in the source "
                        "any more")
            continue
        try:
            compile(text, path_of(entry), "exec")
        except SyntaxError as exc:
            dead.append(f"{label}: the mutated source does not parse ({exc})")
    return dead


_SELFTEST_SOURCE = "x = 1\n"


def _selftest() -> List[str]:
    """Prove the preflight can fire, on synthetic mutations, on every run.

    A preflight that returns an empty list looks exactly like a preflight that
    works, and the failure mode it exists to catch -- a mutation whose target
    text has moved -- is invisible from the outside.  So it is fed three
    mutations whose answers are known: one that applies, one whose target is
    gone, and one that applies but does not parse.  Without this, a
    ``dead_mutations`` that always returned ``[]`` would be indistinguishable
    from a correct one, and every harness would print "all mutations apply" for
    ever.
    """
    problems: List[str] = []
    source_of = lambda _e: _SELFTEST_SOURCE            # noqa: E731
    path_of = lambda _e: "<selftest>"                  # noqa: E731
    good = ("GOOD", "applies", [("x = 1", "x = 2")], False)
    gone = ("GONE", "target removed", [("y = 1", "y = 2")], False)
    broken = ("BROKEN", "does not parse", [("x = 1", "x = (")], False)

    if dead_mutations([good], source_of, path_of):
        problems.append("the preflight rejected a mutation that does apply")
    if not dead_mutations([gone], source_of, path_of):
        problems.append(
            "the preflight passed a mutation whose target text is gone, so a "
            "dead mutation would be reported as coverage")
    if not dead_mutations([broken], source_of, path_of):
        problems.append(
            "the preflight passed a mutation that does not parse, so a "
            "mutation that would abort the run mid-list is not caught")
    return problems


def report(name: str, mutations, dead: List[str]) -> int:
    """Print the preflight verdict and return an exit code for the caller.

    0 means every mutation still has a target.  1 means the harness must not
    run: its list no longer describes what it claims to test, and a run would
    spend its SCF budget proving it.
    """
    total = len(mutations)
    self_test = _selftest()
    if self_test:
        print("=" * 78)
        print(f"PREFLIGHT ({name}): the preflight itself is broken")
        print("=" * 78)
        for s in self_test:
            print("  -", s)
        print()
        print("  Refusing to report on the mutation list until the check that "
              "reports on it works.")
        return 1
    if dead:
        print("=" * 78)
        print(f"PREFLIGHT ({name}): {len(dead)} of {total} mutations have no "
              "target left")
        print("=" * 78)
        for d in dead:
            print("  -", d)
        print()
        print("  Each of these would have been reported as 'mutation not "
              "applicable' at the end of the run,")
        print("  which reads as 'the assertion was not exercised' rather than "
              "'this mutation is dead'.")
        return 1
    print(f"PREFLIGHT ({name}): self-test fired on both a missing target and a "
          f"syntax error; all {total} mutations apply to the current source "
          "and parse")
    return 0
