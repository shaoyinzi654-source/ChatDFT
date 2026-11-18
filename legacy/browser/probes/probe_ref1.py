"""Probe -- why does each unreferenced nucleus have no reference?

The README lists fourteen NMR-active elements with no reference compound
(Al, B, Br, Cl, D, Hg, I, Li, Na, Pb, Pt, S, Se, Sn) and gives no reason.
A list without a reason is the kind of thing that gets copied forward for
years, and the reasons here are not all the same: some elements may simply
not exist in 6-31G*, while others exist in the basis and are excluded because
the IUPAC primary reference for that nucleus is a hydrated ion (Cl-, Li+,
Na+, Al3+) that a gas-phase program has no business computing, or a molecule
too large to fit (BF3.OEt2 for 11B).

This probe asks PySCF the only question that is actually checkable here:
can this build put the element in a 6-31G* basis at all?  Whatever the answer
is, it goes in the README as the reason, so the list stops being folklore.

Changes nothing.

Run:  python probes/probe_ref1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto

UNREF = ("Al", "B", "Br", "Cl", "Hg", "I", "Li", "Na", "Pb", "Pt",
         "S", "Se", "Sn", "D")


def main() -> int:
    # `gto.basis.load` answers the basis question directly.  Building a bare
    # atom does not: the first version of this probe did that and reported
    # "NO BASIS" for Al, B, Br, Cl, Li, Na and D, all of which were really
    # "Electron number 13 and spin 0 are not consistent" -- a closed-shell
    # check firing before the basis was ever looked at.  Seven of the fourteen
    # reasons would have been written down wrong.
    print("=== is the element even available in 6-31G*? ===\n")
    have, lack = [], []
    for sym in UNREF:
        lookup = "H" if sym == "D" else sym
        try:
            shell = gto.basis.load("6-31g*", lookup)
            nfun = sum(1 for row in shell for _ in row[1:])
            have.append(sym)
            print(f"  {sym:3s}  in the basis  ({len(shell)} shells)")
        except Exception as exc:                          # noqa: BLE001
            lack.append(sym)
            print(f"  {sym:3s}  NOT in the basis  "
                  f"({type(exc).__name__}: {str(exc)[:52]})")
    print()
    print(f"  in the basis, so excluded for another reason: {have}")
    print(f"  not in 6-31G* at all:                        {lack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
