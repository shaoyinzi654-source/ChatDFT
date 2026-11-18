"""Probe -- run a real 31P job, the first this build has ever done.

The library had no phosphorus, so 31P had never been exercised end to end:
the reference layer was checked by the contract in isolation, and nothing had
ever taken a P-containing molecule through the shieldings, the reference
subtraction, the grouping and the spectrum.  Two library entries now exist.

What to look for, in order of how much it would hurt to get wrong:

  * a 31P shift exists at all, and is referenced to PH3;
  * the 31P scale note reaches the payload, because 31P is quoted against PH3
    rather than the IUPAC primary 85% H3PO4 and this build does not compute
    that offset;
  * the shift is not zero.  Methylphosphine is not the reference compound, so
    a zero here would mean the reference lookup matched the wrong molecule --
    the failure mode the element-count lookup exists to avoid;
  * the two rungs differ.  Successive methylation is a real 31P trend, and if
    the two molecules come out identical something is very wrong.

Changes nothing.

Run:  python probes/probe_p2.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engine import molecule as molmod
from backend.engine import nmr


def run(key: str, smiles: str) -> dict:
    m = molmod.from_smiles(smiles, name=key)
    atoms = [(a.symbol, a.x, a.y, a.z) for a in m.atoms]
    t0 = time.time()
    res = nmr.compute(nmr._xyz(atoms), basis="6-31g*", charge=int(m.charge),
                      multiplicity=int(m.multiplicity))
    return res, time.time() - t0


def main() -> int:
    print("=== 31P, end to end, for the first time ===\n")
    got = {}
    for key, smiles in (("methylphosphine", "CP"),
                        ("dimethylphosphine", "CPC")):
        res, dt = run(key, smiles)
        got[key] = res
        print(f"--- {key}  ({res['nao']} basis functions, {dt:.1f} s) ---")
        print(f"    active elements      {res['active_elements']}")
        print(f"    unreferenced         {res['unreferenced_elements']}")
        ref = (res.get("references") or {}).get("P") or {}
        print(f"    P reference          {ref.get('label')} "
              f"{ref.get('sigma_iso_ppm')} ppm")
        print(f"    P scale note         "
              f"{(ref.get('scale_note') or 'None')[:96]}")
        p31 = [n for n in res["nuclei"] if n["symbol"] == "P"]
        for n in p31:
            print(f"    delta(31P)           {n['delta_ppm']:.3f} ppm   "
                  f"note field present: {'reference_scale_note' in n}")
        print(f"    spectra              {sorted(res['spectra'])}")
        print()

    a = [n for n in got["methylphosphine"]["nuclei"] if n["symbol"] == "P"]
    b = [n for n in got["dimethylphosphine"]["nuclei"] if n["symbol"] == "P"]
    print("=== verdicts ===")
    ok = True
    if not a or not b:
        print("  FAIL  no 31P nucleus in the payload at all"); ok = False
    else:
        da, db = a[0]["delta_ppm"], b[0]["delta_ppm"]
        print(f"  PH2Me {da:+.3f} ppm   PHMe2 {db:+.3f} ppm")
        if abs(da) < 1e-6 or abs(db) < 1e-6:
            print("  FAIL  a shift is exactly zero -- the reference lookup "
                  "matched the sample"); ok = False
        else:
            print("  ok    neither is zero, so neither was taken as its own "
                  "reference")
        if abs(da - db) < 0.5:
            print(f"  FAIL  the two rungs differ by only {abs(da - db):.3f} "
                  "ppm; methylation should move 31P"); ok = False
        else:
            print(f"  ok    the two rungs differ by {abs(da - db):.3f} ppm, "
                  "so the series is not a flat line")
        note = (got["methylphosphine"].get("references") or {}).get("P", {})
        if not note.get("scale_note"):
            print("  FAIL  the 31P scale note did not reach the payload")
            ok = False
        else:
            print("  ok    the 31P scale note reached the payload")
    print()
    print(json.dumps({"methylphosphine_31P":
                      (a[0]["delta_ppm"] if a else None),
                      "dimethylphosphine_31P":
                      (b[0]["delta_ppm"] if b else None)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
