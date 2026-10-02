# probes/ — the audit's measuring instruments

These are the scripts that were used to *measure* the claims the code makes,
rather than to assert them. `backend/contract.py` asserts; a probe asks
"is this actually true?" and prints what it found, so that an assertion can
be written from evidence instead of from expectation.

They are kept in the repository on purpose. `contract.py`, `nmr.py` and
`planner.py` cite them **by name** in comments — `probe_mem1`, `probe_nmr28`,
`probe_raman3` and others are the provenance of specific constants. A number
whose measurement is not in the tree is a number nobody can re-check.

## Running one

From the project root:

```
python probes/probe_ref2.py
```

Each probe inserts the project root into `sys.path` itself and resolves
`data/` against the project root, so it works from any working directory and
does not need `PYTHONPATH`. The command in each docstring was rewritten when
this directory was created — the originals named a path one level up, and one
of them misspelled the interpreter directory, so they were re-derived from
where the files actually are rather than copied.

## What to expect

* **Most are read-only.** They say so in their docstring ("Changes nothing").
  A few deliberately rewrite a source file to show that an assertion fires;
  those restore it byte-for-byte and are listed in the round reports.
* **Several cost real SCF time.** Anything that builds a molecule runs
  PySCF. The cheap ones are the ones that only read `library.json`, the
  cached references in `data/nmr_cache/`, or a report under `reports/`.
* **Their output is the evidence, not a verdict.** A probe that prints a
  number has not judged it; the judgement lives in the gate that consumes it.

## Naming

The `probe_` prefix is redundant inside this directory and is kept anyway:
the comments that cite these files say `probe_mem1`, and a citation that
does not resolve is worse than a long filename.

## Not here

`.tmp_probe/` (gitignored) holds throwaway scratch — logs, downloaded source
tarballs, one-off scripts that were superseded within the same round. If a
probe is here, it is because a claim in the shipped code depends on it.
