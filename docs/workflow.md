# Automated calculation workflow

Connect ChatDFT to a PySCF server over SSH and configure Codex or a compatible API. Open the AI menu and select automated calculation mode. The workbench builds the requested model, checks the inputs, submits the calculation, and keeps the returned outputs with the job.

## Submit a job

1. Enter a prompt with the composition, geometry constraints, total charge, spin multiplicity, and calculation settings.
2. Supply any requested clarification before submission. Follow the job status in the workbench after confirming the server connection.
3. On completion, reopen the job from the project library to load the initial or final geometry, open its folder, or generate supplemental analysis.

```text
Build one water molecule from SMILES O. Use charge 0 and spin multiplicity 1.
Run a gas-phase HF/STO-3G single-point calculation in PySCF.
Save the final electronic energy, dipole moment, atomic charges, geometry figure,
and calculation report.
```

For a geometry-optimization example, use a urea dimer with explicit water.

```text
Arrange two urea molecules followed by four waters. Use SMILES NC(N)=O for
urea and O for water. Place the first nitrogen of the second urea 3.05 Å from
the carbonyl oxygen of the first. Set the first two water oxygens 2.90 Å from
the first carbonyl oxygen and the other two 2.90 Å from the second one.
Use charge 0 and multiplicity 1. Optimize with gas-phase B3LYP/6-31G(d),
with no additional dispersion correction. Report convergence, RMSD, and the
initial and final contact distances.
```

## Output files

| Output | Use |
| --- | --- |
| Initial and final structures | Reopen the geometry in ChatDFT or exchange XYZ coordinates. |
| Solver input, logs, and JSON | Audit settings, convergence, and raw output. |
| CSV, PNG, and SVG | Inspect the underlying values and export figures for the available results. |
| Calculation report and supplemental AI analysis | Check the source data, inputs, geometry, and interpretation. Supplemental analysis is saved separately. |

[Workbench manual](../standalone/README.md) · [Gallery](gallery.md)
