# ChatDFT Standalone Workbench 5.8.0

ChatDFT is a Windows application for molecular modeling, remote PySCF jobs, and result review. It supports nonperiodic XYZ and XSD input, SMILES-driven model building, structure comparison, and local job archives.

## Installation

Install 64-bit Python 3.13. Download the complete Windows ZIP from Releases, extract it, and run the installer from the extracted folder.

```powershell
.\Install-ChatDFT.ps1
```

The installer prepares the Python environment, configures the native launcher, and adds desktop and Start menu shortcuts. Pass `-Python` to select an existing compatible environment. If you move the installation, run the installer again. Projects, results, and settings are stored under `%LOCALAPPDATA%\ChatDFT-Studio`.

## Model and compare structures

- Import XYZ or nonperiodic XSD, or use SMILES to construct a starting conformer with RDKit.
- Inspect and edit atom positions in ångströms, add or remove atoms, and specify charge and spin multiplicity.
- Switch among ball-and-stick, sticks, wireframe, and space-filling views. Drag to rotate, use the wheel to zoom, and use the middle button or Shift while dragging to pan.
- Measure distances, angles, and dihedrals. Pin a separate reference structure for comparison, or let it follow the active model. To compute the reference structure, make it the primary model first.
- Save the model and calculation settings in a `.chatdft.json` project. Export coordinates as XYZ or save a PNG of the structure view.

Layouts include a single model, two structure views, a four-panel analysis workspace, and tabbed documents. Panels can be docked, grouped, or floated. A selected historical job may have a different structure from the active model. Check the title of each document for its source.

## PySCF server and jobs

In the calculation menu, open connection and model settings. Enter the SSH host, port, username, trusted SHA256 host fingerprint, and work directory. Provide a password for the session or configure a private key. The remote environment needs PySCF, and the client needs the path to that environment's Python executable.

Choose a method, basis, task, solvent, charge, and multiplicity. Supported tasks include single-point energy, geometry optimization, harmonic frequencies, optimization plus frequencies, and vertical excitations. Gas phase and IEF-PCM are available for compatible tasks. MP2 is limited to gas-phase single points. Import external DOS, band, and NEB curves from CSV when you have results from those calculations.

The job monitor shows status and available values. Open server job management to inspect queues, CPU, memory, disk use, and logs. You can cancel jobs, retry failed molecular jobs with their saved inputs, and download completed output. The project library can reopen the initial and final structures, job folder, and saved reports.

## AI modeling and automated runs

Choose local Codex or a compatible API in model settings. The connection test sends a real model request. You can request a structure, geometry edit, or calculation setup separately, or use automated calculation mode to build and validate a model before submitting a job. If a parameter is ambiguous, answer the assistant's follow-up in the same session.

For a multi-fragment request, specify components, quantities, formal charges, overall charge, spin, and any placement constraints. The local assembly checks atom count, overlap, charge, and spin compatibility. It can prepare up to 48 fragments and 500 atoms including hydrogen. Specified distances describe the starting geometry. Optimization may subsequently change them.

Completed automated jobs retain the input, returned data, base figures, and report. Supplemental AI analysis uses available job data, saves a separate version, and can notify you when it finishes. [Automated calculation workflow](../docs/workflow.md)

## Paper workflow

Open the paper workflow in the AI menu. You can import PDF, TXT, or Markdown, or paste text from a paper and its supplement. PDF extraction uses `pdftotext` on PATH. Scanned documents require OCR first.

Check the extracted method, basis, solvent, charge, spin, and page references against the paper. Load the authors' coordinates or the supplied structure file where available. An approximate structure built from SMILES can serve as a starting point if original coordinates are unavailable. The molecular workflow checks the chosen settings before submitting to PySCF. Where the original energy is available and directly comparable, the application can record an explicit comparison.

For periodic PDC solutions use the separate GROMACS workflow for prepared boxes, trajectory analysis, and aggregation statistics. [PDC molecular dynamics](PDC-MD-GUIDE.md)

[Repository home](../README.md) · [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.en.md)
