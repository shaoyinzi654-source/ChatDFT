# ChatDFT

ChatDFT is a Windows workbench for molecular modeling and computational chemistry. Build molecules and hydrated clusters, submit jobs to PySCF, and inspect or export the results.

[**Download Windows 5.8.0**](https://github.com/shaoyinzi654-source/ChatDFT/releases/download/v5.8.0-standalone/ChatDFT-Standalone-5.8.0-Windows.zip) · [Getting started](#getting-started) · [Gallery](docs/gallery.md) · [中文](README.md)

[![ChatDFT 5.8.0 showing a hydrated CO₂–monoethanolamine cluster, calculation results, and the job list.](docs/images/workbench-5.8.png)](docs/images/workbench-5.8.png)

*The hydrated CO₂–monoethanolamine cluster and its single-point calculation in ChatDFT 5.8.0.*

## Getting started

1. Install **64-bit Python 3.13** and extract the ChatDFT Windows package.
2. Run `.Install-ChatDFT.ps1` in PowerShell from the extracted directory, then open the desktop shortcut.
3. Choose **构建 → 从 SMILES 建模** (Build from SMILES) and enter `O` to view a water molecule. To run a calculation, first configure your PySCF server under **计算 → 连接与模型设置** (Connection and Model Settings).

[First calculation guide (Chinese)](docs/workflow.md#提交作业) · [Workbench manual](standalone/README.md)

## Features

- **Modeling** — Build from SMILES, XYZ, or nonperiodic XSD. Assemble molecules, ions, and explicit water. Edit coordinates, measure distances and angles, and compare structures side by side.
- **Quantum chemistry** — Submit PySCF single-point, geometry optimization, harmonic frequency, and vertical excitation jobs over SSH. Use gas-phase or IEF-PCM solvent settings.
- **Jobs and results** — Follow queue status, progress, and logs. Reopen archived structures and results. Save XYZ, JSON, CSV, PNG, SVG, and Markdown per project.
- **Automation** — Use local Codex or a compatible API to plan structures and jobs. The assistant can ask for missing inputs and prepare additional figures or reports from completed calculations.
- **Papers and periodic systems** — Extract modeling and calculation conditions from papers. The [GROMACS guide](standalone/PDC-MD-GUIDE.md) covers building and analyzing periodic PDC solutions.

## Gallery

[![SCF electronic energy, Mulliken atomic charges, and frontier orbital energies for the CO₂–monoethanolamine cluster.](docs/images/calculation-output.png)](docs/images/calculation-output.png)

*Three plots from the same job. The energy trace shows convergence over SCF iterations.*

[View all 7 images](docs/gallery.md) · [Output files (Chinese)](docs/workflow.md#查看输出)

## Project links

[Source](standalone/) · [Releases](https://github.com/shaoyinzi654-source/ChatDFT/releases) · [Issues](https://github.com/shaoyinzi654-source/ChatDFT/issues) · [Website](https://chatdft.com/) · [License](LICENSE) · [Third-party notices](standalone/OPEN_SOURCE.md) · [Browser archive](legacy/browser/)
