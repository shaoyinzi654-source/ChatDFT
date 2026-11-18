# ChatDFT

**A Windows workbench for molecular modeling, remote PySCF calculations, and reproducible analysis.**

[Download Windows 5.8.0](https://github.com/shaoyinzi654-source/ChatDFT/releases/download/v5.8.0-standalone/ChatDFT-Standalone-5.8.0-Windows.zip) · [Releases](https://github.com/shaoyinzi654-source/ChatDFT/releases) · [中文](README.md)

[![ChatDFT 5.8.0 showing a completed PySCF job for a hydrated CO₂–monoethanolamine cluster](docs/images/workbench-5.8.png)](docs/images/workbench-5.8.png)

Use SMILES, XYZ, nonperiodic XSD, or an AI prompt to build a 3D molecule or cluster. Submit a single-point energy, geometry optimization, frequency, or vertical excitation job to your PySCF server over SSH. Job history restores the matching structure and outputs. Automatic mode packages data, figures, and reports from the calculation results.

## Get started

1. Extract the Windows ZIP and install 64-bit Python 3.13.
2. In the extracted directory, run `.\Install-ChatDFT.ps1` in PowerShell. Open the ChatDFT desktop shortcut.
3. Select **构建 → 从 SMILES 建模** (Build from SMILES) and enter `O` to create a water molecule. For remote calculations, configure your SSH host, username, trusted server fingerprint, and PySCF Python path under **计算 → 连接与模型设置** (Connection and model settings).

AI-assisted features can use a signed-in local Codex CLI or a compatible API. The Windows app handles modeling and analysis. PySCF jobs run on the configured Linux server. The optional PDC molecular dynamics workflow uses GROMACS and requires the original research parameters. [PDC guide](standalone/PDC-MD-GUIDE.md)

The [standalone application](standalone/) is the current edition. See the [workflow guide](docs/workflow.md) for examples or the [browser archive](legacy/browser/) for the earlier edition. Application code is distributed under the terms in [LICENSE](LICENSE). Third-party components and research parameters have their own terms. [Third-party notices](standalone/OPEN_SOURCE.md)
