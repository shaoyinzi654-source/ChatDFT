# Third-party notices

| Project | Use in ChatDFT |
| --- | --- |
| [Qt Advanced Docking System](https://github.com/githubuser0xFFFF/Qt-Advanced-Docking-System) and [PySide6-QtAds](https://github.com/mborgerson/pyside6_qtads) | Docking, floating windows, tabs, and layout serialization. LGPL-2.1. |
| [QtAwesome](https://github.com/spyder-ide/qtawesome) | Workbench toolbar icons. QtAwesome uses MIT terms. The bundled Font Awesome 5 Free font has separate terms. |
| [PySide6](https://doc.qt.io/qtforpython-6/) | Python bindings for the Qt desktop interface. See the corresponding installed package for license information. |
| [GROMACS](https://gitlab.com/gromacs/gromacs) | Runs the optional periodic molecular dynamics workflow on the server. The Windows package contains the client and scripts. |
| [MDTraj](https://github.com/mdtraj/mdtraj) and [SciPy](https://github.com/scipy/scipy) | Read trajectories and analyze connected components in the molecular dynamics workflow. |

ChatDFT loads the desktop components through Python packages. The [third-party license directory](third_party/) contains the included notices for docking and toolbar components. Supported dependency versions are listed in [requirements.txt](requirements.txt).

Avogadro, ParaView, and FreeCAD served as references for the workbench layout and interaction. PDC force-field inputs and original publication materials come from the authors and retain their original terms. [PDC molecular dynamics workflow](PDC-MD-GUIDE.md)
