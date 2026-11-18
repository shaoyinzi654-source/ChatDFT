"""
ChatDFT - conversational density functional theory.

``backend.engine`` holds the physics (PySCF driver, geometry handling,
molecular input parsing). ``backend.agent`` holds the natural-language
layer that maps a chemist's question onto that physics.
"""

# Make the native libraries reachable before anything imports pyscf.
from .. import bootstrap

bootstrap.setup()

from . import dft, elements, molecule  # noqa: E402
from .dft import (  # noqa: F401,E402
    BASIS_SETS,
    BENCHMARKS,
    DFTEngine,
    DFTError,
    FUNCTIONALS,
    HARTREE2EV,
)

__all__ = [
    "elements",
    "molecule",
    "dft",
    "DFTEngine",
    "DFTError",
    "FUNCTIONALS",
    "BASIS_SETS",
    "BENCHMARKS",
    "HARTREE2EV",
]
