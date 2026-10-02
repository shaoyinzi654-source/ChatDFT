"""
GIAO nuclear magnetic shielding, written from the definitions.

Why this file exists
--------------------
``pyscf.prop`` is not in this build.  That is not a packaging accident and it
is not fixable by installing something: the module was split into a separate
repository, and neither the 2.6.2 nor the 2.5.0 source distribution on PyPI
contains a single ``pyscf/prop`` path (checked by downloading both sdists and
listing them).  So every property in ``prop`` -- NMR, Raman, the lot -- has to
be written here from its definition.  ``analysis.py`` already did that for
Raman; this module does it for the shielding tensor.

The method
----------
A numerical GIAO (gauge-including atomic orbital) shielding tensor.  The AO
basis carries the field-dependent phase

    chi_mu(B) = exp(-i/2 (B x R_mu) . r) chi_mu

so every one-electron integral becomes field-dependent, and the second
derivative of the energy with respect to the field B_t and a nuclear magnetic
moment mu_s is the shielding tensor:

    sigma_ts = d2E / dB_t dmu_s

The Hamiltonian is expanded to the orders that derivative needs:

    h(B,mu) = h0 - i B_t O^t - i mu_s P^s + B_t mu_s Q^{ts}
    S(B)    = S0 - i B_t S^t
    eri(B)  = eri0 - i B_t G^t

with **every operator real** and the ``-i`` written explicitly.  That
convention is not cosmetic and getting it wrong is silent:

    O^t = 1/2 int1e_giao_irjxp + int1e_ignuc + int1e_igkin
    S^t = int1e_igovlp
    G^t = int2e_ig1 + int2e_ig1(pair-swapped)
    P^s = int1e_ia01p          (origin moved to R_A)
    Q^{ts} = int1e_giao_a11part - delta_ts Tr(int1e_giao_a11part)
             + int1e_a01gp

The consistency test that catches a mixed convention is sharp and free: a
closed shell has **no linear Zeeman term**, so dE/dB must be exactly zero in
all three directions.  Measured with this formulation it is 0.0e+00; with the
field folded in as real and the moment kept imaginary it was -8.7e-2 along
B_x only.  ``dE/dB`` is therefore computed and returned on every run as
``dE_dB``, and a non-zero value is reported as a defect rather than ignored.

Scale
-----
``sigma[ppm] = sigma[atomic units] * alpha^2 * 1e6`` with
``alpha = 1/137.035999084``, i.e. a factor of 53.251355.  No sign flip is
needed; an apparent need for one was an artifact of the mixed convention
above.  Both facts were measured, not assumed.

Level of theory -- read this before quoting a number
----------------------------------------------------
**The shielding is computed at the Hartree-Fock level, always.**  The
requested DFT functional is not used for it, and the payload says so.

This is a hard limit of this build, not a choice.  The numerical GIAO route
needs a complex density, and ``pyscf.dft.numint`` rejects one outright:

    UFuncTypeError: Cannot cast ufunc 'add' output from dtype('complex128')
                    to dtype('float64') with casting rule 'same_kind'

``scf.RHF`` with a complex-symmetric perturbation works and reproduces
experiment; ``dft.RKS`` raises.  A real-arithmetic substitute (put the field
in as real, split the response with a CPHF-style sign) was tried and fails on
both counts: the field SCF does not converge, and the density response comes
out ~0, giving 1H 28.6 / 17O 392 where the validated answers are 31.6 / 345.0.

Accuracy at RHF/6-31G* against gas-phase experiment, measured (probe_nmr20,
re-measured against the current molecule library by probe_se10, which is the
run ``backend.check_shift_scales`` pins):

absolute shieldings (at the geometry the product ships for each molecule;
see "the reference geometry" below, because 17O cares a great deal)
    NH3  14N   262.48  vs 264.0   (-1.52)
    H2O  17O   323.87  vs 344.0   (-20.13)
    NH3  1H     32.78  vs 30.8    (+1.98)
    CH4  13C   200.14  vs 195.0   (+5.14)
    H2O  1H     31.43  vs 30.7    (+0.73)
    TMS  13C   199.18  vs 188.1

chemical shifts vs TMS, delta = sigma(TMS) - sigma(molecule)
    molecule   nuc  probe_nmr20  probe_se10   drift   exptl    error
    acetylene  1H       1.833       1.997    +0.164    1.80   +0.197
    methane    1H       0.442       0.655    +0.213    0.23   +0.425
    ethane     1H       1.030       1.130    +0.100    0.86   +0.270
    ethylene   1H       5.732       5.737    +0.005    5.40   +0.337
    benzene    1H       7.627       7.666    +0.039    7.26   +0.406
    methane    13C     -1.487      -0.205    +1.282   -2.30   +2.095
    ethane     13C      7.597       8.328    +0.731    5.70   +2.628
    benzene    13C    126.309     126.408    +0.099  128.50   -2.092
    ethylene   13C    121.091     120.564    -0.527  123.50   -2.936
    acetylene  13C     65.785      66.175    +0.390   71.90   -5.725
    methanol   1H       0.340       2.668    +2.328    3.35   -0.682
    methanol   13C     29.488      46.457   +16.969   49.50   -3.043

**This table is a record of what happened, not a claim about the product.**
It used to carry a headline count of how many of its rows were inside a band,
and the count changed without anything noticing -- because a count written in
prose is a count that can contradict the list it counts.  The largest drift is
methanol's carbon: 29.488 ppm when the ledger was written, 46.457 now, because
the library's methanol geometry was re-derived in round 17 (all 62 entries were
stamped "MMFF94 (RDKit ETKDGv3)" while re-embedding them from their own SMILES
reproduced only 25).  A claim about the product's accuracy had been describing
a product that no longer existed, and nothing said so.

The twelve rows are therefore data now (``SHIFT_LEDGER`` below), and
``backend.check_shift_scales`` recomputes every one of them through
``scale_geometry`` and the same shielding routine a real job uses.  The
probe_se10 column above was re-measured against the current library and
reproduces exactly, which is the only reason it is still here.

**The three quantities that describe the set are not written down anywhere.**
How many rows are inside the band, the worst error per nucleus, and the mean
absolute error are all *derived* from the twelve rows, so writing them here
would put a number in the one place nothing recomputes -- which is exactly what
had already happened: this paragraph used to give the mean as 1.89 ppm, and
re-measured it is 1.736.  A count in prose is a count that can contradict the
list it counts.  The gate prints all three.

What survives, and is now checked rather than asserted: the *ordering* is right
in every row, the 1H tolerance holds across all six of its rows, and 13C holds
across five of six with acetylene's triple bond named as the exception
(``SHIFT_LEDGER["C"]["exceptions"]``) instead of being a sentence in a
docstring.

Two things that look like accuracy claims and are not:

* **The 17O absolute shielding is 14 ppm from experiment.**  329.60 against
  344.0.  An earlier round recorded 345.046 here and reported a 1.05 ppm
  agreement; that number came from an under-converged SCF at a small field
  step and is an artifact.  See the note on CONV_TOL.  Absolute 17O shieldings
  are strongly basis-set dependent and 6-31G* is simply not a basis to quote
  one from -- but the *shifts* are differences against a reference computed the
  same way, so the systematic part largely cancels.
* **TMS's own 13C is 199.18 against a literature 188.1.**  That gap is the
  basis and the missing correlation, and it is why a shift is quoted against a
  computed reference rather than a tabulated one: a tabulated reference would
  have put an 11 ppm constant on every 13C shift in the output, invisibly.

The reference geometry
----------------------

17O shielding moves by **539 ppm per angstrom of O-H**.  Measured (probe_nmr28,
RHF/6-31G*) on three waters:

    O-H 0.96857 A, H-O-H 104.00 deg   (the molecule library)   323.866 ppm
    O-H 0.95792 A, H-O-H 104.42 deg                            329.604 ppm
    O-H 0.95720 A, H-O-H 104.52 deg   (this module's own r_e)  329.978 ppm

So a reference computed at a geometry the sample does not have is not a
rounding error.  It was one: this module carried its own experimental water
while the molecule library ships a different one, and the two are 6.11 ppm
apart -- which put 6.11 ppm on every 17O shift in the output and made water's
own 17O shift 6.11 ppm instead of 0.00.

The rule now is that **where the molecule library ships the reference
compound, the reference uses the library geometry**, so computing that
molecule gives a shift of exactly zero; where it does not (TMS, PH3, CFCl3) a
documented experimental geometry is used and recorded.  The lookup is by
element count, never by name, for the reason that keeps 19F honest: the
library's chloroform is CHCl3 and the 19F reference is CFCl3.

Because the geometry is an input to the number, it is also part of the cache
key.  The key used to carry the basis, the tolerance and the field step but
not the geometry, so changing one would have reused the old shieldings.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dft import DFTError, _split_xyz

try:                                                # pragma: no cover
    from pyscf import gto, scf
except Exception as exc:                            # pragma: no cover
    gto = None
    scf = None
    _PYSCF_ERROR = exc
else:
    _PYSCF_ERROR = None

Progress = Optional[Callable[[int, str], None]]

# The fine-structure constant CODATA 2018.  sigma_ppm = sigma_au * PPM.
ALPHA = 1.0 / 137.035999084
PPM = ALPHA ** 2 * 1e6

# Finite-difference step for the field, in atomic units.  At conv_tol = 1e-13
# the whole window 3e-3 .. 3e-5 is flat (water 17O 329.6020 .. 329.7286, CH4
# 13C 200.1423 .. 200.1458), so any value in it is equally valid; 3e-4 sits in
# the middle with a decade of margin on each side.  See the warning on
# CONV_TOL: this window only exists at the tighter tolerance.
FIELD_STEP = 3.0e-4

# The perturbed SCF has to converge in its imaginary part too, and that
# converges more slowly than the density -- slowly enough that conv_tol
# changes the ANSWER, not just the noise.
#
# Measured on water at dB = 1e-4 (probe_nmr16): 1e-9 and 1e-10 both give
# 31.6561 for 1H; 1e-11 and 1e-12 give 31.8650; 1e-13 gives 31.8662.  So
# 1e-9/1e-10 are simply not converged.
#
# Worse, at conv_tol = 1e-11 the dB scan has a cliff in it (probe_nmr19):
#
#     dB        iso(17O) @1e-11    iso(17O) @1e-13
#     1e-04         329.7398           329.6280
#     6e-05         332.8352           329.6651
#     4e-05         345.0462           329.6656
#     3e-05         345.0462           329.6316
#     1e-05         345.0462           331.6819
#
# Three decades of bit-identical 345.0462 is not convergence, it is the SCF
# stopping at the wrong point because the perturbation fell below the
# tolerance it was asked to converge to.  345.046 is the value earlier rounds
# recorded as H2O 17O; it is an artifact, and the converged value is 329.63.
# Every 17O chemical shift is delta = sigma(H2O) - sigma(molecule), so that
# would have been a 15 ppm error on every 17O shift the product prints.
#
# 1e-13 removes the cliff (probe_nmr20: flat to 0.13 ppm from 3e-3 to 3e-5).
# It is part of the reference cache key for the same reason: a reference
# computed at a looser tolerance silently offsets every chemical shift.
CONV_TOL = 1.0e-13
MAX_CYCLE = 300

# Nuclei a chemist asks for.  ``gamma`` is the gyromagnetic ratio in MHz/T
# (the Larmor frequency at 1 tesla), which is what turns a ppm axis into a
# Hz axis and therefore what sets the linewidth in the spectrum.  Spin > 1/2
# means a quadrupole moment: the shielding is still well defined and is what
# the literature quotes, but the experimental line is usually broad, and the
# payload says so per nucleus instead of pretending otherwise.
NUCLEI: Dict[str, dict] = {
    "H":  {"isotope": "1H",  "spin": 0.5, "abundance": 0.999885, "gamma": 42.5775},
    "D":  {"isotope": "2H",  "spin": 1.0, "abundance": 0.000115, "gamma": 6.5359},
    "Li": {"isotope": "7Li", "spin": 1.5, "abundance": 0.9241,   "gamma": 16.546},
    "B":  {"isotope": "11B", "spin": 1.5, "abundance": 0.801,    "gamma": 13.660},
    "C":  {"isotope": "13C", "spin": 0.5, "abundance": 0.0107,   "gamma": 10.7084},
    "N":  {"isotope": "14N", "spin": 1.0, "abundance": 0.99636,  "gamma": 3.0777},
    "O":  {"isotope": "17O", "spin": 2.5, "abundance": 0.00038,  "gamma": -5.7720},
    "F":  {"isotope": "19F", "spin": 0.5, "abundance": 1.0,      "gamma": 40.0776},
    "Na": {"isotope": "23Na", "spin": 1.5, "abundance": 1.0,     "gamma": 11.268},
    "Al": {"isotope": "27Al", "spin": 2.5, "abundance": 1.0,     "gamma": 11.103},
    "Si": {"isotope": "29Si", "spin": 0.5, "abundance": 0.04683, "gamma": -8.4650},
    "P":  {"isotope": "31P", "spin": 0.5, "abundance": 1.0,      "gamma": 17.2383},
    "S":  {"isotope": "33S", "spin": 1.5, "abundance": 0.0075,   "gamma": 3.2665},
    "Cl": {"isotope": "35Cl", "spin": 1.5, "abundance": 0.7576,  "gamma": 4.1720},
    "Se": {"isotope": "77Se", "spin": 0.5, "abundance": 0.0763,  "gamma": 8.1530},
    "Br": {"isotope": "81Br", "spin": 1.5, "abundance": 0.4931,  "gamma": 11.487},
    "Sn": {"isotope": "119Sn", "spin": 0.5, "abundance": 0.0859, "gamma": -15.870},
    "I":  {"isotope": "127I", "spin": 2.5, "abundance": 1.0,     "gamma": 8.5550},
    "Pt": {"isotope": "195Pt", "spin": 0.5, "abundance": 0.3378, "gamma": 9.1500},
    "Hg": {"isotope": "199Hg", "spin": 0.5, "abundance": 0.1687, "gamma": 7.6290},
    "Pb": {"isotope": "207Pb", "spin": 0.5, "abundance": 0.2241, "gamma": 8.9200},
}

# Reference compounds.  delta = sigma_ref - sigma_mol is the chemical shift, so
# the reference has to be recomputed at the same level with the same code --
# a tabulated sigma_ref from another program would put a constant offset on
# every number and there would be no way to see it.
#
# Each entry carries its geometry and where the geometry came from.  These are
# experimental structures, deliberately: the validated chemical shifts above
# were obtained against experimental geometries for both the reference and the
# molecule, so keeping the reference experimental keeps that comparison honest.
# A molecule optimised inside ChatDFT therefore carries a small systematic
# offset, and the payload says so.
REFERENCES: Dict[str, dict] = {
    "tms": {
        "label": "TMS  Si(CH3)4",
        "elements": ("Si", "C", "H"),
        "geometry": "Si-C 1.875 A, C-H 1.090 A, tetrahedral (electron diffraction)",
        "standard_for": ("H", "C", "Si"),
    },
    "ammonia": {
        "label": "NH3",
        "elements": ("N", "H"),
        "geometry": "N-H 1.0124 A, H-N-H 106.67 deg (experimental r_e)",
        "standard_for": ("N",),
    },
    "water": {
        "label": "H2O",
        "elements": ("O", "H"),
        "geometry": "O-H 0.9572 A, H-O-H 104.52 deg (experimental r_e)",
        "standard_for": ("O",),
    },
    "phosphine": {
        "label": "PH3",
        "elements": ("P", "H"),
        "geometry": "P-H 1.420 A, H-P-H 93.5 deg (experimental r_e)",
        "standard_for": ("P",),
    },
    "chloroform_f": {
        "label": "CFCl3",
        "elements": ("C", "F", "Cl"),
        "geometry": "C-F 1.330 A, C-Cl 1.760 A, tetrahedral (experimental r_e)",
        "standard_for": ("F",),
    },
    "methane": {
        "label": "CH4",
        "elements": ("C", "H"),
        "geometry": "C-H 1.087 A, tetrahedral (experimental r_e)",
        "standard_for": (),
    },
    # The 77Se reference, computed and reported but NOT used to produce a
    # shift: SCALE_CHECK["Se"] records that this method does not reproduce the
    # 77Se scale, and compute() consults it.  It is in REFERENCES rather than
    # left out because leaving it out is what hid the problem for four rounds
    # -- "selenium has no reference" reads like an omission, while "selenium
    # has a reference and the method cannot use it, here is the measurement"
    # is the actual state of affairs, and it also puts the IUPAC primary
    # compound's own computed shielding in front of the user.
    "dimethyl_selenide": {
        "label": "(CH3)2Se",
        "elements": ("Se", "C", "H"),
        "geometry": "Se-C 1.943 A, C-Se-C 96.2 deg (microwave, Beecher 1966), "
                    "C-H 1.090 A, H-C-Se 109.47 deg",
        "standard_for": ("Se",),
    },
}

# Which reference each NMR-active nucleus is quoted against.  F is the one
# place where the IUPAC standard (CFCl3) is a different molecule from the
# obvious one; using CH3F instead would put every 19F shift on a scale offset
# by the difference between the two, so CFCl3 is used.
REFERENCE_FOR: Dict[str, str] = {
    "H": "tms", "C": "tms", "Si": "tms",
    "N": "ammonia", "O": "water", "P": "phosphine", "F": "chloroform_f",
    "Se": "dimethyl_selenide",
}

# The IUPAC primary reference for each nucleus, from Harris, Becker, Cabral de
# Menezes, Goodfellow & Granger, "NMR nomenclature.  Nuclear spin properties
# and conventions for chemical shifts", Pure Appl. Chem. 73, 1795 (2001).
#
# Five of the seven nuclei above are quoted against theirs.  Two are not, and
# the reason is what the standard costs, not a preference:
#
#   15N  the primary standard is nitromethane, CH3NO2 -- 66 basis functions at
#        6-31G*, which the memory guard prices at 1.3 GB.  Liquid ammonia is a
#        recognised secondary standard, so the number this build prints is not
#        wrong, but it sits 380.2 ppm away from the scale most papers use.
#   31P  the primary standard is 85% H3PO4 *in water*, which is not something
#        this build can compute at all -- it is a mixture, and the shielding
#        of the phosphate in it is a solvation problem.  Phosphoric acid
#        itself, H3PO4, is 85 basis functions and 3.5 GB.
#
# A shift printed without its scale is not a shift.  ``scale_note`` turns this
# table into the sentence each nucleus carries, and it compares against
# ``REFERENCE_FOR`` rather than repeating a flag, so adding the real standard
# later makes the note go away by itself instead of going stale.
IUPAC_PRIMARY: Dict[str, Dict[str, Optional[str]]] = {
    "H": {"key": "tms", "compound": "TMS", "conversion": None},
    "C": {"key": "tms", "compound": "TMS", "conversion": None},
    "Si": {"key": "tms", "compound": "TMS", "conversion": None},
    "O": {"key": "water", "compound": "H2O (liquid)", "conversion": None},
    "F": {"key": "chloroform_f", "compound": "CFCl3", "conversion": None},
    "Se": {"key": "dimethyl_selenide", "compound": "(CH3)2Se",
           "conversion": None},
    "N": {"key": "nitromethane",
          "compound": "CH3NO2 (nitromethane, neat liquid)",
          "conversion": "subtract 380.2 ppm to move the shift onto the "
                        "nitromethane scale"},
    "P": {"key": "phosphoric_acid", "compound": "85% H3PO4 in water",
          "conversion": "the offset is the 31P shift of PH3 itself on the "
                        "H3PO4 scale, which this build does not compute; it "
                        "is not small and must not be assumed to be"},
}


# Why each NMR-active element with no reference compound has none.  The list
# used to be just a list, which is how a gap turns into folklore: element
# symbols copied forward with nobody able to say whether the omission was a
# decision or an oversight.  It was a mixture of both, and the reasons are not
# the same, so each one is recorded here and carried into the warning the user
# sees.  The size of the list is deliberately not written down: this comment
# said "fourteen" until 77Se acquired a reference, and the table itself is the
# count.
#
# Which of them are in 6-31G* at all was measured (probe_ref1, via
# gto.basis.load -- building a bare atom answers a different question, and the
# first version of that probe wrote down seven of these reasons wrong because
# the closed-shell check fired before the basis was ever looked at):
#
#   not in the basis at all        Hg, I, Pb, Pt, Sn
#   in the basis, excluded for a reason
#                                  Al, B, Br, Cl, Li, Na, S
#   not an element this build can build at all
#                                  D
#
# The third category is new and deuterium is the only member.  It used to sit
# in the second, which was wrong twice over: the element table has no
# deuterium, so there is no basis to be in or not be in, and no Molecule can
# carry a 2H atom.  The distinction matters because the two need different
# work -- the second group needs a compound or a bigger budget, the third
# needs the element to exist in the builder first, and until it does no
# reference compound would help.  See UNBUILDABLE_NUCLEI below.
#
# For six of the second group the reason is that the standard is a *solution*:
# 7Li LiCl/D2O, 23Na NaCl/D2O, 27Al Al(NO3)3/D2O, 35Cl NaCl/D2O, 81Br
# NaBr/D2O, and 33S saturated (NH4)2SO4 in D2O.  A gas-phase isolated ion is
# not on that scale, and the halide anions would need diffuse functions that
# 6-31G* does not have, so computing them here would produce a number wearing
# the wrong scale's name -- the exact failure the 15N/31P notes exist to
# prevent.  Boron's standard is a molecule, BF3.OEt2, and the only thing
# stopping it is size: 19 atoms, 146 basis functions, 30.6 GB by the estimate
# below, four times the budget (probe_ref2).
#
# Selenium was in this table until this round, with the reason "no reference is
# provided yet; (CH3)2Se would be computable here" -- a to-do note, which is
# what it was.  It is no longer a to-do.  A reference was computable, so the
# question became whether it would help, and the answer was measured: it would
# not.  Selenium now HAS a reference compound and no chemical shift, because
# the method does not reproduce the 77Se scale (SCALE_CHECK["Se"]), and it has
# left this table for the reason this table exists -- the cause is different
# and it is recorded where that cause belongs.
#
# Deuterium was the last element here in the "not done yet" state, with the
# reason "no reference is provided yet; TMS-d12 would be computable here".
# That is also a to-do note, and it is also about the wrong thing: TMS-d12
# being computable is not what stands between this build and a 2H shift.  No
# job can contain deuterium in the first place (UNBUILDABLE_NUCLEI), so the
# reference was never the first problem.  The reason below says which problem
# it is instead, and it is the same string UNBUILDABLE_NUCLEI carries -- one
# reason, one place, so the two cannot describe the same nucleus differently.
#
# The table also no longer claims to be a list of things to do.  Of the three
# causes in it, two are decisions (a solution standard, an element 6-31G* does
# not carry) and one is a limit of the builder.  What they share is only that
# no reference compound is defined, which is what the name says.
UNBUILDABLE_NUCLEI: Dict[str, str] = {
    "D": "no job can contain it: the molecule builder's element table has no "
         "deuterium, so a 2H atom cannot reach the calculation and no "
         "reference compound would help it.  (The SMILES path used to turn "
         "[2H] into 1H without saying so, which was worse than either.)",
}

NO_REFERENCE_WHY: Dict[str, str] = {
    "Al": "its standard is Al(NO3)3 in D2O, a solution",
    "B": "its standard is BF3.OEt2, 19 atoms and 30.6 GB at this basis",
    "Br": "its standard is NaBr in D2O, a solution",
    "Cl": "its standard is NaCl in D2O, a solution",
    # Same string as UNBUILDABLE_NUCLEI["D"], and the contract asserts they
    # are equal: the user-facing reason and the structural reason for the same
    # nucleus must not be two different stories.
    "D": UNBUILDABLE_NUCLEI["D"],
    "Hg": "not in the 6-31G* basis",
    "I": "not in the 6-31G* basis",
    "Li": "its standard is LiCl in D2O, a solution",
    "Na": "its standard is NaCl in D2O, a solution",
    "Pb": "not in the 6-31G* basis",
    "Pt": "not in the 6-31G* basis",
    "S": "its standard is saturated (NH4)2SO4 in D2O, a solution",
    "Sn": "not in the 6-31G* basis",
}


def scale_note(el: str) -> Optional[str]:
    """The sentence about which scale this element's shift is on, if it is not
    the IUPAC primary one.

    Returns None when the build quotes the nucleus against the IUPAC primary
    reference, because in that case there is nothing to explain.
    """
    prim = IUPAC_PRIMARY.get(el)
    if prim is None:
        return None
    used = REFERENCE_FOR.get(el)
    if used == prim["key"]:
        return None
    label = REFERENCES.get(used, {}).get("label", used)
    out = (f"{el} is quoted against {label}, not the IUPAC primary reference "
           f"{prim['compound']}.")
    conv = prim.get("conversion")
    out += (" " + conv + "." if conv else
            " This build does not know the offset, so the number is on a "
            "different scale from the literature and has to be converted "
            "before it is compared with one.")
    return out


# ======================================================================
#  the accuracy ledger -- does the method reproduce the scale?
# ======================================================================
# A reference compound makes a shift *definable*.  It does not make it
# *correct*, and until this table existed the only statement about accuracy in
# this build was the twelve 1H/13C shifts in the module docstring -- which no
# gate recomputed, and which said nothing at all about 14N, 17O, 19F, 29Si,
# 31P or 77Se even though the product prints shifts for all of them.
#
# That is the same failure as a list of unreferenced elements with no reasons:
# a claim that lives in prose cannot be wrong, so it cannot be trusted.  Each
# entry here is a molecule whose shift against the product's own reference is
# known, computed at RHF/6-31G* by the same code path the product uses
# (``reference_geometry``, so a library geometry wins exactly as it does in
# production), with the measured number recorded.  ``backend/check_shift_scales``
# recomputes every entry, so the number cannot rot.
#
# The verdicts, which do not rot, without the numbers, which do:
#
#   nucleus  reference  sample   verdict
#   1H       TMS        CH4      reproduced
#   13C      TMS        CH4      reproduced
#   14N      NH3        CH3NO2   the right scale, about a quarter high
#   19F      CFCl3      CH3F     the right scale, about a fifth low
#   77Se     (CH3)2Se   H2Se     not a 77Se shift at all
#
# The numbers are in SCALE_CHECK below and nowhere else.  A first draft of
# this comment did repeat them, and by the time the gate first ran it was
# already wrong in two of the five rows -- 1H read +0.44 against the gate's
# +0.66 and 13C read -1.49 against its -0.21, because the comment was written
# from the probe run that measured them and the gate then measured them again
# through the product's own code path.  That is the same rot the module
# docstring's twelve-shift table had, appearing inside a single round.  Prose
# cannot be recomputed; the table it duplicates can.
#
# The two light nuclei are the validated ones and behave as the docstring says.
# 14N and 19F are on the right scale and the right sign, 20-25% out on a shift
# of a few hundred ppm, which is what a 6-31G* Hartree-Fock description of a
# lone-pair-bearing first-row atom is worth and is reported rather than hidden.
#
# 77Se is a different failure and is why this table exists.  Selenium is a
# fourth-row element: 6-31G* has no diffuse functions for it and there is no
# relativistic treatment at all, so the dominant (paramagnetic) term is
# qualitatively wrong.  The evidence does not depend on a literature shift,
# which is deliberate -- a single tabulated number is one more thing to get
# wrong, and this build can measure the failure directly:
#
#   * The reference's own shielding moves by **72.1 ppm** between two
#     defensible geometries of (CH3)2Se -- this build's B3LYP/6-31G*
#     stationary point (Se-C 1.9473 A, C-Se-C 92.22 deg) against the
#     microwave structure (Se-C 1.943 A, C-Se-C 96.2 deg, Beecher, J. Mol.
#     Spectrosc. 21, 414 (1966)).  A reference geometry sets the zero of every
#     shift, so that is 72 ppm of pure choice.
#   * The methyl conformation of the reference moves it by a further 8.4 ppm.
#   * The method's *entire* response to replacing both methyls of the
#     reference with hydrogens -- Me2Se to H2Se, the largest structural change
#     available in the smallest possible molecule -- is **101 ppm**, against a
#     77Se chemical shift range of 3000 ppm (-1000 to 2000, referenced to
#     Me2Se; HUJI 77Se nucleus table).
#
# So the scale the program would print is dominated by geometry choices it
# makes arbitrarily, and its total dynamic range over the hydride is 3% of the
# real one.  Selenium therefore keeps its absolute shielding and gets no
# chemical shift.  The reason lives here rather than in NO_REFERENCE_WHY, and
# the contract asserts that: a nucleus whose scale was measured and not
# reproduced has a different problem from one that has no reference at all,
# and if either table were allowed to hold both, the difference -- one needs a
# better method, the other needs a compound -- would be invisible.
#
# The evidence was measured in probe_se5 (geometry spread), probe_se6 (the
# hydride) and probe_se9 (the whole table).  A first version of those probes
# forgot the au -> ppm factor and reported every number 53.25x too small, and
# a second version built CH3F with a non-orthonormal frame and reported
# sigma(F) 1165 ppm from the right answer -- which read exactly like the method
# having the fluorine scale backwards.  Both were defects in the measurement,
# not in the method; the numbers in the table below are from the corrected
# runs.
SCALE_CHECK: Dict[str, dict] = {
    "H": {
        "nucleus": "1H", "reference": "TMS", "sample": "CH4",
        "reference_geometry": "tms", "sample_geometry": "methane",
        "computed_ppm": 0.66, "experimental_ppm": 0.23,
        # 1.0, not the 2.5 this entry was first given.  2.5 came from the
        # docstring's twelve-shift table, where one band covered 1H and 13C
        # together and the 1H half therefore passed by construction -- 2.5 ppm
        # is a fifth of the whole 1H shift range.  1.0 is set from the worst 1H
        # error in SHIFT_LEDGER, which the gate recomputes; the number is not
        # repeated here, because the whole point of moving the ledger into data
        # was to stop writing derived quantities into prose.
        "usable": True, "tolerance_ppm": 1.0,
        "source": "the worst 1H error over the six rows of SHIFT_LEDGER, gas "
                  "phase, recomputed by backend.check_shift_scales -- not from "
                  "the 2.5 ppm headline the docstring table used to carry, "
                  "which 1H could not fail",
        "note": None,
    },
    "C": {
        "nucleus": "13C", "reference": "TMS", "sample": "CH4",
        "reference_geometry": "tms", "sample_geometry": "methane",
        "computed_ppm": -0.21, "experimental_ppm": -2.30,
        # 3.2, raised from 2.5 in round 19.  The 2.5 was measured on methane
        # alone (2.09 ppm out) and was never true of the set: methanol is 3.04
        # ppm out.  A per-nucleus tolerance measured on one molecule is not a
        # claim about the nucleus, and the gate now checks it against all six
        # rows of SHIFT_LEDGER.  Acetylene's carbon is the one row allowed to
        # exceed it and is named there as an exception.
        "usable": True, "tolerance_ppm": 3.2,
        "source": "the worst 13C error over the six rows of SHIFT_LEDGER "
                  "excluding the named acetylene exception, recomputed by "
                  "backend.check_shift_scales; the previous 2.5 came from "
                  "methane alone",
        "note": None,
    },
    "N": {
        "nucleus": "14N", "reference": "NH3", "sample": "CH3NO2",
        "reference_geometry": "ammonia", "sample_geometry": "ch3no2",
        "computed_ppm": 472.45, "experimental_ppm": 380.20,
        "usable": True, "tolerance_ppm": 120.0,
        "source": "the ammonia-to-nitromethane conversion, 380.2 ppm "
                  "(IUPAC; the same number the scale note quotes)",
        # Interpolated, not typed: this sentence is rendered by ``scale_why``
        # from the fields above, so the two cannot disagree.  The percentage
        # and the 92 ppm difference that used to sit in this sentence are gone
        # rather than given slots -- both are functions of computed_ppm and
        # experimental_ppm, and a derived quantity written down anywhere is a
        # derived quantity that can rot (round 19 found three of them).
        "why": "The 14N/15N scale is reproduced but too high: this build puts "
               "nitromethane {computed:.0f} ppm from ammonia where experiment "
               "puts it {experimental:.0f}. The scale note tells the user to "
               "subtract {experimental:.1f} ppm to reach the nitromethane "
               "scale, and a shift converted that way from this build's "
               "numbers carries the same error with it.",
        "reason_kind": "measurement",
        "why_quotes": ("computed", "experimental"),
    },
    "F": {
        "nucleus": "19F", "reference": "CFCl3", "sample": "CH3F",
        "reference_geometry": "chloroform_f", "sample_geometry": "ch3f",
        "computed_ppm": -212.27, "experimental_ppm": -271.90,
        "usable": True, "tolerance_ppm": 80.0,
        "source": "CH3F is -271.9 ppm against neat CFCl3 (Bruker Almanac "
                  "1991, via the Indiana University 19F shift table)",
        # The series span (209.6 ppm experimental, 183.9 ppm computed) used to
        # be in this sentence and is in no field anywhere, so it had no
        # recomputation behind it -- an orphan number inside the text a user
        # reads.  It is dropped rather than promoted to a field: the ordering
        # claim is what the verdict needs, and nothing in this build measures
        # a span.
        "why": "The 19F scale is reproduced in sign and ordering but too "
               "small: this build puts CH3F {computed:.0f} ppm from CFCl3 "
               "where experiment puts it {experimental:.0f}. The whole "
               "fluoromethane series is ordered correctly, so a shift is "
               "usable with that caveat; the magnitude is not.",
        "reason_kind": "measurement",
        "why_quotes": ("computed", "experimental"),
    },
    "Se": {
        "nucleus": "77Se", "reference": "(CH3)2Se", "sample": "H2Se",
        "reference_geometry": "dimethyl_selenide",
        "sample_geometry": "h2se",
        "computed_ppm": -101.38, "experimental_ppm": None,
        "usable": False, "tolerance_ppm": 60.0,
        # The two numbers the verdict rests on, both measured here.
        "range_ppm": 3000.0,
        "reference_geometry_spread_ppm": 72.10,  # probe_se5

        "conformer_spread_ppm": 8.37,
        "source": "measured in this build (probe_se5, probe_se6, probe_se9); "
                  "the 77Se range and reference compound are from the HUJI "
                  "77Se nucleus table",
        # The sentence names the cause and the measurements separately, because
        # the previous version gave the geometry spread as the *conclusion*:
        # "the answer is dominated by a geometry choice this build makes
        # arbitrarily".  That is a symptom, and it points the user at a remedy
        # that does not exist -- changing the reference geometry does not fix
        # this.  The cause is the method: no diffuse functions on a fourth-row
        # element and no relativistic treatment at all, so the dominant
        # paramagnetic term is qualitatively wrong.  The geometry spread is
        # still here, one clause later, because it says something the cause
        # does not: even a correct method would have an unstable zero at this
        # reference.
        "why": "The method does not produce a 77Se shift, and the cause is the "
               "method rather than the reference: selenium is a fourth-row "
               "element, this basis has no diffuse functions for it, and "
               "there is no relativistic treatment at all, so the dominant "
               "paramagnetic term is qualitatively wrong. Two measurements "
               "say so from different sides -- the method's entire response "
               "to replacing both methyls of the reference with hydrogens is "
               "{computed:.0f} ppm against a {range:.0f} ppm shift range, and "
               "the reference's own shielding moves by {geometry_spread:.1f} "
               "ppm between two defensible geometries, so even a correct "
               "method would have an unstable zero here. The absolute "
               "shielding is reported; no shift is.",
        "reason_kind": "measurement",
        "why_quotes": ("computed", "range", "geometry_spread"),
    },
}

# The nuclei that *are* referenced and whose scale has never been measured
# here.  This is a gap recorded as a gap, for the same reason NO_REFERENCE_WHY
# exists: a list of things nobody has checked is the state in which nobody can
# tell a decision from an oversight.  The contract asserts that SCALE_CHECK and
# SCALE_UNMEASURED together cover REFERENCE_FOR exactly, so adding a reference
# forces one of the two to be filled in.
#
# Each entry carries the *pair* it is about and the value this build computes
# for it, so the gap is a specific missing number rather than a general
# admission.  An earlier version of this table said only "a pair has been
# computed here, but no experimental shift for it has been verified", which
# named no pair and no value: the next reader could not tell which comparison
# was set up, whether the method produced a real scale for the nucleus, or
# which literature number would close it.
#
# The values are here rather than in prose, and that is a deliberate reversal.
# The argument for keeping them out was that numbers rot -- the twelve-shift
# ledger in the module docstring went stale when the library's geometries were
# re-derived underneath it.  But this round established what actually causes
# that: numbers rot when nothing recomputes them.  SCALE_CHECK's numbers have
# not rotted, and they are checked by ``backend/check_shift_scales`` on every
# run.  So these are pinned the same way, and a pinned number is strictly more
# useful than a missing one: when the experimental value finally arrives, the
# computed side is already verified rather than remembered.
#
# Measured here (probe_scale19).  What the three say, and they do not say the
# same thing:
#
#   * 31P produces a real scale.  PH3 -> CH3PH2 is +66.4 ppm and
#     CH3PH2 -> (CH3)2PH a further +40.9, against a 31P range of about 430 ppm.
#     Each methyl deshields phosphorus, which is the experimental ordering, so
#     this is a scale the method carries -- unlike 77Se.
#   * 29Si produces a real scale too: silane is 64.9 ppm upfield of TMS, and
#     silane is upfield of TMS experimentally.  The magnitude is the open
#     question, not the existence.
#   * 17O is a different problem and it is not a missing number.  The IUPAC
#     primary reference is *liquid* water and 17O shifts are measured in
#     solution, where they are strongly solvent-dependent; this build is
#     gas-phase.  A gas-phase calculation compared against a solution
#     measurement confounds the method with the solvent, so no literature
#     number would make this comparison sound on its own.
SCALE_UNMEASURED: Dict[str, dict] = {
    "O": {
        "nucleus": "17O", "reference": "H2O", "sample": "formaldehyde",
        "reference_geometry": "water", "sample_geometry": "formaldehyde",
        "computed_ppm": 716.94,
        "range_ppm": 1100.0,
        # Empty on purpose: this reason is an argument, not a measurement, and
        # the verdict it supports ("this comparison cannot be made soundly") does
        # not rest on the number.  Declaring that is the point -- see
        # ``reason_kind``: an empty list is a decision, an omitted key would be
        # an oversight, and the contract requires both keys so the two cannot be
        # confused.
        "reason_kind": "argument",
        "why_quotes": (),
        "why": "the obstacle is the phase, not a number.  The IUPAC primary "
               "reference is liquid water and every 17O shift is measured in "
               "solution, where it is strongly solvent-dependent; this build "
               "is gas-phase, so a comparison would confound the method with "
               "the solvent.  It needs either a gas-phase standard or an "
               "explicit solvent model -- a literature number alone would not "
               "make it sound",
    },
    "Si": {
        "nucleus": "29Si", "reference": "TMS", "sample": "SiH4",
        "reference_geometry": "tms", "sample_geometry": "silane",
        "computed_ppm": -64.93,
        "range_ppm": 519.0,
        "reason_kind": "measurement",
        "why_quotes": ("computed", "range"),
        "why": "the method produces a real 29Si scale (silane {computed:+.1f} "
               "ppm against TMS, which is the experimental direction, against "
               "a {range:.0f} ppm range), so what is missing is one verified "
               "experimental shift for silane against TMS.  Both are volatile, "
               "so the comparison is well posed and does not have 17O's phase "
               "problem",
    },
    "P": {
        "nucleus": "31P", "reference": "PH3", "sample": "methylphosphine",
        "reference_geometry": "phosphine",
        "sample_geometry": "methylphosphine",
        "computed_ppm": 66.41,
        "range_ppm": 430.0,
        "reason_kind": "measurement",
        "why_quotes": ("computed", "range"),
        # A directional claim, checked rather than asserted.  "Each methyl
        # deshields phosphorus" is the evidence that the method carries a real
        # 31P scale -- one methylation could be a coincidence, two in the right
        # direction is a trend -- but writing the second step's magnitude into
        # the prose would put a number there that nothing recomputes, which is
        # the defect this whole table exists to avoid.  So the claim is stated
        # as a direction and ``check_shift_scales`` recomputes it, including a
        # floor on the size of the step: a "trend" of 0.001 ppm satisfies
        # "deshields" and says nothing.
        "trend": [("dimethylphosphine", +1)],
        "why": "the method produces a real 31P scale: each methyl deshields "
               "phosphorus, PH3 -> CH3PH2 by {computed:+.1f} ppm and then "
               "CH3PH2 -> (CH3)2PH by a further step of the same sign, against "
               "a {range:.0f} ppm range.  So what is missing is a verified "
               "experimental shift for methylphosphine against PH3.  There is "
               "a second gap that no literature value closes: the reference is "
               "PH3, not the primary standard 85% H3PO4, and the offset "
               "between them is the one number this build does not compute",
    },
}

# ======================================================================
#  the twelve-shift ledger -- the measurement the tolerances rest on
# ======================================================================
# The module docstring has carried this table since round 12 and nothing has
# ever recomputed it.  Two of its numbers were already known to have moved
# (methanol's 1H and 13C, when the library's geometries were re-derived in
# round 17) and one more had rotted unnoticed: the table's *mean absolute
# error*.  The docstring used to state it as 1.89 ppm; re-measured through this
# module's own code path it is 1.736 ppm.  That sentence is gone rather than
# corrected, because the mean is derived from the twelve rows and nothing was
# deriving it -- writing the new value down would only reset the clock on the
# same rot.
#
# That is why the rows are data now.  ``backend.check_shift_scales`` walks
# every one of them through ``scale_geometry`` and the same shielding routine a
# real job uses, and checks the computed value against the number recorded
# here.  The three quantities that are *derived* from the rows -- how many are
# inside the band, the worst error per nucleus, the mean -- are deliberately
# **not written down anywhere**: they are printed by the gate.  A count written
# in prose is a count that can contradict the list it counts.
#
# The tolerances in ``SCALE_CHECK`` are set from this table, and until now that
# was true of exactly one row each: the 1H entry says its 1.0 ppm comes from
# "the worst 1H error measured anywhere in that table is 0.68 ppm (methanol)",
# and the 13C entry's 2.5 ppm came from methane alone.  A per-nucleus tolerance
# measured on one molecule is not a claim about the nucleus.  The gate now
# checks each tolerance against every row, and the 13C number had to move:
# methane is 2.09 ppm out but methanol is 3.04, so 2.5 was never true of the
# set.  Acetylene's carbon (5.73 ppm) is the one row that is *allowed* to
# exceed it, and it is named below rather than left as a sentence in a
# docstring -- an exception that is not written down is indistinguishable from
# an oversight.
SHIFT_LEDGER: Dict[str, dict] = {
    "H": {
        "nucleus": "1H",
        "reference": "TMS",
        "reference_geometry": "tms",
        # The band the docstring's "inside 2.5 ppm" headline was about.  One
        # band per nucleus, because a single band covering 1H and 13C together
        # is how the 1H half came to pass by construction.
        "band_ppm": 2.5,
        # (sample, geometry key, computed ppm, experimental ppm)
        "rows": [
            ("acetylene", "acetylene", 1.997, 1.80),
            ("methane", "methane", 0.655, 0.23),
            ("ethane", "ethane", 1.130, 0.86),
            ("ethylene", "ethylene", 5.737, 5.40),
            ("benzene", "benzene", 7.666, 7.26),
            ("methanol", "methanol", 2.668, 3.35),
        ],
        "exceptions": {},
    },
    "C": {
        "nucleus": "13C",
        "reference": "TMS",
        "reference_geometry": "tms",
        "band_ppm": 2.5,
        "rows": [
            ("methane", "methane", -0.205, -2.30),
            ("ethane", "ethane", 8.328, 5.70),
            ("benzene", "benzene", 126.408, 128.50),
            ("ethylene", "ethylene", 120.564, 123.50),
            ("acetylene", "acetylene", 66.175, 71.90),
            ("methanol", "methanol", 46.457, 49.50),
        ],
        # Named, with the reason, because the alternative is a reader deciding
        # for themselves whether 5.7 ppm is a known weakness or a bug.  RHF at
        # 6-31G* has no correlation and no diffuse functions on carbon, and a
        # triple bond is where both matter most; the ordering is still right.
        "exceptions": {
            "acetylene": "the known RHF/6-31G* weakness on a carbon-carbon "
                         "triple bond -- no correlation and no diffuse "
                         "functions, and the ordering is still correct",
        },
    },
}

# A ``usable`` entry whose tolerance still rests on a single molecule.
#
# This table exists so the set cannot grow silently.  ``SHIFT_LEDGER`` is what
# turns "the 1H tolerance is 1.0 ppm" from a memory into a claim checked against
# six rows; every other usable nucleus is checked against one pair, and a
# tolerance measured on one molecule is not a claim about the nucleus -- that is
# exactly how 13C came to carry a 2.5 ppm tolerance that methanol (3.04 ppm out)
# had always falsified.  The contract asserts that the usable entries are
# partitioned into "has a multi-row ledger" and "named here", so adding a usable
# entry without one fails rather than passing unnoticed.
TOLERANCE_UNBACKED: Dict[str, str] = {
    "N": "the tolerance is set from the single pair NH3 vs CH3NO2.  A second "
         "row is available -- the library carries 18 nitrogen compounds "
         "including pyridine, pyrrole and nitrobenzene -- but it has not been "
         "measured.  The pair has to be chosen with care rather than taken at "
         "random: a 14N shift is dominated by the lone pair's local "
         "environment, so a second row measures a different thing unless the "
         "two compounds are comparable.  Recorded as work, not as done",
    "F": "the tolerance is set from the single pair CFCl3 vs CH3F, where CH3F "
         "is constructed rather than taken from the library.  A second row is "
         "available -- the library carries hydrogen fluoride and "
         "trifluoroacetic acid, both fluorinated -- but it has not been "
         "measured.  Recorded as work, not as done",
}


def tolerance_rule(tolerance: float, errors: Dict[str, float],
                   exceptions: Dict[str, str], tag: str = "nmrscale") -> list:
    """The rule on its own, as a function of the numbers it is about.

    Pure and table-free, so it can be driven with synthetic values.  A rule
    that can only be exercised by running six SCFs is a rule that will never be
    exercised -- and ``backend.check_shift_scales.prove()`` drives this one on
    both sides of the boundary in every direction.

    ``errors`` maps a row name to its absolute error.  Two things are checked
    about an exception, not one: that it does exceed the tolerance (otherwise
    the label is stale and hides a row that now passes), and that it is not the
    only row left -- a ledger whose rows are all excepted makes the tolerance a
    claim about nothing.
    """
    fail: list = []
    if not errors:
        return [f"{tag}: the ledger has no rows, so the {tolerance:.2f} ppm "
                "tolerance is a claim about nothing"]
    inliers = {k: v for k, v in errors.items() if k not in exceptions}
    if not inliers:
        return [f"{tag}: every row is listed as an exception, so the "
                f"{tolerance:.2f} ppm tolerance is a claim about nothing"]

    for name in sorted(exceptions):
        if name in errors and errors[name] <= tolerance:
            fail.append(
                f"{tag}: {name} is listed as an exception to the "
                f"{tolerance:.2f} ppm tolerance, but its error is only "
                f"{errors[name]:.3f} ppm -- the label is stale and is hiding a "
                "row that now passes")

    worst_name = max(inliers, key=lambda k: inliers[k])
    worst = inliers[worst_name]
    if worst > tolerance:
        fail.append(
            f"{tag}: the tolerance is {tolerance:.2f} ppm but {worst_name} is "
            f"{worst:.3f} ppm out, so the claim does not hold across the "
            f"ledger ({len(inliers)} row(s) checked, "
            f"{sum(1 for v in inliers.values() if v > tolerance)} over)")
    elif tolerance > 3.0 * worst:
        fail.append(
            f"{tag}: the tolerance is {tolerance:.2f} ppm against a worst "
            f"inlier of {worst:.3f} ppm ({worst_name}) -- more than three times "
            "the error it was measured at, so it would pass a method three "
            "times worse than this one")
    return fail


def tolerance_holds(el: str, errors: Dict[str, float]) -> list:
    """``tolerance_rule`` for one nucleus, reading its own two tables.

    The rule lives in the engine because two callers need it: the contract
    checks it against the numbers the ledger *records* (cheap, no SCF), and
    ``backend.check_shift_scales`` checks it against the numbers it
    *recomputes* (the measurement).  One implementation, two callers -- a rule
    written twice is a rule that can contradict itself.
    """
    entry = SCALE_CHECK.get(el)
    ledger = SHIFT_LEDGER.get(el)
    if entry is None or ledger is None:
        return [f"nmrscale: {el} has a ledger or a tolerance but not both, so "
                "the tolerance is checked against nothing"]
    return tolerance_rule(float(entry.get("tolerance_ppm", 0.0)), errors,
                          dict(ledger.get("exceptions") or {}),
                          tag=f"nmrscale: {el}")


def ledger_errors(el: str) -> Dict[str, float]:
    """The absolute errors the ledger records, without recomputing anything.

    What the contract checks ``tolerance_holds`` against.  The gate recomputes
    the same errors and passes those instead, so a tolerance that is wrong
    relative to the recorded rows and a ledger that has drifted away from the
    method are two different failures rather than one.
    """
    return {str(name): abs(float(c) - float(e))
            for name, _geom, c, e in SHIFT_LEDGER[el]["rows"]}


def scale_usable(el: str) -> bool:
    """Whether this build reports a chemical shift for this element.

    True when the element's scale has been measured and reproduced, and also
    when it has never been measured -- the difference is that the unmeasured
    ones carry a ``accuracy_note`` saying so.  False only when the scale has
    been measured and the method does not reproduce it.
    """
    entry = SCALE_CHECK.get(el)
    return True if entry is None else bool(entry.get("usable", True))


def scale_why(el: str) -> Optional[str]:
    """The reason sentence for an element, with its numbers interpolated.

    One renderer for both tables.  ``SCALE_CHECK`` and ``SCALE_UNMEASURED``
    hold the same kind of claim -- "here is how well this method reproduces
    this nucleus, and here is what the verdict rests on" -- so they render the
    same way: from ``{placeholders}`` in the entry's own ``why``, so the table
    holds each number once and the sentence cannot drift from it.

    Before this, the three ``SCALE_CHECK`` entries with something to say
    carried a hardcoded ``note`` with their own field values typed into it --
    every one of them a second copy of a field a few lines above.  The
    contract *required* those copies: it asserted that the note carried the
    computed and experimental values.  So the duplicate was not an oversight,
    it was enforced, which is why fixing it meant changing the assertion rather
    than the prose.

    This paragraph used to list the duplicated values themselves.  It does not
    any more, and the reason is the one this function exists for: a comment
    that repeats the numbers is a third copy of them, and it is a copy nothing
    recomputes -- so it would have been the last thing in the file still
    claiming a value the table had moved away from.  The fields are named
    instead, which is what the reader needs and what cannot rot.

    Returns None when the entry has nothing to say, which is the case for the
    nuclei whose scale is reproduced inside its tolerance.
    """
    entry = SCALE_CHECK.get(el)
    if entry is None or not entry.get("why"):
        entry = SCALE_UNMEASURED.get(el)
    if entry is None or not entry.get("why"):
        return None
    ctx = {"computed": float(entry["computed_ppm"])}
    if entry.get("range_ppm") is not None:
        ctx["range"] = float(entry["range_ppm"])
    if entry.get("experimental_ppm") is not None:
        ctx["experimental"] = float(entry["experimental_ppm"])
    # The spreads are named for the sentence rather than for the column: a
    # placeholder spelled {reference_geometry_spread_ppm} inside a sentence
    # about ppm reads as a second unit.
    for field, name in (("reference_geometry_spread_ppm", "geometry_spread"),
                        ("conformer_spread_ppm", "conformer_spread")):
        if entry.get(field) is not None:
            ctx[name] = float(entry[field])
    return entry["why"].format(**ctx)


def gap_why(el: str) -> str:
    """The reason a gap is still open, with its numbers filled in.

    The numbers are interpolated from the entry's own fields rather than typed
    into the sentence.  A number written twice is a number that can disagree
    with itself, and this module has already had one ledger go stale that way:
    the twelve-shift table in the docstring carried values that the library's
    geometries later contradicted, because the numbers were written in one place
    and computed in another.

    Only ``computed_ppm`` and ``range_ppm`` have slots.  The ``trend`` entries
    carry a *direction*, not a magnitude -- the step is recomputed by
    ``check_shift_scales`` and deliberately not written down anywhere, so there
    is no number for the sentence to quote.  A first draft of this function
    built a slot for each trend and would have interpolated the sign itself,
    printing "a further 1.0" into a sentence about ppm.
    """
    text = scale_why(el)
    if text is None:
        raise KeyError(f"{el} has no gap reason")
    return text


def accuracy_note(el: str) -> Optional[str]:
    """The sentence about how well the method reproduces this element's scale.

    Returns None when the scale has been measured and reproduced to within its
    tolerance and there is nothing to say.  Returns a sentence in the two cases
    where there is: a measured scale that is off, and a scale that has never
    been measured at all.  A note on all seven nuclei would be a note nobody
    reads, which is the same argument the scale notes are written to.

    Both cases render through ``scale_why``, so the text a user reads is built
    from the table in every case rather than stored beside it.
    """
    if el in SCALE_CHECK:
        return scale_why(el)
    if el not in SCALE_UNMEASURED:
        return None
    return (f"The accuracy of {el} shifts at this level has not been measured: "
            f"{gap_why(el)}.")


def _tick(progress: Progress, pct: int, msg: str) -> None:
    if progress:
        progress(max(0, min(100, int(pct))), msg)


# ======================================================================
#  reference geometries
# ======================================================================
def _tetrahedral(central: str, outer: str, r: float,
                 hydrogens: Optional[Tuple[str, float, float]] = None,
                 angles: Optional[Tuple[float, float, float]] = None):
    """A tetrahedral MX4 frame, optionally with four different substituents.

    ``angles`` is the (a, b, c) set of bond angles, in degrees, used for the
    C3v cases (NH3, PH3, CH3F) where the four substituents are not equivalent.
    """
    t = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                  [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    t /= np.linalg.norm(t, axis=1)[:, None]
    out = [(central, 0.0, 0.0, 0.0)]
    for v in t:
        out.append((outer, *(r * v)))
    return out


def _tms_geometry():
    """Si(CH3)4.

    Three hydrogens per carbon, not one.  An earlier version of this file put
    a single H on each carbon, which is Si(CH)4 -- the giveaway was nao = 33 at
    STO-3G (9 Si + 4*5 C + 4*1 H) where Si(CH3)4 needs 41, and the resulting
    shielding diverged with basis size (13C raw 100.9 -> 62.0 -> 72.8 -> 188.1
    -> 534.1 across five bases).  Check nao against the formula.
    """
    tet = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    tet /= np.linalg.norm(tet, axis=1)[:, None]
    out = [("Si", 0.0, 0.0, 0.0)]
    for u in tet:
        cpos = 1.875 * u
        ref = (np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9
               else np.array([1.0, 0.0, 0.0]))
        w1 = np.cross(u, ref)
        w1 /= np.linalg.norm(w1)
        w2 = np.cross(u, w1)
        out.append(("C", *cpos))
        for k in range(3):
            phi = 2.0 * np.pi * k / 3.0
            # H-C-Si = 109.47 deg, i.e. the H sits at u/3 + (2 sqrt2 / 3) w.
            v = u / 3.0 + (2.0 * math.sqrt(2.0) / 3.0) * (
                math.cos(phi) * w1 + math.sin(phi) * w2)
            out.append(("H", *(cpos + 1.090 * v)))
    return out


def _c3v(central: str, outer: str, r: float, angle: float,
         extra: Optional[Tuple[str, float]] = None):
    """A C3v MX3 frame with the X-M-X angle given in degrees.

    With the C3 axis along z and the X's at polar angle theta,
        cos(angle) = (3 cos^2(theta) - 1) / 2   ==>   cos^2(theta) = (2c+1)/3
    """
    c = math.cos(math.radians(angle))
    cz = math.sqrt(max(0.0, (2.0 * c + 1.0) / 3.0))
    sz = math.sqrt(max(0.0, 1.0 - cz * cz))
    out = []
    if extra is not None:
        out.append((extra[0], 0.0, 0.0, extra[1]))
    else:
        out.append((central, 0.0, 0.0, 0.0))
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        out.append((outer, r * sz * math.cos(phi), r * sz * math.sin(phi),
                    -r * cz))
    return out


def _builtin_reference_geometry(key: str):
    """The module's own geometry for a reference compound, as atom tuples."""
    if key == "tms":
        return _tms_geometry()
    if key == "ammonia":
        return _c3v("N", "H", 1.0124, 106.67)
    if key == "water":
        # O-H 0.9572 A, H-O-H 104.52 deg, bisector along +z.
        half = math.radians(104.52 / 2.0)
        return [("O", 0.0, 0.0, 0.0),
                ("H", 0.9572 * math.sin(half), 0.0, 0.9572 * math.cos(half)),
                ("H", -0.9572 * math.sin(half), 0.0, 0.9572 * math.cos(half))]
    if key == "phosphine":
        return _c3v("P", "H", 1.420, 93.5)
    if key == "chloroform_f":
        # CFCl3: F up the z axis, three Cl at the tetrahedral angle to it.
        out = [("C", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, 1.330)]
        cz = -1.0 / 3.0
        sz = math.sqrt(1.0 - cz * cz)
        for k in range(3):
            phi = 2.0 * math.pi * k / 3.0
            out.append(("Cl", 1.760 * sz * math.cos(phi),
                        1.760 * sz * math.sin(phi), 1.760 * cz))
        return out
    if key == "methane":
        return _tetrahedral("C", "H", 1.087)
    if key == "dimethyl_selenide":
        return _dimethyl_selenide_geometry()
    raise DFTError(f"Unknown NMR reference '{key}'.")


# ----------------------------------------------------------------------
#  geometries for the accuracy ledger
# ----------------------------------------------------------------------
# Same kind of object as the reference geometries above -- experimental bond
# lengths and angles, constructed rather than optimised -- for the molecules
# the ledger compares against.  They live here rather than in a probe because
# ``backend/check_shift_scales`` has to rebuild exactly what the recorded
# number was measured at, and a geometry in a probe is a geometry nobody
# re-runs.
def _c3v_polar(angle_deg: float) -> Tuple[float, float]:
    """(cos, sin) of the polar angle for three substituents at ``angle_deg``.

    With the C3 axis along z: cos(angle) = (3 cos^2(theta) - 1) / 2.
    """
    c = math.cos(math.radians(angle_deg))
    cz = math.sqrt(max(0.0, (2.0 * c + 1.0) / 3.0))
    return cz, math.sqrt(max(0.0, 1.0 - cz * cz))


def _perp_frame(axis):
    """An orthonormal frame (a, w1, w2) with ``a`` along ``axis``.

    Written out because getting it wrong is silent and expensive: a first
    version of the 19F check combined a = (0,0,1) with a frame built
    perpendicular to (1,0,0), which is not perpendicular to a at all, and one
    hydrogen ended up 1.2656 A from the carbon instead of 1.095.  sigma(F) came
    out 1165 ppm from the right answer and read like a sign error in the
    method.  ``assert``ing the orthonormality is cheaper than trusting it.
    """
    a = np.array(axis, dtype=float)
    a = a / np.linalg.norm(a)
    ref = (np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9
           else np.array([1.0, 0.0, 0.0]))
    w1 = np.cross(a, ref)
    w1 = w1 / np.linalg.norm(w1)
    w2 = np.cross(a, w1)
    assert abs(float(np.dot(a, w1))) < 1e-12
    assert abs(float(np.dot(a, w2))) < 1e-12
    assert abs(float(np.dot(w1, w2))) < 1e-12
    return a, w1, w2


def _dimethyl_selenide_geometry():
    """(CH3)2Se -- the 77Se reference compound, from experimental parameters.

    Se-C 1.943 A and C-Se-C 96.2 deg from the microwave structure (Beecher,
    *J. Mol. Spectrosc.* **21**, 414 (1966)); C-H 1.090 A and H-C-Se 109.47
    deg as the idealised methyl used for TMS.

    Built from experiment rather than taken from this build's own B3LYP/6-31G*
    stationary point (Se-C 1.9473 A, C-Se-C 92.22 deg) because every other
    reference in this module is an experimental structure, and because the
    4.0 deg gap is the largest single number in the 77Se ledger: the shielding
    moves by 72.1 ppm between the two (probe_se5).  Selenium gets no chemical
    shift in this build at all, so this geometry is not used by the product --
    it exists so the measurement that says so can be reproduced.
    """
    half = math.radians(96.2 / 2.0)
    out = [("Se", 0.0, 0.0, 0.0)]
    for sgn in (1.0, -1.0):
        u = np.array([sgn * math.sin(half), 0.0, math.cos(half)])
        cpos = 1.943 * u
        out.append(("C", *cpos))
        a, w1, w2 = _perp_frame(-u)
        for k in range(3):
            phi = 2.0 * math.pi * k / 3.0
            v = (math.cos(math.radians(109.47)) * a
                 + math.sin(math.radians(109.47)) * (math.cos(phi) * w1
                                                     + math.sin(phi) * w2))
            out.append(("H", *(cpos + 1.090 * v)))
    return out


def _h2se_geometry():
    """H2Se, experimental r_e: Se-H 1.4599 A, H-Se-H 90.6 deg.

    The molecule the 77Se ledger measures the method against, because it is
    the largest structural change reachable in three atoms.
    """
    half = math.radians(90.6 / 2.0)
    return [("Se", 0.0, 0.0, 0.0),
            ("H", 1.4599 * math.sin(half), 0.0, 1.4599 * math.cos(half)),
            ("H", -1.4599 * math.sin(half), 0.0, 1.4599 * math.cos(half))]


def _ch3f_geometry():
    """CH3F, experimental r_e: C-F 1.382 A, C-H 1.095 A, H-C-H 110.6 deg."""
    cz, sz = _c3v_polar(110.6)
    out = [("C", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, 1.382)]
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        out.append(("H", 1.095 * sz * math.cos(phi), 1.095 * sz * math.sin(phi),
                    -1.095 * cz))
    return out


def _ch3no2_geometry():
    """CH3NO2, experimental r_e.

    C-N 1.489 A, N-O 1.224 A, C-N-O 117.3 deg, C-H 1.094 A, H-C-H 107.3 deg.
    The nitro group is planar; the methyl is tetrahedral about the C-N axis.
    """
    ang = math.radians(117.3)
    out = [("N", 0.0, 0.0, 0.0),
           ("C", 1.489, 0.0, 0.0),
           ("O", 1.224 * math.cos(ang), 1.224 * math.sin(ang), 0.0),
           ("O", 1.224 * math.cos(ang), -1.224 * math.sin(ang), 0.0)]
    cpos = np.array([1.489, 0.0, 0.0])
    a, w1, w2 = _perp_frame([-1.0, 0.0, 0.0])
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        v = (math.cos(math.radians(107.3)) * a
             + math.sin(math.radians(107.3)) * (math.cos(phi) * w1
                                                + math.sin(phi) * w2))
        out.append(("H", *(cpos + 1.094 * v)))
    return out


def _sih4_geometry():
    """SiH4, experimental r_e: Si-H 1.480 A, tetrahedral."""
    return _tetrahedral("Si", "H", 1.480)


def _h2co_geometry():
    """Formaldehyde, experimental r_e: C-O 1.208 A, C-H 1.111 A, H-C-H 116.5
    deg, C2 axis along +z."""
    half = math.radians(116.5 / 2.0)
    return [("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 1.208),
            ("H", 1.111 * math.sin(half), 0.0, -1.111 * math.cos(half)),
            ("H", -1.111 * math.sin(half), 0.0, -1.111 * math.cos(half))]


_SCALE_GEOMETRY = {
    "h2se": _h2se_geometry,
    "ch3f": _ch3f_geometry,
    "ch3no2": _ch3no2_geometry,
    "sih4": _sih4_geometry,
    "h2co": _h2co_geometry,
}


# ----------------------------------------------------------------------
#  which geometry a reference is actually computed at
# ----------------------------------------------------------------------
# A reference compound that the molecule library also ships has ONE geometry
# in this product, and the reference has to be computed at it.
#
# It was not, and the symptom was visible: the library's water is
# O-H 0.96857 A / H-O-H 104.00 deg, while the module carried its own
# experimental water at O-H 0.9572 A / 104.52 deg.  So "water" meant two
# different molecules depending on which code path asked for it, and water's
# own 17O shift -- referenced to water -- came out at 0.374 ppm instead of
# 0.00.  That is the contract's flagship consistency check: the one number
# that says the reference path and the molecule path are the same code, and
# the one an under-converged SCF cannot fake.  A 0.374 ppm offset on every
# 17O shift in the output is small, but it is pure geometry, not chemistry.
#
# The lookup is by ELEMENT COUNT, and only when exactly one library entry has
# that count.  Element counts rather than the formula string, because the
# library writes "NH3" where reference_formula() writes "H3N", and because
# the count is what makes the 19F case safe: the library's chloroform is
# CHCl3, the 19F reference is CFCl3, the counts differ, and the module's own
# experimental geometry is kept.  Matching on the name would have silently
# put every 19F shift on the CHCl3 scale.
#
# Three element counts are shared by two library entries each (hydroxide and
# the hydroxyl radical; butane and isobutane; pyrazine and pyrimidine).  None
# is a reference compound, and the "exactly one" rule means an ambiguous
# count falls back to the built-in geometry rather than picking arbitrarily.
_LIBRARY_GEOMETRY: Optional[Dict[str, dict]] = None


def _library_geometry_index() -> Dict[str, dict]:
    """Library molecules keyed by their sorted element tuple.

    Built once.  A key maps to the single library molecule with that element
    count; an element count shared by two molecules is dropped, so the lookup
    can only ever return an unambiguous answer.
    """
    global _LIBRARY_GEOMETRY
    if _LIBRARY_GEOMETRY is not None:
        return _LIBRARY_GEOMETRY

    index: Dict[str, dict] = {}
    ambiguous: set = set()
    try:
        from .molecule import _load_library
        lib = _load_library()
    except Exception:                                    # noqa: BLE001
        lib = {}

    for name, entry in lib.items():
        xyz = (entry.get("xyz") or "").splitlines()
        if len(xyz) < 2:
            continue
        try:
            natm = int(xyz[0].split()[0])
        except (ValueError, IndexError):
            continue
        atoms = []
        for line in xyz[2:2 + natm]:
            p = line.split()
            if len(p) < 4:
                atoms = []
                break
            atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
        if len(atoms) != natm:
            continue
        sig = ",".join(sorted(a[0] for a in atoms))
        if sig in index:
            ambiguous.add(sig)
        else:
            index[sig] = {"name": name, "atoms": atoms}

    for sig in ambiguous:
        index.pop(sig, None)
    _LIBRARY_GEOMETRY = index
    return index


def _geometry_signature(atoms) -> str:
    """A fingerprint of the coordinates, for the cache key.

    The cache key used to carry only the basis, the tolerance and the field
    step -- not the geometry.  Changing a reference geometry would therefore
    have reused the old shielding values silently, which is the same class of
    mistake as the cache that never wrote: a number that is not what it says
    it is.
    """
    blob = json.dumps([[str(s), round(float(x), 6), round(float(y), 6),
                        round(float(z), 6)] for s, x, y, z in atoms],
                      sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def reference_geometry_source(key: str) -> str:
    """'library' when the molecule library defines this compound, else
    'builtin'."""
    atoms = _builtin_reference_geometry(key)
    sig = ",".join(sorted(a[0] for a in atoms))
    return "library" if sig in _library_geometry_index() else "builtin"


def reference_library_name(key: str) -> Optional[str]:
    """The library entry a reference's geometry came from, or None."""
    atoms = _builtin_reference_geometry(key)
    sig = ",".join(sorted(a[0] for a in atoms))
    hit = _library_geometry_index().get(sig)
    return hit["name"] if hit else None


def reference_geometry(key: str):
    """The geometry a reference compound is computed at, as atom tuples.

    The molecule library wins when it has the compound (see above); the
    module's own experimental geometry is the fallback.
    """
    atoms = _builtin_reference_geometry(key)
    sig = ",".join(sorted(a[0] for a in atoms))
    hit = _library_geometry_index().get(sig)
    return list(hit["atoms"]) if hit else atoms


def reference_geometry_note(key: str) -> str:
    """A description of the geometry actually in use, derived from it.

    Derived, not typed: the hand-written strings in ``REFERENCES`` described
    the built-in geometry and would have gone on describing it after the
    library took over -- which is how the library's own water entry came to
    claim "~104.5 deg" for a geometry that is 104.00.

    The summary is the bonded distances grouped by element pair, plus the
    angles at a unique central atom.  That is enough for a reader to see which
    geometry the shift is quoted against, which is the whole point of
    recording it.
    """
    from . import elements

    atoms = reference_geometry(key)
    src = reference_geometry_source(key)
    coords = np.array([a[1:] for a in atoms], dtype=float)
    syms = [a[0] for a in atoms]

    pairs: Dict[tuple, List[float]] = {}
    neigh: Dict[int, List[int]] = {i: [] for i in range(len(atoms))}
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if 0.4 < d <= elements.bond_cutoff(syms[i], syms[j]):
                pairs.setdefault(tuple(sorted((syms[i], syms[j]))), []).append(d)
                neigh[i].append(j)
                neigh[j].append(i)

    parts = []
    for (a, b), ds in sorted(pairs.items()):
        parts.append(f"{a}-{b} {sum(ds) / len(ds):.4f} A"
                     + (f" x{len(ds)}" if len(ds) > 1 else ""))

    # a unique central atom: every angle at it is one number, and it is the
    # number a reference geometry is usually quoted by
    central = [i for i in neigh if len(neigh[i]) == len(atoms) - 1
               and len(atoms) > 2]
    if len(central) == 1:
        c = central[0]
        nb = neigh[c]
        if len({syms[j] for j in nb}) == 1:
            angs = []
            for x in range(len(nb)):
                for y in range(x + 1, len(nb)):
                    v1 = coords[nb[x]] - coords[c]
                    v2 = coords[nb[y]] - coords[c]
                    cosang = float(np.dot(v1, v2)
                                   / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                    angs.append(math.degrees(
                        math.acos(max(-1.0, min(1.0, cosang)))))
            parts.append(f"{syms[nb[0]]}-{syms[c]}-{syms[nb[0]]} "
                         f"{sum(angs) / len(angs):.2f} deg")

    where = ("the molecule library" if src == "library"
             else "the module's experimental geometry")
    return f"{'; '.join(parts)} ({where})"


def reference_formula(key: str) -> str:
    from collections import Counter
    counts = Counter(sym for sym, *_ in reference_geometry(key))
    order = {"C": 0, "H": 1}
    parts = sorted(counts.items(), key=lambda kv: (order.get(kv[0], 9), kv[0]))
    return "".join(f"{s}{n if n > 1 else ''}" for s, n in parts)


def scale_geometry(key: str):
    """The geometry a scale-check molecule is computed at, as atom tuples.

    Three sources, in this order: a constructed experimental geometry from
    ``_SCALE_GEOMETRY``; a reference compound, which resolves through
    ``reference_geometry`` so a library geometry wins exactly as it does in
    production; and otherwise a molecule library entry by name.  Resolving
    through the same function the product uses is the point -- a ledger entry
    measured on a private geometry would be a statement about the probe.
    """
    fn = _SCALE_GEOMETRY.get(key)
    if fn is not None:
        return fn()
    if key in REFERENCES:
        return reference_geometry(key)
    from .molecule import resolve
    got = resolve(key, kind="name")
    return [(a.symbol, a.x, a.y, a.z) for a in got.atoms]


# ======================================================================
#  operators
# ======================================================================
def _build_mol(atom_xyz: str, basis: str, charge: int, multiplicity: int):
    if _PYSCF_ERROR is not None:                    # pragma: no cover
        raise DFTError(f"PySCF is not importable: {_PYSCF_ERROR}")
    syms, coords = _split_xyz(atom_xyz)
    atoms = [(s, float(c[0]), float(c[1]), float(c[2]))
             for s, c in zip(syms, coords)]
    try:
        # PySCF's default unit is Angstrom; the reference geometries above are
        # in Angstrom too, so no conversion anywhere in this module.
        return gto.M(atom=atoms, basis=basis, charge=int(charge),
                     spin=int(multiplicity) - 1, verbose=0)
    except Exception as exc:
        raise DFTError(f"Could not build the basis for NMR: {exc}")


def build_operators(mol) -> dict:
    """Every field- and moment-derivative integral the shielding needs.

    Two origins matter and only two.  ``with_rinv_orig(R_A)`` sets all three
    of ``int1e_ia01p``, ``int1e_giao_a11part`` and ``int1e_a01gp`` -- measured
    in probe_nmr15 as |A-B| = 1.27e+1, 4.49e-1, 4.40e-1 between two different
    origins.  ``with_common_origin`` changes none of them (|A-B| = 0.0 for all
    three), and building Q under it destroys the shieldings (H2O 1H goes
    31.85 -> 154.63 ppm), so it is not an alternative.
    """
    nao, natm = mol.nao_nr(), mol.natm
    R = mol.atom_coords()
    d = {
        "natm": natm,
        "nao": nao,
        "h0": mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        "s0": mol.intor("int1e_ovlp"),
        "eri0": mol.intor("int2e"),
        "O_h": (0.5 * mol.intor("int1e_giao_irjxp")
                + mol.intor("int1e_ignuc") + mol.intor("int1e_igkin")),
        "O_S": mol.intor("int1e_igovlp"),
    }
    ig1 = mol.intor("int2e_ig1")
    # int2e_ig1 carries only the (mu,nu) pair's phase derivative -- measured
    # antisymmetric under mu<->nu (||ig1 - swap|| / ||ig1|| = 2.0) and
    # symmetric under kappa<->lambda (1.2e-16) -- so the full derivative needs
    # the pair-swapped term added, not the raw integral.
    #
    # The sum is NOT formed here.  ig1 is (3, nao, nao, nao, nao) and at TMS
    # (nao = 98, 6-31G*) that is 2.26 GB on its own; adding the pair-swapped
    # copy and then contracting over t needs three more arrays of the same
    # size, and the resulting peak of about 11 GB is what made the first TMS
    # reference thrash a 15 GB machine for half an hour without finishing.
    # Since every field direction is used on its own, the pair-swapped sum is
    # built per direction in _giao_eri, which needs one extra array rather
    # than four.
    d["ig1"] = ig1

    d["P"] = np.zeros((natm, 3, nao, nao))
    d["Q"] = np.zeros((natm, 3, 3, nao, nao))
    eye = np.eye(3)
    for a in range(natm):
        with mol.with_rinv_orig(R[a]):
            d["P"][a] = mol.intor("int1e_ia01p")
            a11 = mol.intor("int1e_giao_a11part").reshape(3, 3, nao, nao)
            a01 = mol.intor("int1e_a01gp").reshape(3, 3, nao, nao)
        # Q^{ts} = a11part - delta_ts Tr(a11part) + a01gp.  The trace is taken
        # over (t,s) and subtracted as a multiple of the identity, which is
        # what leaves a rank-2 object out of the rank-3 a11part.
        d["Q"][a] = a11 - np.einsum("ts,uv->tsuv", eye, a11.trace(axis1=0, axis2=1))
        d["Q"][a] += a01
    return d


def _giao_eri(ops: dict, bvec) -> np.ndarray:
    """``eri0 - i B_t G^t``, built one field direction at a time.

    ``G^t = ig1[t] + ig1[t](pair-swapped)`` and it is only ever needed
    contracted with a single direction vector, so forming the full (3, nao^4)
    object first would cost three times the memory for nothing.  See the note
    in :func:`build_operators` for the measurement that forced this.
    """
    bvec = np.asarray(bvec, dtype=float)
    eri0 = ops["eri0"]
    if not np.any(bvec):
        return eri0
    term = None
    for t in range(3):
        if abs(bvec[t]) < 1e-15:
            continue
        # ig1[t] is [mu, nu, kappa, lambda]; the pair swap moves (mu,nu) past
        # (kappa,lambda), which is a (2,3,0,1) transpose on this slice.
        g = ops["ig1"][t] + ops["ig1"][t].transpose(2, 3, 0, 1)
        term = bvec[t] * g if term is None else term + bvec[t] * g
        del g
    return eri0 - 1j * term


def _scf_at(mol, ops: dict, bvec: Sequence[float], dm0,
            conv_tol: float = CONV_TOL, max_cycle: int = MAX_CYCLE):
    """One SCF at a finite magnetic field, with the GIAO Hamiltonian."""
    bvec = np.asarray(bvec, dtype=float)
    m = scf.RHF(mol)
    m.verbose = 0
    # h5py's probe of the temp path fails in this sandbox, and ChatDFT keeps
    # everything in memory anyway.
    m.chkfile = None
    m.max_cycle = max_cycle
    m.conv_tol = conv_tol
    m.get_hcore = lambda *a: ops["h0"] - 1j * np.einsum(
        "t,tuv->uv", bvec, ops["O_h"])
    m.get_ovlp = lambda *a: ops["s0"] - 1j * np.einsum(
        "t,tuv->uv", bvec, ops["O_S"])
    m._eri = _giao_eri(ops, bvec)
    m.kernel(dm0)
    return m


def shielding_tensor(mol, ops: dict, dB: float = FIELD_STEP,
                     conv_tol: float = CONV_TOL,
                     progress: Progress = None, base: int = 20,
                     span: int = 70) -> Tuple[np.ndarray, dict]:
    """The shielding tensor of every nucleus, in atomic units.

    Returns ``(sigma, diagnostics)``.  ``sigma`` is symmetrised; see
    :func:`symmetrise` for why that is not a fudge and for the residual it
    hides.
    """
    natm = ops["natm"]

    _tick(progress, base, "Reference SCF (unperturbed)")
    m0 = _scf_at(mol, ops, [0.0, 0.0, 0.0], None, conv_tol)
    if not m0.converged:
        raise DFTError(
            "The unperturbed SCF did not converge, so no shielding can be "
            "trusted. Try a different basis or charge.")
    dm0 = m0.make_rdm1()
    D0 = m0.make_rdm1()

    sigma = np.zeros((natm, 3, 3))
    dE_dB: List[float] = []
    runs: List[dict] = []
    for t in range(3):
        bp = [0.0, 0.0, 0.0]
        bp[t] = dB
        bm = [0.0, 0.0, 0.0]
        bm[t] = -dB
        _tick(progress, base + int(span * (t + 0.5) / 3.0),
              f"Field-perturbed SCF, direction {'xyz'[t]}")
        mp_ = _scf_at(mol, ops, bp, dm0, conv_tol)
        mm_ = _scf_at(mol, ops, bm, dm0, conv_tol)
        runs.append({"direction": "xyz"[t], "converged": bool(mp_.converged
                                                              and mm_.converged),
                     "cycles": [int(getattr(mp_, "cycles", -1)),
                                int(getattr(mm_, "cycles", -1))]})
        # A closed shell has no linear Zeeman term, so this is exactly zero if
        # the perturbation is consistently imaginary.  It is the cheapest
        # available test of the convention and it is kept in the payload.
        dE_dB.append(float((mp_.e_tot.real - mm_.e_tot.real) / (2.0 * dB)))

        dp, dm_ = mp_.make_rdm1(), mm_.make_rdm1()
        for a in range(natm):
            for s in range(3):
                hp = -1j * ops["P"][a][s] + sum(
                    bp[k] * ops["Q"][a][k][s] for k in range(3))
                hm = -1j * ops["P"][a][s] + sum(
                    bm[k] * ops["Q"][a][k][s] for k in range(3))
                sigma[a, t, s] = (
                    np.einsum("uv,vu->", dp, hp).real
                    - np.einsum("uv,vu->", dm_, hm).real) / (2.0 * dB)

    _tick(progress, base + span, "Assembling the shielding tensors")
    raw = sigma.copy()
    sigma, asym = symmetrise(raw)
    diag = {
        "dE_dB": dE_dB,
        "dE_dB_max": float(max(abs(v) for v in dE_dB)),
        "tensor_asymmetry_ppm": asym,
        "tensor_asymmetry_max_ppm": float(np.max(asym)) if asym.size else 0.0,
        "scf_runs": runs,
        "conv_tol": conv_tol,
        "field_step_au": dB,
        "unperturbed_energy": float(m0.e_tot.real),
    }
    return sigma, diag


def symmetrise(sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Replace sigma by its symmetric part, and return the discarded part.

    sigma_ts = d2E/dB_t dmu_s must equal sigma_st: both are mixed second
    partial derivatives of one scalar, so they commute.  The computed tensor
    does not quite satisfy that, and the residual is not numerical -- probe_nmr16
    swept dB over five decades and conv_tol over three and moved it by 0.0003
    ppm, and probe_nmr17 showed it is entirely in the OFF-DIAGONAL elements
    (the diagonal asymmetry is exactly 0.000000 in every molecule tested) and
    appears only where the site symmetry permits an off-diagonal element at
    all: CH4 and benzene are exactly clean, H2O 1H carries 1.648 ppm, NH3 1H
    0.621, NH3 14N 0.0004, H2O 17O 0.0000.

    It splits into two individually asymmetric pieces whose asymmetric parts
    do not quite cancel:

        sigma_ts = Tr(D0 Q^{ts})  +  Tr(D1_t^B (-i P^s))
                   diamagnetic         paramagnetic

    Transposing Q does not fix it (probe_nmr17: A + B^T makes it worse, 3.680
    ppm, against 1.648 for A + B), and neither piece is symmetric alone
    (sym(A) + B leaves 1.016, A + sym(B) leaves 2.664).  Only symmetrising
    both clears it, to 0.000000.

    Symmetrising is therefore the correct assembly, not a cosmetic fix: it is
    the average of the two equivalent orderings of the same mixed derivative,
    which is exactly what the identity above demands.  It is also provably
    harmless to every number the validated tables quote, because the trace --
    and therefore the isotropic shielding and every chemical shift -- is
    invariant under transposition.  Measured: 0.000e+00 delta on H2O, NH3 and
    CH4, for every nucleus.

    The discarded antisymmetric part is returned rather than dropped, so the
    caller can report the residual instead of hiding it.
    """
    sym = 0.5 * (sigma + sigma.transpose(0, 2, 1))
    asym = np.abs(sigma - sigma.transpose(0, 2, 1)).max(axis=(1, 2)) * PPM
    return sym, asym


# ======================================================================
#  tensor invariants
# ======================================================================
def principal_values(tensor: np.ndarray) -> np.ndarray:
    """The three principal shieldings, ascending (sigma_11 <= 22 <= 33)."""
    return np.linalg.eigvalsh(0.5 * (tensor + tensor.T))


def tensor_invariants(tensor_ppm: np.ndarray) -> dict:
    """Span, skew and reduced anisotropy, in the Haeberlen convention.

    ``span``  = sigma33 - sigma11, always >= 0.
    ``skew``  = 3 (sigma22 - sigma_iso) / span, in [-1, 1].
    ``delta_anisotropy`` = sigma33 - (sigma11 + sigma22) / 2.

    These are the three numbers a solid-state NMR paper quotes, and they are
    stated here in terms of the SHIELDING.  A paper reporting shift
    anisotropy quotes the same quantities with the opposite sign, because
    delta = sigma_ref - sigma; the sign is not a convention that can be
    guessed, so the payload labels every value as a shielding.
    """
    p = principal_values(tensor_ppm)
    iso = float(np.trace(tensor_ppm) / 3.0)
    span = float(p[2] - p[0])
    skew = float(3.0 * (p[1] - iso) / span) if span > 1e-12 else 0.0
    return {
        "eigenvalues_ppm": [float(p[0]), float(p[1]), float(p[2])],
        "sigma_iso_ppm": iso,
        "span_ppm": span,
        "skew": max(-1.0, min(1.0, skew)),
        "anisotropy_ppm": float(p[2] - 0.5 * (p[0] + p[1])),
    }


# ======================================================================
#  reference shieldings, with a disk cache
# ======================================================================
def _cache_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    path = os.path.join(root, "data", "nmr_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _method_signature() -> str:
    """A fingerprint of the code that produces the shielding number.

    The cache key already carried the basis, the tolerance, the field step and
    a fingerprint of the geometry, and each of those was added *after* a
    concrete failure showed that omitting it silently reused a wrong number.
    The one thing still missing was the code itself: edit the GIAO solve and
    every cached reference stays exactly where it was, computed by the old
    code, and every chemical shift the product prints inherits the difference
    as a constant offset that nothing in the output can reveal.

    That is not hypothetical here.  ``data/nmr_cache`` is **shipped as data**
    -- ``gen_nmr_cache`` writes it and it travels with the tree -- so a user
    who upgrades the code keeps the old numbers until something invalidates
    them.  The module's own docstring already makes this argument about
    ``conv_tol``: "a reference computed at the wrong tolerance would put a
    constant offset on every shift the product prints and there would be no way
    to see it from the output."  A reference computed by the wrong *code* is
    the same failure with no way to see it either.

    So the version is not a constant anyone has to remember to bump -- a
    constant that must be remembered is one that will be forgotten, and
    ``CACHE_FORMAT`` had already been bumped by hand twice.  It is derived from
    the source of the functions that determine the value, plus the constants
    they use, so editing any of them invalidates every entry that depends on
    it and nothing else does.

    ``inspect.getsource`` reads the file from disk, so a mutation harness that
    rewrites this module gets a different signature and cannot read the shipped
    cache -- which is the same hazard the harnesses' port guard exists for.
    """
    import hashlib
    import inspect

    parts = []
    for fn in (_xyz, _build_mol, build_operators, shielding_tensor):
        try:
            parts.append(inspect.getsource(fn))
        except (OSError, TypeError):        # frozen build, or a stub
            parts.append(getattr(fn, "__name__", "?"))
    parts.append(f"PPM={PPM!r}")
    parts.append(f"FIELD_STEP={FIELD_STEP!r}")
    parts.append(f"CONV_TOL={CONV_TOL!r}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _cache_key(key: str, basis: str, conv_tol: float, dB: float,
               geometry: str = "", method: str = None) -> str:
    """The cache key: everything the shielding value depends on.

    ``geometry`` is a fingerprint of the coordinates, not a description.  It
    was added after the water reference was found to be 0.374 ppm off: the key
    carried the basis, the tolerance and the field step but not the geometry,
    so swapping a reference geometry would have silently reused the old
    numbers.  An empty geometry means "the built-in one", which is what every
    key written before this field existed means -- those entries are
    recomputed, not misread.

    ``method`` is the same argument one level up: the value also depends on the
    code that computes it.  It is a parameter rather than a call to
    ``_method_signature`` inside the body so that the gate can hand it two
    different values and watch the key change -- the only way to show that a
    key really carries something is to vary that thing and see the key move.
    """
    if method is None:
        method = _method_signature()
    blob = json.dumps([key, basis, conv_tol, dB, geometry, method],
                      sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:20]


# The version of the cache entry layout.  Bumped whenever the set of fields the
# reader depends on changes, so an old entry is recomputed instead of being
# half-read.
#
# This is the *layout* version, and it is the only thing here that still has to
# be remembered by hand -- the value version is derived from the code, see
# _method_signature.  2 -> 3: entries carry the method signature they were
# computed with, so a file that somehow lands at a matching path from another
# code version is rejected instead of read.
CACHE_FORMAT = 3

# The fields a cached reference must carry for the rest of the module to work.
# Checked on load, because a truncated or legacy file that happens to parse is
# far worse than a missing one: it would be used, and the numbers in it would
# be silently wrong or absent.
#
# The three geometry fields are required as well.  They are what tells the
# caller *which* molecule the shielding belongs to -- without them a cache
# entry is an anonymous number, and the 0.374 ppm offset on every 17O shift
# was exactly that: a value computed at one water geometry and read as if it
# belonged to another.
#
# ``method`` is required for the same reason one level up: it is what tells the
# caller *which code* produced the number.
_CACHE_REQUIRED = ("key", "label", "basis", "conv_tol", "field_step_au",
                   "natm", "nao", "element_shielding_ppm",
                   "geometry", "geometry_source", "geometry_signature",
                   "method")


def _json_default(obj):
    """Make the payload JSON-serialisable, numpy included.

    ``mol.nao_nr()`` returns ``numpy.int64``, not ``int``.  ``json.dump``
    raises ``TypeError`` on it -- *after* it has already written the key, so
    the file on disk ends mid-object at ``"nao": `` and every later run finds
    a file that exists but does not parse.  That is exactly what happened
    here: five reference files, all truncated at the same byte, and a cache
    that had never once been hit.

    Casting at the point of construction is the primary fix; this is the
    belt-and-braces, because the next numpy scalar added to the payload would
    otherwise break it again in the same silent way.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serialisable")


def _write_cache_atomic(path: str, payload: dict) -> Optional[str]:
    """Write ``payload`` to ``path`` atomically, returning an error or None.

    Atomic because a half-written entry that parses is worse than no entry:
    write to a temporary file in the same directory and rename over the
    target, so a reader sees either the old file or the complete new one.

    The return value exists because the previous version wrapped this in
    ``except Exception: pass``.  A cache that silently never writes costs the
    user the full reference cost -- TMS is 80 minutes on a busy machine -- on
    every single job, and nothing in the output says so.
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return None
    except Exception as exc:                                  # noqa: BLE001
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return f"{type(exc).__name__}: {exc}"


def _read_cache(path: str) -> Optional[dict]:
    """A usable cache entry, or None.

    A truncated file, an entry from an older layout, one missing a field the
    caller needs, or one computed by different code all come back as None,
    which means "recompute" -- never as a half-valid dict.

    The method check is the second of two guards and it is not redundant.  The
    first is the path: the method is in the key, so a foreign entry is normally
    at a different filename and is never opened.  This one covers the case
    where a file is handed to the reader by something that is not the key --
    a directory walk, a repair script, a person -- and it is the difference
    between "the cache is keyed by the code" and "the cache is keyed by the
    code *and* says so when asked".
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(cached, dict):
        return None
    if cached.get("format") != CACHE_FORMAT:
        return None
    for field in _CACHE_REQUIRED:
        if field not in cached:
            return None
    if cached.get("method") != _method_signature():
        return None
    return cached


def cache_inventory(bases: Sequence[str] = ("6-31g*", "sto-3g")) -> dict:
    """Which files in the cache directory the current code can actually read.

    Returns ``{"live": [...], "stale": [...]}`` as absolute paths.  A file is
    live when its name is the path ``reference_cache_path`` produces for some
    known reference at this basis -- i.e. when the engine would find it.

    The rest are not read by anything: an entry from an older key, an older
    format, or a geometry that has since been re-derived.  They are harmless to
    the engine and not harmless to a person.  Two ammonia files were sitting in
    here, one live and one from a superseded geometry, and a probe written to
    ask "did the shipped cache match the code?" paired old and new by the
    ``key`` field inside the payload, compared the orphan, and reported a
    1.15 ppm error that no run had ever produced.  The instrument defect looked
    exactly like the method defect it was looking for.

    So the ambiguity is reported rather than left to be rediscovered: this
    function is what ``check_shift_scales`` and ``gen_nmr_cache`` print.
    """
    import glob

    d = _cache_dir()
    live = set()
    for key in REFERENCES:
        for basis in bases:
            live.add(os.path.abspath(reference_cache_path(key, basis)))
    out = {"live": [], "stale": []}
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        (out["live"] if os.path.abspath(path) in live
         else out["stale"]).append(path)
    return out


def reference_cache_path(key: str, basis: str, conv_tol: float = CONV_TOL,
                         dB: float = FIELD_STEP) -> str:
    """Where a reference's shielding lives on disk.

    One function, because there were two.  ``reference_shieldings`` and
    ``gen_nmr_cache`` each built this path, and the script is the one that
    *writes* the file the engine later *reads*: if they ever disagree, the
    cache is silently never hit and every first job pays five minutes of SCF
    with nothing saying why.  The method signature made that concrete -- it is
    a new term in the key, and two copies of the expression is one more place
    to forget it.
    """
    atoms = reference_geometry(key)
    return os.path.join(
        _cache_dir(),
        _cache_key(key, basis, conv_tol, dB, _geometry_signature(atoms),
                   _method_signature()) + ".json")


def reference_shieldings(key: str, basis: str, conv_tol: float = CONV_TOL,
                         dB: float = FIELD_STEP,
                         progress: Progress = None, use_cache: bool = True,
                         base: int = 0, span: int = 20) -> dict:
    """Absolute shieldings of every element in a reference compound.

    Cached on disk: a reference costs the same seven SCF runs a molecule does,
    and TMS is 17 atoms / 98 basis functions at 6-31G*, so recomputing it for
    every job would dominate the runtime.  The cache key carries the basis,
    the convergence tolerance and the field step, because the shielding moves
    by 0.21 ppm between conv_tol 1e-10 and 1e-11 -- a reference computed at
    the wrong tolerance would silently offset every chemical shift.
    """
    if key not in REFERENCES:
        raise DFTError(f"Unknown NMR reference '{key}'.")

    atoms = reference_geometry(key)
    geom_sig = _geometry_signature(atoms)
    method = _method_signature()
    cpath = reference_cache_path(key, basis, conv_tol, dB)
    if use_cache:
        cached = _read_cache(cpath)
        if cached is not None:
            cached["cached"] = True
            return cached

    info = REFERENCES[key]
    from collections import Counter
    counts = Counter(sym for sym, *_ in atoms)
    # A reference has to be a closed shell with the electron count the formula
    # implies.  Checking it here catches a bad geometry before the SCF does,
    # and it is how the Si(CH)4 mistake above was found.
    expected = sum(_Z(s) for s, *_ in atoms)
    if expected % 2:
        raise DFTError(
            f"The reference {info['label']} has an odd electron count "
            f"({expected}); it cannot be treated with RHF.")

    _tick(progress, base, f"Reference {info['label']} ({basis})")
    mol = _build_mol(_xyz(atoms), basis, 0, 1)
    check_memory(mol.nao_nr(), basis)
    ops = build_operators(mol)
    sigma, diag = shielding_tensor(mol, ops, dB=dB, conv_tol=conv_tol,
                                   progress=progress, base=base, span=span)

    out: Dict[str, object] = {
        "format": CACHE_FORMAT,
        "key": key,
        "label": info["label"],
        "formula": reference_formula(key),
        # Derived from the coordinates actually used, so it cannot describe a
        # geometry the job did not run at.
        "geometry": reference_geometry_note(key),
        "geometry_source": reference_geometry_source(key),
        "geometry_signature": geom_sig,
        "geometry_documented": info["geometry"],
        # Which code computed this.  In the key, so a different one cannot
        # reach this file at all; recorded here too, so a human reading the
        # entry can see whether it predates the method they are running.
        "method": method,
        "basis": basis,
        "conv_tol": conv_tol,
        "field_step_au": dB,
        # int() is load-bearing: nao_nr() is a numpy.int64, and json.dump
        # cannot serialise one.  See _json_default.
        "natm": int(mol.natm),
        "nao": int(mol.nao_nr()),
        "atoms_per_element": {str(k): int(v) for k, v in counts.items()},
        "element_shielding_ppm": {},
        "dE_dB_max": float(diag["dE_dB_max"]),
        "tensor_asymmetry_max_ppm": float(diag["tensor_asymmetry_max_ppm"]),
        "cached": False,
    }
    for el in info["elements"]:
        idx = [a for a in range(mol.natm) if mol.atom_symbol(a) == el]
        if not idx:
            continue
        vals = [float(np.trace(sigma[a]) / 3.0 * PPM) for a in idx]
        out["element_shielding_ppm"][el] = {
            "sigma_iso_ppm": float(np.mean(vals)),
            "spread_ppm": float(max(vals) - min(vals)),
            "n_equivalent": int(len(idx)),
            "per_atom_ppm": vals,
        }
    if use_cache:
        # Reported, not swallowed.  A cache that never writes turns a five
        # minute reference into an eighty minute one on every job, and the
        # only way the user could tell was the clock.
        err = _write_cache_atomic(cpath, out)
        if err is not None:
            out["cache_error"] = err
    return out


def _Z(symbol: str) -> int:
    """Atomic number, for the electron-count sanity check.

    Deuterium is hydrogen as far as the electron count is concerned, and the
    periodic table in ``elements.py`` does not carry it, so it is special
    cased rather than allowed to raise.
    """
    if symbol in ("D", "T"):
        return 1
    try:
        from . import elements
        return int(elements.symbol_to_z(symbol))
    except Exception:
        return 0


def _xyz(atoms) -> str:
    body = "".join(f"{s} {x:.10f} {y:.10f} {z:.10f}\n" for s, x, y, z in atoms)
    return f"{len(atoms)}\nnmr\n{body}"


# ======================================================================
#  spectrum
# ======================================================================
def group_equivalent(entries: Sequence[dict],
                     tol_ppm: float = 0.05) -> List[dict]:
    """Group nuclei that are equivalent by their computed shift.

    This is a heuristic, and it is labelled as one.  Real equivalence is a
    symmetry property; here two nuclei of the same element count as equivalent
    when their isotropic shifts agree within ``tol_ppm``.  For a molecule
    whose distinct sites are further apart than the tolerance -- which is the
    normal case -- it reproduces the integration a paper prints (3H, 2H, 1H),
    and it is transparent about the one way it can fail: two genuinely
    different sites that happen to coincide.  ``spread_ppm`` is reported for
    every group so a suspicious merge is visible.

    **The merge is anchored, not chained, and the entries are sorted first.**
    The first version walked the entries in the order the atoms happened to be
    listed and compared each one against the group's running MEAN.  That makes
    the answer depend on the input order, which for a paper's integration
    table means the same molecule can print different numbers depending on how
    the file was written.  Measured (probe_nmr22): shifts 0.00 / 0.03 / 0.06 /
    0.09 with tol 0.05 come out as counts (1,3) in 18 of the 24 orderings and
    (2,2) in the other 6.

    Anchoring on the group's first member fixes that, and is the more
    conservative rule: no two nuclei are ever called equivalent when they are
    further apart than the tolerance, which chaining cannot promise.  The sort
    key carries the atom index as a tie-break, so the result is a function of
    the multiset of shifts and nothing else.
    """
    groups: List[dict] = []
    # Filter BEFORE sorting.  A nucleus whose reference could not be computed
    # carries delta_ppm = None, and a sort key of float(None) raises -- which
    # is the one case where the missing-reference path is actually reached, so
    # it is the last place that can afford to crash.
    usable = [e for e in entries if e.get("delta_ppm") is not None]
    ordered = sorted(usable, key=lambda e: (-float(e["delta_ppm"]),
                                            int(e.get("index", 0))))
    for e in ordered:
        placed = False
        for g in groups:
            if g["element"] != e["symbol"]:
                continue
            # the anchor is the group's highest shift, which is the first
            # member because the list is sorted descending
            if abs(g["_anchor"] - float(e["delta_ppm"])) <= tol_ppm:
                g["indices"].append(e["index"])
                g["_vals"].append(float(e["delta_ppm"]))
                g["_sigmas"].append(float(e["sigma_iso_ppm"]))
                g["delta_ppm"] = float(np.mean(g["_vals"]))
                g["sigma_iso_ppm"] = float(np.mean(g["_sigmas"]))
                placed = True
                break
        if not placed:
            groups.append({
                "element": e["symbol"],
                "isotope": e.get("isotope"),
                "indices": [e["index"]],
                "delta_ppm": float(e["delta_ppm"]),
                "sigma_iso_ppm": float(e["sigma_iso_ppm"]),
                "_anchor": float(e["delta_ppm"]),
                "_vals": [float(e["delta_ppm"])],
                "_sigmas": [float(e["sigma_iso_ppm"])],
            })
    out = []
    for g in groups:
        vals = g.pop("_vals")
        g.pop("_sigmas", None)
        g.pop("_anchor", None)
        g["count"] = len(g["indices"])
        g["spread_ppm"] = float(max(vals) - min(vals))
        g["integral"] = len(g["indices"])
        g["tol_ppm"] = float(tol_ppm)
        out.append(g)
    out.sort(key=lambda g: -g["delta_ppm"])
    total = max(1, sum(g["count"] for g in out))
    biggest = max((g["count"] for g in out), default=1)
    for g in out:
        # The height a stick is drawn at.  NMR integration is the one
        # quantitative thing in the figure, so it is a field of the payload
        # and not something the chart invents.
        g["rel_intensity"] = float(g["count"]) / float(max(1, biggest))
        g["fraction"] = float(g["count"]) / float(total)
    return out


def larmor_mhz(element: str, spectrometer_mhz: float) -> float:
    """Larmor frequency of ``element`` on an instrument quoted for 1H.

    A spectrometer is named by its 1H frequency -- "a 400 MHz machine" -- and
    the field that implies is B0 = 400 / gamma(1H).  Every other nucleus
    precesses at gamma(X) * B0, so

        13C at 400 MHz(1H) = 400 * 10.7084 / 42.5775 = 100.6 MHz

    This matters because the ppm width of a line is (linewidth in Hz) /
    (Larmor frequency in MHz).  A 1 Hz line is 0.0025 ppm wide for 1H and
    0.0099 ppm wide for 13C: a factor of four.  Using the 1H frequency for a
    13C spectrum makes every 13C line four times too narrow, and since the
    linewidth is what the grid has to resolve, it also decides how many points
    the spectrum needs.
    """
    g1 = NUCLEI["H"]["gamma"]
    info = NUCLEI.get(element)
    if info is None:
        return float(spectrometer_mhz)
    return float(abs(info["gamma"]) / g1 * float(spectrometer_mhz))


# How many grid points per half-width at half maximum.  A Lorentzian sampled at
# a worst-case offset of half a step is at 1/(1 + (step/2hwhm)^2) of its true
# height, so 5 points per HWHM puts the worst case within 1% -- and the ratio
# between two equal lines within 1% as well.  See probe_nmr24/25.
POINTS_PER_HWHM = 5.0

# Ceiling on the curve, so the JSON payload stays a sensible size.  When the
# resolution requirement exceeds it the drawn line is WIDENED to the narrowest
# width the grid can represent and the payload reports both the requested and
# the drawn width -- rather than drawing a line that is not the one the caption
# names.
MAX_SPECTRUM_POINTS = 30000


def nmr_spectrum(groups: Sequence[dict], linewidth_hz: float = 1.0,
                 spectrometer_mhz: float = 400.0,
                 wmin: Optional[float] = None,
                 wmax: Optional[float] = None,
                 points: int = 2000,
                 isotope: Optional[str] = None,
                 larmor_mhz_: Optional[float] = None,
                 points_per_hwhm: float = POINTS_PER_HWHM,
                 max_points: int = MAX_SPECTRUM_POINTS) -> dict:
    """A Lorentzian envelope over the computed shifts.

    ``linewidth_hz`` is the full width at half maximum in Hz **at the Larmor
    frequency of the nucleus being observed**, which is ``larmor_mhz_`` if
    given and ``spectrometer_mhz`` otherwise.  ``compute`` passes the correct
    per-isotope frequency; the default keeps the function usable on its own.

    The number of points is chosen from the linewidth, not fixed.  A spectrum
    is a sum of Lorentzians, so a grid step larger than the line is not a
    coarse picture of the line, it is a different function: each line is
    sampled somewhere on its own flank, and since the curve is normalised by
    its maximum, two lines that are equal come out at different heights.
    Measured before this was fixed (probe_nmr25): two 1H nuclei, one line
    each, 0.5 ppm apart in a 12 ppm window, drawn with a height ratio between
    1.01 and 1.72 depending only on where they fell relative to the grid.  The
    grid is now built so that the worst-case error is 1% (see
    ``POINTS_PER_HWHM``), and the payload reports the step, the points per
    half-width and whether the cap forced the line to be widened.

    The axis runs from high ppm to low, the way an NMR spectrum is plotted.
    """
    base = {
        "shift_ppm": [], "intensity": [], "sticks": [],
        "linewidth_hz": float(linewidth_hz),
        "linewidth_hz_requested": float(linewidth_hz),
        "linewidth_ppm": None,
        "spectrometer_mhz": float(spectrometer_mhz),
        "larmor_mhz": float(larmor_mhz_ if larmor_mhz_ else spectrometer_mhz),
        "isotope": isotope,
        "points": 0, "points_requested": 0,
        "grid_step_ppm": None, "points_per_hwhm": None,
        "resolved": True, "wmin": None, "wmax": None,
    }
    if not groups:
        return base

    larmor = float(larmor_mhz_ if larmor_mhz_ else spectrometer_mhz)
    hwhm_ppm = max(1e-9, float(linewidth_hz) / max(1e-9, larmor))
    drawn_hz = float(linewidth_hz)

    shifts = [float(g["delta_ppm"]) for g in groups]
    lo = min(shifts) if wmin is None else float(wmin)
    hi = max(shifts) if wmax is None else float(wmax)
    pad = max(0.5, 0.08 * max(1.0, hi - lo))
    lo -= pad
    hi += pad
    window = hi - lo

    # how many points the line needs, and what the cap allows
    need = int(np.ceil(window / max(1e-12, hwhm_ppm / float(points_per_hwhm))))
    n = int(min(max_points, max(int(points), need)))
    resolved = need <= max_points
    if not resolved:
        # widen the line to the narrowest one this grid can draw, and say so
        hwhm_ppm = window * float(points_per_hwhm) / float(max_points)
        drawn_hz = hwhm_ppm * larmor

    grid = np.linspace(lo, hi, n)
    step = float(grid[1] - grid[0]) if n > 1 else float(window)
    y = np.zeros_like(grid)
    for g in groups:
        d = grid - float(g["delta_ppm"])
        y += g["count"] * hwhm_ppm ** 2 / (d ** 2 + hwhm_ppm ** 2)
    peak = float(y.max()) if y.size else 0.0
    if peak > 0:
        y = y / peak

    biggest = max((int(g["count"]) for g in groups), default=1)
    return {
        "shift_ppm": [float(v) for v in grid],
        "intensity": [float(v) for v in y],
        "sticks": [{"delta_ppm": float(g["delta_ppm"]),
                    "count": int(g["count"]),
                    "rel_intensity": float(g["count"]) / float(max(1, biggest)),
                    "element": g["element"],
                    "isotope": g.get("isotope")} for g in groups],
        "linewidth_hz": drawn_hz,
        "linewidth_hz_requested": float(linewidth_hz),
        "linewidth_ppm": float(hwhm_ppm * 2.0),
        "spectrometer_mhz": float(spectrometer_mhz),
        "larmor_mhz": larmor,
        "isotope": isotope,
        "points": int(n),
        "points_requested": int(need),
        "grid_step_ppm": step,
        "points_per_hwhm": float(hwhm_ppm / step) if step else 0.0,
        "resolved": bool(resolved),
        "wmin": float(lo), "wmax": float(hi),
    }


# ======================================================================
#  the job entry point
# ======================================================================
MAX_ATOMS = 20

# The GIAO machinery is O(nao^4) in memory, and the constant is not small.
# The list of what is live at the worst point --
#   real ERIs                    8 bytes per nao^4
#   the three GIAO ERI derivatives 24
#   one pair-swapped direction     8
#   the complex ERIs of a perturbed SCF 16
# -- adds to 56, and for a long time 56 was the number in the estimate.  It is
# about a quarter too low, and it was measured, not guessed, that it is:
# probe_mem1 sampled the peak resident set of ``shielding_tensor`` alone (the
# baseline taken after ``build_operators``, so the operators themselves are not
# in it) for water, ammonia, formaldehyde and ethanol -- nao = 18, 20, 32, 54,
# a 55x range in nao^4 -- and the four points fall on a straight line of slope
# 68.8 bytes per nao^4 with an intercept of +0.2 MB and R^2 = 0.99997.  The
# missing quarter is the temporaries of the einsum that forms each direction,
# which the list above does not count.
#
# The underestimate is not academic, and the same file used to record its
# consequences without noticing the cause: the first TMS reference was
# estimated at 5165 MB against a 6000 MB budget, passed, and then took 4829 s
# instead of 285 on a machine with 1 GB free.  At 68.8 the same nao = 98 is
# 6346 MB -- over the budget, which is exactly why it should have been
# refused.  A guard that under-bids by a quarter will under-refuse, and the
# 1.25x free-memory margin below does not cover a 1.23x error by much.
#
# Rounded up to 69.  At TMS (nao = 98, 6-31G*) that is 6.2 GB; the naive
# implementation that formed the full (3, nao^4) derivative needed three times
# that and thrashed a 15 GB machine for over half an hour without finishing.
# Refusing early with a number the user can act on is much better than that,
# so the estimate is checked before anything is allocated.
# Overridable, because the right budget depends on the machine.
_BYTES_PER_NAO4 = 69

# The budget has to admit the references.  It used not to, and correcting the
# constant is what exposed it: at 56 bytes per nao^4 TMS came to 5165 MB and
# sat 16% under a 6000 MB budget, and at the measured 69 it comes to 6364 MB
# and sits 6% *over*.  The comparison `need > MEMORY_BUDGET_MB` never looks at
# free memory, so that refusal would have fired on a 128 GB machine -- and it
# would only have fired on a *cold* cache, because reference_shieldings reads
# the cache before it reaches check_memory.  Every gate runs with the cache
# warm, so a fresh install would have lost every 1H, 13C and 29Si shift with
# no gate noticing.  Found by probe_mem4 after the constant changed, not by
# the suite.
#
# 7500 MB keeps the same ~18% headroom over the largest reference that 6000 MB
# used to give, and still refuses the sizes that motivated the budget: nao =
# 200 is 110 GB.  nmr_sections asserts that every reference fits, so this
# cannot drift out from under TMS again.
MEMORY_BUDGET_MB = int(os.environ.get("CHATDFT_NMR_MEMORY_MB", "7500"))


def memory_estimate_mb(nao: int) -> float:
    """Peak memory of the shielding machinery, in MB."""
    return _BYTES_PER_NAO4 * (int(nao) ** 4) / 1e6


def available_mb() -> Optional[float]:
    """Physical memory the OS says is free, in MB, or None if it cannot tell.

    This exists because a fixed budget is not enough.  The first TMS reference
    was *estimated* at 5165 MB against a 6000 MB budget -- which passed -- on a
    machine that happened to have 1 GB free at the time, so it swapped and took
    4829 seconds instead of the 285 the same calculation took on an idle
    machine.  The estimate was the low one (see ``_BYTES_PER_NAO4``); at the
    measured constant the same job is 6346 MB and would have been refused
    outright, which is the outcome that was wanted all along.  A refusal that
    names the shortfall is far more useful than an hour of disk thrashing.
    """
    try:
        import ctypes

        class MEMSTATUS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMSTATUS()
        st.dwLength = ctypes.sizeof(MEMSTATUS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st.ullAvailPhys / 1e6
    except Exception:
        return None


# Set CHATDFT_NMR_ALLOW_SWAP=1 to run even when the machine is short of
# physical memory.  The free-memory guard exists because swapping turns a five
# minute reference into an eighty minute one -- but it also means that on a
# busy machine the 1H/13C reference cannot be computed at all, and a shift
# with no reference is not a shift.  Generating the cache once, deliberately,
# is worth an hour of swapping; the default stays strict.
ALLOW_SWAP = os.environ.get("CHATDFT_NMR_ALLOW_SWAP", "") not in ("", "0")


def check_memory(nao: int, basis: str) -> float:
    """Refuse a calculation that would not fit, with the numbers in the error."""
    need = memory_estimate_mb(nao)
    if need > MEMORY_BUDGET_MB:
        raise DFTError(
            f"NMR at {basis} needs {need / 1024:.1f} GB for this molecule "
            f"({nao} basis functions) and the budget is "
            f"{MEMORY_BUDGET_MB / 1024:.1f} GB. The GIAO two-electron "
            "derivative is O(nao^4), so the fix is a smaller basis or a "
            "smaller molecule. Raise the budget with CHATDFT_NMR_MEMORY_MB "
            "if the machine has the RAM.")
    free = available_mb()
    if free is not None and free < need * 1.25 and not ALLOW_SWAP:
        raise DFTError(
            f"NMR at {basis} needs about {need / 1024:.1f} GB for this "
            f"molecule ({nao} basis functions) but only "
            f"{free / 1024:.1f} GB of physical memory is free right now. "
            "Running it anyway would swap, which turns a five-minute "
            "calculation into an hour. Free some memory, or use a smaller "
            "basis. To generate a reference cache deliberately, accept the "
            "wait with CHATDFT_NMR_ALLOW_SWAP=1.")
    return need


def compute(atom_xyz: str, basis: str = "6-31g*", charge: int = 0,
            multiplicity: int = 1, functional: str = "b3lyp",
            progress: Progress = None, linewidth_hz: float = 1.0,
            spectrometer_mhz: float = 400.0, use_cache: bool = True,
            conv_tol: float = CONV_TOL, dB: float = FIELD_STEP,
            equivalence_tol_ppm: float = 0.05) -> dict:
    """Absolute shieldings, chemical shifts and a spectrum for one molecule."""
    if multiplicity != 1:
        raise DFTError(
            f"NMR shielding is implemented for closed shells only; this "
            f"molecule has multiplicity {multiplicity}. The GIAO formulation "
            "for open shells needs a spin-resolved response that this build "
            "cannot run (see engine/nmr.py).")
    t0 = time.time()
    _tick(progress, 2, "Building basis")
    mol = _build_mol(atom_xyz, basis, charge, multiplicity)
    if mol.natm > MAX_ATOMS:
        raise DFTError(
            f"NMR needs seven SCF runs; {mol.natm} atoms in {basis} is too "
            f"heavy for an interactive session (limit {MAX_ATOMS}).")
    check_memory(mol.nao_nr(), basis)
    ops = build_operators(mol)

    sigma, diag = shielding_tensor(mol, ops, dB=dB, conv_tol=conv_tol,
                                   progress=progress, base=5, span=75)

    syms = [mol.atom_symbol(a) for a in range(mol.natm)]
    entries: List[dict] = []
    active: Dict[str, int] = {}
    for a in range(mol.natm):
        sym = syms[a]
        iso = NUCLEI.get(sym)
        tensor_ppm = sigma[a] * PPM
        inv = tensor_invariants(tensor_ppm)
        entry = {
            "index": a,
            "symbol": sym,
            "nmr_active": iso is not None,
            "isotope": (iso or {}).get("isotope"),
            "spin": (iso or {}).get("spin"),
            "abundance": (iso or {}).get("abundance"),
            "quadrupolar": bool((iso or {}).get("spin", 0.5) > 0.5),
            "gamma_mhz_per_t": (iso or {}).get("gamma"),
            "sigma_iso_ppm": inv["sigma_iso_ppm"],
            "tensor_ppm": [[float(v) for v in row] for row in tensor_ppm],
            "tensor_asymmetry_ppm": float(diag["tensor_asymmetry_ppm"][a]),
            "delta_ppm": None,
            "reference": None,
        }
        entry.update(inv)
        entries.append(entry)
        if iso is not None:
            active[sym] = active.get(sym, 0) + 1

    # ---- reference shieldings for every active element -----------------
    needed = sorted({s for s in syms if s in NUCLEI and s in REFERENCE_FOR})
    refs: Dict[str, dict] = {}
    warnings: List[str] = []
    for i, el in enumerate(needed):
        key = REFERENCE_FOR[el]
        try:
            refs[el] = reference_shieldings(
                key, basis, conv_tol=conv_tol, dB=dB, use_cache=use_cache,
                progress=progress, base=80 + int(18 * i / max(1, len(needed))),
                span=max(1, int(18 / max(1, len(needed)))))
        except DFTError as exc:
            warnings.append(f"No reference for {el}: {exc}")
    _tick(progress, 99, "Reference shieldings")

    for e in entries:
        el = e["symbol"]
        ref = refs.get(el)
        if ref is None:
            continue
        info = ref["element_shielding_ppm"].get(el)
        if info is None:
            continue
        # The reference is computed and reported either way: it is the scale
        # the element is on, and hiding it would hide the reason.  What the
        # accuracy ledger decides is whether a SHIFT is reported.  A reference
        # compound makes a shift definable; only a measurement makes it
        # correct, and for 77Se the measurement says the difference against
        # (CH3)2Se is not a chemical shift.
        usable = scale_usable(el)
        if usable:
            e["delta_ppm"] = float(info["sigma_iso_ppm"] - e["sigma_iso_ppm"])
        else:
            e["delta_ppm"] = None
            e["delta_unavailable_reason"] = (
                f"the reference {ref['label']} was computed, but this build "
                f"does not reproduce the {SCALE_CHECK[el]['nucleus']} scale, "
                "so the difference is not a chemical shift")
        e["shift_usable"] = usable
        e["reference"] = ref["label"]
        e["reference_geometry"] = ref["geometry"]
        e["reference_sigma_ppm"] = float(info["sigma_iso_ppm"])
        e["reference_cached"] = bool(ref.get("cached"))
        # Present and None rather than absent: a caller that checks for the
        # key and finds it missing cannot tell "this nucleus is on the IUPAC
        # scale" from "nobody thought about the scale".
        e["reference_scale_note"] = scale_note(el)
        # How well the method reproduces this element's scale -- None where it
        # has been measured and reproduced, a sentence where it has been
        # measured and is off, and a sentence where it has never been measured.
        e["accuracy_note"] = accuracy_note(el)

    groups = group_equivalent(entries, tol_ppm=equivalence_tol_ppm)

    # ---- one spectrum per isotope ---------------------------------------
    # An NMR spectrum is acquired for ONE nucleus.  A 1H window is about
    # 12 ppm wide and a 13C window about 220, so a single axis carrying both
    # is not a spectrum anybody can compare with a measurement -- and the
    # previous version did exactly that, with one linewidth in Hz at the 1H
    # Larmor frequency, which additionally made every 13C line four times too
    # narrow.  Each isotope now gets its own axis and its own Larmor
    # frequency.
    by_iso: Dict[str, List[dict]] = {}
    for g in groups:
        by_iso.setdefault(str(g.get("isotope") or g["element"]), []).append(g)
    spectra: Dict[str, dict] = {}
    for iso, gs in sorted(by_iso.items()):
        el = gs[0]["element"]
        spectra[iso] = nmr_spectrum(
            gs, linewidth_hz=linewidth_hz, spectrometer_mhz=spectrometer_mhz,
            isotope=iso, larmor_mhz_=larmor_mhz(el, spectrometer_mhz))
    # The one the panel opens on: 1H if the molecule has any, because that is
    # what a chemist asks for first, otherwise the richest spectrum.
    if "1H" in spectra:
        default_iso: Optional[str] = "1H"
    elif spectra:
        default_iso = max(spectra, key=lambda k: (len(by_iso[k]), k))
    else:
        default_iso = None

    # ---- warnings the user has to see ---------------------------------
    for iso, sp in spectra.items():
        if not sp.get("resolved", True):
            warnings.append(
                f"The {iso} curve was drawn with a line of "
                f"{sp['linewidth_hz']:.3f} Hz rather than the requested "
                f"{sp['linewidth_hz_requested']:.3f} Hz: resolving the "
                f"requested width over this window would need "
                f"{sp['points_requested']} points and the cap is "
                f"{MAX_SPECTRUM_POINTS}. The drawn width is the one in the "
                "figure.")
    # Which nuclei are on a scale that is not the IUPAC primary one.  Naming
    # the reference compound is not enough on its own: a 15N shift against NH3
    # and the same shift against nitromethane differ by 380 ppm, and a user
    # who compares ours with a paper that used the standard will read that
    # whole gap as an error in the calculation.
    for el in sorted(refs):
        note = scale_note(el)
        if note:
            warnings.append(note)
    # How well the method reproduces each scale, for the nuclei where it has
    # been measured and is off.  15N is 24% high and 19F 22% low: both still
    # produce a shift, and the error is stated next to it rather than left for
    # the user to discover by comparing with a paper.  The nuclei that have
    # never been measured are carried per nucleus and in the references block
    # instead of here -- a warning on every job that contains oxygen is a
    # warning nobody reads, and the panel shows the note where the number is.
    for el in sorted(refs):
        entry = SCALE_CHECK.get(el)
        if entry is None or not scale_usable(el):
            continue
        if entry.get("note"):
            warnings.append(str(entry["note"]))
    # Nuclei whose reference was computed but whose scale this method does not
    # reproduce.  They are a *different* case from the unreferenced ones below
    # and are reported separately: the compound is known and computed, and the
    # reason there is no shift is a measurement about the method.
    unusable = sorted(el for el in refs if not scale_usable(el))
    if unusable:
        why = "; ".join(
            f"{el}: {SCALE_CHECK[el]['note']}" for el in unusable
            if el in SCALE_CHECK)
        warnings.append(
            f"The reference compound for {', '.join(unusable)} was computed and "
            "is reported, but this build does not reproduce that nucleus's "
            "scale, so those nuclei have an absolute shielding and NO chemical "
            "shift, and they are absent from the spectrum. The shift is not "
            f"zero -- it is unavailable. Why -- {why}")
    # Which nuclei have no chemical shift, and why.  The set used to be built
    # as
    #     {s for s in syms if s in NUCLEI and s in REFERENCE_FOR} - set(refs)
    # and the `s in REFERENCE_FOR` term excludes exactly the elements that have
    # no reference *defined* -- which is the whole point of the field.
    # Fourteen NMR-active elements have none in this build (Al, B, Br, Cl, D,
    # Hg, I, Li, Na, Pb, Pt, S, Se, Sn), so thiophene's sulfur was dropped
    # without a word and the list came back empty on the one molecule where it
    # should not have been.  `refs` only ever holds elements whose reference
    # was computed, so the subtraction needs no extra filter.
    missing = sorted(set(active) - set(refs))
    undefined = [s for s in missing if s not in REFERENCE_FOR]
    failed = [s for s in missing if s in REFERENCE_FOR]
    if undefined:
        # The reason goes in the warning, not just the name.  A user who is
        # told "no reference for S" will reasonably assume it is an oversight;
        # a user told why can decide whether to switch basis, switch molecule,
        # or convert from the literature.  Every element in this branch has an
        # entry in NO_REFERENCE_WHY and nmr_sections fails if one does not.
        why = "; ".join(f"{el}: {NO_REFERENCE_WHY[el]}" for el in undefined
                        if el in NO_REFERENCE_WHY)
        warnings.append(
            f"No reference compound is defined for {', '.join(undefined)} in "
            "this build, so those nuclei have an absolute shielding but NO "
            "chemical shift, and they are absent from the spectrum. The "
            "shifts are not zero -- they are unavailable, and quoting the "
            "absolute shielding as if it were a shift would be wrong by the "
            "whole scale (33S spans 964 ppm, from -290 to 674). Why each is "
            f"missing -- {why}.")
    if failed:
        warnings.append(
            f"A reference compound is defined for {', '.join(failed)} but "
            "could not be computed, so those nuclei have an absolute "
            "shielding but NO chemical shift, and they are absent from the "
            "spectrum. The shifts are not zero -- they are unavailable.")
    warnings.append(
        "The shielding is computed at the HARTREE-FOCK level. The requested "
        f"functional ({functional}) is not used for it: this build's DFT "
        "integrator cannot take the complex density the GIAO response needs, "
        "and a real-arithmetic substitute does not converge. Absolute "
        "shieldings at RHF/6-31G* are accurate to a few ppm, and chemical "
        "shifts to about 2 ppm for hydrocarbons -- but a carbon bearing a "
        "lone pair (methanol, carbonyls) can be off by 20 ppm at this level.")
    if diag["dE_dB_max"] > 1e-6:
        warnings.append(
            "The dE/dB consistency check did not come out at zero "
            f"({diag['dE_dB_max']:.2e}); a closed shell has no linear Zeeman "
            "term, so a non-zero value means the perturbation convention is "
            "inconsistent and the shieldings should not be trusted.")
    if not all(r["converged"] for r in diag["scf_runs"]):
        warnings.append(
            "At least one field-perturbed SCF did not converge; the affected "
            "tensor elements are unreliable.")
    if diag["tensor_asymmetry_max_ppm"] > 0.01:
        warnings.append(
            "The shielding tensor was symmetrised: the raw tensor's "
            f"antisymmetric part reached {diag['tensor_asymmetry_max_ppm']:.3f} "
            "ppm. The isotropic shielding and every chemical shift are "
            "unaffected (the trace is invariant under transposition), but "
            "treat the reported anisotropy of the affected nuclei with care.")
    if not refs:
        warnings.append(
            "No reference compound could be computed, so only absolute "
            "shieldings are available. Chemical shifts need a reference at "
            "the same level of theory.")

    result = {
        "job_type": "nmr",
        "level": {
            "scf": "RHF",
            "functional_requested": functional,
            "functional_used_for_shielding": None,
            "basis": basis,
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "conv_tol": conv_tol,
            "field_step_au": dB,
            "scale": {"alpha": ALPHA, "alpha_squared_1e6": PPM,
                      "formula": "sigma[ppm] = sigma[au] * alpha^2 * 1e6"},
        },
        "natoms": int(mol.natm),
        "nao": int(mol.nao_nr()),
        "nuclei": entries,
        "active_elements": sorted(active),
        "unreferenced_elements": missing,
        # A reference was computed and the shift is still not reported, because
        # the method does not reproduce the scale.  Separate from the list
        # above because the cause is different and so is what a user should do
        # about it: an unreferenced nucleus needs a reference compound, this
        # one needs a different level of theory.
        "scale_unusable_elements": sorted(
            el for el in refs if not scale_usable(el)),
        "references": {el: {
            "key": REFERENCE_FOR.get(el),
            "label": r["label"],
            "formula": r["formula"],
            "geometry": r["geometry"],
            "basis": r["basis"],
            "sigma_iso_ppm": r["element_shielding_ppm"][el]["sigma_iso_ppm"],
            "n_equivalent": r["element_shielding_ppm"][el]["n_equivalent"],
            "spread_ppm": r["element_shielding_ppm"][el]["spread_ppm"],
            "cached": bool(r.get("cached")),
            # Which scale this element's shifts are on, or None when it is
            # the IUPAC primary one.  It sits here as well as on every nucleus
            # so the panel can put it beside the reference compound's name,
            # which is where a user looks when a number disagrees with a
            # paper -- and because a field that nothing reads is a field that
            # rots.
            "scale_note": scale_note(el),
            "accuracy_note": accuracy_note(el),
            "shift_usable": scale_usable(el),
        } for el, r in refs.items() if el in r["element_shielding_ppm"]},
        "groups": groups,
        "groups_by_isotope": {iso: gs for iso, gs in sorted(by_iso.items())},
        # ``spectrum`` is kept for the single-isotope case and for callers that
        # predate the split; ``spectra`` is the real answer.
        "spectra": spectra,
        "spectrum": (spectra.get(default_iso) if default_iso
                     else nmr_spectrum([], linewidth_hz=linewidth_hz,
                                       spectrometer_mhz=spectrometer_mhz)),
        "spectrum_isotope": default_iso,
        "spectrum_isotopes": sorted(spectra),
        "diagnostics": {
            "dE_dB": diag["dE_dB"],
            "dE_dB_max": diag["dE_dB_max"],
            "tensor_asymmetry_max_ppm": diag["tensor_asymmetry_max_ppm"],
            "scf_runs": diag["scf_runs"],
            "unperturbed_energy_hartree": diag["unperturbed_energy"],
        },
        "warnings": warnings,
        "elapsed_s": round(time.time() - t0, 2),
    }
    return result
