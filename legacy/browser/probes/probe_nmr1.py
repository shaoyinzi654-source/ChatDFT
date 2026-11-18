"""probe_nmr1.py -- what does this PySCF build actually give us for NMR?

pyscf.prop does not exist in this build, so GIAO NMR has to be written from
the definitions.  Before writing anything, find out which of the magnetic
integrals libcint actually exposes here, and with what shape and convention.

The discipline of the last few rounds: measure the contract, do not assume it.
"""
import numpy as np

import backend.bootstrap
backend.bootstrap.setup()

from pyscf import gto, scf, dft

mol = gto.M(
    atom=[("O", 0.0, 0.0, 0.0), ("H", 0.0, -0.757, 0.587), ("H", 0.0, 0.757, 0.587)],
    basis="sto-3g", unit="Angstrom", verbose=0)

print("=" * 72)
print("  libcint integrals relevant to a magnetic field / GIAO")
print("=" * 72)

candidates = [
    # plain angular momentum and magnetic-dipole operators
    "int1e_r", "int1e_r2", "int1e_cg_irxp", "int1e_irp", "int1e_rxp",
    "int1e_prinvxp",
    # GIAO (gauge-including) one-electron integrals
    "int1e_igovlp", "int1e_igkin", "int1e_ignuc", "int1e_igirjxp",
    "int1e_govlp", "int1e_ig1ovlp",
    # Zeeman / field-derivative of the Fock pieces
    "int1e_irp", "int1e_ipovlp",
    # spin-orbit style
    "int1e_pnucxp",
]
ok = {}
for name in candidates:
    try:
        v = mol.intor(name)
        ok[name] = v.shape
        print(f"  OK    {name:20s} shape={v.shape}")
    except Exception as exc:
        msg = str(exc).split("\n")[0][:70]
        print(f"  --    {name:20s} {type(exc).__name__}: {msg}")

print()
print("=" * 72)
print("  second-derivative / derivative integrals (cint1e_)")
print("=" * 72)
for name in ["cint1e_igovlp_sph", "cint1e_igkin_sph", "cint1e_ignuc_sph",
             "cint1e_igirjxp_sph", "cint1e_irp_sph", "cint1e_rxp_sph"]:
    try:
        v = mol.intor(name)
        print(f"  OK    {name:24s} shape={v.shape}")
    except Exception as exc:
        print(f"  --    {name:24s} {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")

print()
print("=" * 72)
print("  is there a GIAO-capable SCF / density-fitting path?")
print("=" * 72)
for mod, attr in [("pyscf.scf.cphf", "solve"),
                  ("pyscf.scf._response_functions", "_gen_rhf_response"),
                  ("pyscf.dft.numint", "nr_rks_fxc"),
                  ("pyscf.gto.mole", "intor"),
                  ("pyscf.scf.rohf", "ROHF"),
                  ("pyscf.scf.uhf", "UHF")]:
    try:
        m = __import__(mod, fromlist=[attr])
        print(f"  OK    {mod}.{attr} -> {hasattr(m, attr)}")
    except Exception as exc:
        print(f"  --    {mod}.{attr} {type(exc).__name__}")

print()
print("=" * 72)
print("  sanity: does int1e_r really return r in bohr?  (He at 1 Angstrom)")
print("=" * 72)
he = gto.M(atom=[("He", 0.0, 0.0, 1.0)], basis="sto-3g", unit="Angstrom", verbose=0)
r = he.intor("int1e_r")
print(f"  <z> = {r[2, 0, 0]:.8f}  (expect 1.8897261246 = 1 Angstrom in bohr)")

print()
print("=" * 72)
print("  magnetic-dipole integral: int1e_irp should give <mu| r x p |nu>")
print("  check its antisymmetry and that it is purely imaginary-like real")
print("=" * 72)
if "int1e_irp" in ok or True:
    for name in ["int1e_irp", "int1e_cg_irxp", "int1e_rxp"]:
        try:
            v = mol.intor(name)
            print(f"  {name:16s} shape={v.shape}  "
                  f"max|asym|={np.max(np.abs(v + np.transpose(v, (0, 2, 1)))):.3e}  "
                  f"max|sym|={np.max(np.abs(v - np.transpose(v, (0, 2, 1)))):.3e}")
        except Exception as exc:
            print(f"  {name:16s} {type(exc).__name__}")
