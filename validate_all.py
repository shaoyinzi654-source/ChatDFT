"""Full physical correctness validation for all key DFY system functions."""
from diatomic_engine import (
    solve_multi_atom_scf, compute_dipole_moment_multi,
    compute_mulliken_charges, compute_bond_orders
)
from dft_engine import solve_1d_dft
import numpy as np
import sys

PASS = []
FAIL = []

def check(condition, name, details=""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name} -- {details}")
    sys.stdout.flush()

print("=" * 60)
print("TEST 1: H2 (2e, STO-3G RHF)")
atoms_h2 = [{'name':'H','pos':[0,0,-0.7]},{'name':'H','pos':[0,0,0.7]}]
r = solve_multi_atom_scf(atoms_h2, 2)
check(r['converged'], "H2 converged")
check(-1.3 < r['E_tot'] < -0.7, "H2 E_tot range", f"got {r['E_tot']:.4f}")
mu, muD = compute_dipole_moment_multi(atoms_h2, r)
check(abs(muD) < 0.2, "H2 dipole ~ 0 (symmetric)", f"got {muD:.4f}")
mop, bs = compute_bond_orders(atoms_h2, r)
print(f"  H2: E={r['E_tot']:.4f} Ha, dipole={muD:.4f} D, bond_strength={bs[0,1]:.4f}")

print("=" * 60)
print("TEST 2: H2O (10e, STO-3G RHF)")
atoms_w = [{'name':'O','pos':[0,0,0.12]},{'name':'H','pos':[0,1.43,-0.98]},{'name':'H','pos':[0,-1.43,-0.98]}]
rw = solve_multi_atom_scf(atoms_w, 10)
check(rw['converged'], "H2O converged")
check(-100 < rw['E_tot'] < -40, "H2O E_tot range", f"got {rw['E_tot']:.4f}")
muw, muwD = compute_dipole_moment_multi(atoms_w, rw)
check(0.3 < muwD < 6.0, "H2O dipole 0.3~6 D", f"got {muwD:.4f}")
chw = compute_mulliken_charges(atoms_w, rw)
O_q = next(c['net_charge'] for c in chw if c['name'] == 'O')
H_qs = [c['net_charge'] for c in chw if c['name'] == 'H']
print(f"  H2O: E={rw['E_tot']:.4f} Ha, dipole={muwD:.4f} D")
print(f"  Charges: O={O_q:+.4f}, H={H_qs[0]:+.4f},{H_qs[1]:+.4f}")

print("=" * 60)
print("TEST 3: N2 (14e, triple bond)")
atoms_n2 = [{'name':'N','pos':[0,0,-1.04]},{'name':'N','pos':[0,0,1.04]}]
rn = solve_multi_atom_scf(atoms_n2, 14)
check(rn['converged'], "N2 converged")
mun, munD = compute_dipole_moment_multi(atoms_n2, rn)
check(abs(munD) < 0.1, "N2 dipole ~0 (homonuclear)", f"got {munD:.6f}")
print(f"  N2: E={rn['E_tot']:.4f} Ha, dipole={munD:.6e} D (expect ~0)")

print("=" * 60)
print("TEST 4: LiF (12e, ionic)")
atoms_lif = [{'name':'LI','pos':[0,0,-1.55]},{'name':'F','pos':[0,0,1.55]}]
rl = solve_multi_atom_scf(atoms_lif, 12)
check(rl['converged'], "LiF converged")
mul, mulD = compute_dipole_moment_multi(atoms_lif, rl)
check(mulD > 2.0, "LiF dipole large (ionic)", f"got {mulD:.4f}")
chl = compute_mulliken_charges(atoms_lif, rl)
print(f"  LiF: E={rl['E_tot']:.4f} Ha, dipole={mulD:.4f} D")

print("=" * 60)
print("TEST 5: CO2 (22e, linear, dipole=0)")
atoms_co2 = [{'name':'C','pos':[0,0,0]},{'name':'O','pos':[0,0,-2.19]},{'name':'O','pos':[0,0,2.19]}]
rc = solve_multi_atom_scf(atoms_co2, 22)
check(rc['converged'], "CO2 converged")
muc, mucD = compute_dipole_moment_multi(atoms_co2, rc)
check(abs(mucD) < 0.5, "CO2 dipole ~0 (centrosymmetric)", f"got {mucD:.4f}")
print(f"  CO2: E={rc['E_tot']:.4f} Ha, dipole={mucD:.4f} D (expect ~0)")

print("=" * 60)
print("TEST 6: 1D DFT He (2e, LDA)")
def vhe(x): return -2.0 / np.sqrt(x**2 + 1.0)
rhe = solve_1d_dft(vhe, num_electrons=2, L=10.0, N=200, max_iter=100, tol=1e-6, alpha=0.2, softening=1.0, functional="LDA")
check(rhe['converged'], "1D He converged")
ne_integrated = np.sum(rhe['density']) * (20.0 / 200)
check(abs(ne_integrated - 2.0) < 0.2, "1D He density integrates to 2e", f"got {ne_integrated:.4f}")
print(f"  1D He: E={rhe['energies']['E_tot']:.4f} Ha, Ne_int={ne_integrated:.4f}")

print("=" * 60)
print("TEST 7: 1D DFT double-well (2e)")
def vdw(x): return 0.08 * (x**2 - 3.0)**2
rdw = solve_1d_dft(vdw, num_electrons=2, L=10.0, N=200, max_iter=100, tol=1e-6, alpha=0.2, softening=0.5, functional="LDA")
check(rdw['converged'], "Double-well converged")
print(f"  Double-well: E={rdw['energies']['E_tot']:.4f} Ha")

print("=" * 60)
print("TEST 8: 1D DFT LDA Kronig-Penney (8e, Linear mixing)")
def vkp(x): return -2.5 * (np.cos(2*np.pi*x/5.0))**2
rkp = solve_1d_dft(vkp, num_electrons=8, L=15.0, N=300, max_iter=200, tol=1e-6,
    alpha=0.15, softening=0.5, functional="LDA", mixing_method="Linear")
check(rkp['converged'], "Kronig-Penney LDA+Linear converged")
print(f"  K-P: E={rkp['energies']['E_tot']:.4f} Ha, converged={rkp['converged']}")

print()
print("=" * 60)
print(f"SUMMARY: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("FAILED TESTS:", FAIL)
else:
    print("ALL TESTS PASSED!")
