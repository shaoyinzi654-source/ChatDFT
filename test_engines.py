import numpy as np
from dft_engine import solve_1d_dft
from diatomic_engine import solve_diatomic_scf

def test_1d():
    print("Testing 1D DFT Solver...")
    # Harmonic oscillator potential: Vext = 0.5 * k * x^2
    k = 1.0
    Vext_fn = lambda x: 0.5 * k * (x ** 2)
    
    result = solve_1d_dft(Vext_fn, num_electrons=2, L=5.0, N=100, max_iter=50, tol=1e-6, alpha=0.3)
    print(f"1D DFT Converged: {result['converged']} in {result['iterations']} iterations")
    print(f"Total Energy: {result['energies']['E_tot']:.6f} Hartree")
    print(f"Energy components: {result['energies']}")
    print("------------------------------------------")

def test_3d():
    print("Testing 3D STO-3G Diatomic SCF Solver...")
    # H2 molecule at R = 1.4 Bohr
    R = 1.4
    atom1_pos = [0.0, 0.0, -R/2]
    atom2_pos = [0.0, 0.0, R/2]
    
    result = solve_diatomic_scf('H', atom1_pos, 'H', atom2_pos, num_electrons=2, max_iter=50, tol=1e-6)
    print(f"3D SCF H2 Converged: {result['converged']} in {result['iterations']} iterations")
    print(f"Total Energy: {result['E_tot']:.6f} Hartree")
    print(f"Electronic Energy: {result['E_elec']:.6f} Hartree")
    print(f"Nuclear Repulsion: {result['E_nuc']:.6f} Hartree")
    print(f"Energy components: Kin={result['E_kin']:.6f}, Ext={result['E_ext']:.6f}, EE={result['E_ee']:.6f}")
    
    # HeH+ molecule at R = 1.46 Bohr
    print("\nTesting HeH+ ion...")
    result_heh = solve_diatomic_scf('HE', [0.0, 0.0, 0.0], 'H', [0.0, 0.0, 1.46], num_electrons=2, max_iter=50, tol=1e-6)
    print(f"3D SCF HeH+ Converged: {result_heh['converged']} in {result_heh['iterations']} iterations")
    print(f"Total Energy: {result_heh['E_tot']:.6f} Hartree")
    print(f"Electronic Energy: {result_heh['E_elec']:.6f} Hartree")
    print(f"Nuclear Repulsion: {result_heh['E_nuc']:.6f} Hartree")
    print(f"Energy components: Kin={result_heh['E_kin']:.6f}, Ext={result_heh['E_ext']:.6f}, EE={result_heh['E_ee']:.6f}")
    print("------------------------------------------")

if __name__ == "__main__":
    test_1d()
    test_3d()
