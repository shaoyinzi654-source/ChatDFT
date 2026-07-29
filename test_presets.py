from dft_engine import solve_1d_dft
import numpy as np

def v_ssh(x):
    return (-1.8/np.sqrt((x+9)**2+0.5) - 0.9/np.sqrt((x+6)**2+0.5)
           - 1.8/np.sqrt((x+3)**2+0.5) - 0.9/np.sqrt((x)**2+0.5)
           - 1.8/np.sqrt((x-3)**2+0.5) - 0.9/np.sqrt((x-6)**2+0.5)
           - 1.8/np.sqrt((x-9)**2+0.5))

r = solve_1d_dft(v_ssh, num_electrons=8, L=15.0, N=300, max_iter=120,
    tol=1e-6, alpha=0.15, softening=1.0, functional='LDA', mixing_method='Anderson')
print("SSH chain: E=", r['energies']['E_tot'], "converged=", r['converged'])

def v_wqd(x):
    return 0.15 * x**2

r2 = solve_1d_dft(v_wqd, num_electrons=8, L=12.0, N=300, max_iter=150,
    tol=1e-7, alpha=0.1, softening=0.5, functional='GGA-PBE', mixing_method='Anderson')
print("Wigner QD: E=", r2['energies']['E_tot'], "converged=", r2['converged'])

# Test Morse preset
def v_morse(x):
    return 2.5 * (1.0 - np.exp(-(x+3.0)))**2 - 2.5

r3 = solve_1d_dft(v_morse, num_electrons=2, L=15.0, N=300, max_iter=120,
    tol=1e-6, alpha=0.2, softening=0.5, functional='Exchange-Only', mixing_method='Anderson')
print("Morse: E=", r3['energies']['E_tot'], "converged=", r3['converged'])

# Test asymmetric DW
def v_adw(x):
    return 0.06*(x**4 - 8*x**2) + 0.3*x

r4 = solve_1d_dft(v_adw, num_electrons=2, L=12.0, N=200, max_iter=100,
    tol=1e-6, alpha=0.15, softening=0.8, functional='LDA', mixing_method='Anderson')
print("Asymmetric DW: E=", r4['energies']['E_tot'], "converged=", r4['converged'])
