import numpy as np
from dft_engine import solve_1d_periodic_dft

# Setup a Kronig-Penney periodic lattice
# V(x) = -2.5 * cos(2*pi*x / L)^2
L = 6.0
N = 100
num_electrons = 4

def vkp(x):
    return -2.5 * (np.cos(2 * np.pi * x / L))**2

r = solve_1d_periodic_dft(
    vkp, 
    num_electrons=num_electrons, 
    L=L, 
    N=N, 
    max_iter=100, 
    tol=1e-5, 
    alpha=0.15, 
    softening=0.5, 
    functional="LDA", 
    nkpoints=15
)

print("Converged:", r["converged"])
print("Iterations:", r["iterations"])
print("Total Energy:", r["energies"]["E_tot"], "Hartree")
print("Bands shape:", r["bands"].shape) # expected (N, nkpoints) -> (100, 15)
print("Lowest band at k=0 (Gamma point):", r["bands"][0, 7])
print("Lowest band at BZ boundary (k = pi/L):", r["bands"][0, 0])
print("Band dispersion width:", abs(r["bands"][0, 7] - r["bands"][0, 0]), "Hartree")
