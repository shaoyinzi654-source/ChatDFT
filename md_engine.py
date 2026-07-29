import numpy as np

class MolecularDynamics:
    """
    3D Classical Molecular Dynamics (MD) engine for multi-molecular systems.
    Uses LAMMPS 'metal' units:
      - Distance: Angstrom (A)
      - Time: picosecond (ps)
      - Energy: electron-volt (eV)
      - Mass: atomic mass unit (AMU)
      - Temperature: Kelvin (K)
      - Force: eV/A
    """
    def __init__(self, box_length=15.0, temp=300.0, dt=0.001):
        self.L = box_length
        self.T_target = temp
        self.dt = dt  # in ps (typically 0.001 ps = 1 fs)
        self.conversion_factor = 9648.53  # converts (eV / (A * AMU)) to A/ps^2
        
        self.positions = None
        self.velocities = None
        self.forces = None
        self.masses = None
        self.names = None
        self.bonds = []  # list of tuples (i, j, r0, kb)
        
    def init_argon_box(self, n_atoms=64):
        """Initializes a box of Argon atoms on a cubic grid with Maxwell-Boltzmann velocities."""
        self.names = ["Ar"] * n_atoms
        self.masses = np.ones(n_atoms) * 39.948
        
        # Grid layout
        n_side = int(np.ceil(n_atoms ** (1.0 / 3.0)))
        spacing = self.L / n_side
        
        pos = []
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_side):
                    if len(pos) < n_atoms:
                        pos.append([i * spacing + spacing/2, j * spacing + spacing/2, k * spacing + spacing/2])
        self.positions = np.array(pos)
        
        # Initialize velocities (Maxwell-Boltzmann)
        # k_B = 8.61733e-5 eV/K. In our units, std = sqrt(k_B * T / mass)
        kB = 8.61733e-5
        std_vel = np.sqrt(kB * self.T_target / 39.948) * 100.0  # std in A/ps (approx. std_vel)
        # Standard velocity scaling factor: std_vel = sqrt(kB * T / m)
        # Since std_vel in A/ps: kB in eV/K, T in K, m in AMU.
        # Let's check: kB/m in eV / (K * AMU). Conversion to (A/ps)^2 is kB * conversion_factor
        std_vel = np.sqrt(kB * self.T_target * self.conversion_factor / 39.948)
        
        self.velocities = np.random.normal(0.0, std_vel, (n_atoms, 3))
        # Remove net momentum
        self.velocities -= np.mean(self.velocities, axis=0)
        
        self.forces = np.zeros_like(self.positions)
        self.bonds = []

    def init_nitrogen_box(self, n_molecules=32):
        """Initializes a box of Nitrogen (N2) molecules with flexible harmonic bonds."""
        n_atoms = n_molecules * 2
        self.names = ["N"] * n_atoms
        self.masses = np.ones(n_atoms) * 14.007
        
        # Grid layout for molecule centers
        n_side = int(np.ceil(n_molecules ** (1.0 / 3.0)))
        spacing = self.L / n_side
        
        pos = []
        bond_length = 1.10  # Nitrogen bond length in A
        self.bonds = []
        
        m_idx = 0
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_side):
                    if m_idx < n_molecules:
                        cx = i * spacing + spacing/2
                        cy = j * spacing + spacing/2
                        cz = k * spacing + spacing/2
                        
                        # Diatomic orientation (random vector)
                        theta = np.random.uniform(0, np.pi)
                        phi = np.random.uniform(0, 2*np.pi)
                        dx = np.sin(theta) * np.cos(phi) * (bond_length / 2)
                        dy = np.sin(theta) * np.sin(phi) * (bond_length / 2)
                        dz = np.cos(theta) * (bond_length / 2)
                        
                        pos.append([cx - dx, cy - dy, cz - dz])
                        pos.append([cx + dx, cy + dy, cz + dz])
                        
                        # Add harmonic bond: kb = 140 eV/A^2 (strong bond)
                        self.bonds.append((2*m_idx, 2*m_idx + 1, bond_length, 140.0))
                        m_idx += 1
                        
        self.positions = np.array(pos)
        
        # Velocities
        kB = 8.61733e-5
        std_vel = np.sqrt(kB * self.T_target * self.conversion_factor / 14.007)
        self.velocities = np.random.normal(0.0, std_vel, (n_atoms, 3))
        self.velocities -= np.mean(self.velocities, axis=0)
        self.forces = np.zeros_like(self.positions)

    def compute_forces(self):
        """Computes forces using Lennard-Jones potential and harmonic bonds with periodic boundary conditions."""
        n_atoms = len(self.positions)
        self.forces = np.zeros_like(self.positions)
        pot_energy = 0.0
        
        # 1. Non-bonded Lennard-Jones forces (pairwise)
        # Parameters for N and Ar:
        # Argon: epsilon = 0.0103 eV, sigma = 3.40 A
        # Nitrogen: epsilon = 0.0032 eV, sigma = 3.31 A
        is_argon = self.names[0] == "Ar"
        epsilon = 0.0103 if is_argon else 0.0032
        sigma = 3.40 if is_argon else 3.31
        
        sig_sq = sigma ** 2
        rcut = 2.5 * sigma
        rcut_sq = rcut ** 2
        
        # Minimum image convention pairwise loop
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # Check if bonded to skip non-bonded force (1-2 exclusion)
                is_bonded = False
                for bi, bj, _, _ in self.bonds:
                    if (bi == i and bj == j) or (bi == j and bj == i):
                        is_bonded = True
                        break
                
                # Pairwise distance vector
                dr = self.positions[i] - self.positions[j]
                # Periodic image convention
                dr -= np.round(dr / self.L) * self.L
                
                r_sq = np.sum(dr ** 2)
                if r_sq < rcut_sq:
                    r_inv_sq = 1.0 / r_sq
                    sig_r_sq = sig_sq * r_inv_sq
                    s6 = sig_r_sq ** 3
                    s12 = s6 ** 2
                    
                    # Force magnitude: F = 24*eps/r^2 * (2*s12 - s6)
                    f_mag = (24.0 * epsilon * r_inv_sq) * (2.0 * s12 - s6)
                    self.forces[i] += f_mag * dr
                    self.forces[j] -= f_mag * dr
                    
                    # Potential energy (shifted to be 0 at rcut)
                    s6_cut = (sig_sq / rcut_sq) ** 3
                    s12_cut = s6_cut ** 2
                    v_cut = 4.0 * epsilon * (s12_cut - s6_cut)
                    pot_energy += 4.0 * epsilon * (s12 - s6) - v_cut
                    
        # 2. Bonded forces (harmonic bonds)
        for i, j, r0, kb in self.bonds:
            dr = self.positions[i] - self.positions[j]
            dr -= np.round(dr / self.L) * self.L
            r = np.linalg.norm(dr)
            
            if r > 1e-5:
                # V = 0.5 * kb * (r - r0)^2
                # F = -kb * (r - r0) * dr / r
                f_mag = -kb * (r - r0) / r
                self.forces[i] += f_mag * dr
                self.forces[j] -= f_mag * dr
                pot_energy += 0.5 * kb * (r - r0) ** 2
                
        return pot_energy

    def integrate_verlet_step1(self):
        """First part of Velocity Verlet integration: update positions and half-update velocities."""
        acc = (self.forces / self.masses[:, np.newaxis]) * self.conversion_factor
        self.positions += self.velocities * self.dt + 0.5 * acc * (self.dt ** 2)
        # Apply periodic boundary conditions (wrap positions back into [0, L])
        self.positions = np.mod(self.positions, self.L)
        # Half-step velocity update
        self.velocities += 0.5 * acc * self.dt

    def integrate_verlet_step2(self):
        """Second part of Velocity Verlet integration: update velocities using new forces."""
        acc = (self.forces / self.masses[:, np.newaxis]) * self.conversion_factor
        self.velocities += 0.5 * acc * self.dt

    def compute_kinetic_energy(self):
        """Computes total kinetic energy and temperature of the system."""
        n_atoms = len(self.positions)
        # KE = sum 0.5 * mass * vel^2. In metal units, kinetic energy is:
        # KE = 0.5 * mass * vel^2 / conversion_factor (to get eV)
        ke = 0.5 * np.sum(self.masses[:, np.newaxis] * (self.velocities ** 2)) / self.conversion_factor
        
        # Temperature: T = 2 * KE / (3 * N * k_B)
        # k_B = 8.61733e-5 eV/K. Degrees of freedom = 3 * N - 3 (removing net translation)
        kB = 8.61733e-5
        dof = 3 * n_atoms - 3
        temp = (2.0 * ke) / (dof * kB)
        return ke, temp

    def apply_thermostat(self, current_temp):
        """Simple Velocity Rescaling Thermostat to keep temperature near target."""
        if current_temp > 1e-3:
            factor = np.sqrt(self.T_target / current_temp)
            # Gentle rescaling (Berendsen-like scaling parameter lambda=0.1)
            scale = 1.0 + 0.1 * (factor - 1.0)
            self.velocities *= scale

    def calculate_rdf(self, dr_bin=0.1, r_max=6.0):
        """Calculates Radial Distribution Function g(r) for structural analysis."""
        n_atoms = len(self.positions)
        n_bins = int(r_max / dr_bin)
        hist = np.zeros(n_bins)
        
        # Compute all pairwise distances
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dr = self.positions[i] - self.positions[j]
                dr -= np.round(dr / self.L) * self.L
                r = np.linalg.norm(dr)
                if r < r_max:
                    bin_idx = int(r / dr_bin)
                    if bin_idx < n_bins:
                        hist[bin_idx] += 2  # count both i-j and j-i
                        
        # Normalize g(r)
        # g(r) = hist(r) / (density * shell_volume * N)
        rho_density = n_atoms / (self.L ** 3)
        r_vals = (np.arange(n_bins) + 0.5) * dr_bin
        
        g_r = np.zeros_like(r_vals)
        for i in range(n_bins):
            r_inner = i * dr_bin
            r_outer = (i + 1) * dr_bin
            shell_vol = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
            expected_pairs = rho_density * shell_vol * n_atoms
            g_r[i] = hist[i] / expected_pairs if expected_pairs > 0 else 0.0
            
        return r_vals, g_r
