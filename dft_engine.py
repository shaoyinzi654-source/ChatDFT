import numpy as np
from scipy.linalg import eigh

def solve_1d_dft(Vext_fn, num_electrons, L=10.0, N=200, max_iter=100, tol=1e-6, alpha=0.2, softening=1.0, functional="LDA", mixing_method="Linear"):
    """
    Solves the 1D Kohn-Sham DFT self-consistently.
    - Vext_fn: A function that takes a grid array 'x' and returns Vext(x).
    - functional: "Hartree", "Exchange-Only", "LDA", or "GGA-PBE".
    - mixing_method: "Linear" or "Anderson".
    """
    # 1. Grid setup
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    
    # 2. Kinetic energy operator (Three-point finite difference)
    # T = -0.5 * d^2/dx^2
    T = np.zeros((N, N))
    for i in range(N):
        T[i, i] = 1.0 / (dx ** 2)
        if i > 0:
            T[i, i - 1] = -0.5 / (dx ** 2)
        if i < N - 1:
            T[i, i + 1] = -0.5 / (dx ** 2)
            
    # 3. External potential
    Vext = Vext_fn(x)
    
    # 4. Electron-electron interaction matrix (soft Coulomb)
    X_i, X_j = np.meshgrid(x, x, indexing='ij')
    V_ee_mat = 1.0 / np.sqrt((X_i - X_j) ** 2 + softening ** 2)
    
    # 5. Helper functions for Exchange-Correlation
    cx = (3.0 / np.pi) ** (1.0 / 3.0)
    
    def get_Vxc(rho):
        if functional == "Hartree":
            return np.zeros_like(rho)
            
        rho_clipped = np.clip(rho, 1e-15, None)
        Vx = -cx * (rho_clipped ** (1.0 / 3.0))
        
        # LDA correlation (Wigner)
        Vc = -0.058 * (rho_clipped ** 2 + 24.3 * rho_clipped) / ((rho_clipped + 12.15) ** 2)
        
        if functional == "Exchange-Only":
            return Vx
        elif functional == "LDA":
            return Vx + Vc
        elif functional == "GGA-PBE":
            # Gradient correction in 1D (Becke 88 / PBE-like)
            # Smooth gradient with 3-point averaging to reduce grid noise
            drho_raw = np.gradient(rho_clipped, dx)
            drho = np.abs(drho_raw)
            # Pad-average to reduce boundary oscillations
            drho_smooth = np.convolve(drho, [0.25, 0.5, 0.25], mode='same')
            
            # Reduced gradient s, clamped to avoid divergence in low-density tails
            kF = (3.0 * (np.pi ** 2)) ** (1.0 / 3.0)
            s = drho_smooth / (2.0 * kF * (rho_clipped ** (4.0 / 3.0)) + 1e-10)
            s = np.clip(s, 0.0, 10.0)  # cap s to prevent divergence
            
            # Becke 88 exchange enhancement, beta=0.006
            beta = 0.006
            arcsinh_s = np.arcsinh(s)
            denom = 1.0 + 6.0 * beta * s * arcsinh_s + 1e-10
            corr_factor = np.clip((beta * s ** 2) / denom, 0.0, 1.0)  # cap enhancement
            Vx_gga = Vx * (1.0 + corr_factor)
            
            return Vx_gga + Vc
        else:
            return Vx + Vc
            
    def get_Exc_density(rho):
        if functional == "Hartree":
            return np.zeros_like(rho)
            
        rho_clipped = np.clip(rho, 1e-15, None)
        eps_x = -0.75 * cx * (rho_clipped ** (1.0 / 3.0))
        eps_c = -0.058 * rho_clipped / (rho_clipped + 12.15)
        
        if functional == "Exchange-Only":
            return eps_x
        elif functional == "LDA":
            return eps_x + eps_c
        elif functional == "GGA-PBE":
            drho_raw = np.gradient(rho_clipped, dx)
            drho_smooth = np.abs(np.convolve(np.abs(drho_raw), [0.25, 0.5, 0.25], mode='same'))
            kF = (3.0 * (np.pi ** 2)) ** (1.0 / 3.0)
            s = np.clip(drho_smooth / (2.0 * kF * (rho_clipped ** (4.0 / 3.0)) + 1e-10), 0.0, 10.0)
            beta = 0.006
            arcsinh_s = np.arcsinh(s)
            corr_factor = np.clip((beta * s ** 2) / (1.0 + 6.0 * beta * s * arcsinh_s + 1e-10), 0.0, 1.0)
            eps_x_gga = eps_x * (1.0 + corr_factor)
            return eps_x_gga + eps_c
        else:
            return eps_x + eps_c
            
    # 6. Initial guess
    Veff = Vext.copy()
    rho = np.zeros(N)
    rho_old = np.ones(N) * (num_electrons / (2.0 * L))
    
    # Lists for Anderson mixing
    rho_in_hist = []
    f_hist = []
    
    history = []
    converged = False
    
    # 7. SCF Loop
    for iteration in range(max_iter):
        H = T + np.diag(Veff)
        eigenvalues, eigenvectors = eigh(H)
        orbitals = eigenvectors / np.sqrt(dx)
        
        new_rho = np.zeros(N)
        remaining = num_electrons
        occupations = np.zeros(N)
        
        for i in range(N):
            if remaining <= 0:
                break
            occ = min(2.0, remaining)
            occupations[i] = occ
            new_rho += occ * (orbitals[:, i] ** 2)
            remaining -= occ
            
        # Density mixing
        if mixing_method == "Anderson":
            # Residual
            f_k = new_rho - rho_old
            rho_in_hist.append(rho_old.copy())
            f_hist.append(f_k.copy())
            
            # Keep only last M=3 steps
            if len(rho_in_hist) > 3:
                rho_in_hist.pop(0)
                f_hist.pop(0)
                
            m = len(rho_in_hist)
            if m == 1:
                rho = alpha * new_rho + (1.0 - alpha) * rho_old
            else:
                # Solve Anderson coefficient minimization
                A = np.zeros((m-1, m-1))
                b = np.zeros(m-1)
                df = [f_hist[i] - f_hist[-1] for i in range(m-1)]
                f_m = f_hist[-1]
                
                for i in range(m-1):
                    for j in range(m-1):
                        A[i, j] = np.sum(df[i] * df[j]) * dx
                    b[i] = np.sum(f_m * df[i]) * dx
                    
                try:
                    theta = np.linalg.solve(A + 1e-8 * np.eye(m-1), -b)
                    all_theta = list(theta) + [1.0 - sum(theta)]
                except np.linalg.LinAlgError:
                    all_theta = [0.0] * (m-1) + [1.0]
                    
                rho_mixed = np.zeros_like(rho_old)
                f_mixed = np.zeros_like(rho_old)
                for i in range(m):
                    rho_mixed += all_theta[i] * rho_in_hist[i]
                    f_mixed += all_theta[i] * f_hist[i]
                    
                rho = rho_mixed + alpha * f_mixed
        else:
            # Linear Mixing
            rho = alpha * new_rho + (1.0 - alpha) * rho_old
            
        # Check convergence
        density_diff = np.sum(np.abs(rho - rho_old)) * dx
        
        # Calculate Hartree potential
        VH = V_ee_mat.dot(rho) * dx
        
        # Exchange correlation potential
        Vxc = get_Vxc(rho)
        
        # Update Effective Potential
        Veff = Vext + VH + Vxc
        
        # Calculate Energy components
        E_kin = sum(occupations[i] * np.dot(eigenvectors[:, i], T.dot(eigenvectors[:, i])) for i in range(N) if occupations[i] > 0)
        E_ext = np.sum(Vext * rho) * dx
        E_H = 0.5 * np.sum(VH * rho) * dx
        E_xc = np.sum(get_Exc_density(rho) * rho) * dx
        E_tot = E_kin + E_ext + E_H + E_xc
        
        history.append(E_tot)
        
        # Dual convergence criterion:
        # 1) Density residual < tol (primary), OR
        # 2) Energy change < tol*0.01 for last 5 steps (energy-converged, e.g. near-degenerate periodic systems)
        if density_diff < tol:
            converged = True
            break
        if len(history) >= 6:
            recent_E_changes = [abs(history[-k] - history[-k-1]) for k in range(1, 6)]
            if max(recent_E_changes) < tol * 0.01:
                converged = True
                break

            
        rho_old = rho.copy()
        
    # Get final outputs
    potentials = {
        'Vext': Vext,
        'VH': VH,
        'Vxc': Vxc,
        'Veff': Veff
    }
    
    return {
        'x': x,
        'density': rho,
        'orbitals': orbitals[:, :int(np.ceil(num_electrons / 2))],
        'eigenvalues': eigenvalues[:int(np.ceil(num_electrons / 2))],
        'potentials': potentials,
        'energies': {
            'E_kin': E_kin,
            'E_ext': E_ext,
            'E_H': E_H,
            'E_xc': E_xc,
            'E_tot': E_tot
        },
        'iterations': len(history),
        'converged': converged,
        'history': history
    }

def solve_1d_periodic_dft(Vext_fn, num_electrons, L=10.0, N=100, max_iter=100, tol=1e-6, alpha=0.2, softening=1.0, functional="LDA", nkpoints=21):
    """
    Solves the 1D Periodic Kohn-Sham DFT self-consistently.
    - nkpoints: Number of k-points in the 1st Brillouin zone.
    """
    # 1. Grid setup for one unit cell: x in [-L/2, L/2 - dx]
    dx = L / N
    x = np.linspace(-L/2.0, L/2.0 - dx, N)
    
    # 2. k-point sampling in 1st BZ: k in [-pi/L, pi/L]
    k_points = np.linspace(-np.pi / L, np.pi / L, nkpoints)
    
    # 3. Periodic Coulomb interaction (Hartree matrix with 5 periodic images)
    V_ee_mat = np.zeros((N, N))
    for m in range(-5, 6):
        X_i, X_j = np.meshgrid(x, x, indexing='ij')
        V_ee_mat += 1.0 / np.sqrt((X_i - X_j - m * L) ** 2 + softening ** 2)
        
    # 4. External potential
    Vext = Vext_fn(x)
    
    # Helper functions for Exchange-Correlation (same as 1D DFT)
    cx = (3.0 / np.pi) ** (1.0 / 3.0)
    
    def get_Vxc(rho):
        rho_clipped = np.clip(rho, 1e-15, None)
        Vx = -cx * (rho_clipped ** (1.0 / 3.0))
        Vc = -0.058 * (rho_clipped ** 2 + 24.3 * rho_clipped) / ((rho_clipped + 12.15) ** 2)
        if functional == "Exchange-Only":
            return Vx
        elif functional == "LDA":
            return Vx + Vc
        elif functional == "GGA-PBE":
            drho = np.abs(np.gradient(rho_clipped, dx))
            drho_smooth = np.convolve(drho, [0.25, 0.5, 0.25], mode='same')
            kF = (3.0 * (np.pi ** 2)) ** (1.0 / 3.0)
            s = np.clip(drho_smooth / (2.0 * kF * (rho_clipped ** (4.0 / 3.0)) + 1e-10), 0.0, 10.0)
            beta = 0.006
            arcsinh_s = np.arcsinh(s)
            corr_factor = np.clip((beta * s ** 2) / (1.0 + 6.0 * beta * s * arcsinh_s + 1e-10), 0.0, 1.0)
            return Vx * (1.0 + corr_factor) + Vc
        else:
            return np.zeros_like(rho)
            
    def get_Exc_density(rho):
        rho_clipped = np.clip(rho, 1e-15, None)
        eps_x = -0.75 * cx * (rho_clipped ** (1.0 / 3.0))
        eps_c = -0.058 * rho_clipped / (rho_clipped + 12.15)
        if functional == "Exchange-Only":
            return eps_x
        elif functional == "LDA":
            return eps_x + eps_c
        elif functional == "GGA-PBE":
            drho = np.gradient(rho_clipped, dx)
            drho_smooth = np.abs(np.convolve(np.abs(drho), [0.25, 0.5, 0.25], mode='same'))
            kF = (3.0 * (np.pi ** 2)) ** (1.0 / 3.0)
            s = np.clip(drho_smooth / (2.0 * kF * (rho_clipped ** (4.0 / 3.0)) + 1e-10), 0.0, 10.0)
            beta = 0.006
            arcsinh_s = np.arcsinh(s)
            corr_factor = np.clip((beta * s ** 2) / (1.0 + 6.0 * beta * s * arcsinh_s + 1e-10), 0.0, 1.0)
            return eps_x * (1.0 + corr_factor) + eps_c
        else:
            return np.zeros_like(rho)

    # 5. Initial Guess
    Veff = Vext.copy()
    rho = np.ones(N) * (num_electrons / L)
    converged = False
    history = []
    
    # 6. SCF Loop
    for iteration in range(max_iter):
        k_eigenvalues = []
        k_eigenvectors = []
        
        for k in k_points:
            H = np.zeros((N, N), dtype=complex)
            
            # Diagonal terms: kinetic (1/dx^2) + potential
            for i in range(N):
                H[i, i] = 1.0 / (dx ** 2) + Veff[i]
                
            # Off-diagonal kinetic terms with periodic boundary conditions
            for i in range(N):
                j1 = (i - 1) % N
                H[i, j1] += -0.5 / (dx ** 2) * (np.exp(-1j * k * L) if i == 0 else 1.0)
                
                j2 = (i + 1) % N
                H[i, j2] += -0.5 / (dx ** 2) * (np.exp(1j * k * L) if i == N - 1 else 1.0)
                
            # Diagonalize complex Hermitian matrix
            eps, C = np.linalg.eigh(H)
            k_eigenvalues.append(eps)
            k_eigenvectors.append(C)
            
        k_eigenvalues = np.array(k_eigenvalues) # shape (nkpoints, N)
        k_eigenvectors = np.array(k_eigenvectors) # shape (nkpoints, N, N)
        
        # Determine filling of bands by finding the Fermi level (chemical potential)
        all_states = []
        for ik in range(nkpoints):
            for ib in range(N):
                all_states.append((k_eigenvalues[ik, ib], ik, ib))
        all_states.sort(key=lambda x: x[0])
        
        occupations = np.zeros((nkpoints, N))
        remaining = num_electrons
        for energy, ik, ib in all_states:
            if remaining <= 0:
                break
            occ = min(2.0, remaining)
            occupations[ik, ib] = occ
            remaining -= occ
            
        # Reconstruct new density
        new_rho = np.zeros(N)
        for ik in range(nkpoints):
            for ib in range(N):
                occ = occupations[ik, ib]
                if occ > 0:
                    new_rho += (occ / nkpoints) * (np.abs(k_eigenvectors[ik, :, ib]) ** 2) / dx
                    
        # Check convergence
        density_diff = np.sum(np.abs(new_rho - rho)) * dx
        rho_old = rho.copy()
        
        # Linear mixing
        rho = alpha * new_rho + (1.0 - alpha) * rho_old
        
        # Calculate potentials
        VH = V_ee_mat.dot(rho) * dx
        Vxc = get_Vxc(rho)
        Veff = Vext + VH + Vxc
        
        # Calculate periodic energies
        E_kin = 0.0
        for ik in range(nkpoints):
            for ib in range(N):
                occ = occupations[ik, ib]
                if occ > 0:
                    vec = k_eigenvectors[ik, :, ib]
                    eps_val = k_eigenvalues[ik, ib]
                    v_expect = np.sum(np.abs(vec) ** 2 * Veff)
                    E_kin += (occ / nkpoints) * (eps_val - v_expect)
                    
        E_ext = np.sum(Vext * rho) * dx
        E_H = 0.5 * np.sum(VH * rho) * dx
        E_xc = np.sum(get_Exc_density(rho) * rho) * dx
        E_tot = E_kin + E_ext + E_H + E_xc
        
        history.append(E_tot)
        
        if density_diff < tol:
            converged = True
            break
            
        if len(history) >= 6:
            recent_E_changes = [abs(history[-m] - history[-m-1]) for m in range(1, 6)]
            if max(recent_E_changes) < tol * 0.01:
                converged = True
                break
                
    return {
        'converged': converged,
        'iterations': len(history),
        'history': history,
        'x': x,
        'density': rho,
        'k_points': k_points,
        'bands': k_eigenvalues.T, # shape (N, nkpoints)
        'occupations': occupations.T, # shape (N, nkpoints)
        'energies': {
            'E_tot': E_tot,
            'E_kin': E_kin,
            'E_ext': E_ext,
            'E_H': E_H,
            'E_xc': E_xc
        }
    }

