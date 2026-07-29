import numpy as np
from scipy.linalg import eigh

# Boys function F0(x)
def BoysF0(x):
    if x < 1e-8:
        return 1.0 - x / 3.0
    return 0.5 * np.sqrt(np.pi / x) * scipy_erf(np.sqrt(x))

# Simple approximation for Error Function to avoid dependency on scipy
def scipy_erf(x):
    sign = np.sign(x)
    x = np.abs(x)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-x * x))
    return sign * y

# Primitive Gaussian integral definitions
def overlap_primitives(a, A, b, B):
    dist_sq = np.sum((A - B) ** 2)
    pre = (np.pi / (a + b)) ** 1.5
    return pre * np.exp(-a * b / (a + b) * dist_sq)

def kinetic_primitives(a, A, b, B):
    dist_sq = np.sum((A - B) ** 2)
    reduced_exp = a * b / (a + b)
    pre = reduced_exp * (3.0 - 2.0 * reduced_exp * dist_sq) * (np.pi / (a + b)) ** 1.5
    return pre * np.exp(-reduced_exp * dist_sq)

def nuclear_attraction_primitives(a, A, b, B, C, Z):
    dist_sq = np.sum((A - B) ** 2)
    gamma = a + b
    P = (a * A + b * B) / gamma
    PC_sq = np.sum((P - C) ** 2)
    
    val = - (2.0 * np.pi * Z / gamma) * np.exp(-a * b / gamma * dist_sq) * BoysF0(gamma * PC_sq)
    return val

def electron_repulsion_primitives(a, A, b, B, c, C, d, D):
    AB_sq = np.sum((A - B) ** 2)
    CD_sq = np.sum((C - D) ** 2)
    
    g1 = a + b
    g2 = c + d
    
    P = (a * A + b * B) / g1
    Q = (c * C + d * D) / g2
    PQ_sq = np.sum((P - Q) ** 2)
    
    val = (2.0 * (np.pi ** 2.5) / (g1 * g2 * np.sqrt(g1 + g2))) * \
          np.exp(-a * b / g1 * AB_sq) * \
          np.exp(-c * d / g2 * CD_sq) * \
          BoysF0((g1 * g2 / (g1 + g2)) * PQ_sq)
    return val

# Contracted orbital definitions (STO-3G fit)
class STO3GOrbital:
    def __init__(self, center, zeta, charge):
        self.center = np.array(center, dtype=float)
        self.zeta = zeta
        self.charge = charge
        
        # Standard coefficients and exponents for STO-3G 1s orbital
        # mapped to zeta=1.0. Exponents scale as alphas = standard_alphas * zeta^2.
        self.alphas = np.array([0.109818, 0.405771, 2.22766]) * (zeta ** 2)
        self.coeffs = np.array([0.444635, 0.535328, 0.154329])

# Contracted integral helper functions
def overlap_contracted(o1, o2):
    val = 0.0
    for i in range(3):
        for j in range(3):
            val += o1.coeffs[i] * o2.coeffs[j] * overlap_primitives(o1.alphas[i], o1.center, o2.alphas[j], o2.center)
    return val

def kinetic_contracted(o1, o2):
    val = 0.0
    for i in range(3):
        for j in range(3):
            val += o1.coeffs[i] * o2.coeffs[j] * kinetic_primitives(o1.alphas[i], o1.center, o2.alphas[j], o2.center)
    return val

def nuclear_attraction_contracted(o1, o2, C, Z):
    val = 0.0
    for i in range(3):
        for j in range(3):
            val += o1.coeffs[i] * o2.coeffs[j] * nuclear_attraction_primitives(o1.alphas[i], o1.center, o2.alphas[j], o2.center, C, Z)
    return val

def electron_repulsion_contracted(o1, o2, o3, o4):
    val = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    val += o1.coeffs[i] * o2.coeffs[j] * o3.coeffs[k] * o4.coeffs[l] * \
                           electron_repulsion_primitives(o1.alphas[i], o1.center,
                                                         o2.alphas[j], o2.center,
                                                         o3.alphas[k], o3.center,
                                                         o4.alphas[l], o4.center)
    return val

# Extended periodic table shells mapping (up to Magnesium)
def get_element_orbitals(name, pos):
    """
    Returns a list of normalized STO3GOrbital objects representing the atomic shells,
    along with the true nuclear charge Z.
    """
    ELEMENT_SHELLS = {
        'H':  ([1.24], 1.0),
        'HE': ([1.69], 2.0),
        'LI': ([2.69, 0.65], 3.0),
        'BE': ([3.68, 0.975], 4.0),
        'B':  ([4.68, 1.30, 0.5], 5.0),
        'C':  ([5.67, 1.72, 0.6], 6.0),
        'N':  ([6.67, 1.95, 0.7, 0.3], 7.0),
        'O':  ([7.66, 2.225, 0.8, 0.35], 8.0),
        'F':  ([8.65, 2.60, 0.9, 0.4, 0.2], 9.0),
        'NE': ([9.64, 2.925, 1.0, 0.45, 0.25], 10.0),
        'NA': ([10.63, 3.25, 1.1, 0.5, 0.3, 0.15], 11.0),
        'MG': ([11.62, 3.575, 1.2, 0.55, 0.35, 0.2], 12.0),
        'G':  ([1.0], 0.0)
    }
    nm = name.upper()
    if nm not in ELEMENT_SHELLS:
        raise ValueError(f"Unsupported atom name: {name}. Supported: H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, G")
        
    zetas, Z = ELEMENT_SHELLS[nm]
    orbs = []
    for zeta in zetas:
        orbs.append(STO3GOrbital(pos, zeta, Z))
    return orbs, Z

def solve_multi_atom_scf(atoms, num_electrons, max_iter=100, tol=1e-6, multiplicity=None):
    """
    Solves the 3D Unrestricted Hartree-Fock (UHF) SCF for an arbitrary multi-atom system in 3D.
    Supports any spin multiplicity (multiplicity = 2S + 1).
    - atoms: List of dicts, e.g. [{"name": "O", "pos": [0,0,0]}, {"name": "H", "pos": [0,1,1]}, ...]
    """
    basis = []
    nuclei = []
    for item in atoms:
        name = item["name"]
        pos = item["pos"]
        o_list, Z = get_element_orbitals(name, pos)
        basis.extend(o_list)
        nuclei.append((np.array(pos, dtype=float), Z))
        
    nbasis = len(basis)
    
    # Normalize basis functions
    for p in range(nbasis):
        spp = 0.0
        for i in range(3):
            for j in range(3):
                spp += basis[p].coeffs[i] * basis[p].coeffs[j] * ((np.pi / (basis[p].alphas[i] + basis[p].alphas[j])) ** 1.5)
        basis[p].coeffs /= np.sqrt(spp)
        
    # Build matrices
    S = np.zeros((nbasis, nbasis))
    T = np.zeros((nbasis, nbasis))
    V = np.zeros((nbasis, nbasis))
    
    for p in range(nbasis):
        for q in range(nbasis):
            S[p, q] = overlap_contracted(basis[p], basis[q])
            T[p, q] = kinetic_contracted(basis[p], basis[q])
            for C, Z in nuclei:
                V[p, q] += nuclear_attraction_contracted(basis[p], basis[q], C, Z)
                
    Hcore = T + V
    
    ERI = np.zeros((nbasis, nbasis, nbasis, nbasis))
    for p in range(nbasis):
        for q in range(nbasis):
            for r in range(nbasis):
                for s in range(nbasis):
                    ERI[p, q, r, s] = electron_repulsion_contracted(basis[p], basis[q], basis[r], basis[s])
                    
    # Orthogonalize basis
    s_val, s_vec = np.linalg.eigh(S)
    s_val = np.clip(s_val, 1e-12, None)
    X = np.dot(s_vec, np.dot(np.diag(1.0 / np.sqrt(s_val)), s_vec.T))
    
    # Determine spin population
    if multiplicity is None:
        multiplicity = 1 if (num_electrons % 2 == 0) else 2
        
    # multiplicity = 2S + 1 => S = (multiplicity - 1) / 2
    # n_alpha - n_beta = 2S = multiplicity - 1
    # n_alpha + n_beta = num_electrons
    n_alpha = (num_electrons + multiplicity - 1) // 2
    n_beta = num_electrons - n_alpha
    
    if n_alpha < 0 or n_beta < 0 or (num_electrons + multiplicity - 1) % 2 != 0:
        raise ValueError(f"电子数 ({num_electrons}) 与自旋多重度 ({multiplicity}) 物理上不兼容！")
        
    if n_alpha > nbasis or n_beta > nbasis:
        raise ValueError(f"泡利不相容限值：当前体系基组仅包含 {nbasis} 个空间轨道，无法容纳 {max(n_alpha, n_beta)} 个相同自旋方向的电子！请降低多重度或添加更多原子。")
        
    P_alpha = np.zeros((nbasis, nbasis))
    P_beta = np.zeros((nbasis, nbasis))
    
    # Initial guess from Hcore diagonalization
    F_prime = np.dot(X.T, np.dot(Hcore, X))
    eps_guess, C_prime_guess = np.linalg.eigh(F_prime)
    C_guess = np.dot(X, C_prime_guess)
    
    for i in range(n_alpha):
        P_alpha += np.outer(C_guess[:, i], C_guess[:, i])
    for i in range(n_beta):
        P_beta += np.outer(C_guess[:, i], C_guess[:, i])
        
    history = []
    converged = False
    
    # DIIS history lists
    diis_F_alpha = []
    diis_F_beta = []
    diis_e_alpha = []
    diis_e_beta = []
    max_diis = 6
    
    # Compute nuclear repulsion energy
    E_nuc = 0.0
    num_nuclei = len(nuclei)
    for i in range(num_nuclei):
        for j in range(i+1, num_nuclei):
            C_i, Z_i = nuclei[i]
            C_j, Z_j = nuclei[j]
            dist = np.linalg.norm(C_i - C_j)
            if dist > 1e-6:
                E_nuc += (Z_i * Z_j) / dist
                
    for iteration in range(max_iter):
        G_alpha = np.zeros((nbasis, nbasis))
        G_beta = np.zeros((nbasis, nbasis))
        P = P_alpha + P_beta
        
        # Build UHF Fock matrices:
        # G^alpha_pq = sum_rs P_rs (pq|rs) - sum_rs P^alpha_rs (ps|rq)
        # G^beta_pq  = sum_rs P_rs (pq|rs) - sum_rs P^beta_rs  (ps|rq)
        for p in range(nbasis):
            for q in range(nbasis):
                for r in range(nbasis):
                    for s in range(nbasis):
                        val_J = P[r, s] * ERI[p, q, r, s]
                        G_alpha[p, q] += val_J - P_alpha[r, s] * ERI[p, s, r, q]
                        G_beta[p, q] += val_J - P_beta[r, s] * ERI[p, s, r, q]
                        
        F_alpha = Hcore + G_alpha
        F_beta = Hcore + G_beta
        
        # Compute DIIS error matrices: e = F @ P @ S - S @ P @ F
        e_alpha = F_alpha @ P_alpha @ S - S @ P_alpha @ F_alpha
        e_beta = F_beta @ P_beta @ S - S @ P_beta @ F_beta
        max_e = max(np.max(np.abs(e_alpha)), np.max(np.abs(e_beta)))
        
        # Append to DIIS history
        diis_F_alpha.append(F_alpha.copy())
        diis_F_beta.append(F_beta.copy())
        diis_e_alpha.append(e_alpha.copy())
        diis_e_beta.append(e_beta.copy())
        if len(diis_F_alpha) > max_diis:
            diis_F_alpha.pop(0)
            diis_F_beta.pop(0)
            diis_e_alpha.pop(0)
            diis_e_beta.pop(0)
            
        # Determine whether to run DIIS extrapolation
        # DIIS is typically turned on once the error norm is reasonably small (e.g., < 0.2 a.u.)
        diis_active = False
        F_alpha_eval = F_alpha.copy()
        F_beta_eval = F_beta.copy()
        
        if max_e < 0.2 and len(diis_F_alpha) >= 2:
            n_diis = len(diis_F_alpha)
            B = np.zeros((n_diis + 1, n_diis + 1))
            for r in range(n_diis):
                for s in range(n_diis):
                    B[r, s] = np.sum(diis_e_alpha[r] * diis_e_alpha[s]) + np.sum(diis_e_beta[r] * diis_e_beta[s])
            B[:-1, -1] = -1.0
            B[-1, :-1] = -1.0
            B[-1, -1] = 0.0
            
            rhs = np.zeros(n_diis + 1)
            rhs[-1] = -1.0
            
            try:
                c_coeffs = np.linalg.solve(B + 1e-10 * np.eye(n_diis + 1), rhs)[:-1]
                F_alpha_eval = sum(c_coeffs[i] * diis_F_alpha[i] for i in range(n_diis))
                F_beta_eval = sum(c_coeffs[i] * diis_F_beta[i] for i in range(n_diis))
                diis_active = True
            except np.linalg.LinAlgError:
                pass # fallback to F_alpha_eval = F_alpha
                
        # Diagonalize F_alpha
        F_prime_alpha = np.dot(X.T, np.dot(F_alpha_eval, X))
        eps_alpha, C_prime_alpha = np.linalg.eigh(F_prime_alpha)
        C_alpha = np.dot(X, C_prime_alpha)
        
        # Diagonalize F_beta
        F_prime_beta = np.dot(X.T, np.dot(F_beta_eval, X))
        eps_beta, C_prime_beta = np.linalg.eigh(F_prime_beta)
        C_beta = np.dot(X, C_prime_beta)
        
        # Construct new density matrices
        P_alpha_new = np.zeros((nbasis, nbasis))
        P_beta_new = np.zeros((nbasis, nbasis))
        for i in range(n_alpha):
            P_alpha_new += np.outer(C_alpha[:, i], C_alpha[:, i])
        for i in range(n_beta):
            P_beta_new += np.outer(C_beta[:, i], C_beta[:, i])
            
        E_elec = 0.5 * (np.sum(P_alpha_new * (Hcore + F_alpha)) + np.sum(P_beta_new * (Hcore + F_beta)))
        E_tot = E_elec + E_nuc
        
        diff = np.sqrt(np.sum((P_alpha_new - P_alpha) ** 2) + np.sum((P_beta_new - P_beta) ** 2))
        history.append(E_tot)
        
        if diff < tol:
            converged = True
            P_alpha = P_alpha_new.copy()
            P_beta = P_beta_new.copy()
            break
            
        if diis_active:
            # Extrapolated density is directly taken for quadratic convergence
            P_alpha = P_alpha_new.copy()
            P_beta = P_beta_new.copy()
        else:
            # Standard adaptive damping
            mix_frac = min(0.3 + iteration * 0.02, 0.7)
            P_alpha = mix_frac * P_alpha_new + (1.0 - mix_frac) * P_alpha
            P_beta = mix_frac * P_beta_new + (1.0 - mix_frac) * P_beta
        
    # Recompute final energies self-consistently
    P = P_alpha + P_beta
    G_alpha_final = np.zeros((nbasis, nbasis))
    G_beta_final = np.zeros((nbasis, nbasis))
    for p in range(nbasis):
        for q in range(nbasis):
            for r in range(nbasis):
                for s in range(nbasis):
                    val_J = P[r, s] * ERI[p, q, r, s]
                    G_alpha_final[p, q] += val_J - P_alpha[r, s] * ERI[p, s, r, q]
                    G_beta_final[p, q] += val_J - P_beta[r, s] * ERI[p, s, r, q]
                    
    F_alpha_final = Hcore + G_alpha_final
    F_beta_final = Hcore + G_beta_final
    
    E_elec = 0.5 * (np.sum(P_alpha * (Hcore + F_alpha_final)) + np.sum(P_beta * (Hcore + F_beta_final)))
    E_tot = E_elec + E_nuc
    E_kin = np.sum(P * T)
    E_ext = np.sum(P * V)
    E_ee = E_elec - E_kin - E_ext
    
    return {
        'converged': converged,
        'iterations': len(history),
        'history': history,
        'E_tot': E_tot,
        'E_elec': E_elec,
        'E_nuc': E_nuc,
        'E_kin': E_kin,
        'E_ext': E_ext,
        'E_ee': E_ee,
        'C': C_alpha,
        'eps': eps_alpha,
        'C_alpha': C_alpha,
        'C_beta': C_beta,
        'eps_alpha': eps_alpha,
        'eps_beta': eps_beta,
        'P_alpha': P_alpha,
        'P_beta': P_beta,
        'P': P,
        'S': S,
        'basis': basis,
        'nuclei': nuclei,
        'n_alpha': n_alpha,
        'n_beta': n_beta,
        'multiplicity': multiplicity
    }

def solve_diatomic_scf(atom1_name, atom1_pos, atom2_name, atom2_pos, num_electrons, max_iter=100, tol=1e-6, multiplicity=None):
    """
    Diatomic wrapper for solve_multi_atom_scf to maintain backward compatibility.
    """
    atoms = [
        {"name": atom1_name, "pos": atom1_pos},
        {"name": atom2_name, "pos": atom2_pos}
    ]
    return solve_multi_atom_scf(atoms, num_electrons, max_iter, tol, multiplicity)


def compute_dipole_moment_3d(atom1_name, atom1_pos, atom2_name, atom2_pos, res):
    """
    Computes the molecular dipole moment along the Z-axis (in a.u. and Debye) for 3D systems.
    """
    o1_list, Z1 = get_element_orbitals(atom1_name, atom1_pos)
    o2_list, Z2 = get_element_orbitals(atom2_name, atom2_pos)
    basis = o1_list + o2_list
    nbasis = len(basis)
    
    # Normalize basis
    for p in range(nbasis):
        spp = 0.0
        for i in range(3):
            for j in range(3):
                spp += basis[p].coeffs[i] * basis[p].coeffs[j] * ((np.pi / (basis[p].alphas[i] + basis[p].alphas[j])) ** 1.5)
        basis[p].coeffs /= np.sqrt(spp)
        
    p1 = np.array(atom1_pos, dtype=float)
    p2 = np.array(atom2_pos, dtype=float)
    
    # Nuclear dipole
    mu_nuc = Z1 * p1[2] + Z2 * p2[2]
    
    # Dipole matrix
    D = np.zeros((nbasis, nbasis))
    for p in range(nbasis):
        for q in range(nbasis):
            val = 0.0
            for i in range(3):
                for j in range(3):
                    a = basis[p].alphas[i]
                    b = basis[q].alphas[j]
                    cp = basis[p].coeffs[i]
                    cq = basis[q].coeffs[j]
                    
                    dist_sq = np.sum((basis[p].center - basis[q].center)**2)
                    S_prim = ((np.pi / (a + b)) ** 1.5) * np.exp(-a * b / (a + b) * dist_sq)
                    Pz = (a * basis[p].center[2] + b * basis[q].center[2]) / (a + b)
                    
                    val += cp * cq * S_prim * Pz
            D[p, q] = val
            
    P = res['P']
    mu_elec = np.sum(P * D)
    
    mu_tot = mu_nuc - mu_elec
    mu_debye = mu_tot * 2.541746
    
    return mu_tot, mu_debye

def compute_dipole_moment_multi(atoms, res):
    """
    Computes the molecular dipole vector (x, y, z) and its magnitude for any multi-atomic system.
    """
    basis = res['basis']
    nbasis = len(basis)
    P = res['P']
    
    # Nuclear dipole vector
    mu_nuc = np.zeros(3)
    for name, pos in [(item["name"], item["pos"]) for item in atoms]:
        _, Z = get_element_orbitals(name, pos)
        mu_nuc += Z * np.array(pos, dtype=float)
        
    # Dipole matrix for X, Y, Z
    Dx = np.zeros((nbasis, nbasis))
    Dy = np.zeros((nbasis, nbasis))
    Dz = np.zeros((nbasis, nbasis))
    
    for p in range(nbasis):
        for q in range(nbasis):
            val_x = val_y = val_z = 0.0
            for i in range(3):
                for j in range(3):
                    a = basis[p].alphas[i]
                    b = basis[q].alphas[j]
                    cp = basis[p].coeffs[i]
                    cq = basis[q].coeffs[j]
                    
                    dist_sq = np.sum((basis[p].center - basis[q].center)**2)
                    S_prim = ((np.pi / (a + b)) ** 1.5) * np.exp(-a * b / (a + b) * dist_sq)
                    P_vec = (a * basis[p].center + b * basis[q].center) / (a + b)
                    
                    val_x += cp * cq * S_prim * P_vec[0]
                    val_y += cp * cq * S_prim * P_vec[1]
                    val_z += cp * cq * S_prim * P_vec[2]
            Dx[p, q] = val_x
            Dy[p, q] = val_y
            Dz[p, q] = val_z
            
    mu_elec_x = np.sum(P * Dx)
    mu_elec_y = np.sum(P * Dy)
    mu_elec_z = np.sum(P * Dz)
    
    mu_tot = mu_nuc - np.array([mu_elec_x, mu_elec_y, mu_elec_z])
    magnitude_debye = np.linalg.norm(mu_tot) * 2.541746
    
    return mu_tot, magnitude_debye

def compute_mulliken_charges(atoms, res):
    """
    Computes Mulliken population analysis net charges for each atom.
    q_A = Z_A - sum_{mu in A} (P * S)_{mu, mu}
    """
    basis = res['basis']
    P = res['P']
    S = res['S']
    nbasis = len(basis)
    
    # Diagonal of PS matrix represents orbital gross population
    PS = np.dot(P, S)
    
    charges = []
    for idx, item in enumerate(atoms):
        name = item["name"]
        pos = np.array(item["pos"], dtype=float)
        _, Z = get_element_orbitals(name, pos)
        
        pop = 0.0
        for p in range(nbasis):
            # Check if orbital belongs to this atom (by center overlap)
            if np.allclose(basis[p].center, pos, atol=1e-3):
                pop += PS[p, p]
                
        net_charge = Z - pop
        charges.append({
            "index": idx + 1,
            "name": name,
            "pos": pos.tolist(),
            "population": pop,
            "net_charge": net_charge
        })
    return charges

def compute_bond_orders(atoms, res):
    """
    Computes Mulliken Overlap Populations (MOP) between all atom pairs.
    
    MOP(A,B) = sum_{mu in A, nu in B} 2 * P_{mu,nu} * S_{mu,nu}
    
    This is the standard Mulliken bond overlap population. Positive values indicate
    covalent bonding (electron density shared between atoms), negative values indicate
    anti-bonding contributions, and ~zero means no direct orbital interaction.
    
    Returns:
        mop_matrix: (N_atoms x N_atoms) Mulliken overlap population matrix (signed)
        bond_strength: (N_atoms x N_atoms) absolute MOP (bond strength indicator)
    """
    basis = res['basis']
    P = res['P']
    S = res['S']
    nbasis = len(basis)
    num_atoms = len(atoms)
    
    # Initialize output matrices
    mop_matrix = np.zeros((num_atoms, num_atoms))
    bond_strength = np.zeros((num_atoms, num_atoms))
    
    # Map basis functions to atoms by center proximity
    atom_orbitals = []
    for at in atoms:
        pos = np.array(at["pos"], dtype=float)
        orbs = [p for p in range(nbasis) if np.allclose(basis[p].center, pos, atol=1e-3)]
        atom_orbitals.append(orbs)
    
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                mop_matrix[i, j] = 0.0
                bond_strength[i, j] = 0.0
                continue
            mop_val = 0.0
            for mu in atom_orbitals[i]:
                for nu in atom_orbitals[j]:
                    # Mulliken Overlap Population: 2 * P_{mu,nu} * S_{mu,nu}
                    mop_val += 2.0 * P[mu, nu] * S[mu, nu]
            mop_matrix[i, j] = mop_val
            bond_strength[i, j] = abs(mop_val)
    
    return mop_matrix, bond_strength

