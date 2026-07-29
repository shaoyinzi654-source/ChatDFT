import numpy as np
import streamlit as st
from dft_engine import solve_1d_dft
from diatomic_engine import solve_diatomic_scf, STO3GOrbital, get_element_orbitals

def get_element_zeta(name):
    ELEMENT_ZETAS = {
        'H':  1.24,
        'HE': 1.69,
        'LI': 0.65,
        'BE': 0.975,
        'B':  1.30,
        'C':  1.625,
        'N':  1.95,
        'O':  2.275,
        'F':  2.60,
        'NE': 2.925,
        'NA': 0.73,
        'MG': 0.95,
        'G':  1.0
    }
    nm = name.upper()
    return ELEMENT_ZETAS.get(nm, 1.0)

def calculate_dos(eigenvalues, occupations, E_grid, sigma=0.05):
    """
    Computes the total Density of States (DOS) using Gaussian broadening.
    DOS(E) = sum_i occ_i * (1 / (sigma * sqrt(2*pi))) * exp(-(E - eps_i)^2 / (2 * sigma^2))
    """
    dos = np.zeros_like(E_grid)
    for eps, occ in zip(eigenvalues, occupations):
        gaussian = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((E_grid - eps) ** 2) / (2.0 * (sigma ** 2)))
        dos += occ * gaussian
    return dos

def calculate_pdos_3d(eigenvalues, C, S, E_grid, sigma=0.05):
    """
    Computes the Projected Density of States (PDOS) onto the atomic orbitals for 3D diatomic systems.
    Using Mulliken orbital population:
    Q_A^(i) = sum_{mu in A} sum_{nu} C_{mu, i} * C_{nu, i} * S_{mu, nu}
    """
    if "config" not in st.session_state:
        # 命令行独立运行脚本（如 run_and_plot.py 运行 H2 分子）时的备用方案
        n1 = 1
    else:
        params = st.session_state.config["params"]
        def get_num_orbitals(name):
            ELEMENT_ORBS = {
                'H': 1, 'HE': 1, 'LI': 2, 'BE': 2, 'B': 3, 'C': 3,
                'N': 4, 'O': 4, 'F': 5, 'NE': 5, 'NA': 6, 'MG': 6, 'G': 1
            }
            return ELEMENT_ORBS.get(name.upper(), 1)
        n1 = get_num_orbitals(params.get("atom1_name", "H"))
    
    pdos_atom1 = np.zeros_like(E_grid)
    pdos_atom2 = np.zeros_like(E_grid)
    
    nbasis = C.shape[0]
    for i in range(nbasis):
        eps = eigenvalues[i]
        
        # Mulliken population on Atom 1 (orbitals 0 to n1-1)
        q1 = 0.0
        for mu in range(0, min(n1, nbasis)):
            for nu in range(nbasis):
                q1 += C[mu, i] * C[nu, i] * S[mu, nu]
                
        # Mulliken population on Atom 2 (orbitals n1 to nbasis-1)
        q2 = 0.0
        for mu in range(n1, nbasis):
            for nu in range(nbasis):
                q2 += C[mu, i] * C[nu, i] * S[mu, nu]
                
        gaussian = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((E_grid - eps) ** 2) / (2.0 * (sigma ** 2)))
        pdos_atom1 += q1 * gaussian
        pdos_atom2 += q2 * gaussian
        
    return pdos_atom1, pdos_atom2

def calculate_pdos_multi(eigenvalues, C, S, basis, atoms, E_grid, sigma=0.05):
    """
    Computes the Projected Density of States (PDOS) onto each atom in a multi-atom system.
    Using Mulliken orbital population mapping by orbital centers.
    """
    num_atoms = len(atoms)
    nbasis = len(basis)
    
    # Map basis indices to atom indices by matching physical orbital center to nearest atom
    orbital_to_atom = []
    for mu in range(nbasis):
        center = np.array(basis[mu].center)
        best_atom_idx = 0
        min_dist = float('inf')
        for j, at in enumerate(atoms):
            dist = np.linalg.norm(center - np.array(at["pos"]))
            if dist < min_dist:
                min_dist = dist
                best_atom_idx = j
        orbital_to_atom.append(best_atom_idx)
        
    pdos_list = [np.zeros_like(E_grid) for _ in range(num_atoms)]
    
    for i in range(nbasis):
        eps = eigenvalues[i]
        gaussian = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((E_grid - eps) ** 2) / (2.0 * (sigma ** 2)))
        
        # Calculate Mulliken charge distribution on each atom for MO i
        for j in range(num_atoms):
            q = 0.0
            for mu in range(nbasis):
                if orbital_to_atom[mu] == j:
                    for nu in range(nbasis):
                        q += C[mu, i] * C[nu, i] * S[mu, nu]
            pdos_list[j] += q * gaussian
            
    return pdos_list

def eval_density_z(res, z_arr):
    """
    Evaluates the 3D density of the diatomic/multi-atomic system along the Z-axis (X=0, Y=0).
    """
    basis = res['basis']
    nbasis = len(basis)
    P = res['P']
    
    phi = []
    for p in range(nbasis):
        vals = np.zeros_like(z_arr)
        cz = basis[p].center[2]
        cx = basis[p].center[0]
        cy = basis[p].center[1]
        for k in range(3):
            a = basis[p].alphas[k]
            d = basis[p].coeffs[k]
            norm = (2.0 * a / np.pi) ** 0.75
            # evaluation along Z-axis (x=0, y=0)
            r_sq = cx**2 + cy**2 + (z_arr - cz) ** 2
            vals += d * norm * np.exp(-a * r_sq)
        phi.append(vals)
        
    rho = np.zeros_like(z_arr)
    for p in range(nbasis):
        for q in range(nbasis):
            rho += P[p, q] * phi[p] * phi[q]
    return rho

def calculate_cdd_3d(atoms, num_electrons, mol_res, z_grid):
    """
    Computes the 3D Charge Density Difference along the Z-axis:
    Delta rho(z) = rho_molecule(z) - sum_i rho_isolated_atom_i(z)
    """
    rho_mol = eval_density_z(mol_res, z_grid)
    
    rho_atoms_sum = np.zeros_like(z_grid)
    for item in atoms:
        name = item["name"]
        pos = item["pos"]
        
        # Isolated reference: put it at its pos, and put a Ghost (G) at a far-away pos [pos[0]+50.0, pos[1], pos[2]]
        far_pos = [pos[0] + 50.0, pos[1], pos[2]]
        
        # The number of electrons of this isolated atom is its atomic number
        def get_atomic_number(n):
            ELEMENT_Z = {'H':1, 'HE':2, 'LI':3, 'BE':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'NE':10, 'NA':11, 'MG':12, 'G':0}
            return ELEMENT_Z.get(n.upper(), 0)
        ne_atom = get_atomic_number(name)
        
        res_atom = solve_diatomic_scf(name, pos, 'G', far_pos, num_electrons=ne_atom, max_iter=50, tol=1e-6)
        rho_atom = eval_density_z(res_atom, z_grid)
        rho_atoms_sum += rho_atom
        
    cdd = rho_mol - rho_atoms_sum
    return cdd, rho_mol, rho_atoms_sum

def calculate_cdd_1d(params, mol_res):
    """
    Computes the 1D Charge Density Difference for double-well potentials.
    """
    expr = params["potential_expr"]
    L = params["L"]
    N = params["N"]
    softening = params["softening"]
    x = mol_res["x"]
    
    # Parse potential to extract centers
    centers = []
    import re
    matches = re.findall(r'\(x\s*([+-]\s*\d+\.?\d*)\)', expr)
    for m in matches:
        val = float(m.replace(" ", ""))
        centers.append(-val)
        
    if len(centers) == 2:
        x1, x2 = centers
        Vext1_fn = lambda x_arr: -1.0 / np.sqrt((x_arr - x1)**2 + softening**2)
        Vext2_fn = lambda x_arr: -1.0 / np.sqrt((x_arr - x2)**2 + softening**2)
        
        res_a1 = solve_1d_dft(Vext1_fn, num_electrons=1, L=L, N=N, max_iter=80, tol=1e-5, alpha=0.3, softening=softening)
        res_a2 = solve_1d_dft(Vext2_fn, num_electrons=1, L=L, N=N, max_iter=80, tol=1e-5, alpha=0.3, softening=softening)
        
        rho_mol = mol_res['density']
        rho_a1 = res_a1['density']
        rho_a2 = res_a2['density']
        
        cdd = rho_mol - rho_a1 - rho_a2
        return cdd, rho_mol, rho_a1, rho_a2
    else:
        return np.zeros_like(x), mol_res['density'], mol_res['density'], np.zeros_like(x)

def eval_density_2d(res, Y, Z):
    """
    Evaluates the 2D density of the diatomic/multi-atomic system in the YZ-plane (where X=0).
    """
    basis = res['basis']
    nbasis = len(basis)
    P = res['P']
    
    phi = []
    for p in range(nbasis):
        vals = np.zeros_like(Y)
        cz = basis[p].center[2]
        cx = basis[p].center[0]
        cy = basis[p].center[1]
        for k in range(3):
            a = basis[p].alphas[k]
            d = basis[p].coeffs[k]
            norm = (2.0 * a / np.pi) ** 0.75
            # Evaluate at X=0
            r_sq = cx**2 + (Y - cy)**2 + (Z - cz)**2
            vals += d * norm * np.exp(-a * r_sq)
        phi.append(vals)
        
    rho = np.zeros_like(Y)
    for p in range(nbasis):
        for q in range(nbasis):
            rho += P[p, q] * phi[p] * phi[q]
    return rho

def calculate_cdd_2d(atoms, num_electrons, mol_res, Y, Z):
    """
    Computes the 2D Charge Density Difference in the YZ-plane (X=0):
    Delta rho(y,z) = rho_molecule(y,z) - sum_i rho_isolated_atom_i(y,z)
    """
    rho_mol = eval_density_2d(mol_res, Y, Z)
    
    rho_atoms_sum = np.zeros_like(Y)
    for item in atoms:
        name = item["name"]
        pos = item["pos"]
        
        far_pos = [pos[0] + 50.0, pos[1], pos[2]]
        def get_atomic_number(n):
            ELEMENT_Z = {'H':1, 'HE':2, 'LI':3, 'BE':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'NE':10, 'NA':11, 'MG':12, 'G':0}
            return ELEMENT_Z.get(n.upper(), 0)
        ne_atom = get_atomic_number(name)
        
        res_atom = solve_diatomic_scf(name, pos, 'G', far_pos, num_electrons=ne_atom, max_iter=50, tol=1e-6)
        rho_atom = eval_density_2d(res_atom, Y, Z)
        rho_atoms_sum += rho_atom
        
    cdd = rho_mol - rho_atoms_sum
    return cdd, rho_mol, rho_atoms_sum

def eval_density_3d_grid(res, X, Y, Z):
    """
    Evaluates the 3D density of the diatomic/multi-atomic system on a 3D grid (X, Y, Z).
    """
    basis = res['basis']
    nbasis = len(basis)
    P = res['P']
    
    phi = []
    for p in range(nbasis):
        vals = np.zeros_like(X)
        cx, cy, cz = basis[p].center
        for k in range(3):
            a = basis[p].alphas[k]
            d = basis[p].coeffs[k]
            norm = (2.0 * a / np.pi) ** 0.75
            r_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
            vals += d * norm * np.exp(-a * r_sq)
        phi.append(vals)
        
    rho = np.zeros_like(X)
    for p in range(nbasis):
        for q in range(nbasis):
            rho += P[p, q] * phi[p] * phi[q]
    return rho

def calculate_cdd_3d_grid(atoms, num_electrons, mol_res, X, Y, Z):
    """
    Computes the 3D CDD on a 3D grid:
    Delta rho(x,y,z) = rho_molecule(x,y,z) - sum_i rho_isolated_atom_i(x,y,z)
    """
    rho_mol = eval_density_3d_grid(mol_res, X, Y, Z)
    
    rho_atoms_sum = np.zeros_like(X)
    for item in atoms:
        name = item["name"]
        pos = item["pos"]
        
        far_pos = [pos[0] + 50.0, pos[1], pos[2]]
        def get_atomic_number(n):
            ELEMENT_Z = {'H':1, 'HE':2, 'LI':3, 'BE':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'NE':10, 'NA':11, 'MG':12, 'G':0}
            return ELEMENT_Z.get(n.upper(), 0)
        ne_atom = get_atomic_number(name)
        
        res_atom = solve_diatomic_scf(name, pos, 'G', far_pos, num_electrons=ne_atom, max_iter=50, tol=1e-6)
        rho_atom = eval_density_3d_grid(res_atom, X, Y, Z)
        rho_atoms_sum += rho_atom
        
    cdd = rho_mol - rho_atoms_sum
    return cdd, rho_mol, rho_atoms_sum

def calculate_mep_grid_2d(atoms, res, Y_grid, Z_grid):
    """
    Computes the EXACT analytical molecular electrostatic potential (MEP) in the YZ-plane (X=0).
    MEP(y, z) = V_nuclei(y, z) + V_electrons(y, z)
    """
    from scipy.special import erf
    import copy
    
    # 1. Nuclear contribution
    V_nuc = np.zeros_like(Y_grid)
    for at in atoms:
        ELEMENT_Z = {'H':1, 'HE':2, 'LI':3, 'BE':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'NE':10, 'NA':11, 'MG':12, 'G':0}
        Z = ELEMENT_Z.get(at["name"].upper(), 1.0)
        pos = np.array(at["pos"])
        dist = np.sqrt(pos[0]**2 + (Y_grid - pos[1])**2 + (Z_grid - pos[2])**2)
        V_nuc += Z / np.clip(dist, 1e-6, None)
        
    # 2. Electronic contribution (Analytical STO-3G electrostatic potential)
    basis = res["basis"]
    nbasis = len(basis)
    P = res["P"]
    
    V_elec = np.zeros_like(Y_grid)
    
    for p in range(nbasis):
        bp = basis[p]
        for q in range(nbasis):
            bq = basis[q]
            p_val = P[p, q]
            if abs(p_val) < 1e-6:
                continue
                
            for i in range(3):
                alpha = bp.alphas[i]
                d_p = bp.coeffs[i]
                norm_p = (2.0 * alpha / np.pi) ** 0.75
                
                for j in range(3):
                    beta = bq.alphas[j]
                    d_q = bq.coeffs[j]
                    norm_q = (2.0 * beta / np.pi) ** 0.75
                    
                    p_exp = alpha + beta
                    P_center = (alpha * np.array(bp.center) + beta * np.array(bq.center)) / p_exp
                    
                    dist_AB_sq = np.sum((np.array(bp.center) - np.array(bq.center))**2)
                    pref = d_p * norm_p * d_q * norm_q * np.exp(-alpha * beta / p_exp * dist_AB_sq)
                    
                    weight = p_val * pref * (np.pi / p_exp) ** 1.5
                    
                    dist_grid = np.sqrt(P_center[0]**2 + (Y_grid - P_center[1])**2 + (Z_grid - P_center[2])**2)
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        pot_term = erf(np.sqrt(p_exp) * dist_grid) / dist_grid
                    pot_term[dist_grid < 1e-8] = 2.0 * np.sqrt(p_exp / np.pi)
                    
                    V_elec -= weight * pot_term
                    
    return V_nuc + V_elec
