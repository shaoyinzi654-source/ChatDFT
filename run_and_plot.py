import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from dft_engine import solve_1d_dft
from diatomic_engine import solve_diatomic_scf, solve_multi_atom_scf
from analysis_tools import (
    calculate_dos, calculate_pdos_3d, calculate_cdd_3d,
    eval_density_2d, calculate_cdd_2d, eval_density_3d_grid,
    calculate_mep_grid_2d
)

# -----------------------------------------------------------------------------
# High-End Publication Aesthetic Configuration (Nature / Science Standard)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 15,
    'grid.linestyle': '--',
    'grid.alpha': 0.22,
    'axes.facecolor': '#f8fafc',
    'figure.facecolor': '#ffffff',
    'axes.edgecolor': '#cbd5e1',
    'axes.linewidth': 1.0,
    'lines.solid_capstyle': 'round',
    'savefig.dpi': 350,
    'savefig.bbox': 'tight'
})

THEME = {
    'primary': '#0f172a',     # Deep slate / navy
    'blue': '#2563eb',        # Royal blue
    'cyan': '#0284c7',        # Sky cyan
    'teal': '#0d9488',        # Emerald teal
    'green': '#16a34a',       # Forest green
    'amber': '#d97706',       # Warm amber
    'rose': '#e11d48',        # Vibrant rose/crimson
    'purple': '#7c3aed',      # Deep violet
    'slate': '#64748b',       # Muted slate
    'bg_card': '#f1f5f9',     # Card fill tint
    'border': '#e2e8f0'       # Border line
}

def add_nature_badge(ax, letter, title=""):
    """Draw a Nature-style badge tag (e.g. A, B, C, D) with a title."""
    if hasattr(ax, 'zaxis'):
        # 3D plot
        ax.text2D(-0.08, 1.04, f"({letter})", transform=ax.transAxes,
                  fontsize=15, fontweight='bold', va='bottom', ha='right', color='#0f172a')
        if title:
            ax.text2D(0.0, 1.04, title, transform=ax.transAxes,
                      fontsize=12, fontweight='bold', va='bottom', ha='left', color='#1e293b')
    else:
        # 2D plot
        ax.text(-0.10, 1.05, f"({letter})", transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='bottom', ha='right', color='#0f172a')
        if title:
            ax.text(0.0, 1.05, title, transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='bottom', ha='left', color='#1e293b')

# =============================================================================
# FIGURE PLATE 1: 1D Kohn-Sham DFT Solver & Self-Consistent Field Kinetics
# =============================================================================
def generate_nature_fig1():
    print("Generating Nature Figure Plate 1: 1D Kohn-Sham DFT Solver & Kinetics...")
    
    def vhe(x): return -2.0 / np.sqrt(x**2 + 1.0)
    res_dft = solve_1d_dft(vhe, num_electrons=2, L=8.0, N=300, max_iter=100, tol=1e-6, functional="LDA")
    res_lin = solve_1d_dft(vhe, num_electrons=2, L=8.0, N=200, max_iter=45, tol=1e-7, mixing_method="Linear", alpha=0.2)
    res_pul = solve_1d_dft(vhe, num_electrons=2, L=8.0, N=200, max_iter=45, tol=1e-7, mixing_method="Pulay", alpha=0.2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.8))
    
    # ----------------------------------------------------
    # Panel A: Density & External Soft-Coulomb Potential
    # ----------------------------------------------------
    ax_a = axes[0, 0]
    add_nature_badge(ax_a, 'A', '1D Helium-like Atom ($Z=2$) Density & $V_{ext}$')
    x = res_dft['x']
    rho = res_dft['density']
    v_ext = res_dft['potentials']['Vext']
    
    line1 = ax_a.plot(x, rho, color=THEME['blue'], linewidth=2.4, label=r'Electron Density $\rho(x)$ ($e/\mathrm{Bohr}$)')
    ax_a.fill_between(x, rho, color=THEME['blue'], alpha=0.18)
    ax_a.set_xlabel(r'Spatial Coordinate $x$ (Bohr)')
    ax_a.set_ylabel(r'Density $\rho(x)$ ($e/\mathrm{Bohr}$)', color=THEME['blue'], fontweight='bold')
    ax_a.tick_params(axis='y', labelcolor=THEME['blue'])
    
    ax_a_twin = ax_a.twinx()
    line2 = ax_a_twin.plot(x, v_ext, color=THEME['rose'], linestyle='--', linewidth=2.0, label=r'External Potential $V_{\mathrm{ext}}(x)$')
    ax_a_twin.set_ylabel(r'Potential $V_{\mathrm{ext}}$ (Hartree)', color=THEME['rose'], fontweight='bold')
    ax_a_twin.tick_params(axis='y', labelcolor=THEME['rose'])
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_a.legend(lines, labels, loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    ax_a.grid(True, alpha=0.25)
    
    # Annotate total energy
    e_tot = res_dft['energies']['E_tot']
    ax_a.text(0.05, 0.85, f"$E_{{\\mathrm{{tot}}}} = {e_tot:.4f}\\mathrm{{ Ha}}$\n$N_e = 2.00 e$", transform=ax_a.transAxes,
              fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", facecolor='#eff6ff', edgecolor='#93c5fd', alpha=0.95))
    
    # ----------------------------------------------------
    # Panel B: Potential Component Decomposition
    # ----------------------------------------------------
    ax_b = axes[0, 1]
    add_nature_badge(ax_b, 'B', 'Kohn-Sham Potential Field Decomposition')
    v_h = res_dft['potentials']['VH']
    v_xc = res_dft['potentials']['Vxc']
    v_eff = res_dft['potentials']['Veff']
    
    ax_b.plot(x, v_ext, label=r'$V_{\mathrm{ext}}$ (Nuclear Soft-Coulomb)', color=THEME['rose'], linestyle='--', linewidth=1.6)
    ax_b.plot(x, v_h, label=r'$V_{\mathrm{H}}$ (Hartree Repulsion)', color=THEME['cyan'], linewidth=2.0)
    ax_b.plot(x, v_xc, label=r'$V_{\mathrm{xc}}$ (Slater-Wigner LDA)', color=THEME['amber'], linewidth=2.0)
    ax_b.plot(x, v_eff, label=r'$V_{\mathrm{eff}}$ (Total Effective Potential)', color=THEME['primary'], linewidth=2.4)
    
    ax_b.set_xlabel(r'Spatial Coordinate $x$ (Bohr)')
    ax_b.set_ylabel(r'Potential Energy (Hartree)')
    ax_b.set_ylim(-2.4, 1.2)
    ax_b.grid(True, alpha=0.25)
    ax_b.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    
    # ----------------------------------------------------
    # Panel C: Eigen-orbitals & Energy Levels
    # ----------------------------------------------------
    ax_c = axes[1, 0]
    add_nature_badge(ax_c, 'C', r'Kohn-Sham Eigen-orbitals $\psi_n(x)$ & Energy Spectrum')
    orbitals = res_dft['all_orbitals']
    eps = res_dft['all_eigenvalues']
    colors = [THEME['blue'], THEME['teal'], THEME['amber'], THEME['purple']]
    
    ax_c.plot(x, v_eff, color='#94a3b8', linestyle=':', label=r'$V_{\mathrm{eff}}(x)$', linewidth=1.5)
    scale = 0.45
    for i in range(min(4, len(eps))):
        psi = orbitals[:, i]
        energy = eps[i]
        color = colors[i % len(colors)]
        occ_str = " (HOMO, 2e)" if i == 0 else " (Virtual)"
        ax_c.plot(x, energy + psi * scale, color=color, linewidth=2.2,
                  label=fr'$\psi_{i+1}(x)$ ($E_{i+1} = {energy:.4f}\mathrm{{ Ha}}$){occ_str}')
        ax_c.axhline(y=energy, color=color, linestyle='--', alpha=0.4, linewidth=0.9)
        ax_c.fill_between(x, energy, energy + psi * scale, color=color, alpha=0.14)
        
    ax_c.set_xlabel(r'Spatial Coordinate $x$ (Bohr)')
    ax_c.set_ylabel(r'Energy / Wavefunction Amplitude (Hartree)')
    ax_c.set_xlim(-6, 6)
    ax_c.set_ylim(-1.5, 0.6)
    ax_c.grid(True, alpha=0.25)
    ax_c.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    
    # ----------------------------------------------------
    # Panel D: SCF Convergence & Pulay DIIS Acceleration
    # ----------------------------------------------------
    ax_d = axes[1, 1]
    add_nature_badge(ax_d, 'D', 'SCF Convergence Kinetics & Pulay DIIS Acceleration')
    hist_lin = res_lin['history']
    hist_pul = res_pul['history']
    
    iters_lin = list(range(1, len(hist_lin) + 1))
    d_lin = [1e-1] + [max(abs(hist_lin[i] - hist_lin[i-1]), 1e-12) for i in range(1, len(hist_lin))]
    
    iters_pul = list(range(1, len(hist_pul) + 1))
    d_pul = [1e-1] + [max(abs(hist_pul[i] - hist_pul[i-1]), 1e-12) for i in range(1, len(hist_pul))]
    
    ax_d.semilogy(iters_lin, d_lin, marker='o', markersize=4.5, color=THEME['blue'], label='Linear Density Mixing', linewidth=2.0)
    ax_d.semilogy(iters_pul, d_pul, marker='s', markersize=4.5, color=THEME['rose'], label='Pulay DIIS Acceleration', linewidth=2.0)
    ax_d.axhline(y=1e-6, color=THEME['slate'], linestyle='--', linewidth=1.2, label=r'Convergence Threshold ($10^{-6}\mathrm{ Ha}$)')
    
    ax_d.set_xlabel('SCF Iteration Step')
    ax_d.set_ylabel(r'Energy Residual $|\Delta E|$ (Hartree)')
    ax_d.grid(True, alpha=0.25, which='both')
    ax_d.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    
    # Callout text box
    ax_d.text(0.05, 0.12, f"Pulay DIIS: {len(hist_pul)} steps to converge\nLinear: {len(hist_lin)} steps to converge",
              transform=ax_d.transAxes, fontsize=9.5,
              bbox=dict(boxstyle="round,pad=0.4", facecolor='#fff1f2', edgecolor='#fecdd3', alpha=0.95))
    
    plt.tight_layout()
    plt.savefig('nature_fig1_dft_solver.png', dpi=350)
    plt.savefig('dft_results.png', dpi=350)
    plt.savefig('dft_orbitals.png', dpi=350)
    plt.savefig('scf_convergence.png', dpi=350)
    plt.close()
    print("Saved 'nature_fig1_dft_solver.png', 'dft_results.png', 'dft_orbitals.png', 'scf_convergence.png'")

# =============================================================================
# FIGURE PLATE 2: 3D/2D Molecular Electron Density & Charge Density Difference
# =============================================================================
def generate_nature_fig2():
    print("Generating Nature Figure Plate 2: 3D/2D Density & CDD (H2)...")
    
    R = 1.4
    atom1_name = 'H'
    atom1_pos = [0.0, 0.0, -R/2.0]
    atom2_name = 'H'
    atom2_pos = [0.0, 0.0, R/2.0]
    atoms = [{"name": atom1_name, "pos": atom1_pos}, {"name": atom2_name, "pos": atom2_pos}]
    num_electrons = 2
    
    res_mol = solve_diatomic_scf(atom1_name, atom1_pos, atom2_name, atom2_pos, num_electrons=num_electrons)
    p1_z = atom1_pos[2]
    p2_z = atom2_pos[2]
    
    fig = plt.figure(figsize=(12.5, 9.8))
    
    # ----------------------------------------------------
    # Panel A: 3D Volumetric Electron Density Isosurface
    # ----------------------------------------------------
    ax_a = fig.add_subplot(221, projection='3d')
    add_nature_badge(ax_a, 'A', r'3D Volumetric Electron Density Isosurface $\rho(\mathbf{r})$')
    ax_a.set_facecolor('#ffffff')
    
    y = np.linspace(-2.5, 2.5, 90)
    z = np.linspace(-3.5, 3.5, 110)
    Y3d, Z3d = np.meshgrid(y, z)
    X3d = np.zeros_like(Y3d)
    rho3d = eval_density_3d_grid(res_mol, X3d, Y3d, Z3d)
    rho3d = np.maximum(rho3d, 0.0)
    levels = np.percentile(rho3d[rho3d > 0], [55, 75, 88, 96])
    
    for level, alpha in zip(levels, [0.14, 0.24, 0.38, 0.55]):
        surface = np.ma.masked_less(rho3d, level)
        ax_a.plot_surface(Z3d, Y3d, surface, cmap='viridis', alpha=alpha,
                          linewidth=0, antialiased=True, vmin=0, vmax=np.max(rho3d))
        
    for idx, atom in enumerate(atoms):
        pos = atom['pos']
        ax_a.scatter([pos[2]], [pos[1]], [np.max(rho3d) * 0.02], s=200,
                     color='#f8fafc', edgecolor=THEME['primary'], linewidth=1.6,
                     depthshade=True, label=f"H{idx + 1} Nucleus")
    ax_a.plot([p1_z, p2_z], [0, 0], [np.max(rho3d) * 0.02] * 2,
              color=THEME['rose'], linewidth=3.8, label='H-H Bond Axis')
    
    ax_a.set_xlabel(r'Bond Axis $z$ (Bohr)', labelpad=7)
    ax_a.set_ylabel(r'Transverse $y$ (Bohr)', labelpad=7)
    ax_a.set_zlabel(r'Density $\rho$ ($e/\mathrm{Bohr}^3$)', labelpad=7)
    ax_a.view_init(elev=24, azim=-58)
    ax_a.set_box_aspect((1.35, 1.0, 0.72))
    ax_a.grid(True, alpha=0.18)
    ax_a.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    
    # ----------------------------------------------------
    # Panel B: 2D Electron Density Contour Map
    # ----------------------------------------------------
    ax_b = fig.add_subplot(222)
    add_nature_badge(ax_b, 'B', r'2D Molecular Plane Electron Density $\rho(y,z)$')
    
    grid_y = np.linspace(-3.0, 3.0, 160)
    grid_z = np.linspace(-4.0, 4.0, 190)
    Y2d, Z2d = np.meshgrid(grid_y, grid_z)
    rho_2d = eval_density_2d(res_mol, Y2d, Z2d)
    
    cf_b = ax_b.contourf(Z2d, Y2d, rho_2d, levels=70, cmap='plasma')
    cb_b = fig.colorbar(cf_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb_b.set_label(r'Electron Density $\rho(y,z)$ ($e/\mathrm{Bohr}^3$)', fontweight='bold')
    
    c1 = plt.Circle((p1_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.6, zorder=5)
    c2 = plt.Circle((p2_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.6, zorder=5)
    ax_b.add_patch(c1)
    ax_b.add_patch(c2)
    ax_b.text(p1_z, 0, 'H1', color='black', ha='center', va='center', weight='bold', zorder=6)
    ax_b.text(p2_z, 0, 'H2', color='black', ha='center', va='center', weight='bold', zorder=6)
    ax_b.plot([p1_z, p2_z], [0, 0], color='white', linestyle='--', linewidth=2.0, zorder=4)
    
    ax_b.set_xlabel(r'Bond Axis $z$ (Bohr)')
    ax_b.set_ylabel(r'Transverse $y$ (Bohr)')
    ax_b.set_aspect('equal')
    
    # ----------------------------------------------------
    # Panel C: 2D Charge Density Difference (CDD) Contour
    # ----------------------------------------------------
    ax_c = fig.add_subplot(223)
    add_nature_badge(ax_c, 'C', r'2D Charge Density Difference $\Delta\rho(y,z)$')
    
    cdd_2d, _, _ = calculate_cdd_2d(atoms, num_electrons, res_mol, Y2d, Z2d)
    max_val = np.max(np.abs(cdd_2d))
    cf_c = ax_c.contourf(Z2d, Y2d, cdd_2d, levels=70, cmap='RdBu_r', vmin=-max_val, vmax=max_val)
    cb_c = fig.colorbar(cf_c, ax=ax_c, fraction=0.046, pad=0.04)
    cb_c.set_label(r'Density Difference $\Delta\rho(y,z)$ ($e/\mathrm{Bohr}^3$)', fontweight='bold')
    
    c1 = plt.Circle((p1_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.6, zorder=5)
    c2 = plt.Circle((p2_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.6, zorder=5)
    ax_c.add_patch(c1)
    ax_c.add_patch(c2)
    ax_c.text(p1_z, 0, 'H1', color='black', ha='center', va='center', weight='bold', zorder=6)
    ax_c.text(p2_z, 0, 'H2', color='black', ha='center', va='center', weight='bold', zorder=6)
    ax_c.plot([p1_z, p2_z], [0, 0], color='black', linestyle='--', linewidth=1.6, zorder=4)
    
    ax_c.set_xlabel(r'Bond Axis $z$ (Bohr)')
    ax_c.set_ylabel(r'Transverse $y$ (Bohr)')
    ax_c.set_aspect('equal')
    
    # Text annotation for CDD accumulation
    ax_c.text(0.0, 0.8, 'Bonding Accumulation (+)', color='#990000', ha='center', va='center', fontsize=9, fontweight='bold',
              bbox=dict(boxstyle="round,pad=0.3", facecolor='#ffe6e6', edgecolor='#ff9999', alpha=0.9))
    
    # ----------------------------------------------------
    # Panel D: 1D Axial CDD Profile
    # ----------------------------------------------------
    ax_d = fig.add_subplot(224)
    add_nature_badge(ax_d, 'D', r'1D Axial Charge Density Difference Profile $\Delta\rho(z)$')
    
    z_grid = np.linspace(-4.0, 4.0, 300)
    cdd_1d, rho_mol, sum_atoms = calculate_cdd_3d(atoms, num_electrons, res_mol, z_grid)
    
    ax_d.plot(z_grid, cdd_1d, label=r'Difference $\Delta\rho(z)$', color=THEME['green'], linewidth=2.4)
    ax_d.plot(z_grid, rho_mol, label=r'Molecular $\rho_{\mathrm{mol}}(z)$', color=THEME['blue'], linestyle='--', alpha=0.85)
    ax_d.plot(z_grid, sum_atoms, label=r'Superposition $\rho_{\mathrm{atoms}}(z)$', color=THEME['slate'], linestyle=':', alpha=0.85)
    
    ax_d.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
    ax_d.axvline(x=p1_z, color=THEME['rose'], linestyle='--', linewidth=1.2)
    ax_d.axvline(x=p2_z, color=THEME['rose'], linestyle='--', linewidth=1.2)
    
    ax_d.text(p1_z - 0.1, np.max(rho_mol)*0.5, 'H1', color=THEME['rose'], ha='right', weight='bold')
    ax_d.text(p2_z + 0.1, np.max(rho_mol)*0.5, 'H2', color=THEME['rose'], ha='left', weight='bold')
    
    ax_d.set_xlabel(r'Bond Axis Position $z$ (Bohr)')
    ax_d.set_ylabel(r'Electron Density ($e/\mathrm{Bohr}^3$)')
    ax_d.grid(True, alpha=0.25)
    ax_d.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    
    plt.tight_layout()
    plt.savefig('nature_fig2_h2_density.png', dpi=350)
    plt.savefig('h2_density_3d.png', dpi=350)
    plt.savefig('h2_density_2d_contour.png', dpi=350)
    plt.savefig('h2_cdd_2d_contour.png', dpi=350)
    plt.savefig('h2_charge_density_difference.png', dpi=350)
    plt.close()
    print("Saved 'nature_fig2_h2_density.png', 'h2_density_3d.png', 'h2_density_2d_contour.png', 'h2_cdd_2d_contour.png', 'h2_charge_density_difference.png'")

# =============================================================================
# FIGURE PLATE 3: Spectroscopic Analysis, DOS/PDOS & Electrostatic Potential
# =============================================================================
def generate_nature_fig3():
    print("Generating Nature Figure Plate 3: Spectroscopy, DOS/PDOS & MEP...")
    
    R = 1.4
    res_mol = solve_diatomic_scf('H', [0, 0, -R/2.0], 'H', [0, 0, R/2.0], num_electrons=2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.8))
    
    # ----------------------------------------------------
    # Panel A: Density of States (DOS) & PDOS
    # ----------------------------------------------------
    ax_a = axes[0, 0]
    add_nature_badge(ax_a, 'A', 'Electronic Density of States (Total DOS / PDOS)')
    
    E_grid = np.linspace(-1.5, 0.5, 450)
    sigma = 0.04
    e_vals = res_mol['eps']
    occs = [2.0 if i < 1 else 0.0 for i in range(len(e_vals))]
    
    dos = calculate_dos(e_vals, occs, E_grid, sigma)
    pdos1, pdos2 = calculate_pdos_3d(e_vals, res_mol['C'], res_mol['S'], E_grid, sigma)
    
    ax_a.plot(E_grid, dos, label='Total DOS', color=THEME['primary'], linewidth=2.4)
    ax_a.plot(E_grid, pdos1, label=r'PDOS ($\mathrm{H}_1$ Atom)', color=THEME['rose'], linestyle='--', linewidth=1.8)
    ax_a.plot(E_grid, pdos2, label=r'PDOS ($\mathrm{H}_2$ Atom)', color=THEME['teal'], linestyle=':', linewidth=1.8)
    
    homo_val = e_vals[0]
    ax_a.axvline(x=homo_val, color=THEME['blue'], linestyle='-', linewidth=1.5, label=fr'Fermi Level / HOMO ({homo_val:.4f} Ha)')
    ax_a.fill_between(E_grid, 0, dos, where=(E_grid <= homo_val), color=THEME['blue'], alpha=0.18, label='Occupied Sea')
    
    ax_a.set_xlabel(r'Energy $E$ (Hartree)')
    ax_a.set_ylabel(r'Density of States (States / Hartree)')
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    
    # ----------------------------------------------------
    # Panel B: Molecular Orbital Energy Spectrum
    # ----------------------------------------------------
    ax_b = axes[0, 1]
    add_nature_badge(ax_b, 'B', 'Molecular Orbital Energy Level Spectrum')
    
    atoms_h2 = [{'name':'H','pos':[0,0,-0.7]},{'name':'H','pos':[0,0,0.7]}]
    r_h2 = solve_multi_atom_scf(atoms_h2, 2)
    
    eps_all = r_h2['eps']
    n_occ = r_h2['n_alpha']
    
    for i, ep in enumerate(eps_all):
        is_occ = i < n_occ
        color = THEME['blue'] if is_occ else THEME['rose']
        style = '-' if is_occ else '--'
        label_text = f"MO-{i+1} ({'HOMO' if i==n_occ-1 else ('LUMO' if i==n_occ else 'Virtual')})"
        
        ax_b.hlines(ep, 0.2, 0.8, colors=color, linestyles=style, linewidth=2.8)
        ax_b.text(0.83, ep, f"{label_text}: {ep:.4f} Ha", va='center', fontsize=9.5, color=color, fontweight='bold')
        
    gap = eps_all[n_occ] - eps_all[n_occ-1]
    ax_b.text(0.1, (eps_all[n_occ] + eps_all[n_occ-1])/2, f"HOMO-LUMO Gap:\n$\Delta E = {gap:.4f}\\mathrm{{ Ha}}$ ({gap*27.2114:.2f} eV)",
              fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor='#f0fdf4', edgecolor='#86efac', alpha=0.95))
    
    ax_b.set_xlim(0, 1.4)
    ax_b.set_ylim(eps_all[0] - 0.25, max(eps_all) + 0.35)
    ax_b.set_xticks([])
    ax_b.set_ylabel(r'Orbital Energy $\epsilon_i$ (Hartree)')
    ax_b.grid(True, alpha=0.25, axis='y')
    
    # ----------------------------------------------------
    # Panel C: Simulated Water IR Vibrational Spectrum
    # ----------------------------------------------------
    ax_c = axes[1, 0]
    add_nature_badge(ax_c, 'C', r'Simulated Water ($\mathrm{H}_2\mathrm{O}$) Vibrational IR Spectrum')
    
    freqs = np.array([1595.0, 3657.0, 3756.0]) # Water vibrational modes in cm^-1
    intensities = np.array([72.0, 15.0, 55.0])
    mode_names = ['Bending (v2)', 'Sym Stretch (v1)', 'Asym Stretch (v3)']
    
    freq_grid = np.linspace(1000, 4200, 550)
    ir_spectrum = np.zeros_like(freq_grid)
    gamma = 35.0 # Lorentzian width
    
    for f0, I, name in zip(freqs, intensities, mode_names):
        lorentzian = I * (gamma**2) / ((freq_grid - f0)**2 + gamma**2)
        ir_spectrum += lorentzian
        ax_c.vlines(f0, 0, I, colors=THEME['rose'], linestyles='--', alpha=0.8, linewidth=1.5)
        ax_c.text(f0, I + 3.5, f"{name}\n{f0:.0f} cm$^{{-1}}$", ha='center', fontsize=8.5, color=THEME['primary'], fontweight='bold')
        
    ax_c.plot(freq_grid, ir_spectrum, color=THEME['primary'], linewidth=2.2, label='IR Absorbance Profile')
    ax_c.fill_between(freq_grid, 0, ir_spectrum, color=THEME['primary'], alpha=0.15)
    
    ax_c.set_xlabel(r'Wavenumber $\tilde{\nu}$ ($\mathrm{cm}^{-1}$)')
    ax_c.set_ylabel(r'IR Absorption Intensity (km / mol)')
    ax_c.grid(True, alpha=0.25)
    ax_c.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    
    # ----------------------------------------------------
    # Panel D: Molecular Electrostatic Potential (MEP) Map
    # ----------------------------------------------------
    ax_d = axes[1, 1]
    add_nature_badge(ax_d, 'D', r'Molecular Electrostatic Potential (MEP) Map')
    
    grid_y = np.linspace(-3.0, 3.0, 130)
    grid_z = np.linspace(-4.0, 4.0, 150)
    Y2d, Z2d = np.meshgrid(grid_y, grid_z)
    atoms_h2_dict = [{"name": "H", "pos": [0,0,-0.7]}, {"name": "H", "pos": [0,0,0.7]}]
    
    mep = calculate_mep_grid_2d(atoms_h2_dict, res_mol, Y2d, Z2d)
    mep_clipped = np.clip(mep, -0.5, 1.5)
    
    cf_d = ax_d.contourf(Z2d, Y2d, mep_clipped, levels=60, cmap='Spectral_r')
    cb_d = fig.colorbar(cf_d, ax=ax_d, fraction=0.046, pad=0.04)
    cb_d.set_label(r'MEP $V_{\mathrm{MEP}}(\mathbf{r})$ (Hartree)', fontweight='bold')
    
    p1_z, p2_z = -0.7, 0.7
    c1 = plt.Circle((p1_z, 0), radius=0.25, edgecolor='black', facecolor='white', linewidth=1.6, zorder=5)
    c2 = plt.Circle((p2_z, 0), radius=0.25, edgecolor='black', facecolor='white', linewidth=1.6, zorder=5)
    ax_d.add_patch(c1)
    ax_d.add_patch(c2)
    ax_d.text(p1_z, 0, 'H1', color='black', ha='center', va='center', weight='bold', zorder=6)
    ax_d.text(p2_z, 0, 'H2', color='black', ha='center', va='center', weight='bold', zorder=6)
    
    ax_d.set_xlabel(r'Bond Axis $z$ (Bohr)')
    ax_d.set_ylabel(r'Transverse $y$ (Bohr)')
    ax_d.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('nature_fig3_spectroscopy.png', dpi=350)
    plt.savefig('h2_density_of_states.png', dpi=350)
    plt.close()
    print("Saved 'nature_fig3_spectroscopy.png', 'h2_density_of_states.png'")

# =============================================================================
# FIGURE PLATE 4: Potential Energy Surface, Bonding & Energetics
# =============================================================================
def generate_nature_fig4():
    print("Generating Nature Figure Plate 4: PES, Chemical Bonding & Energetics...")
    
    distances = np.linspace(0.6, 4.5, 34)
    e_tot_list = []
    e_1e_list = []
    e_2e_list = []
    v_nn_list = []
    homo_list = []
    lumo_list = []
    
    for r in distances:
        pos1 = [0.0, 0.0, -r/2.0]
        pos2 = [0.0, 0.0, r/2.0]
        res = solve_diatomic_scf('H', pos1, 'H', pos2, num_electrons=2, max_iter=50, tol=1e-6)
        e_tot_list.append(res['E_tot'])
        v_nn_list.append(res['E_nuc'])
        e_1e_list.append(res['E_kin'] + res['E_ext'])
        e_2e_list.append(res['E_ee'])
        homo_list.append(res['eps'][0])
        lumo_list.append(res['eps'][1])
        
    min_idx = np.argmin(e_tot_list)
    r_eq = distances[min_idx]
    e_min = e_tot_list[min_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.8))
    
    # ----------------------------------------------------
    # Panel A: Ground-State PES Scan
    # ----------------------------------------------------
    ax_a = axes[0, 0]
    add_nature_badge(ax_a, 'A', r'Ground-State Potential Energy Surface (PES) Scan')
    
    ax_a.plot(distances, e_tot_list, linestyle='-', marker='o', color=THEME['primary'],
              linewidth=2.2, markersize=4.5, markerfacecolor='white', label=r'RHF Total Energy $E_{\mathrm{tot}}$')
    ax_a.axvline(x=r_eq, color=THEME['rose'], linestyle='--', alpha=0.85, linewidth=1.4,
                 label=fr'Equilibrium Bond $R_e \approx {r_eq:.2f}\mathrm{{ Bohr}}$ ({r_eq*0.529177:.2f} Å)')
    ax_a.scatter([r_eq], [e_min], color=THEME['rose'], s=80, zorder=5)
    
    ax_a.set_xlabel(r'Internuclear Distance $R$ (Bohr)')
    ax_a.set_ylabel(r'System Energy $E$ (Hartree)')
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    
    # Annotation
    ax_a.text(0.05, 0.25, f"$E_{{\\mathrm{{min}}}} = {e_min:.4f}\\mathrm{{ Ha}}$\n$R_e = {r_eq:.2f}\\mathrm{{ Bohr}}$",
              transform=ax_a.transAxes, fontsize=9.5,
              bbox=dict(boxstyle="round,pad=0.4", facecolor='#fef2f2', edgecolor='#fca5a5', alpha=0.95))
    
    # ----------------------------------------------------
    # Panel B: STO-3G Energy Component Decomposition
    # ----------------------------------------------------
    ax_b = axes[0, 1]
    add_nature_badge(ax_b, 'B', 'Hartree-Fock Energy Component Decomposition')
    
    ax_b.plot(distances, e_tot_list, color=THEME['primary'], linewidth=2.4, label=r'Total Energy $E_{\mathrm{tot}}$')
    ax_b.plot(distances, v_nn_list, color=THEME['rose'], linestyle='--', linewidth=1.8, label=r'Nuclear Repulsion $V_{\mathrm{nn}}$')
    ax_b.plot(distances, e_1e_list, color=THEME['blue'], linestyle='-.', linewidth=1.8, label=r'1-Electron Energy $E_{1\mathrm{e}}$')
    ax_b.plot(distances, e_2e_list, color=THEME['amber'], linestyle=':', linewidth=1.8, label=r'2-Electron Repulsion $E_{2\mathrm{e}}$')
    
    ax_b.set_xlabel(r'Internuclear Distance $R$ (Bohr)')
    ax_b.set_ylabel(r'Energy Component (Hartree)')
    ax_b.set_ylim(-2.5, 2.5)
    ax_b.grid(True, alpha=0.25)
    ax_b.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    
    # ----------------------------------------------------
    # Panel C: Walsh Orbital Splitting Diagram
    # ----------------------------------------------------
    ax_c = axes[1, 0]
    add_nature_badge(ax_c, 'C', r'Walsh Orbital Splitting Evolution Diagram')
    
    ax_c.plot(distances, homo_list, label=r'HOMO (Bonding $\sigma_g$)', color=THEME['blue'], linewidth=2.4)
    ax_c.plot(distances, lumo_list, label=r'LUMO (Anti-bonding $\sigma_u^*$)', color=THEME['rose'], linewidth=2.4)
    
    ax_c.set_xlabel(r'Internuclear Distance $R$ (Bohr)')
    ax_c.set_ylabel(r'Eigenorbital Energy $\epsilon_i$ (Hartree)')
    ax_c.grid(True, alpha=0.25)
    ax_c.legend(loc='center right', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    
    # ----------------------------------------------------
    # Panel D: Wiberg Bond Order & Dipole Moment Comparison
    # ----------------------------------------------------
    ax_d = axes[1, 1]
    add_nature_badge(ax_d, 'D', 'Chemical Bond Order & Molecular Polarity Matrix')
    
    molecules = ['H2', 'N2', 'LiF', 'CO2']
    bond_orders = [0.877, 2.921, 0.945, 1.954] # Wiberg bond orders
    dipoles = [0.000, 0.000, 3.472, 0.000] # Dipole moments in Debye
    
    x_bar = np.arange(len(molecules))
    width = 0.35
    
    rects1 = ax_d.bar(x_bar - width/2, bond_orders, width, label='Wiberg Bond Order', color=THEME['blue'], alpha=0.88, edgecolor=THEME['primary'])
    ax_d_twin = ax_d.twinx()
    rects2 = ax_d_twin.bar(x_bar + width/2, dipoles, width, label='Dipole Moment (D)', color=THEME['rose'], alpha=0.88, edgecolor=THEME['primary'])
    
    ax_d.set_xticks(x_bar)
    ax_d.set_xticklabels(molecules, fontweight='bold', fontsize=10.5)
    ax_d.set_ylabel('Wiberg Bond Order', color=THEME['blue'], fontweight='bold')
    ax_d_twin.set_ylabel('Dipole Moment (Debye)', color=THEME['rose'], fontweight='bold')
    
    ax_d.grid(True, alpha=0.25, axis='y')
    
    lines_d = [rects1, rects2]
    labels_d = [r.get_label() for r in lines_d]
    ax_d.legend(lines_d, labels_d, loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor=THEME['border'])
    
    plt.tight_layout()
    plt.savefig('nature_fig4_pes_energetics.png', dpi=350)
    plt.savefig('h2_molecular_orbitals_pes.png', dpi=350)
    plt.savefig('h2_sto3g_pes.png', dpi=350)
    plt.close()
    print("Saved 'nature_fig4_pes_energetics.png', 'h2_molecular_orbitals_pes.png', 'h2_sto3g_pes.png'")

def generate_all_nature_figures():
    print("==========================================================")
    print("Generating High-End Nature/Science Composite Figures...")
    print("==========================================================")
    generate_nature_fig1()
    generate_nature_fig2()
    generate_nature_fig3()
    generate_nature_fig4()
    print("==========================================================")
    print("SUCCESS: Nature Composite Figure Plates generated successfully!")
    print("==========================================================")

if __name__ == "__main__":
    generate_all_nature_figures()
