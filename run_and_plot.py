import numpy as np
import matplotlib.pyplot as plt
from dft_engine import solve_1d_dft
from diatomic_engine import solve_diatomic_scf, STO3GOrbital
from analysis_tools import calculate_dos, calculate_pdos_3d, calculate_cdd_3d, eval_density_2d, calculate_cdd_2d, eval_density_3d_grid

# Configure scientific publication-quality plotting style (SCI style) with Chinese support
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, # Correct display of minus signs
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'grid.linestyle': '--',
    'grid.alpha': 0.22,
    'axes.facecolor': '#f7f9fc',
    'figure.facecolor': 'white',
    'axes.edgecolor': '#b8c2d1',
    'axes.linewidth': 0.8,
    'lines.solid_capstyle': 'round',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

PALETTE = {
    'navy': '#17324d',
    'teal': '#087f8c',
    'coral': '#e76f51',
    'gold': '#e9a23b',
    'blue': '#277da1',
    'ink': '#263238',
}

def generate_3d_density_plot(res_mol, atoms):
    """Create a readable 3D density surface for the README and examples."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    y = np.linspace(-2.8, 2.8, 100)
    z = np.linspace(-3.8, 3.8, 120)
    Y, Z = np.meshgrid(y, z)
    X = np.zeros_like(Y)
    rho = eval_density_3d_grid(res_mol, X, Y, Z)
    rho = np.maximum(rho, 0.0)
    levels = np.percentile(rho[rho > 0], [55, 72, 86, 95])

    fig = plt.figure(figsize=(10, 7), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f7f9fc')
    for level, alpha in zip(levels, [0.12, 0.2, 0.3, 0.42]):
        surface = np.ma.masked_less(rho, level)
        ax.plot_surface(Z, Y, surface, cmap='viridis', alpha=alpha,
                        linewidth=0, antialiased=True, vmin=0, vmax=np.max(rho))

    for idx, atom in enumerate(atoms):
        pos = atom['pos']
        color = '#d9e2ec' if atom['name'].upper() == 'H' else PALETTE['coral']
        ax.scatter([pos[2]], [pos[1]], [np.max(rho) * 0.02], s=150,
                   color=color, edgecolor=PALETTE['ink'], linewidth=1.2,
                   depthshade=True, label=f"{atom['name']} {idx + 1}")
    ax.plot([atoms[0]['pos'][2], atoms[1]['pos'][2]],
            [atoms[0]['pos'][1], atoms[1]['pos'][1]],
            [np.max(rho) * 0.02] * 2, color=PALETTE['coral'], linewidth=3,
            label='H-H bond')

    ax.set_title('H2 electron density | 3D density surface', pad=18,
                 fontsize=15, color=PALETTE['ink'], weight='bold')
    ax.set_xlabel('Bond axis z (Bohr)', labelpad=9)
    ax.set_ylabel('Transverse y (Bohr)', labelpad=9)
    ax.set_zlabel('Density rho (e/Bohr^3)', labelpad=9)
    ax.view_init(elev=25, azim=-58)
    ax.set_box_aspect((1.35, 1.0, 0.72))
    ax.grid(True, alpha=0.18)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#d6dee8')
    fig.tight_layout()
    fig.savefig('h2_density_3d.png', dpi=240, bbox_inches='tight')
    plt.close(fig)
    print("Saved 'h2_density_3d.png'")

def generate_academic_plots():
    print("Running calculations and generating SCI publication-grade plots (including atomic structures)...")
    
    # ----------------------------------------------------
    # 1. Molecular Ground State Calculation (H2 at R=1.4 Bohr)
    # ----------------------------------------------------
    R = 1.4
    atom1_name = 'H'
    atom1_pos = [0.0, 0.0, -R/2.0]
    atom2_name = 'H'
    atom2_pos = [0.0, 0.0, R/2.0]
    num_electrons = 2
    
    res_mol = solve_diatomic_scf(
        atom1_name=atom1_name,
        atom1_pos=atom1_pos,
        atom2_name=atom2_name,
        atom2_pos=atom2_pos,
        num_electrons=num_electrons,
        max_iter=50,
        tol=1e-6
    )
    
    p1_z = atom1_pos[2]
    p2_z = atom2_pos[2]
    
    # Generate 2D grids (YZ plane, X=0)
    grid_y = np.linspace(-3.0, 3.0, 150)
    grid_z = np.linspace(-4.0, 4.0, 180)
    Y, Z = np.meshgrid(grid_y, grid_z)
    
    # ----------------------------------------------------
    # Plot 1: 2D Electron Density Contour Map with Atom Overlay
    # ----------------------------------------------------
    print("Generating H2 2D Electron Density contour plot with atoms...")
    rho_2d = eval_density_2d(res_mol, Y, Z)
    
    fig, ax = plt.subplots(figsize=(7.5, 6))
    contour = ax.contourf(Z, Y, rho_2d, levels=60, cmap='viridis')
    cbar = fig.colorbar(contour)
    cbar.set_label('电子密度 $\\rho(y,z)$ ($e$/Bohr³)')
    
    # Overlay H1 and H2 atom shapes
    c1 = plt.Circle((p1_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.5, zorder=5)
    c2 = plt.Circle((p2_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.5, zorder=5)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.text(p1_z, 0, 'H1', color='black', ha='center', va='center', weight='bold', zorder=6)
    ax.text(p2_z, 0, 'H2', color='black', ha='center', va='center', weight='bold', zorder=6)
    
    # Overlay chemical bond (dashed line)
    ax.plot([p1_z, p2_z], [0, 0], color='white', linestyle='--', linewidth=1.8, zorder=4)
    
    ax.set_xlabel('Z轴核键方向位置 $z$ (Bohr)')
    ax.set_ylabel('Y轴径向位置 $y$ (Bohr)')
    ax.set_title('氢分子 ($H_2$) 二维电子等密度线与原子结构重叠图')
    ax.set_aspect('equal')
    plt.savefig('h2_density_2d_contour.png')
    plt.close()
    print("Saved 'h2_density_2d_contour.png'")

    # ----------------------------------------------------
    # Plot 2: 2D Charge Density Difference (CDD) with Atom Overlay
    # ----------------------------------------------------
    print("Generating H2 2D Charge Density Difference (CDD) contour plot with atoms...")
    atoms = [{"name": atom1_name, "pos": atom1_pos}, {"name": atom2_name, "pos": atom2_pos}]
    generate_3d_density_plot(res_mol, atoms)
    cdd_2d, _, _ = calculate_cdd_2d(atoms, num_electrons, res_mol, Y, Z)
    
    fig, ax = plt.subplots(figsize=(7.5, 6))
    # CDD uses divergent 'RdBu' cmap
    max_val = np.max(np.abs(cdd_2d))
    contour = ax.contourf(Z, Y, cdd_2d, levels=60, cmap='RdBu', vmin=-max_val, vmax=max_val)
    cbar = fig.colorbar(contour)
    cbar.set_label('电荷密度差 $\\Delta\\rho(y,z)$ ($e$/Bohr³)')
    
    # Overlay atoms
    c1 = plt.Circle((p1_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.5, zorder=5)
    c2 = plt.Circle((p2_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.5, zorder=5)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.text(p1_z, 0, 'H1', color='black', ha='center', va='center', weight='bold', zorder=6)
    ax.text(p2_z, 0, 'H2', color='black', ha='center', va='center', weight='bold', zorder=6)
    
    # Bond line
    ax.plot([p1_z, p2_z], [0, 0], color='black', linestyle='--', linewidth=1.5, zorder=4)
    
    ax.set_xlabel('Z轴核键方向位置 $z$ (Bohr)')
    ax.set_ylabel('Y轴径向位置 $y$ (Bohr)')
    ax.set_title('氢分子 ($H_2$) 二维差分电荷密度与原子结构重叠图')
    ax.set_aspect('equal')
    plt.savefig('h2_cdd_2d_contour.png')
    plt.close()
    print("Saved 'h2_cdd_2d_contour.png'")

    # ----------------------------------------------------
    # Plot 3: 1D CDD projection profile
    # ----------------------------------------------------
    print("Generating H2 1D Charge Density Difference profile...")
    z_grid = np.linspace(-4.0, 4.0, 300)
    cdd_1d, rho_mol, sum_atoms = calculate_cdd_3d(atoms, num_electrons, res_mol, z_grid)
    
    plt.figure(figsize=(7, 5.5))
    plt.plot(z_grid, cdd_1d, label='差分电荷密度 $\\Delta\\rho(z)$', color='#2ca02c', linewidth=2.0)
    plt.plot(z_grid, rho_mol, label='分子总电荷密度 $\\rho_{\\mathrm{mol}}(z)$', color='#1f77b4', linestyle='--', alpha=0.7)
    plt.plot(z_grid, sum_atoms, label='原子叠加参考密度 $\\rho_{\\mathrm{atoms}}(z)$', color='#7f7f7f', linestyle=':', alpha=0.7)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.axvline(x=p1_z, color='#d62728', linestyle='--', linewidth=1.0)
    plt.axvline(x=p2_z, color='#d62728', linestyle='--', linewidth=1.0)
    
    plt.text(p1_z - 0.1, np.max(rho_mol)*0.5, 'H1', color='#d62728', ha='right', weight='bold')
    plt.text(p2_z + 0.1, np.max(rho_mol)*0.5, 'H2', color='#d62728', ha='left', weight='bold')
    
    plt.xlabel('Z轴核键方向位置 $z$ (Bohr)')
    plt.ylabel('电子云密度 $e$/Bohr³')
    plt.title('氢分子 ($H_2$) 沿键轴方向的差分电荷密度 (CDD)')
    plt.grid(True)
    plt.legend(frameon=True, edgecolor='black', fancybox=False)
    plt.savefig('h2_charge_density_difference.png')
    plt.close()
    print("Saved 'h2_charge_density_difference.png'")
    
    # ----------------------------------------------------
    # Plot 4: Density of States (DOS) & PDOS
    # ----------------------------------------------------
    print("Generating H2 Density of States (DOS/PDOS) plot...")
    E_grid = np.linspace(-1.5, 0.5, 400)
    sigma = 0.04
    e_vals = res_mol['eps']
    occs = [2.0 if i < 1 else 0.0 for i in range(len(e_vals))]
    
    dos = calculate_dos(e_vals, occs, E_grid, sigma)
    pdos1, pdos2 = calculate_pdos_3d(e_vals, res_mol['C'], res_mol['S'], E_grid, sigma)
    
    plt.figure(figsize=(7, 5.5))
    plt.plot(E_grid, dos, label='总态密度 (DOS)', color='black', linewidth=2.0)
    plt.plot(E_grid, pdos1, label='原子 H1 投影态密度 (PDOS)', color='#d62728', linestyle='--', linewidth=1.5)
    plt.plot(E_grid, pdos2, label='原子 H2 投影态密度 (PDOS)', color='#2ca02c', linestyle=':', linewidth=1.5)
    
    homo_val = e_vals[0]
    plt.axvline(x=homo_val, color='blue', linestyle='-', linewidth=1.2, label=f'费米面/HOMO ({homo_val:.4f} Ha)')
    
    plt.xlabel('能量 $E$ (Hartree)')
    plt.ylabel('态密度 (States/Hartree)')
    plt.title('氢分子 ($H_2$) 态密度 (DOS) 与投影态密度 (PDOS)')
    plt.grid(True)
    plt.legend(frameon=True, edgecolor='black', fancybox=False, loc='upper left')
    plt.savefig('h2_density_of_states.png')
    plt.close()
    print("Saved 'h2_density_of_states.png'")
    
    # ----------------------------------------------------
    # Plot 5: Dual Panel - PES Scan and HOMO-LUMO Energy Splitting
    # ----------------------------------------------------
    print("Generating H2 potential energy surface scan and orbital splitting curve...")
    distances = np.linspace(0.6, 4.5, 30)
    energies = []
    homo_energies = []
    lumo_energies = []
    
    for r in distances:
        pos1 = [0.0, 0.0, -r/2.0]
        pos2 = [0.0, 0.0, r/2.0]
        res = solve_diatomic_scf(
            atom1_name='H',
            atom1_pos=pos1,
            atom2_name='H',
            atom2_pos=pos2,
            num_electrons=2,
            max_iter=50,
            tol=1e-6
        )
        energies.append(res['E_tot'])
        homo_energies.append(res['eps'][0])
        lumo_energies.append(res['eps'][1])
        
    min_idx = np.argmin(energies)
    eq_dist = distances[min_idx]
    eq_energy = energies[min_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left Subplot: PES Curve
    ax1.plot(distances, energies, linestyle='-', marker='o', color='black', linewidth=1.5, markersize=5, markerfacecolor='none', markeredgecolor='black', label='自洽场总能量')
    ax1.axvline(x=eq_dist, color='#d62728', linestyle='--', alpha=0.8, linewidth=1.2, label=f'平衡键长 $R_e \\approx {eq_dist:.2f}$ Bohr')
    ax1.scatter([eq_dist], [eq_energy], color='#d62728', s=60, zorder=5)
    
    ax1.set_xlabel('核间距 $R$ (Bohr)')
    ax1.set_ylabel('系统总能量 $E$ (Hartree)')
    ax1.set_title('(a) 氢分子基态自洽势能曲线 (PES)')
    ax1.grid(True)
    ax1.legend(frameon=True, edgecolor='black', fancybox=False)
    
    # Right Subplot: Orbital splitting
    ax2.plot(distances, homo_energies, label='HOMO (成键轨道 - $\\sigma_g$)', color='#1f77b4', linewidth=2.0)
    ax2.plot(distances, lumo_energies, label='LUMO (反键轨道 - $\\sigma_u^*$)', color='#d62728', linewidth=2.0)
    
    ax2.set_xlabel('核间距 $R$ (Bohr)')
    ax2.set_ylabel('本征轨道能量 (Hartree)')
    ax2.set_title('(b) 分子轨道能级杂化分裂演化 (Walsh 图)')
    ax2.grid(True)
    ax2.legend(frameon=True, edgecolor='black', fancybox=False)
    
    plt.suptitle('氢分子自洽场电子能级演化与能量扫描图', y=0.98)
    plt.tight_layout()
    plt.savefig('h2_molecular_orbitals_pes.png')
    plt.close()
    print("Saved 'h2_molecular_orbitals_pes.png'")

if __name__ == "__main__":
    generate_academic_plots()
    print("\nAll SCI academic-quality plots generated successfully!")
