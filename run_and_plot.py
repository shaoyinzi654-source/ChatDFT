import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dft_engine import solve_1d_dft
from diatomic_engine import solve_diatomic_scf, solve_multi_atom_scf
from analysis_tools import (
    calculate_dos, calculate_pdos_3d, calculate_cdd_3d,
    eval_density_2d, calculate_cdd_2d, eval_density_3d_grid
)

# -----------------------------------------------------------------------------
# Global Publication-Grade Plotting Configuration (SCI Style)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial'],
    'axes.unicode_minus': False,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'grid.linestyle': '--',
    'grid.alpha': 0.25,
    'axes.facecolor': '#f8fafc',
    'figure.facecolor': 'white',
    'axes.edgecolor': '#94a3b8',
    'axes.linewidth': 1.0,
    'lines.solid_capstyle': 'round',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

PALETTE = {
    'navy': '#1e3a8a',
    'blue': '#2563eb',
    'sky': '#0284c7',
    'teal': '#0d9488',
    'green': '#16a34a',
    'amber': '#d97706',
    'coral': '#ea580c',
    'crimson': '#dc2626',
    'purple': '#9333ea',
    'slate': '#475569',
    'dark': '#0f172a'
}

def generate_dft_1d_results():
    """1. Generate 1D Kohn-Sham DFT Results Plot (dft_results.png)."""
    print("Generating 1D DFT ground state potential & density plot (dft_results.png)...")
    def vhe(x): return -2.0 / np.sqrt(x**2 + 1.0)
    res = solve_1d_dft(vhe, num_electrons=2, L=8.0, N=300, max_iter=100, tol=1e-6, functional="LDA")
    
    x = res['x']
    rho = res['density']
    v_ext = res['potentials']['Vext']
    v_h = res['potentials']['VH']
    v_xc = res['potentials']['Vxc']
    v_eff = res['potentials']['Veff']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Left: Density & External Potential
    ax1.plot(x, rho, color=PALETTE['navy'], linewidth=2.2, label=r'电子密度 $\rho(x)$ ($e$/Bohr)')
    ax1.fill_between(x, rho, color=PALETTE['navy'], alpha=0.15)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, v_ext, color=PALETTE['crimson'], linestyle='--', linewidth=1.8, label=r'核外势场 $V_{\mathrm{ext}}(x)$ (Ha)')
    ax1_twin.set_ylabel(r'外势 $V_{\mathrm{ext}}$ (Hartree)', color=PALETTE['crimson'])
    ax1_twin.tick_params(axis='y', labelcolor=PALETTE['crimson'])
    
    ax1.set_xlabel('一维空间坐标 $x$ (Bohr)')
    ax1.set_ylabel(r'电子密度 $\rho(x)$ ($e$/Bohr)', color=PALETTE['navy'])
    ax1.set_title('(a) 1D 类氦原子基态电子密度与外势')
    ax1.grid(True, alpha=0.3)
    
    # Combined legend for ax1
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, facecolor='white')
    
    # Right: Potential Components
    ax2.plot(x, v_ext, label=r'外势 $V_{\mathrm{ext}}$', color=PALETTE['crimson'], linestyle='--')
    ax2.plot(x, v_h, label=r'Hartree 势 $V_{\mathrm{H}}$', color=PALETTE['sky'], linewidth=1.8)
    ax2.plot(x, v_xc, label=r'交换相关势 $V_{\mathrm{xc}}$ (LDA)', color=PALETTE['amber'], linewidth=1.8)
    ax2.plot(x, v_eff, label=r'有效势 $V_{\mathrm{eff}}$', color=PALETTE['dark'], linewidth=2.2)
    
    ax2.set_xlabel('一维空间坐标 $x$ (Bohr)')
    ax2.set_ylabel('势能 $V(x)$ (Hartree)')
    ax2.set_title('(b) Kohn-Sham 有效势各分量分解')
    ax2.set_ylim(-2.5, 1.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', frameon=True, facecolor='white')
    
    plt.suptitle(r'一维 Kohn-Sham DFT 计算结果 ($Z=2, N_e=2$, LDA 泛函)', y=0.98, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('dft_results.png', dpi=300)
    plt.close()
    print("Saved 'dft_results.png'")

def generate_dft_1d_orbitals():
    """2. Generate 1D Kohn-Sham Orbitals Plot (dft_orbitals.png)."""
    print("Generating 1D DFT Kohn-Sham orbitals plot (dft_orbitals.png)...")
    def vhe(x): return -2.0 / np.sqrt(x**2 + 1.0)
    res = solve_1d_dft(vhe, num_electrons=2, L=8.0, N=300, max_iter=100, tol=1e-6)
    
    x = res['x']
    orbitals = res['all_orbitals']
    eps = res['all_eigenvalues']
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = [PALETTE['blue'], PALETTE['teal'], PALETTE['amber'], PALETTE['purple']]
    
    # Plot potential curve for reference
    ax.plot(x, res['potentials']['Veff'], color='#94a3b8', linestyle=':', label=r'有效势 $V_{\mathrm{eff}}(x)$', linewidth=1.5)
    
    # Plot top 4 orbitals, offset by their eigenvalues
    scale = 0.5
    for i in range(min(4, len(eps))):
        psi = orbitals[:, i]
        energy = eps[i]
        color = colors[i % len(colors)]
        occ_str = " (占据, 2e)" if i == 0 else " (未占据)"
        
        ax.plot(x, energy + psi * scale, color=color, linewidth=2.0,
                label=fr'$\psi_{i+1}(x)$ (E = {energy:.4f} Ha){occ_str}')
        ax.axhline(y=energy, color=color, linestyle='--', alpha=0.5, linewidth=0.8)
        ax.fill_between(x, energy, energy + psi * scale, color=color, alpha=0.12)
    
    ax.set_xlabel('一维空间坐标 $x$ (Bohr)')
    ax.set_ylabel('能量 / 波函数振幅 (Hartree)')
    ax.set_title(r'一维 Kohn-Sham 本征轨道波函数与能级谱分布 $\psi_n(x)$', weight='bold')
    ax.set_ylim(-1.5, 0.6)
    ax.set_xlim(-6, 6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=True, facecolor='white')
    
    plt.tight_layout()
    plt.savefig('dft_orbitals.png', dpi=300)
    plt.close()
    print("Saved 'dft_orbitals.png'")

def generate_scf_convergence():
    """3. Generate SCF Energy Convergence Plot (scf_convergence.png)."""
    print("Generating SCF convergence analysis plot (scf_convergence.png)...")
    def vhe(x): return -2.0 / np.sqrt(x**2 + 1.0)
    res_linear = solve_1d_dft(vhe, num_electrons=2, L=8.0, N=200, max_iter=40, tol=1e-7, mixing_method="Linear", alpha=0.2)
    res_pulay = solve_1d_dft(vhe, num_electrons=2, L=8.0, N=200, max_iter=40, tol=1e-7, mixing_method="Pulay", alpha=0.2)
    
    hist_lin = res_linear['history']
    hist_pul = res_pulay['history']
    
    iters_lin = list(range(1, len(hist_lin) + 1))
    e_lin = hist_lin
    d_lin = [1e-1] + [max(abs(hist_lin[i] - hist_lin[i-1]), 1e-12) for i in range(1, len(hist_lin))]
    
    iters_pul = list(range(1, len(hist_pul) + 1))
    e_pul = hist_pul
    d_pul = [1e-1] + [max(abs(hist_pul[i] - hist_pul[i-1]), 1e-12) for i in range(1, len(hist_pul))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Energy vs iteration
    ax1.plot(iters_lin, e_lin, marker='o', markersize=4, color=PALETTE['blue'], label='Linear Density Mixing', linewidth=1.8)
    ax1.plot(iters_pul, e_pul, marker='s', markersize=4, color=PALETTE['crimson'], label='Pulay DIIS Acceleration', linewidth=1.8)
    ax1.set_xlabel('SCF 迭代步数')
    ax1.set_ylabel('总能量 $E_{\mathrm{tot}}$ (Hartree)')
    ax1.set_title('(a) 自洽场 (SCF) 总能量演化曲线')
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, facecolor='white')
    
    # Energy error vs iteration (log scale)
    ax2.semilogy(iters_lin, d_lin, marker='o', markersize=4, color=PALETTE['blue'], label='Linear Mixing', linewidth=1.8)
    ax2.semilogy(iters_pul, d_pul, marker='s', markersize=4, color=PALETTE['crimson'], label='Pulay DIIS Acceleration', linewidth=1.8)
    ax2.axhline(y=1e-6, color='gray', linestyle='--', label=r'收敛阈值 ($10^{-6}$ Ha)')
    ax2.set_xlabel('SCF 迭代步数')
    ax2.set_ylabel(r'能量残差 $|\Delta E|$ (Hartree)')
    ax2.set_title(r'(b) 对数能量残差收敛速率对比 ($\log_{10}|\Delta E|$)')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(frameon=True, facecolor='white')
    
    plt.suptitle('自洽场 (SCF) 算法收敛性能分析', y=0.98, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('scf_convergence.png', dpi=300)
    plt.close()
    print("Saved 'scf_convergence.png'")

def generate_3d_density_plot(res_mol, atoms):
    """4. Create 3D density surface (h2_density_3d.png)."""
    print("Generating 3D electron density volume plot (h2_density_3d.png)...")
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
    ax.set_facecolor('#f8fafc')
    for level, alpha in zip(levels, [0.12, 0.22, 0.35, 0.50]):
        surface = np.ma.masked_less(rho, level)
        ax.plot_surface(Z, Y, surface, cmap='viridis', alpha=alpha,
                        linewidth=0, antialiased=True, vmin=0, vmax=np.max(rho))

    for idx, atom in enumerate(atoms):
        pos = atom['pos']
        color = '#e2e8f0' if atom['name'].upper() == 'H' else PALETTE['coral']
        ax.scatter([pos[2]], [pos[1]], [np.max(rho) * 0.02], s=180,
                   color=color, edgecolor=PALETTE['dark'], linewidth=1.5,
                   depthshade=True, label=f"{atom['name']} {idx + 1}")
    ax.plot([atoms[0]['pos'][2], atoms[1]['pos'][2]],
            [atoms[0]['pos'][1], atoms[1]['pos'][1]],
            [np.max(rho) * 0.02] * 2, color=PALETTE['crimson'], linewidth=3.5,
            label='H-H 化学键')

    ax.set_title('氢分子 ($H_2$) 3D 电子密度空间等值面分布', pad=18,
                 fontsize=14, color=PALETTE['dark'], weight='bold')
    ax.set_xlabel('键轴方向 $z$ (Bohr)', labelpad=9)
    ax.set_ylabel('径向方向 $y$ (Bohr)', labelpad=9)
    ax.set_zlabel(r'电子密度 $\rho$ ($e$/Bohr³)', labelpad=9)
    ax.view_init(elev=25, azim=-58)
    ax.set_box_aspect((1.35, 1.0, 0.72))
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cbd5e1')
    fig.tight_layout()
    fig.savefig('h2_density_3d.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved 'h2_density_3d.png'")

def generate_h2_sto3g_pes():
    """5. Generate STO-3G PES Energy Components Scan (h2_sto3g_pes.png)."""
    print("Generating H2 STO-3G PES energy decomposition scan (h2_sto3g_pes.png)...")
    distances = np.linspace(0.6, 4.0, 35)
    e_tot_list = []
    e_1e_list = []
    e_2e_list = []
    v_nn_list = []
    
    for r in distances:
        pos1 = [0.0, 0.0, -r/2.0]
        pos2 = [0.0, 0.0, r/2.0]
        res = solve_diatomic_scf('H', pos1, 'H', pos2, num_electrons=2, max_iter=50, tol=1e-6)
        e_tot_list.append(res['E_tot'])
        v_nn_list.append(res['E_nuc'])
        e_1e_list.append(res['E_kin'] + res['E_ext'])
        e_2e_list.append(res['E_ee'])
        
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(distances, e_tot_list, color=PALETTE['dark'], linewidth=2.5, label=r'自洽场总能量 $E_{\mathrm{tot}}$')
    ax.plot(distances, v_nn_list, color=PALETTE['crimson'], linestyle='--', linewidth=1.8, label=r'核核排斥能 $V_{\mathrm{nn}}$')
    ax.plot(distances, e_1e_list, color=PALETTE['blue'], linestyle='-.', linewidth=1.8, label=r'单电子能量 $E_{1\mathrm{e}}$')
    ax.plot(distances, e_2e_list, color=PALETTE['amber'], linestyle=':', linewidth=1.8, label=r'双电子排斥能 $E_{2\mathrm{e}}$')
    
    min_idx = np.argmin(e_tot_list)
    r_eq = distances[min_idx]
    e_min = e_tot_list[min_idx]
    
    ax.axvline(x=r_eq, color=PALETTE['crimson'], linestyle=':', alpha=0.7)
    ax.scatter([r_eq], [e_min], color=PALETTE['crimson'], s=70, zorder=5, label=fr'极小值点 $R_e = {r_eq:.2f}$ Bohr ($E = {e_min:.4f}$ Ha)')
    
    ax.set_xlabel('核间距 $R$ (Bohr)')
    ax.set_ylabel('能量 (Hartree)')
    ax.set_title('氢分子 ($H_2$) Hartree-Fock (STO-3G) 能量分量随键长演化', weight='bold')
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=True, facecolor='white')
    
    plt.tight_layout()
    plt.savefig('h2_sto3g_pes.png', dpi=300)
    plt.close()
    print("Saved 'h2_sto3g_pes.png'")

def generate_h2_contours_and_analysis():
    """6. Generate 2D contours, CDD, DOS, and PES plots."""
    R = 1.4
    atom1_name = 'H'
    atom1_pos = [0.0, 0.0, -R/2.0]
    atom2_name = 'H'
    atom2_pos = [0.0, 0.0, R/2.0]
    num_electrons = 2
    
    res_mol = solve_diatomic_scf(atom1_name, atom1_pos, atom2_name, atom2_pos, num_electrons=num_electrons)
    p1_z = atom1_pos[2]
    p2_z = atom2_pos[2]
    
    grid_y = np.linspace(-3.0, 3.0, 150)
    grid_z = np.linspace(-4.0, 4.0, 180)
    Y, Z = np.meshgrid(grid_y, grid_z)
    
    # 2D Density Contour
    print("Generating H2 2D Electron Density contour plot...")
    rho_2d = eval_density_2d(res_mol, Y, Z)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    contour = ax.contourf(Z, Y, rho_2d, levels=60, cmap='viridis')
    cbar = fig.colorbar(contour)
    cbar.set_label(r'电子密度 $\rho(y,z)$ ($e$/Bohr³)')
    
    c1 = plt.Circle((p1_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.5, zorder=5)
    c2 = plt.Circle((p2_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.5, zorder=5)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.text(p1_z, 0, 'H1', color='black', ha='center', va='center', weight='bold', zorder=6)
    ax.text(p2_z, 0, 'H2', color='black', ha='center', va='center', weight='bold', zorder=6)
    ax.plot([p1_z, p2_z], [0, 0], color='white', linestyle='--', linewidth=1.8, zorder=4)
    
    ax.set_xlabel('Z轴键方向位置 $z$ (Bohr)')
    ax.set_ylabel('Y轴径向位置 $y$ (Bohr)')
    ax.set_title('氢分子 ($H_2$) 二维电子密度等值线图 (YZ截面)', weight='bold')
    ax.set_aspect('equal')
    plt.savefig('h2_density_2d_contour.png', dpi=300)
    plt.close()
    print("Saved 'h2_density_2d_contour.png'")

    # 2D Charge Density Difference Contour
    print("Generating H2 2D CDD contour plot...")
    atoms = [{"name": atom1_name, "pos": atom1_pos}, {"name": atom2_name, "pos": atom2_pos}]
    generate_3d_density_plot(res_mol, atoms)
    cdd_2d, _, _ = calculate_cdd_2d(atoms, num_electrons, res_mol, Y, Z)
    
    fig, ax = plt.subplots(figsize=(7.5, 6))
    max_val = np.max(np.abs(cdd_2d))
    contour = ax.contourf(Z, Y, cdd_2d, levels=60, cmap='RdBu_r', vmin=-max_val, vmax=max_val)
    cbar = fig.colorbar(contour)
    cbar.set_label(r'差分电荷密度 $\Delta\rho(y,z)$ ($e$/Bohr³)')
    
    c1 = plt.Circle((p1_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.5, zorder=5)
    c2 = plt.Circle((p2_z, 0), radius=0.28, edgecolor='black', facecolor='white', linewidth=1.5, zorder=5)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.text(p1_z, 0, 'H1', color='black', ha='center', va='center', weight='bold', zorder=6)
    ax.text(p2_z, 0, 'H2', color='black', ha='center', va='center', weight='bold', zorder=6)
    ax.plot([p1_z, p2_z], [0, 0], color='black', linestyle='--', linewidth=1.5, zorder=4)
    
    ax.set_xlabel('Z轴键方向位置 $z$ (Bohr)')
    ax.set_ylabel('Y轴径向位置 $y$ (Bohr)')
    ax.set_title('氢分子 ($H_2$) 二维差分电荷密度分布图 (成键电子富集)', weight='bold')
    ax.set_aspect('equal')
    plt.savefig('h2_cdd_2d_contour.png', dpi=300)
    plt.close()
    print("Saved 'h2_cdd_2d_contour.png'")

    # 1D CDD Profile
    print("Generating H2 1D CDD axial profile...")
    z_grid = np.linspace(-4.0, 4.0, 300)
    cdd_1d, rho_mol, sum_atoms = calculate_cdd_3d(atoms, num_electrons, res_mol, z_grid)
    
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(z_grid, cdd_1d, label=r'差分电荷密度 $\Delta\rho(z)$', color=PALETTE['green'], linewidth=2.2)
    plt.plot(z_grid, rho_mol, label=r'分子总电子密度 $\rho_{\mathrm{mol}}(z)$', color=PALETTE['blue'], linestyle='--', alpha=0.8)
    plt.plot(z_grid, sum_atoms, label=r'原子叠加参考密度 $\rho_{\mathrm{atoms}}(z)$', color=PALETTE['slate'], linestyle=':', alpha=0.8)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.axvline(x=p1_z, color=PALETTE['crimson'], linestyle='--', linewidth=1.2)
    plt.axvline(x=p2_z, color=PALETTE['crimson'], linestyle='--', linewidth=1.2)
    
    plt.text(p1_z - 0.1, np.max(rho_mol)*0.5, 'H1', color=PALETTE['crimson'], ha='right', weight='bold')
    plt.text(p2_z + 0.1, np.max(rho_mol)*0.5, 'H2', color=PALETTE['crimson'], ha='left', weight='bold')
    
    plt.xlabel('Z轴键方向位置 $z$ (Bohr)')
    plt.ylabel(r'电子密度 ($e$/Bohr³)')
    plt.title('氢分子 ($H_2$) 沿键轴方向 1D 差分电荷密度剖面', weight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor='white', loc='upper right')
    plt.tight_layout()
    plt.savefig('h2_charge_density_difference.png', dpi=300)
    plt.close()
    print("Saved 'h2_charge_density_difference.png'")
    
    # DOS / PDOS
    print("Generating H2 DOS/PDOS plot...")
    E_grid = np.linspace(-1.5, 0.5, 400)
    sigma = 0.04
    e_vals = res_mol['eps']
    occs = [2.0 if i < 1 else 0.0 for i in range(len(e_vals))]
    
    dos = calculate_dos(e_vals, occs, E_grid, sigma)
    pdos1, pdos2 = calculate_pdos_3d(e_vals, res_mol['C'], res_mol['S'], E_grid, sigma)
    
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(E_grid, dos, label='总态密度 (Total DOS)', color=PALETTE['dark'], linewidth=2.2)
    plt.plot(E_grid, pdos1, label='原子 H1 投影态密度 (PDOS)', color=PALETTE['crimson'], linestyle='--', linewidth=1.8)
    plt.plot(E_grid, pdos2, label='原子 H2 投影态密度 (PDOS)', color=PALETTE['teal'], linestyle=':', linewidth=1.8)
    
    homo_val = e_vals[0]
    plt.axvline(x=homo_val, color=PALETTE['blue'], linestyle='-', linewidth=1.4, label=fr'费米面 / HOMO 能级 ({homo_val:.4f} Ha)')
    
    plt.xlabel('能量 $E$ (Hartree)')
    plt.ylabel('态密度 (States / Hartree)')
    plt.title('氢分子 ($H_2$) 态密度 (DOS) 与投影态密度 (PDOS)', weight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor='white', loc='upper left')
    plt.tight_layout()
    plt.savefig('h2_density_of_states.png', dpi=300)
    plt.close()
    print("Saved 'h2_density_of_states.png'")
    
    # Dual Panel: PES Scan & Walsh Diagram
    print("Generating H2 PES scan and Walsh diagram (h2_molecular_orbitals_pes.png)...")
    distances = np.linspace(0.6, 4.5, 32)
    energies = []
    homo_energies = []
    lumo_energies = []
    
    for r in distances:
        pos1 = [0.0, 0.0, -r/2.0]
        pos2 = [0.0, 0.0, r/2.0]
        res = solve_diatomic_scf('H', pos1, 'H', pos2, num_electrons=2, max_iter=50, tol=1e-6)
        energies.append(res['E_tot'])
        homo_energies.append(res['eps'][0])
        lumo_energies.append(res['eps'][1])
        
    min_idx = np.argmin(energies)
    eq_dist = distances[min_idx]
    eq_energy = energies[min_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))
    
    # PES Curve
    ax1.plot(distances, energies, linestyle='-', marker='o', color=PALETTE['dark'], linewidth=1.8, markersize=4, markerfacecolor='white', label='自洽场总能量')
    ax1.axvline(x=eq_dist, color=PALETTE['crimson'], linestyle='--', alpha=0.8, linewidth=1.2, label=fr'平衡键长 $R_e \approx {eq_dist:.2f}$ Bohr')
    ax1.scatter([eq_dist], [eq_energy], color=PALETTE['crimson'], s=70, zorder=5)
    
    ax1.set_xlabel('核间距 $R$ (Bohr)')
    ax1.set_ylabel('系统总能量 $E$ (Hartree)')
    ax1.set_title('(a) 氢分子势能曲面扫描 (PES)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, facecolor='white')
    
    # Orbital Splitting
    ax2.plot(distances, homo_energies, label=r'HOMO (成键轨道 $\sigma_g$)', color=PALETTE['blue'], linewidth=2.2)
    ax2.plot(distances, lumo_energies, label=r'LUMO (反键轨道 $\sigma_u^*$)', color=PALETTE['crimson'], linewidth=2.2)
    
    ax2.set_xlabel('核间距 $R$ (Bohr)')
    ax2.set_ylabel('本征轨道能量 (Hartree)')
    ax2.set_title('(b) 分子轨道能级杂化分裂演化 (Walsh 图)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, facecolor='white')
    
    plt.suptitle('氢分子自洽场电子能级演化与能量扫描图', y=0.98, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('h2_molecular_orbitals_pes.png', dpi=300)
    plt.close()
    print("Saved 'h2_molecular_orbitals_pes.png'")

def generate_all_plots():
    print("==========================================================")
    print("Generating ALL SCI publication-grade plots for ChatDFT...")
    print("==========================================================")
    generate_dft_1d_results()
    generate_dft_1d_orbitals()
    generate_scf_convergence()
    generate_h2_sto3g_pes()
    generate_h2_contours_and_analysis()
    print("==========================================================")
    print("SUCCESS: All software figures generated cleanly & optimized!")
    print("==========================================================")

if __name__ == "__main__":
    generate_all_plots()
