import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# 导入计算引擎与AI模块
from dft_engine import solve_1d_dft
from diatomic_engine import solve_diatomic_scf, STO3GOrbital, compute_dipole_moment_3d
from ai_helper import parse_user_request
from analysis_tools import calculate_dos, calculate_pdos_3d, calculate_pdos_multi, calculate_cdd_3d, calculate_cdd_1d, calculate_mep_grid_2d

# 页面配置
st.set_page_config(
    page_title="Chat DFT - AI自洽量子化学计算分析软件",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 纯净学术级界面样式（白底蓝灰色调，符合SCI论文审稿人审美）
st.markdown("""
<style>
    /* 全局背景与字体 */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Microsoft YaHei", sans-serif;
    }
    
    /* 标题样式 */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-family: 'SimHei', 'Times New Roman', sans-serif !important;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.3rem;
    }
    
    /* 系统主标题 */
    .app-title {
        font-size: 2.2rem !important;
        color: #1e3a8a !important;
        font-weight: 700;
        margin-bottom: 0.2rem;
        border-bottom: 2px solid #1e3a8a;
    }
    
    .app-subtitle {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    
    /* 学术卡片样式（纯白背景、细实线边框、无阴影） */
    .academic-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #cbd5e1;
    }
    
    /* 学术提示框 */
    .academic-alert {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        color: #1e40af;
        padding: 0.75rem 1rem;
        margin-bottom: 1.25rem;
        border-radius: 2px;
        font-size: 0.95rem;
    }
    
    /* 运行按钮样式（学术深蓝） */
    div.stButton > button {
        background-color: #1e3a8a !important;
        color: white !important;
        border: 1px solid #1e3a8a !important;
        border-radius: 4px !important;
        padding: 0.4rem 1.2rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* 标签页样式 */
    button[data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1e3a8a !important;
        border-bottom-color: #1e3a8a !important;
    }
</style>
""", unsafe_allow_html=True)

def polish_3d_figure(fig, height=540):
    """Apply one consistent, high-contrast style to interactive 3D figures."""
    axis_style = dict(
        showbackground=True,
        backgroundcolor="#f5f7fb",
        gridcolor="#d8e0ea",
        zerolinecolor="#b7c3d0",
        showspikes=False,
        title_font=dict(size=12, color="#26384a"),
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#26384a", size=12),
        margin=dict(l=0, r=0, t=42, b=0),
        height=height,
        legend=dict(bgcolor="rgba(255,255,255,0.86)", bordercolor="#d8e0ea", borderwidth=1),
    )
    fig.update_scenes(xaxis=axis_style, yaxis=axis_style, zaxis=axis_style,
                      camera=dict(eye=dict(x=1.55, y=1.55, z=1.15)))
    return fig

# 安全计算用户输入的物理势能表达式
def make_potential_fn(expr_str):
    allowed_names = {
        'x': None,
        'np': np,
        'numpy': np,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi
    }
    def potential_fn(x):
        local_vars = allowed_names.copy()
        local_vars['x'] = x
        try:
            return eval(expr_str, {"__builtins__": {}}, local_vars)
        except Exception as e:
            st.error(f"势能函数表达式解析失败: {e}。系统已默认降级为谐振子势: 0.5 * x^2")
            return 0.5 * (x ** 2)
    return potential_fn

def run_vibrational_analysis(atoms, num_electrons, multiplicity=None):
    """
    计算分子体系的数值 Hessian 矩阵，进行质量加权正交化，并求解红外振动频率与强度。
    """
    from diatomic_engine import solve_multi_atom_scf, compute_dipole_moment_multi
    import copy
    
    num_at = len(atoms)
    # 原子质量 (AMU)
    MASSES = {
        'H': 1.008, 'HE': 4.0026, 'LI': 6.94, 'BE': 9.0122, 'B': 10.81, 'C': 12.011,
        'N': 14.007, 'O': 15.999, 'F': 18.998, 'NE': 20.180, 'NA': 22.990, 'MG': 24.305, 'G': 1.0
    }
    
    m = np.array([MASSES.get(at["name"].upper(), 1.0) for at in atoms])
    
    # 双侧微扰差分计算力常数矩阵 (Hessian, 3M x 3M)
    d = 0.006
    H = np.zeros((3*num_at, 3*num_at))
    dipole_deriv = np.zeros((3*num_at, 3)) # dMu/dX
    
    for j in range(num_at):
        for b in range(3):
            col_idx = 3*j + b
            
            # 位移 +d
            atoms_p = copy.deepcopy(atoms)
            atoms_p[j]["pos"][b] += d
            res_p = solve_multi_atom_scf(atoms_p, num_electrons=num_electrons, max_iter=45, tol=1e-5, multiplicity=multiplicity)
            mu_p, _ = compute_dipole_moment_multi(atoms_p, res_p)
            
            # 位移 -d
            atoms_m = copy.deepcopy(atoms)
            atoms_m[j]["pos"][b] -= d
            res_m = solve_multi_atom_scf(atoms_m, num_electrons=num_electrons, max_iter=45, tol=1e-5, multiplicity=multiplicity)
            mu_m, _ = compute_dipole_moment_multi(atoms_m, res_m)
            
            # 在 +d 和 -d 处数值计算受力 F = -dE/dX
            forces_p = np.zeros((num_at, 3))
            forces_m = np.zeros((num_at, 3))
            fd_step = 0.003
            
            for i in range(num_at):
                for a in range(3):
                    # +d
                    atoms_p[i]["pos"][a] += fd_step
                    res_pp = solve_multi_atom_scf(atoms_p, num_electrons=num_electrons, max_iter=25, tol=1e-4, multiplicity=multiplicity)
                    atoms_p[i]["pos"][a] -= 2*fd_step
                    res_pm = solve_multi_atom_scf(atoms_p, num_electrons=num_electrons, max_iter=25, tol=1e-4, multiplicity=multiplicity)
                    atoms_p[i]["pos"][a] += fd_step
                    forces_p[i, a] = -(res_pp["E_tot"] - res_pm["E_tot"]) / (2*fd_step)
                    
                    # -d
                    atoms_m[i]["pos"][a] += fd_step
                    res_mp = solve_multi_atom_scf(atoms_m, num_electrons=num_electrons, max_iter=25, tol=1e-4, multiplicity=multiplicity)
                    atoms_m[i]["pos"][a] -= 2*fd_step
                    res_mm = solve_multi_atom_scf(atoms_m, num_electrons=num_electrons, max_iter=25, tol=1e-4, multiplicity=multiplicity)
                    atoms_m[i]["pos"][a] += fd_step
                    forces_m[i, a] = -(res_mp["E_tot"] - res_mm["E_tot"]) / (2*fd_step)
            
            # H_ia, jb = -(F_ia_plus - F_ia_minus) / 2d
            for i in range(num_at):
                for a in range(3):
                    row_idx = 3*i + a
                    H[row_idx, col_idx] = -(forces_p[i, a] - forces_m[i, a]) / (2*d)
            
            # 偶极矩一阶导数 dMu/dX_jb
            dipole_deriv[col_idx, :] = (mu_p - mu_m) / (2*d)
            
    # 强制对称化
    H = 0.5 * (H + H.T)
    
    # 质量加权 Hessian 矩阵 M
    M = np.zeros_like(H)
    for i in range(num_at):
        for a in range(3):
            for j in range(num_at):
                for b in range(3):
                    r = 3*i + a
                    c = 3*j + b
                    M[r, c] = H[r, c] / np.sqrt(m[i] * m[j])
                    
    # 求解本征值与本征向量
    eigvals, eigvecs = np.linalg.eigh(M)
    
    frequencies = []
    intensities = []
    
    # 频率转换因子 (Hartree/(AMU*Bohr^2))^0.5 -> cm^-1
    conv = 1085.7
    
    for k in range(3*num_at):
        val = eigvals[k]
        if val <= 1e-5:
            freq = 0.0
        else:
            freq = conv * np.sqrt(val)
            
        # 红外强度: dMu/dQ = sum_ia (dMu/dX_ia) * (1/sqrt(m_i)) * e_{ia, k}
        dmu_dq = np.zeros(3)
        for i in range(num_at):
            for a in range(3):
                row_idx = 3*i + a
                dmu_dq += dipole_deriv[row_idx, :] * (1.0 / np.sqrt(m[i])) * eigvecs[row_idx, k]
                
        intensity = np.sum(dmu_dq ** 2) * 42.256
        frequencies.append(freq)
        intensities.append(intensity)
        
    return frequencies, intensities, eigvecs

# 顶部主标题
st.markdown('<div class="app-title">Chat DFT</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">基于自洽密度泛函理论 (DFT) 和量子化学自洽场方法的智能分析与绘图系统</div>', unsafe_allow_html=True)

def sync_config_to_sidebar_state():
    if "config" in st.session_state and st.session_state.config.get("solver_type") == "3d_diatomic":
        params = st.session_state.config.get("params", {})
        if "atoms" in params:
            xyz_lines = []
            for at in params["atoms"]:
                xyz_lines.append(f"{at['name']} {at['pos'][0]} {at['pos'][1]} {at['pos'][2]}")
            xyz_str = "\n".join(xyz_lines)
            st.session_state["sidebar_xyz_str"] = xyz_str
            st.session_state["sidebar_ne_3d"] = int(params.get("num_electrons", 2))
            st.session_state["sidebar_max_it_3d"] = int(params.get("max_iter", 50))
            
            # 同步多重度
            num_e = int(params.get("num_electrons", 2))
            default_mult = int(params.get("multiplicity", 1 if (num_e % 2 == 0) else 2))
            st.session_state["sidebar_mult_3d"] = default_mult
            
            st.session_state["preset_select_widget"] = "自定义 (Custom)"
            st.session_state["coord_mode_radio"] = "多原子 XYZ 编辑器"
            st.session_state.prev_preset = "自定义 (Custom)"

# 自洽迭代计算配置缓存初始化
if "config" not in st.session_state:
    st.session_state.config = {
        "solver_type": "1d_dft",
        "explanation": "默认加载：一维类氦原子软库仑势模型。",
        "params": {
            "num_electrons": 2,
            "L": 10.0,
            "N": 200,
            "max_iter": 100,
            "tol": 1e-6,
            "alpha": 0.2,
            "softening": 1.0,
            "potential_expr": "-2.0 / np.sqrt(x ** 2 + 1.0)",
            "potential_description": "一维类氦原子模型 (Z=2, 软化常数 a=1.0)"
        }
    }
if "calc_results" not in st.session_state:
    st.session_state.calc_results = None
if "last_solver_type" not in st.session_state:
    st.session_state.last_solver_type = st.session_state.config["solver_type"]
if "auto_optimize" not in st.session_state:
    st.session_state.auto_optimize = False

# ----------------- 侧边栏：AI 智能助手模型选择 -----------------
st.sidebar.markdown("### 🤖 AI 智能模型选择")
ai_model_select = st.sidebar.selectbox(
    "选择 AI 翻译模型 (Tokken.cc)",
    options=["gpt-5.6-luna", "gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "deepseek-chat"],
    index=0,
    help="您可以选择更强大的大语言模型来翻译和解析您的自然语言指令。"
)

# ----------------- 侧边栏：手动参数调节面板 -----------------
st.sidebar.markdown("### 🛠️ 量子化学计算设置 (手动)")
solver_select = st.sidebar.selectbox(
    "选择求解计算引擎",
    options=["1d_dft", "3d_diatomic", "3d_md"],
    format_func=lambda x: (
        "1D Kohn-Sham 密度泛函求解器" if x == "1d_dft" else (
        "3D STO-3G 分子自洽场求解器 (UHF)" if x == "3d_diatomic" else 
        "3D 经典分子动力学 (MD) 大规模模拟引擎"
    )),
    index=0 if st.session_state.config["solver_type"] == "1d_dft" else (1 if st.session_state.config["solver_type"] == "3d_diatomic" else 2)
)

# 检测求解器切换，若切换则清空历史计算结果以防数据格式冲突
if solver_select != st.session_state.last_solver_type:
    st.session_state.calc_results = None
    st.session_state.ads_results = None
    st.session_state.last_solver_type = solver_select

st.session_state.config["solver_type"] = solver_select

if solver_select == "1d_dft":
    st.sidebar.markdown("#### 1. 网格离散化参数")
    num_e = st.sidebar.number_input("系统电子总数 (N_e)", min_value=1, max_value=100, value=int(st.session_state.config["params"].get("num_electrons", 2)))
    grid_L = st.sidebar.slider("网格边界 L (Bohr)", min_value=3.0, max_value=20.0, value=float(st.session_state.config["params"].get("L", 10.0)), step=0.5)
    grid_N = st.sidebar.slider("网格离散点数 (N)", min_value=50, max_value=300, value=int(st.session_state.config["params"].get("N", 200)), step=10)
    
    st.sidebar.markdown("#### 2. 自洽迭代与泛函控制")
    max_it = st.sidebar.number_input("最大自洽场迭代次数", min_value=10, max_value=200, value=int(st.session_state.config["params"].get("max_iter", 100)))
    mix_alpha = st.sidebar.slider("电荷密度混合率 (α)", min_value=0.05, max_value=1.0, value=float(st.session_state.config["params"].get("alpha", 0.2)), step=0.05)
    soft_val = st.sidebar.slider("静电势软化因子 (a)", min_value=0.1, max_value=3.0, value=float(st.session_state.config["params"].get("softening", 1.0)), step=0.1)
    
    func_options = ["LDA", "GGA-PBE", "Exchange-Only", "Hartree"]
    curr_func = st.session_state.config["params"].get("functional", "LDA")
    func_idx = func_options.index(curr_func) if curr_func in func_options else 0
    func_select = st.sidebar.selectbox("交换关联泛函 (XC Functional)", func_options, index=func_idx)
    
    mix_options = ["Linear", "Anderson"]
    curr_mix = st.session_state.config["params"].get("mixing_method", "Linear")
    mix_idx = mix_options.index(curr_mix) if curr_mix in mix_options else 0
    mix_select = st.sidebar.selectbox("电荷混能方法 (Mixing Scheme)", mix_options, index=mix_idx)
    
    st.sidebar.markdown("#### 3. 外部核吸引势 V_ext(x)")
    pot_expr = st.sidebar.text_input("V_ext(x) 数学表达式", value=st.session_state.config["params"].get("potential_expr", "-2.0 / np.sqrt(x**2 + 1.0)"))
    pot_desc = st.sidebar.text_input("势场物理描述", value=st.session_state.config["params"].get("potential_description", "一维类氦原子"))
    
    st.session_state.config["params"] = {
        "num_electrons": num_e,
        "L": grid_L,
        "N": grid_N,
        "max_iter": max_it,
        "alpha": mix_alpha,
        "softening": soft_val,
        "functional": func_select,
        "mixing_method": mix_select,
        "potential_expr": pot_expr,
        "potential_description": pot_desc,
        "tol": 1e-6
    }
elif solver_select == "3d_diatomic":
    st.sidebar.markdown("#### 1. 3D 原子坐标模式")
    coord_mode = st.sidebar.radio("配置模式", ["双原子快捷滑块", "多原子 XYZ 编辑器"], index=0, key="coord_mode_radio")
    
    if coord_mode == "双原子快捷滑块":
        elements = ["H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG"]
        atom1_val = st.session_state.config["params"].get("atom1_name", "H").upper()
        atom2_val = st.session_state.config["params"].get("atom2_name", "H").upper()
        idx1 = elements.index(atom1_val) if atom1_val in elements else 0
        idx2 = elements.index(atom2_val) if atom2_val in elements else 0
        
        atom1 = st.sidebar.selectbox("原子 1 类型", elements, index=idx1, key="sidebar_atom1")
        atom2 = st.sidebar.selectbox("原子 2 类型", elements, index=idx2, key="sidebar_atom2")
        r_dist = st.sidebar.slider("核间距 R (Bohr)", min_value=0.2, max_value=6.0, value=1.4, step=0.1, key="sidebar_rdist")
        
        atom1_pos = [0.0, 0.0, -r_dist/2.0]
        atom2_pos = [0.0, 0.0, r_dist/2.0]
        
        atoms = [
            {"name": atom1, "pos": atom1_pos},
            {"name": atom2, "pos": atom2_pos}
        ]
    else:
        # 多原子 XYZ 编辑器与预设库
        PRESETS = {
            "自定义 (Custom)": None,
            "H2O (水分子, 10e)": {
                "xyz": "O 0.0 0.0 0.12\nH 0.0 1.43 -0.98\nH 0.0 -1.43 -0.98",
                "ne": 10
            },
            "CO2 (二氧化碳, 22e)": {
                "xyz": "C 0.0 0.0 0.0\nO 0.0 0.0 -2.19\nO 0.0 0.0 2.19",
                "ne": 22
            },
            "NH3 (氨分子, 10e)": {
                "xyz": "N 0.0 0.0 0.25\nH 0.0 1.77 -0.58\nH 1.53 -0.88 -0.58\nH -1.53 -0.88 -0.58",
                "ne": 10
            },
            "CH4 (甲烷, 10e)": {
                "xyz": "C 0.0 0.0 0.0\nH 1.18 1.18 1.18\nH -1.18 -1.18 1.18\nH -1.18 1.18 -1.18\nH 1.18 -1.18 -1.18",
                "ne": 10
            },
            "CO (一氧化碳, 14e)": {
                "xyz": "C 0.0 0.0 -1.06\nO 0.0 0.0 1.06",
                "ne": 14
            },
            "N2 (氮气, 14e)": {
                "xyz": "N 0.0 0.0 -1.04\nN 0.0 0.0 1.04",
                "ne": 14
            },
            "O2 (氧气, 16e)": {
                "xyz": "O 0.0 0.0 -1.13\nO 0.0 0.0 1.13",
                "ne": 16
            },
            "HF (氟化氢, 10e)": {
                "xyz": "H 0.0 0.0 -1.63\nF 0.0 0.0 0.18",
                "ne": 10
            },
            "LiF (氟化锂, 12e)": {
                "xyz": "LI 0.0 0.0 -1.50\nF 0.0 0.0 1.50",
                "ne": 12
            }
        }
        
        preset_name = st.sidebar.selectbox("📂 选择分子结构预设", list(PRESETS.keys()), index=0, key="preset_select_widget")
        
        if "prev_preset" not in st.session_state:
            st.session_state.prev_preset = "自定义 (Custom)"
            
        curr_atoms = st.session_state.config["params"].get("atoms", [
            {"name": "O", "pos": [0.0, 0.0, 0.12]},
            {"name": "H", "pos": [0.0, 1.43, -0.98]},
            {"name": "H", "pos": [0.0, -1.43, -0.98]}
        ])
        xyz_lines = []
        for at in curr_atoms:
            xyz_lines.append(f"{at['name']} {at['pos'][0]} {at['pos'][1]} {at['pos'][2]}")
        default_xyz_str = "\n".join(xyz_lines)
        
        if preset_name != "自定义 (Custom)" and preset_name != st.session_state.prev_preset:
            st.session_state["sidebar_xyz_str"] = PRESETS[preset_name]["xyz"]
            st.session_state["sidebar_ne_3d"] = PRESETS[preset_name]["ne"]
            st.session_state.prev_preset = preset_name
            default_xyz_str = PRESETS[preset_name]["xyz"]
        elif preset_name == "自定义 (Custom)":
            st.session_state.prev_preset = "自定义 (Custom)"
            
        xyz_str = st.sidebar.text_area("输入 XYZ 格式坐标 (元素 X Y Z，单位 Bohr)", value=default_xyz_str, height=150, key="sidebar_xyz_str")
        
        uploaded_xyz = st.sidebar.file_uploader("📤 导入本地 .xyz 文件", type=["xyz", "txt"], key="xyz_uploader")
        if uploaded_xyz is not None:
            content_bytes = uploaded_xyz.read()
            content_str = content_bytes.decode("utf-8")
            lines = [l.strip() for l in content_str.split("\n") if l.strip()]
            parsed_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) == 4:
                    parsed_lines.append(line)
            if parsed_lines:
                new_xyz_str = "\n".join(parsed_lines)
                st.session_state["sidebar_xyz_str"] = new_xyz_str
                xyz_str = new_xyz_str
                st.sidebar.success("🎉 .xyz 坐标导入成功！")
                
        atoms = []
        for line in xyz_str.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 4:
                try:
                    name = parts[0].upper()
                    pos = [float(parts[1]), float(parts[2]), float(parts[3])]
                    atoms.append({"name": name, "pos": pos})
                except ValueError:
                    pass
        if not atoms:
            atoms = curr_atoms
            
        # --- 坐标单位一键转换 & 质心对齐快捷工具 ---
        st.sidebar.markdown("##### ⚙️ 坐标编辑辅助工具")
        col_c1, col_c2 = st.sidebar.columns(2)
        with col_c1:
            if st.sidebar.button("🔄 转为埃 (Å)", use_container_width=True, key="btn_to_angstrom"):
                try:
                    lines = st.session_state["sidebar_xyz_str"].strip().split("\n")
                    new_lines = []
                    for l in lines:
                        parts = l.strip().split()
                        if len(parts) == 4:
                            name = parts[0]
                            pos = [float(p) / 1.889726 for p in parts[1:4]]
                            new_lines.append(f"{name} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
                    st.session_state["sidebar_xyz_str"] = "\n".join(new_lines)
                    st.sidebar.success("已成功转换为埃 (Å)！")
                    st.rerun()
                except Exception:
                    st.sidebar.error("格式错误")
        with col_c2:
            if st.sidebar.button("🔄 转为 Bohr", use_container_width=True, key="btn_to_bohr"):
                try:
                    lines = st.session_state["sidebar_xyz_str"].strip().split("\n")
                    new_lines = []
                    for l in lines:
                        parts = l.strip().split()
                        if len(parts) == 4:
                            name = parts[0]
                            pos = [float(p) * 1.889726 for p in parts[1:4]]
                            new_lines.append(f"{name} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
                    st.session_state["sidebar_xyz_str"] = "\n".join(new_lines)
                    st.sidebar.success("已成功转换为 Bohr！")
                    st.rerun()
                except Exception:
                    st.sidebar.error("格式错误")
                    
        if st.sidebar.button("📍 一键对齐分子质心到原点", use_container_width=True, key="btn_com_align"):
            try:
                lines = st.session_state["sidebar_xyz_str"].strip().split("\n")
                atoms_temp = []
                for l in lines:
                    parts = l.strip().split()
                    if len(parts) == 4:
                        atoms_temp.append({"name": parts[0], "pos": [float(p) for p in parts[1:4]]})
                if atoms_temp:
                    coords = np.array([at["pos"] for at in atoms_temp])
                    mean_coord = np.mean(coords, axis=0)
                    new_lines = []
                    for at in atoms_temp:
                        new_pos = np.array(at["pos"]) - mean_coord
                        new_lines.append(f"{at['name']} {new_pos[0]:.6f} {new_pos[1]:.6f} {new_pos[2]:.6f}")
                    st.session_state["sidebar_xyz_str"] = "\n".join(new_lines)
                    st.sidebar.success("质心对齐成功！")
                    st.rerun()
            except Exception:
                st.sidebar.error("格式错误")
                
        # --- 几何构型测量工具 ---
        with st.sidebar.expander("🔍 几何键长/键角测量工具"):
            st.markdown("**1-based 原子索引列表：**")
            for idx, at in enumerate(atoms):
                st.markdown(f"- **#{idx+1}** : `{at['name']}` ({at['pos'][0]:.2f}, {at['pos'][1]:.2f}, {at['pos'][2]:.2f})")
                
            m_type = st.radio("测量对象", ["键长 (距离)", "键角 (夹角)"], key="measure_type_radio")
            if m_type == "键长 (距离)":
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    m_idx1 = st.number_input("原子 A", min_value=1, max_value=len(atoms), value=1, step=1, key="meas_len_idx1")
                with col_m2:
                    m_idx2 = st.number_input("原子 B", min_value=1, max_value=len(atoms), value=min(2, len(atoms)), step=1, key="meas_len_idx2")
                
                if m_idx1 != m_idx2:
                    p1 = np.array(atoms[m_idx1 - 1]["pos"])
                    p2 = np.array(atoms[m_idx2 - 1]["pos"])
                    d_bohr = np.linalg.norm(p1 - p2)
                    d_ang = d_bohr / 1.889726
                    st.success(f"📏 **测量结果**：\n* **{d_bohr:.4f} Bohr**\n* **{d_ang:.4f} Å**")
            else:
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    m_idx1 = st.number_input("顶点 A", min_value=1, max_value=len(atoms), value=1, step=1, key="meas_ang_idx1")
                with col_m2:
                    m_idx2 = st.number_input("顶点 B (中)", min_value=1, max_value=len(atoms), value=min(2, len(atoms)), step=1, key="meas_ang_idx2")
                with col_m3:
                    m_idx3 = st.number_input("顶点 C", min_value=1, max_value=len(atoms), value=min(3, len(atoms)), step=1, key="meas_ang_idx3")
                
                if m_idx1 != m_idx2 and m_idx2 != m_idx3 and m_idx1 != m_idx3:
                    p1 = np.array(atoms[m_idx1 - 1]["pos"])
                    p2 = np.array(atoms[m_idx2 - 1]["pos"])
                    p3 = np.array(atoms[m_idx3 - 1]["pos"])
                    v21 = p1 - p2
                    v23 = p3 - p2
                    norm_prod = np.linalg.norm(v21) * np.linalg.norm(v23)
                    if norm_prod > 1e-8:
                        cos_theta = np.dot(v21, v23) / norm_prod
                        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                        theta_deg = np.degrees(theta_rad)
                        st.success(f"📐 **测量结果**：\n* **{theta_deg:.2f}°**")
            
    num_e_3d = st.sidebar.number_input("系统总电子数", min_value=1, max_value=24, value=int(st.session_state.config["params"].get("num_electrons", 2)), key="sidebar_ne_3d")
    max_it_3d = st.sidebar.number_input("最大自洽场迭代次数", min_value=10, max_value=100, value=int(st.session_state.config["params"].get("max_iter", 50)), key="sidebar_max_it_3d")
    
    # 动态确定物理可行的自旋多重度选项 (2S+1)
    if num_e_3d % 2 == 0:
        mult_options = [1, 3, 5, 7]
    else:
        mult_options = [2, 4, 6, 8]
        
    default_mult = int(st.session_state.config["params"].get("multiplicity", mult_options[0]))
    if default_mult not in mult_options:
        default_mult = mult_options[0]
        
    mult_idx = mult_options.index(default_mult)
    mult_3d = st.sidebar.selectbox(
        "自旋多重度 (2S+1)", 
        options=mult_options, 
        index=mult_idx, 
        key="sidebar_mult_3d",
        help="对于开壳层体系（如单电子自由基、三重态氧分子），请选择匹配的自旋多重度以启动 UHF (无限制性 Hartree-Fock) 求解器。"
    )
    
    st.session_state.config["params"] = {
        "atoms": atoms,
        "num_electrons": num_e_3d,
        "max_iter": max_it_3d,
        "multiplicity": mult_3d,
        "tol": 1e-6
    }
    # 兼容旧组件的单原子读取
    if len(atoms) >= 2:
        st.session_state.config["params"]["atom1_name"] = atoms[0]["name"]
        st.session_state.config["params"]["atom1_pos"] = atoms[0]["pos"]
        st.session_state.config["params"]["atom2_name"] = atoms[1]["name"]
        st.session_state.config["params"]["atom2_pos"] = atoms[1]["pos"]
else:
    st.sidebar.markdown("#### 1. 大规模分子动力学系统配置")
    md_sys = st.sidebar.selectbox("模拟分子/材料体系", ["Argon (液氩/气氩, 单原子流体)", "Nitrogen (N2, 柔性双原子分子)"], key="md_sys_select")
    
    # 粒子数设置
    if md_sys.startswith("Argon"):
        n_parts = st.sidebar.slider("原子数量 (Atoms)", min_value=16, max_value=128, value=64, step=8, key="md_n_parts")
    else:
        n_parts = st.sidebar.slider("分子数量 (Molecules)", min_value=8, max_value=64, value=32, step=8, key="md_n_parts")
        
    md_t = st.sidebar.slider("初始平衡温度 (K)", min_value=10.0, max_value=800.0, value=150.0, step=10.0, key="md_t_target")
    md_box = st.sidebar.slider("周期晶胞盒子长度 L (Å)", min_value=10.0, max_value=30.0, value=14.0, step=1.0, key="md_box_len")
    md_dt = st.sidebar.slider("积分时间步长 dt (ps)", min_value=0.0005, max_value=0.004, value=0.001, step=0.0005, key="md_timestep")
    md_steps = st.sidebar.slider("运行总仿真步数 (Steps)", min_value=100, max_value=2000, value=600, step=100, key="md_sim_steps")
    
    st.session_state.config["params"] = {
        "system_type": md_sys,
        "n_particles": n_parts,
        "temp": md_t,
        "box_length": md_box,
        "dt": md_dt,
        "max_iter": md_steps,
        "tol": 1e-6
    }

# 全局获取当前配置和求解器类型，确保各组件在任何 rerun 中都能正常读取，防止 NameError
solver_mode = st.session_state.config["solver_type"]
params = st.session_state.config["params"]

# ----------------- 主界面布局 -----------------
col_prompt, col_info = st.columns([2, 1])

with col_prompt:
    st.markdown('<div class="academic-card">', unsafe_allow_html=True)
    st.markdown("### 📋 物理计算体系智能设置 (AI翻译输入)")
    user_prompt = st.text_area(
        "请在此用中文描述您想要计算和分析的量子物理系统（AI将自动翻译为底层求解参数，例如：'模拟一个包含4个电子的谐振子量子点势能阱'，或者'扫描H2分子的基态能量分布'）：",
        placeholder="例如：建立一个一维H2分子模型，两个核吸引势中心间距为3.0 Bohr，电子数为2...",
        key="user_prompt_text_area"
    )
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        ai_submit = st.button("✨ 运行 AI 一键计算", key="btn_ai_calc_plain")
    with col_btn2:
        manual_submit = st.button("🏃 直接运行当前手动设置", key="btn_manual_calc_plain")
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    st.markdown('<div class="academic-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown("### 🔍 底层物理模拟参数配置")
    st.markdown(f"**求解器类型**: `{st.session_state.config['solver_type']}`")
    st.markdown(f"**模型解释**: *{st.session_state.config['explanation']}*")
    
    if st.session_state.config["solver_type"] == "1d_dft":
        st.markdown(f"- **总电子数**: `{st.session_state.config['params'].get('num_electrons')} e`")
        st.markdown(f"- **核电荷势**: `{st.session_state.config['params'].get('potential_description')}`")
        st.markdown(f"- **势函数表达式**: `{st.session_state.config['params'].get('potential_expr')}`")
    else:
        st.markdown(f"- **总电子数**: `{st.session_state.config['params'].get('num_electrons')} e`")
        
        # 实时分子结构3D立体结构预览
        st.markdown("##### ⚛️ 3D 分子构型实时预览 (Structure Preview)")
        preview_atoms = st.session_state.config["params"].get("atoms", [])
        
        fig_prev = go.Figure()
        for idx, at in enumerate(preview_atoms):
            name = at["name"]
            pos = at["pos"]
            color = 'lightgrey' if name == 'H' else 'pink'
            fig_prev.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers+text',
                marker=dict(size=12, color=color, symbol='circle'),
                text=[name],
                textposition="top center",
                name=f"{name} #{idx+1}"
            ))
            
        num_at = len(preview_atoms)
        for i in range(num_at):
            for j in range(i+1, num_at):
                pos_i = np.array(preview_atoms[i]["pos"])
                pos_j = np.array(preview_atoms[j]["pos"])
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < 3.5:
                    fig_prev.add_trace(go.Scatter3d(
                        x=[pos_i[0], pos_j[0]], y=[pos_i[1], pos_j[1]], z=[pos_i[2], pos_j[2]],
                        mode='lines',
                        line=dict(color='gray', width=3),
                        showlegend=False
                    ))
                    
        # 若已计算出结果，则在 3D 预览图中绘制偶极矩矢量箭头
        if st.session_state.calc_results is not None and "P" in st.session_state.calc_results:
            try:
                from diatomic_engine import compute_dipole_moment_multi
                dip_vec, dip_debye = compute_dipole_moment_multi(preview_atoms, st.session_state.calc_results)
                if dip_debye > 1e-3:
                    # 质心位置作为矢量起点
                    coords = np.array([at["pos"] for at in preview_atoms])
                    com = np.mean(coords, axis=0)
                    
                    # 矢量缩放倍率，便于在 preview 视窗中观察
                    scale_factor = 2.0 / max(dip_debye, 1.0)
                    arr_end = com + dip_vec * scale_factor
                    
                    fig_prev.add_trace(go.Scatter3d(
                        x=[com[0], arr_end[0]],
                        y=[com[1], arr_end[1]],
                        z=[com[2], arr_end[2]],
                        mode='lines+markers',
                        line=dict(color='#8b5cf6', width=5), # 紫色箭头
                        marker=dict(size=[0, 8], color='#8b5cf6', symbol='cone'),
                        name=f"偶极矩 μ = {dip_debye:.3f} D"
                    ))
            except Exception:
                pass
                
        fig_prev.update_layout(
            scene=dict(
                xaxis=dict(title='X (Bohr)', range=[-4, 4]),
                yaxis=dict(title='Y (Bohr)', range=[-4, 4]),
                zaxis=dict(title='Z (Bohr)', range=[-5, 5]),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            height=260,
            showlegend=False
        )
        polish_3d_figure(fig_prev, height=330)
        st.plotly_chart(fig_prev, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 标准模板预设
st.markdown("### 💡 快速学术模型模板 (Categorized Presets Library)")

# 定义分类好的模型模板库
TEMPLATES = {
    "【一维密度泛函 (1D DFT) 基础模型组】": {

        "类氦原子 (Z=2, 1D He)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：一维类氦原子 (Z=2)，两个电子在软库仑势中的自洽场计算。是测试 XC 泛函的经典基准体系。",
            "params": {"num_electrons": 2, "L": 10.0, "N": 200, "max_iter": 100, "tol": 1e-6, "alpha": 0.2, "softening": 1.0, "functional": "LDA", "mixing_method": "Linear",
                "potential_expr": "-2.0 / np.sqrt(x ** 2 + 1.0)",
                "potential_description": "一维类氦原子 (Z=2, 软库仑势)"}
        },
        "类铍原子 (Z=4, 1D Be)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：一维类铍原子 (Z=4)，四个电子的自洽场计算，展示壳层填充结构和 Anderson 加速混合的效果。",
            "params": {"num_electrons": 4, "L": 10.0, "N": 200, "max_iter": 100, "tol": 1e-6, "alpha": 0.2, "softening": 1.0, "functional": "LDA", "mixing_method": "Anderson",
                "potential_expr": "-4.0 / np.sqrt(x ** 2 + 1.0)",
                "potential_description": "一维类铍原子 (Z=4)"}
        },
        "一维氢分子 (1D H2, d=2.0)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：一维双核氢分子，质子间距 d=2.0 Bohr。展示成键与反键电子态的密度分布。",
            "params": {"num_electrons": 2, "L": 12.0, "N": 200, "max_iter": 100, "tol": 1e-6, "alpha": 0.2, "softening": 1.0, "functional": "LDA", "mixing_method": "Linear",
                "potential_expr": "-1.0/np.sqrt((x-1.0)**2 + 1.0) - 1.0/np.sqrt((x+1.0)**2 + 1.0)",
                "potential_description": "一维 H2 分子 (质子间距 2.0 Bohr)"}
        },
        "一维 HF 极性分子 (H-F 键)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：一维模拟高极性 H-F 键，氟原子核电荷 Z=9 显著大于 H 的 Z=1，展示极不对称的电子云极化现象。",
            "params": {"num_electrons": 10, "L": 12.0, "N": 200, "max_iter": 150, "tol": 1e-6, "alpha": 0.15, "softening": 1.0, "functional": "GGA-PBE", "mixing_method": "Anderson",
                "potential_expr": "-1.0/np.sqrt((x+1.5)**2 + 1.0) - 9.0/np.sqrt((x-1.5)**2 + 0.5)",
                "potential_description": "一维 H-F 极性分子 (Z_H=1, Z_F=9)"}
        },
    },
    "【一维量子系统 (1D Advanced Physics) 模型组】": {
        "双势阱量子隧穿 (Double Well Tunneling)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：双对称势阱模型，展示经典禁区内的量子隧穿效应——电子可以穿越中间势垒在两个势阱之间发生量子共振隧穿。",
            "params": {"num_electrons": 2, "L": 10.0, "N": 200, "max_iter": 100, "tol": 1e-6, "alpha": 0.2, "softening": 0.5, "functional": "LDA", "mixing_method": "Linear",
                "potential_expr": "0.5 * (x**2 - 3.0)**2 * 0.08",
                "potential_description": "双势阱量子隧穿系统 (V = 0.08*(x²-3)²)"}
        },
        "谐振量子点 (Harmonic Quantum Dot, 4e)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：强谐振限域多电子量子点，模拟半导体量子点器件中的电子壳层结构（类氢轨道序列）和 Wigner 晶化现象前兆。",
            "params": {"num_electrons": 4, "L": 8.0, "N": 200, "max_iter": 100, "tol": 1e-6, "alpha": 0.15, "softening": 0.8, "functional": "LDA", "mixing_method": "Anderson",
                "potential_expr": "0.25 * x**2",
                "potential_description": "谐振量子点 (ω=0.5 Hartree)"}
        },
        "锂-氢-锂 分子链 (1D Li-H-Li)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：一维交替锂氢链，Li 的 Z=3 核吸引势远强于 H 的 Z=1，展示金属-非金属键合的轨道杂化态和电荷转移现象。",
            "params": {"num_electrons": 4, "L": 12.0, "N": 200, "max_iter": 100, "tol": 1e-6, "alpha": 0.15, "softening": 1.0, "functional": "LDA", "mixing_method": "Anderson",
                "potential_expr": "-3.0/np.sqrt((x+3)**2+1.0) - 1.0/np.sqrt(x**2+1.0) - 3.0/np.sqrt((x-3)**2+1.0)",
                "potential_description": "一维 Li-H-Li 分子链"}
        },
        "周期晶格点缺陷 (Lattice + Impurity)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：模拟一维周期性晶格中存在单个点缺陷/杂质时，局域缺陷态如何在禁带中间形成孤立能级（类比半导体掺杂态）。",
            "params": {"num_electrons": 6, "L": 15.0, "N": 200, "max_iter": 100, "tol": 1e-6, "alpha": 0.2, "softening": 1.0, "functional": "LDA", "mixing_method": "Anderson",
                "potential_expr": "-3.0 * np.cos(np.pi * x / 3.0)**2 - 1.5 * np.exp(-x**2)",
                "potential_description": "缺陷晶格 (周期势 + 杂质态)"}
        },
        "外电场极化效应 (Electric Field Stark Effect)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：类氦原子置于外加线性电场 E=0.1 a.u. 中，展示量子 Stark 效应——外场使原本对称的电子云发生极化变形，轨道能级发生 Stark 分裂。",
            "params": {"num_electrons": 2, "L": 12.0, "N": 200, "max_iter": 100, "tol": 1e-6, "alpha": 0.2, "softening": 1.0, "functional": "GGA-PBE", "mixing_method": "Anderson",
                "potential_expr": "-2.0/np.sqrt(x**2+1.0) + 0.1*x",
                "potential_description": "类氦原子 + 外电场 (E=0.1 a.u., Stark效应)"}
        },
        "Kronig-Penney 能带模型 (Periodic Crystal)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：经典 Kronig-Penney 周期势，用余弦势阱近似模拟一维晶体电子态，在 Brillouin 区边界产生能带间隙，是能带论最基础的演示模型。",
            "params": {"num_electrons": 8, "L": 15.0, "N": 300, "max_iter": 200, "tol": 1e-6, "alpha": 0.15, "softening": 0.5, "functional": "LDA", "mixing_method": "Linear",
                "potential_expr": "-2.5 * (np.cos(2*np.pi*x/5.0))**2",
                "potential_description": "Kronig-Penney 一维晶格 (周期 5 Bohr)"}
        },
        "Morse 分子势能 (Diatomic Vibration)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：用 Morse 势模拟真实双原子分子的非谐振振动，可观察到量子振动能级（Vibrational Levels）的非等间距特征，是分子光谱学的核心模型。",
            "params": {"num_electrons": 2, "L": 15.0, "N": 300, "max_iter": 120, "tol": 1e-6, "alpha": 0.2, "softening": 0.5, "functional": "Exchange-Only", "mixing_method": "Anderson",
                "potential_expr": "2.5 * (1.0 - np.exp(-(x+3.0)))**2 - 2.5",
                "potential_description": "Morse 非谐振子 (D_e=2.5, α=1.0)"}
        },
        "非对称双势阱 (Asymmetric Double-Well)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：左右不对称的双势阱，模拟质子在分子间氢键中的非等价转移（如 DNA 碱基对中的 Proton Transfer），展示隧穿态的非对称极化现象。",
            "params": {"num_electrons": 2, "L": 12.0, "N": 200, "max_iter": 100, "tol": 1e-6, "alpha": 0.15, "softening": 0.8, "functional": "LDA", "mixing_method": "Anderson",
                "potential_expr": "0.06*(x**4 - 8*x**2) + 0.3*x",
                "potential_description": "非对称双势阱 (Asymmetric DW, 质子转移)"}
        },
    },
    "【三维分子自洽场 (3D HF) 基础分子组】": {
        "氢分子 (3D H₂, 2e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：三维 H₂ 分子，平衡键长 1.4 Bohr，STO-3G RHF 计算。理论 E_tot ≈ -1.117 Hartree，是量化计算的经典标准基准。",
            "params": {"atoms": [{"name": "H", "pos": [0.0, 0.0, -0.7]}, {"name": "H", "pos": [0.0, 0.0, 0.7]}],
                "num_electrons": 2, "max_iter": 50, "tol": 1e-6}
        },
        "氦氢离子 (3D HeH⁺, 2e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：宇宙中最早形成的分子 HeH⁺，键长 1.46 Bohr。这是极性最强的双原子分子之一，展示强电荷转移和离子共价键混合特征。",
            "params": {"atoms": [{"name": "HE", "pos": [0.0, 0.0, -0.87]}, {"name": "H", "pos": [0.0, 0.0, 0.59]}],
                "num_electrons": 2, "max_iter": 50, "tol": 1e-6}
        },
        "氟化氢 (3D HF, 10e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：极强极性共价键 HF 分子，键长 1.73 Bohr，偶极矩约 1.82 Debye。展示电负性差异导致的极端电子云偏移和 Mulliken 净电荷分布。",
            "params": {"atoms": [{"name": "H", "pos": [0.0, 0.0, -1.73]}, {"name": "F", "pos": [0.0, 0.0, 0.0]}],
                "num_electrons": 10, "max_iter": 60, "tol": 1e-6}
        },
        "一氧化碳 (3D CO, 14e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：CO 三键分子，键长 2.12 Bohr。CO 是著名的反直觉极性分子——碳端为负（因孤对电子）。展示极化分析与 Wiberg 三键 (≈3.0) 键级。",
            "params": {"atoms": [{"name": "C", "pos": [0.0, 0.0, -1.06]}, {"name": "O", "pos": [0.0, 0.0, 1.06]}],
                "num_electrons": 14, "max_iter": 60, "tol": 1e-6}
        },
        "氮气 (3D N₂, 14e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：N≡N 三键，键长 2.07 Bohr，是自然界最强的双原子共价键之一，HOMO-LUMO gap 大，化学惰性极强。Wiberg 键指数应接近 3.0。",
            "params": {"atoms": [{"name": "N", "pos": [0.0, 0.0, -1.04]}, {"name": "N", "pos": [0.0, 0.0, 1.04]}],
                "num_electrons": 14, "max_iter": 60, "tol": 1e-6}
        },
        "氧气 (3D O₂, 16e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：O=O 双键，键长 2.28 Bohr（RHF 对开壳层描述不完美），展示高 HOMO-LUMO gap 和强共价键合特征。实验参考 E_tot ≈ -147.63 Hartree。",
            "params": {"atoms": [{"name": "O", "pos": [0.0, 0.0, -1.14]}, {"name": "O", "pos": [0.0, 0.0, 1.14]}],
                "num_electrons": 16, "max_iter": 60, "tol": 1e-6}
        },
        "水分子 (3D H₂O, 10e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：弯折形 H₂O 分子，键角 104.5°，O-H 键长 1.809 Bohr。偶极矩 ≈ 1.85 Debye，展示氧原子的孤对电子云及极性。",
            "params": {"atoms": [{"name": "O", "pos": [0.0, 0.0, 0.12]}, {"name": "H", "pos": [0.0, 1.43, -0.98]}, {"name": "H", "pos": [0.0, -1.43, -0.98]}],
                "num_electrons": 10, "max_iter": 60, "tol": 1e-6}
        },
        "氨气 (3D NH₃, 10e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：三角锥形 NH₃ 分子，N 原子顶部有强孤对电子云（HOMO 轨道），偶极矩 ≈ 1.47 Debye，是碱性最强的简单分子之一。",
            "params": {"atoms": [{"name": "N", "pos": [0.0, 0.0, 0.25]}, {"name": "H", "pos": [0.0, 1.77, -0.58]}, {"name": "H", "pos": [1.53, -0.88, -0.58]}, {"name": "H", "pos": [-1.53, -0.88, -0.58]}],
                "num_electrons": 10, "max_iter": 60, "tol": 1e-6}
        },
        "甲烷 (3D CH₄, 10e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：正四面体 CH₄，sp³ 杂化轨道，键角 109.5°，C-H 键长 2.05 Bohr。高度对称，偶极矩 = 0。展示纯 sp³ 轨道的电子密度分布。",
            "params": {"atoms": [{"name": "C", "pos": [0.0, 0.0, 0.0]}, {"name": "H", "pos": [1.18, 1.18, 1.18]}, {"name": "H", "pos": [-1.18, -1.18, 1.18]}, {"name": "H", "pos": [-1.18, 1.18, -1.18]}, {"name": "H", "pos": [1.18, -1.18, -1.18]}],
                "num_electrons": 10, "max_iter": 60, "tol": 1e-6}
        },
        "氟化锂 (3D LiF, 12e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：高极性 LiF 离子型化合物，偶极矩超过 6 Debye，Li 几乎转移全部价电子给 F。是离子键与共价键混合特征的极端案例。",
            "params": {"atoms": [{"name": "LI", "pos": [0.0, 0.0, -1.55]}, {"name": "F", "pos": [0.0, 0.0, 1.55]}],
                "num_electrons": 12, "max_iter": 60, "tol": 1e-6}
        },
        "二氧化碳 (3D CO₂, 22e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：线性对称 CO₂ 分子，两个 C=O 双键。由于中心对称性，偶极矩 = 0，但差分电荷密度和静电势展示极不对称的局域极化特征（亲电碳，亲核氧）。",
            "params": {"atoms": [{"name": "C", "pos": [0.0, 0.0, 0.0]}, {"name": "O", "pos": [0.0, 0.0, -2.19]}, {"name": "O", "pos": [0.0, 0.0, 2.19]}],
                "num_electrons": 22, "max_iter": 70, "tol": 1e-6}
        },
        "三氟化硼 (3D BF₃ 近似, 24e)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：模拟平面三角形 B-F₃ 缺电子分子（Lewis 酸），B-F 键长 2.48 Bohr，用 3D SCF 展示硼原子的空 p 轨道亲电性特征。",
            "params": {"atoms": [{"name": "B", "pos": [0.0, 0.0, 0.0]}, {"name": "F", "pos": [2.48, 0.0, 0.0]}, {"name": "F", "pos": [-1.24, 2.148, 0.0]}, {"name": "F", "pos": [-1.24, -2.148, 0.0]}],
                "num_electrons": 24, "max_iter": 70, "tol": 1e-6}
        },
    },
    "【强相互作用与催化 (Catalysis & Correlation) 模型组】": {
        "Be₂ 分子 (弱 van der Waals 键合)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：Be₂ 是通过极弱的 van der Waals 色散力结合的特殊双原子分子，RHF 预测为非束缚（unbound），展示 RHF 方法的局限性（需要相关能修正 MP2/CCSD）。",
            "params": {"atoms": [{"name": "BE", "pos": [0.0, 0.0, -2.5]}, {"name": "BE", "pos": [0.0, 0.0, 2.5]}],
                "num_electrons": 8, "max_iter": 60, "tol": 1e-6}
        },
        "O₂ 在金属 Be 簇上吸附": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：模拟 O₂ 分子端接吸附（end-on）在 Be 金属单原子位点上，展示过渡金属表面催化活化 O₂ 的基本图像——电子从金属 d 轨道转移至 O₂ 反键轨道。",
            "params": {"atoms": [{"name": "BE", "pos": [0.0, 0.0, -2.6]}, {"name": "O", "pos": [0.0, 0.0, 0.0]}, {"name": "O", "pos": [0.0, 0.0, 2.28]}],
                "num_electrons": 20, "max_iter": 70, "tol": 1e-6}
        },
        "H 原子在 C 基底上吸附 (C-H)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：单个 H 原子吸附到碳基底上，模拟氢在碳材料（石墨烯、金刚石）表面的化学吸附过程，分析 C-H 键的形成与 Mulliken 电荷转移。",
            "params": {"atoms": [{"name": "C", "pos": [0.0, 0.0, -1.2]}, {"name": "H", "pos": [0.0, 0.0, 0.9]}],
                "num_electrons": 7, "max_iter": 60, "tol": 1e-6}
        },
        "水合 Li⁺ 离子团簇 (Li·H₂O)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：Li⁺ 离子与一个水分子的配位溶剂化团簇，展示离子-偶极相互作用对水分子几何构型的影响和电荷重新分配。",
            "params": {"atoms": [{"name": "LI", "pos": [0.0, 0.0, -3.5]}, {"name": "O", "pos": [0.0, 0.0, 0.12]}, {"name": "H", "pos": [0.0, 1.43, -0.98]}, {"name": "H", "pos": [0.0, -1.43, -0.98]}],
                "num_electrons": 13, "max_iter": 70, "tol": 1e-6}
        },
        "C₂H₂ 乙炔 (线性三键分子)": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：线性乙炔分子 H-C≡C-H，sp 杂化，C≡C 三键，键长 2.27 Bohr (C-C)，C-H 键长 2.0 Bohr。展示 π 轨道的圆柱对称电子分布。",
            "params": {"atoms": [{"name": "H", "pos": [0.0, 0.0, -4.27]}, {"name": "C", "pos": [0.0, 0.0, -2.27]}, {"name": "C", "pos": [0.0, 0.0, 2.27]}, {"name": "H", "pos": [0.0, 0.0, 4.27]}],
                "num_electrons": 14, "max_iter": 70, "tol": 1e-6}
        },
        "Be-H₂ 插入反应过渡态": {
            "solver_type": "3d_diatomic",
            "explanation": "中文预设：模拟 Be 原子插入 H-H 键的过渡态结构，H...Be...H 角度为 180°。展示反应过渡态时的差分电荷密度再分配——Be 从 H₂ 获取电子密度。",
            "params": {"atoms": [{"name": "H", "pos": [0.0, 0.0, -2.6]}, {"name": "BE", "pos": [0.0, 0.0, 0.0]}, {"name": "H", "pos": [0.0, 0.0, 2.6]}],
                "num_electrons": 6, "max_iter": 60, "tol": 1e-6}
        },
    },
    "【1D 强关联与拓扑效应模型组】": {
        "SSH 聚乙炔链 (Topological Edge States)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：Su-Schrieffer-Heeger (SSH) 模型——交替强弱耦合的一维原子链，是最简单的一维拓扑绝缘体模型。弱键一侧的边界态在拓扑相中出现孤立的缺陷态。",
            "params": {"num_electrons": 8, "L": 15.0, "N": 300, "max_iter": 120, "tol": 1e-6, "alpha": 0.15, "softening": 1.0, "functional": "LDA", "mixing_method": "Anderson",
                "potential_expr": "-1.8/np.sqrt((x+9)**2+0.5) - 0.9/np.sqrt((x+6)**2+0.5) - 1.8/np.sqrt((x+3)**2+0.5) - 0.9/np.sqrt((x)**2+0.5) - 1.8/np.sqrt((x-3)**2+0.5) - 0.9/np.sqrt((x-6)**2+0.5) - 1.8/np.sqrt((x-9)**2+0.5)",
                "potential_description": "SSH 链 (交替强弱键，拓扑绝缘体模型)"}
        },
        "Anderson 局域化 (无序势中电子局域)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：在均匀晶格势中引入随机无序（随机杂质），模拟 Anderson 电子局域化效应。随机势打破平移对称性，导致扩展态变为局域态，是凝聚态绝缘-金属转变的核心机制。",
            "params": {"num_electrons": 6, "L": 15.0, "N": 300, "max_iter": 120, "tol": 1e-6, "alpha": 0.2, "softening": 0.5, "functional": "LDA", "mixing_method": "Anderson",
                "potential_expr": "-2.0*np.cos(2*np.pi*x/4.0)**2 + 0.6*(np.sin(1.7*x) + np.cos(2.3*x) + np.sin(3.1*x+0.5))",
                "potential_description": "Anderson 无序势 (周期势 + 准随机无序)"}
        },
        "Wigner 晶化前兆 (High-density QD)": {
            "solver_type": "1d_dft",
            "explanation": "中文预设：高密度强谐振量子点（8个电子）中，当电子间排斥强于动能时，体系趋向于形成 Wigner 晶体——电子在空间中形成准周期性排列。",
            "params": {"num_electrons": 8, "L": 12.0, "N": 300, "max_iter": 150, "tol": 1e-7, "alpha": 0.1, "softening": 0.5, "functional": "GGA-PBE", "mixing_method": "Anderson",
                "potential_expr": "0.15 * x**2",
                "potential_description": "强限域量子点 (Wigner晶化前兆, 8e)"}
        },
    }
}


col_grp, col_mod, col_load = st.columns([2, 2, 1])

with col_grp:
    selected_group = st.selectbox("📂 选择模型大类", list(TEMPLATES.keys()), key="preset_group_select")
with col_mod:
    selected_model = st.selectbox("🔬 选择具体学术模型", list(TEMPLATES[selected_group].keys()), key="preset_model_select")
with col_load:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    load_clicked = st.button("🚀 载入该学术模型配置", use_container_width=True, key="preset_load_btn")

if load_clicked:
    chosen_preset = TEMPLATES[selected_group][selected_model]
    st.session_state.config = {
        "solver_type": chosen_preset["solver_type"],
        "explanation": chosen_preset["explanation"],
        "params": chosen_preset["params"]
    }
    st.session_state.calc_results = None
    st.session_state.ads_results = None
    st.session_state.last_solver_type = chosen_preset["solver_type"]
    
    # 同步更新侧边栏 Widget 状态
    sync_config_to_sidebar_state()
    
    st.success(f"成功载入预设模型：{selected_model}！")
    st.rerun()

# 触发计算控制
run_now = False
if ai_submit and user_prompt:
    with st.spinner(f"AI ({ai_model_select}) 正在根据指令解析参数并生成物理配置..."):
        ai_res = parse_user_request(user_prompt, model_name=ai_model_select)
        st.session_state.config = ai_res
        st.session_state.calc_results = None
        st.session_state.ads_results = None
        st.session_state.last_solver_type = ai_res["solver_type"]
        
        # 同步更新侧边栏 Widget 状态
        sync_config_to_sidebar_state()
        
        st.success(f"AI ({ai_model_select}) 成功设置模拟参数！")
        run_now = True
        
        # Check if AI requested structure optimization auto-trigger
        if ai_res.get("launch_optimization", False):
            st.session_state.auto_optimize = True
            st.session_state.opt_history = None
            st.session_state.opt_final_R = None
else:
    run_now = False

if manual_submit or run_now:
    solver = st.session_state.config["solver_type"]
    params = st.session_state.config["params"]
    
    with st.status("自洽场（SCF）非线性薛定谔方程组正在求解中...", expanded=True) as status:
        if solver == "1d_dft":
            st.write("步骤 1: 建立离散网格空间并评估外部势能算符...")
            Vext_fn = make_potential_fn(params["potential_expr"])
            st.write("步骤 2: 自洽迭代运行网格科恩-沈（Kohn-Sham）方程求解...")
            
            result = solve_1d_dft(
                Vext_fn=Vext_fn,
                num_electrons=params["num_electrons"],
                L=params["L"],
                N=params["N"],
                max_iter=params["max_iter"],
                tol=params["tol"],
                alpha=params["alpha"],
                softening=params["softening"],
                functional=params.get("functional", "LDA"),
                mixing_method=params.get("mixing_method", "Linear")
            )
            st.session_state.calc_results = result
        elif solver == "3d_diatomic":
            st.write(f"步骤 1: 求解解析高斯积分（单电子哈密顿量、动能与双电子排斥 ERI，基于 STO-3G 多原子基组）...")
            st.write("步骤 2: 自洽求解 Roothaan 方程组...")
            
            # 兼容旧版本的单原子位置配置
            if "atoms" not in params:
                params["atoms"] = [
                    {"name": params["atom1_name"], "pos": params["atom1_pos"]},
                    {"name": params["atom2_name"], "pos": params["atom2_pos"]}
                ]
                
            # 物理容量验证：防止设置电子数超限导致Roothaan方程无实数解
            from diatomic_engine import get_element_orbitals
            nbasis = 0
            for item in params["atoms"]:
                o_list, _ = get_element_orbitals(item["name"], item["pos"])
                nbasis += len(o_list)
            if params["num_electrons"] > 2 * nbasis:
                st.error(f"❌ **电子数超出基组容量极限**：当前分子体系总共包含 {nbasis} 个原子轨道，由于泡利不相容原理（每个空间轨道最多容纳2个自旋相反电子），当前体系最多只能填充 **{2 * nbasis}** 个电子。而当前输入了 **{params['num_electrons']}** 个电子。\n\n**建议解决方案**：\n1. 调小侧边栏的“系统总电子数”；\n2. 或者通过 XYZ 编辑器添加更多原子以引入更多轨道能级。")
                st.stop()
                
            from diatomic_engine import solve_multi_atom_scf
            result = solve_multi_atom_scf(
                atoms=params["atoms"],
                num_electrons=params["num_electrons"],
                max_iter=params["max_iter"],
                tol=params["tol"],
                multiplicity=params.get("multiplicity", None)
            )
            st.session_state.calc_results = result
            
        if result["converged"]:
            status.update(label=f"计算成功！自洽迭代在第 {result['iterations']} 步完美收敛。", state="complete")
        else:
            status.update(label="计算已达最大步数，未完全收敛，请检查混合因子等参数。", state="error")

# 展示计算数据与结果
# 展示计算数据与结果
solver_mode = st.session_state.config["solver_type"]

# ----------------- 分子动力学 (3D MD) 大规模多分子模拟舱 -----------------
if solver_mode == "3d_md":
    st.markdown('<div class="app-subtitle">🌐 经典分子动力学 (MD) 大规模多分子模拟舱</div>', unsafe_allow_html=True)
    st.write("分子动力学（Molecular Dynamics）模拟是通过经典力学牛顿方程描述多体粒子随时间演化的技术，支持数百个原子的相互作用，适用于探索流体结构、多分子聚集态及扩散性质。")
    
    # 模拟控制面板
    params = st.session_state.config["params"]
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        st.info(f"**当前体系**：{params['system_type']}\n* 包含粒子数: **{params['n_particles']}**\n* 平衡温度: **{params['temp']} K**\n* 模拟边长: **{params['box_length']} Å**")
    with col_ctrl2:
        run_md = st.button("🚀 启动 Verlet 动力学时间步积分 (Run MD Simulation)", use_container_width=True)
        
    if "md_results" not in st.session_state:
        st.session_state.md_results = None
        
    if run_md:
        with st.spinner("多体 Verlet 时间演化积分中..."):
            from md_engine import MolecularDynamics
            import time
            
            md = MolecularDynamics(box_length=params["box_length"], temp=params["temp"], dt=params["dt"])
            if params["system_type"].startswith("Argon"):
                md.init_argon_box(n_atoms=params["n_particles"])
            else:
                md.init_nitrogen_box(n_molecules=params["n_particles"])
                
            step_list = []
            ke_list = []
            pe_list = []
            temp_list = []
            msd_list = []
            
            init_pos = md.positions.copy()
            max_steps = params["max_iter"]
            
            # Progress bar
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            start_time = time.time()
            for step in range(max_steps):
                md.integrate_verlet_step1()
                pot_e = md.compute_forces()
                md.integrate_verlet_step2()
                
                kin_e, curr_temp = md.compute_kinetic_energy()
                
                if step % 10 == 0:
                    md.apply_thermostat(curr_temp)
                    
                if step % 10 == 0:
                    step_list.append(step)
                    ke_list.append(kin_e)
                    pe_list.append(pot_e)
                    temp_list.append(curr_temp)
                    
                    # Compute MSD
                    dr = md.positions - init_pos
                    dr -= np.round(dr / params["box_length"]) * params["box_length"]
                    msd = np.mean(np.sum(dr ** 2, axis=1))
                    msd_list.append(msd)
                    
                if step % 50 == 0:
                    progress_bar.progress(step / max_steps)
                    status_text.text(f"Verlet 动力学演化中... 步数: {step}/{max_steps}，温度: {curr_temp:.2f} K")
                    
            progress_bar.progress(1.0)
            status_text.text(f"仿真演化完成！耗时: {time.time() - start_time:.2f} 秒。")
            
            # Compute RDF
            r_vals, g_r = md.calculate_rdf()
            
            st.session_state.md_results = {
                "steps": step_list,
                "ke": ke_list,
                "pe": pe_list,
                "te": (np.array(ke_list) + np.array(pe_list)).tolist(),
                "temp": temp_list,
                "msd": msd_list,
                "r": r_vals.tolist(),
                "g": g_r.tolist(),
                "final_positions": md.positions.tolist(),
                "names": md.names,
                "bonds": md.bonds
            }
            
    if st.session_state.md_results is not None:
        md_res = st.session_state.md_results
        
        # 结果可视化卡片
        st.markdown("### 📊 多分子体系经典动力学物理分析报告")
        
        # Tab 分类显示
        md_tab1, md_tab2, md_tab3, md_tab4 = st.tabs([
            "🔮 3D 粒子运动轨迹", 
            "📈 能量守恒与温度", 
            "🧬 径向分布函数 g(r)", 
            "🚀 扩散性质与 MSD"
        ])
        
        with md_tab1:
            st.markdown("##### 三维周期性晶格盒子内多分子构型空间分布")
            
            pos_arr = np.array(md_res["final_positions"])
            fig_3d = go.Figure()
            
            # 绘制原子
            is_argon = md_res["names"][0] == "Ar"
            fig_3d.add_trace(go.Scatter3d(
                x=pos_arr[:, 0], y=pos_arr[:, 1], z=pos_arr[:, 2],
                mode='markers',
                marker=dict(
                    size=6 if is_argon else 5,
                    color='royalblue' if is_argon else 'crimson',
                    opacity=0.85
                ),
                name="Argon 原子" if is_argon else "Nitrogen 原子"
            ))
            
            # 绘制共价键 (如果是双原子分子)
            if not is_argon and md_res["bonds"]:
                bond_x, bond_y, bond_z = [], [], []
                for bi, bj, _, _ in md_res["bonds"]:
                    bond_x.extend([pos_arr[bi, 0], pos_arr[bj, 0], None])
                    bond_y.extend([pos_arr[bi, 1], pos_arr[bj, 1], None])
                    bond_z.extend([pos_arr[bi, 2], pos_arr[bj, 2], None])
                fig_3d.add_trace(go.Scatter3d(
                    x=bond_x, y=bond_y, z=bond_z,
                    mode='lines',
                    line=dict(color='black', width=2),
                    name="N-N 键"
                ))
                
            # 绘制线框盒子 (Box Boundary Wireframe)
            bl = params["box_length"]
            box_corners = [
                [0,0,0], [bl,0,0], [bl,bl,0], [0,bl,0], [0,0,0],
                [0,0,bl], [bl,0,bl], [bl,bl,bl], [0,bl,bl], [0,0,bl],
                [bl,0,bl], [bl,0,0], [bl,bl,0], [bl,bl,bl], [0,bl,bl], [0,bl,0]
            ]
            box_corners = np.array(box_corners)
            fig_3d.add_trace(go.Scatter3d(
                x=box_corners[:, 0], y=box_corners[:, 1], z=box_corners[:, 2],
                mode='lines',
                line=dict(color='grey', width=1.5, dash='dash'),
                name="周期性盒子边界"
            ))
            
            fig_3d.update_layout(
                margin=dict(l=0, r=0, b=0, t=30),
                scene=dict(
                    xaxis=dict(title="X (Å)", range=[0, bl]),
                    yaxis=dict(title="Y (Å)", range=[0, bl]),
                    zaxis=dict(title="Z (Å)", range=[0, bl])
                )
            )
            polish_3d_figure(fig_3d)
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with md_tab2:
            st.markdown("##### 热力学能量演化与微正则系综守恒分析 (Conservation of Energy)")
            col_en1, col_en2 = st.columns(2)
            with col_en1:
                fig_en = go.Figure()
                fig_en.add_trace(go.Scatter(x=md_res["steps"], y=md_res["pe"], name="势能 PE", line=dict(color='#d62728')))
                fig_en.add_trace(go.Scatter(x=md_res["steps"], y=md_res["ke"], name="动能 KE", line=dict(color='#2ca02c')))
                fig_en.add_trace(go.Scatter(x=md_res["steps"], y=md_res["te"], name="总能量 TE", line=dict(color='#1f77b4', width=2.5)))
                fig_en.update_layout(
                    xaxis_title="弛豫时间步数 (Steps)",
                    yaxis_title="能量 (eV)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_en, use_container_width=True)
            with col_en2:
                fig_te = go.Figure()
                fig_te.add_trace(go.Scatter(x=md_res["steps"], y=md_res["temp"], name="系统实时温度 (T)", line=dict(color='#ff7f0e')))
                fig_te.add_hline(y=params["temp"], line_dash="dash", line_color="black")
                fig_te.update_layout(
                    xaxis_title="时间步数 (Steps)",
                    yaxis_title="温度 T (K)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_te, use_container_width=True)
                
        with md_tab3:
            st.markdown("##### 结构分析：多体径向分布函数 (Radial Distribution Function - RDF)")
            st.write("径向分布函数 $g(r)$ 描述了在距离某一参考原子 $r$ 处发现另一个原子的相对概率。它是从分子尺度探索物质相态（固、液、气）的最重要工具。")
            
            fig_rdf = go.Figure()
            fig_rdf.add_trace(go.Scatter(x=md_res["r"], y=md_res["g"], name="g(r)", line=dict(color='#9467bd', width=2.5)))
            fig_rdf.add_hline(y=1.0, line_dash="dash", line_color="grey")
            fig_rdf.update_layout(
                xaxis_title="距离距离 r (Å)",
                yaxis_title="g(r) 分布值",
                template="plotly_white"
            )
            st.plotly_chart(fig_rdf, use_container_width=True)
            st.info("💡 **径向分布函数物理分析**：第一个高耸峰代表**第一配位层（最近邻距离，约为 3.4 Å）**。如果 $g(r)$ 在远处长程内趋近于 1，说明体系处于无序的**液体或气体**相态；若有多个清晰的尖锐峰，说明长程有序，处于**固体晶体**态。")
            
        with md_tab4:
            st.markdown("##### 动力学分析：均方位移与自扩散系数 (MSD & Self-Diffusion Coefficient)")
            st.write("均方位移（Mean Square Displacement, MSD）记录了粒子随时间的随机扩散位移。根据 Einstein 扩散方程，在长程极限下：$\\text{MSD}(t) \\approx 6 D t$，其中 $D$ 为自扩散系数。")
            
            fig_msd = go.Figure()
            time_ps = np.array(md_res["steps"]) * params["dt"]
            fig_msd.add_trace(go.Scatter(x=time_ps, y=md_res["msd"], name="MSD(t)", line=dict(color='#8c564b', width=2.5)))
            fig_msd.update_layout(
                xaxis_title="物理模拟时间 t (ps)",
                yaxis_title="均方位移 MSD (Å²)",
                template="plotly_white"
            )
            st.plotly_chart(fig_msd, use_container_width=True)
            
            # 计算自扩散系数
            half_idx = len(time_ps) // 2
            slope, intercept = np.polyfit(time_ps[half_idx:], md_res["msd"][half_idx:], 1)
            D_val = (slope / 6.0) * 1e-4 # cm^2/s
            
            st.success(f"📈 **扩散迁移率分析成果**：\n* **一维拟合斜率 (MSD Slope)**: `{slope:.4f} Å²/ps`\n* **解析自扩散系数 D (Self-Diffusion)**: `{D_val:.4e} cm²/s` (典型常温水/液体通常为 $10^{-5}$ 数量级，气体通常为 $10^{-1}$ 数量级)")

    st.stop()

res = st.session_state.calc_results
if res is not None:
    # 额外增加一层安全验证，确保数据格式与当前求解器模式严格匹配，防止切换求解器时页面崩溃
    solver_mode = st.session_state.config["solver_type"]
    is_dft_res = "energies" in res
    is_3d_res = "E_tot" in res
    
    if (solver_mode == "1d_dft" and not is_dft_res) or (solver_mode == "3d_diatomic" and not is_3d_res):
        st.session_state.calc_results = None
        st.rerun()
        
    st.markdown('<div class="academic-alert">', unsafe_allow_html=True)
    st.markdown("### 📊 自洽计算总能量收敛与分项分解分析")
    
    solver_mode = st.session_state.config["solver_type"]
    
    if solver_mode == "1d_dft":
        col_r1, col_r2, col_r3 = st.columns(3)
        total_energy = res["energies"]["E_tot"]
        col_r1.metric("系统总能量 (E_total)", f"{total_energy:.6f} Hartree")
        col_r2.metric("自洽场迭代次数", f"{res['iterations']}")
        col_r3.metric("网格离散采样点数", f"{len(res['x'])}")
        
        # 计算 1D HOMO-LUMO Gap
        eps_1d = res["eigenvalues"]
        n_occ_1d = int(np.ceil(params["num_electrons"] / 2.0))
        gap_str_1d = "N/A (全空轨)"
        if n_occ_1d < len(eps_1d) and n_occ_1d > 0:
            gap_val_1d = eps_1d[n_occ_1d] - eps_1d[n_occ_1d - 1]
            gap_str_1d = f"**{gap_val_1d:.4f} Hartree** ({gap_val_1d*27.2114:.3f} eV)"
        st.markdown(f"🔑 **HOMO-LUMO 能级隙 (Energy Gap)**: {gap_str_1d}")
        
        st.markdown("#### 能量分解分项表格 (Energy Components Decomposition)")
        e_comps = res["energies"]
        energy_data = {
            "能量贡献项 (Energy Components)": ["动能项 (Kinetic Energy - E_kin)", "外部吸引势项 (External Confinement - E_ext)", "哈特里静电能项 (Hartree Repulsion - E_H)", "量子交换关联能项 (Exchange-Correlation - E_xc)", "基态总能量 (Ground State Energy - E_tot)"],
            "计算值 (Hartree)": [f"{e_comps['E_kin']:.6f}", f"{e_comps['E_ext']:.6f}", f"{e_comps['E_H']:.6f}", f"{e_comps['E_xc']:.6f}", f"{e_comps['E_tot']:.6f}"],
            "物理学含义说明": [
                "电子在网格空间中的动能期望值。",
                "正电中心/外电场/约束势阱对电子的吸引势能能量。",
                "经典平均场近似下的电子与自身电荷云的库仑排斥能量。",
                "局域密度近似（LDA）下的泡利不相容原理交换作用与强电子关联能和。",
                "体系所有能量项加和，为对应变分基态能量的值。"
            ]
        }
        st.table(energy_data)
    else:
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        total_energy = res["E_tot"]
        col_r1.metric("分子基态总能量 (E_total)", f"{total_energy:.6f} Hartree")
        col_r2.metric("自洽场迭代次数", f"{res['iterations']}")
        col_r3.metric("电子总能量 (E_electronic)", f"{res['E_elec']:.6f} Hartree")
        
        if "atoms" not in params:
            params["atoms"] = [
                {"name": params["atom1_name"], "pos": params["atom1_pos"]},
                {"name": params["atom2_name"], "pos": params["atom2_pos"]}
            ]
        from diatomic_engine import compute_dipole_moment_multi, compute_mulliken_charges, get_element_orbitals
        mu_vec, mu_debye = compute_dipole_moment_multi(params["atoms"], res)
        col_r4.metric("分子偶极矩 (Dipole)", f"{abs(mu_debye):.4f} Debye")
        
        # 计算 3D HOMO-LUMO Gap (分自旋通道计算)
        multiplicity = res.get("multiplicity", 1)
        n_alpha = res.get("n_alpha", int(np.ceil(params["num_electrons"] / 2.0)))
        n_beta = res.get("n_beta", params["num_electrons"] - n_alpha)
        
        eps_alpha = res["eps_alpha"]
        eps_beta = res["eps_beta"]
        
        gap_alpha_str = "N/A"
        if 0 < n_alpha < len(eps_alpha):
            gap_alpha_val = eps_alpha[n_alpha] - eps_alpha[n_alpha - 1]
            gap_alpha_str = f"**{gap_alpha_val:.4f} Hartree** ({gap_alpha_val*27.2114:.3f} eV)"
            
        gap_beta_str = "N/A"
        if 0 < n_beta < len(eps_beta):
            gap_beta_val = eps_beta[n_beta] - eps_beta[n_beta - 1]
            gap_beta_str = f"**{gap_beta_val:.4f} Hartree** ({gap_beta_val*27.2114:.3f} eV)"
            
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if multiplicity > 1:
                st.markdown(f"- 🧭 **自旋多重度 (Spin Multiplicity)**: `{multiplicity}` (UHF 开壳层自旋极化)")
                st.markdown(f"- 🔑 **Alpha 自旋通道 (Spin Up) HOMO-LUMO Gap**: {gap_alpha_str} (占据数: {n_alpha})")
                st.markdown(f"- 🔑 **Beta 自旋通道 (Spin Down) HOMO-LUMO Gap**: {gap_beta_str} (占据数: {n_beta})")
            else:
                st.markdown(f"- 🧭 **自旋多重度 (Spin Multiplicity)**: `1` (RHF 闭壳层)")
                st.markdown(f"- 🔑 **HOMO-LUMO 能级隙 (Energy Gap)**: {gap_alpha_str} (双重占据轨道数: {n_alpha})")
            st.markdown(f"- 🧭 **三维偶极矩极化分量 (Dipole Vector)**: `[X={mu_vec[0]*2.541746:.3f}, Y={mu_vec[1]*2.541746:.3f}, Z={mu_vec[2]*2.541746:.3f}]` Debye")
        with col_g2:
            # 下载 XYZ 坐标文件按钮
            xyz_content = f"{len(params['atoms'])}\nOptimized molecular geometry by Chat DFT\n"
            for at in params["atoms"]:
                xyz_content += f"{at['name']} {at['pos'][0]:.6f} {at['pos'][1]:.6f} {at['pos'][2]:.6f}\n"
            st.download_button("💾 导出并下载该分子的 XYZ 坐标文件", data=xyz_content, file_name="molecule.xyz", mime="chemical/x-xyz", key="xyz_download_btn")
            
        st.markdown("#### ⚛️ Mulliken 原子净电荷与电荷转移分析 (Mulliken Net Charges)")
        m_charges = compute_mulliken_charges(params["atoms"], res)
        charge_data = {
            "原子序号": [f"#{item['index']}" for item in m_charges],
            "元素类型": [item["name"] for item in m_charges],
            "原子核电荷 Z": [f"{get_element_orbitals(item['name'], [0,0,0])[1]:.0f}" for item in m_charges],
            "Mulliken 分布电子数 (e)": [f"{item['population']:.4f}" for item in m_charges],
            "净电荷 (Net Charge, a.u.)": [f"{item['net_charge']:+.4f}" for item in m_charges]
        }
        st.table(charge_data)
        
        # 化学键成键分析 (Mulliken Overlap Population)
        st.markdown("#### 🔗 原子间成键重叠布居分析 (Mulliken Overlap Population)")
        st.write("Mulliken 轨道重叠布居 (MOP) = Σ 2·P_μν·S_μν，是量化两原子间共价轨道混合强度的核心指标。正值代表成键性贡献，负值代表反键性贡献，绝对值越大说明轨道重叠越强。")
        from diatomic_engine import compute_bond_orders
        mop_matrix, bond_strength = compute_bond_orders(params["atoms"], res)
        
        # 建立展示的 DataFrame 并应用渐变高亮
        atom_names = [f"{at['name']}#{idx+1}" for idx, at in enumerate(params["atoms"])]
        import pandas as pd
        df_mop = pd.DataFrame(mop_matrix, index=atom_names, columns=atom_names)
        df_strength = pd.DataFrame(bond_strength, index=atom_names, columns=atom_names)
        
        col_bo1, col_bo2 = st.columns(2)
        with col_bo1:
            st.markdown("**📊 MOP 有符号矩阵** (正=成键, 负=反键)")
            # Diverging colormap centered at 0: RdBu_r (red for negative, blue for positive)
            max_abs = max(abs(mop_matrix).max(), 0.01)
            st.dataframe(df_mop.style.format("{:.4f}").background_gradient(cmap='RdYlBu', vmin=-max_abs, vmax=max_abs), use_container_width=True)
        with col_bo2:
            st.markdown("**💪 成键强度绝对值** (越大=共价作用越强)")
            st.dataframe(df_strength.style.format("{:.4f}").background_gradient(cmap='Blues', vmin=0.0, vmax=max(bond_strength.max(), 0.01)), use_container_width=True)
        
        # 提取最强键对
        strongest_pairs = []
        n_at = len(params["atoms"])
        for i in range(n_at):
            for j in range(i+1, n_at):
                if bond_strength[i, j] > 0.001:
                    strongest_pairs.append((params["atoms"][i]["name"], i+1, params["atoms"][j]["name"], j+1, mop_matrix[i,j], bond_strength[i,j]))
        if strongest_pairs:
            strongest_pairs.sort(key=lambda x: -x[5])
            bond_summary = "| 键对 | MOP 值 | 成键强度 | 键型判断 |\n|------|--------|----------|----------|\n"
            for pair in strongest_pairs[:6]:
                n1, i1, n2, i2, mop_val, strength = pair
                if mop_val > 0.3:
                    bond_type = "🟢 强共价键"
                elif mop_val > 0.05:
                    bond_type = "🟡 弱共价/极性键"
                elif mop_val < -0.05:
                    bond_type = "🔴 反键贡献"
                else:
                    bond_type = "⚪ 无明显化学键"
                bond_summary += f"| {n1}#{i1} — {n2}#{i2} | {mop_val:+.4f} | {strength:.4f} | {bond_type} |\n"
            st.markdown(bond_summary)
        st.info("💡 **基组说明**：本软件使用扩展 s 型 STO-3G 基组（无 p 轨道）。对于多重键（如 N≡N、C=O），MOP 反映的是 σ 键成分；π 键成分需要引入 p 型基函数（将在未来版本中支持）。H₂ 的 MOP 应在 0.7~1.0 范围，H₂O 的 O-H MOP 约 0.4~0.8。")
        

        
        st.markdown("#### 能量分解分项表格")
        energy_data = {
            "能量贡献项 (Energy Components)": ["动能项 (Kinetic - E_kin)", "原子核吸引能项 (Nuclear Confinement - E_ext)", "电子间相互作用项 (Electronic Repulsion - E_ee)", "核间库仑排斥项 (Nuclear Repulsion - E_nuc)", "基态分子总能量 (Total Molecular Energy - E_tot)"],
            "计算值 (Hartree)": [f"{res['E_kin']:.6f}", f"{res['E_ext']:.6f}", f"{res['E_ee']:.6f}", f"{res['E_nuc']:.6f}", f"{res['E_tot']:.6f}"],
            "物理学含义说明": [
                "由高斯轨道计算出的电子运动的总动能。",
                "电子受两个原子核正电荷中心库仑力作用产生的吸引能总和。",
                "多电子体系下由 ERI 积分算出的 Coulomb 与 Exchange 斥力贡献。",
                "正电原子核之间的核外经典静电排斥能量 ($Z_1 Z_2 / R$)。",
                "电子总能量与原子核间排斥能量的和。"
            ]
        }
        st.table(energy_data)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 建立学术报告标签页
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "📈 自洽能量收敛", 
        "📊 电荷密度与有效势", 
        "🌀 占据轨道波函数", 
        "📉 差分电荷密度 (CDD)", 
        "🌌 态密度分析 (DOS/PDOS)", 
        "🧬 分子势能面扫描",
        "📥 吸附能计算",
        "⚙️ 结构优化",
        "🎵 振动分析与红外光谱 (IR)",
        "⚡ 静电势分析 (MEP)",
        "🌐 周期晶格与能带结构"
    ])
    
    with tab1:
        st.markdown("#### 自洽迭代(SCF)系统总能量收敛曲线 (SCI 风格)")
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(range(len(res['history']))),
            y=res['history'],
            mode='lines+markers',
            line=dict(color='#1e293b', width=1.5),
            marker=dict(size=5, symbol='square-open', color='#1f77b4'),
            name='自洽能量'
        ))
        fig_conv.update_layout(
            xaxis_title="自洽场迭代次数 (Iteration)",
            yaxis_title="能量 E (Hartree)",
            template="plotly_white",
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig_conv, use_container_width=True)
        
    with tab2:
        if solver_mode == "1d_dft":
            st.markdown("#### 网格科恩-沈（Kohn-Sham）势能与电子云密度图 (双轴 SCI 规范)")
            fig_dp = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 各项势能
            fig_dp.add_trace(go.Scatter(x=res['x'], y=res['potentials']['Vext'], name='$V_{\\mathrm{ext}}$ (吸引势)', line=dict(color='#d62728', dash='dash', width=1.5)), secondary_y=False)
            fig_dp.add_trace(go.Scatter(x=res['x'], y=res['potentials']['VH'], name='$V_{\\mathrm{H}}$ (哈特里静电势)', line=dict(color='#ff7f0e', dash='dot', width=1.5)), secondary_y=False)
            fig_dp.add_trace(go.Scatter(x=res['x'], y=res['potentials']['Veff'], name='$V_{\\mathrm{eff}}$ (有效单体势)', line=dict(color='black', width=2.0)), secondary_y=False)
            
            # 电子密度
            fig_dp.add_trace(go.Scatter(x=res['x'], y=res['density'], name='$\\rho(x)$ (电子密度)', line=dict(color='#1f77b4', width=2.5)), secondary_y=True)
            
            fig_dp.update_layout(
                xaxis_title="网格位置 x (Bohr)",
                template="plotly_white",
                margin=dict(l=40, r=40, t=20, b=40)
            )
            fig_dp.update_yaxes(title_text="势能 V(x) (Hartree)", secondary_y=False)
            fig_dp.update_yaxes(title_text="电荷密度 ρ(x) (e/Bohr)", secondary_y=True)
            
            st.plotly_chart(fig_dp, use_container_width=True)
        else:
            st.markdown("#### 双原子分子电荷密度分布图")
            
            # 使用左右两个分栏：左边为 1D 投影切片，右边为 2D 二维电荷密度等值线与原子重叠图
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.markdown("##### 1. 沿键轴方向的电荷密度投影 (1D 剖面)")
                z = np.linspace(-4.0, 4.0, 300)
                if "atoms" not in params:
                    params["atoms"] = [
                        {"name": params["atom1_name"], "pos": params["atom1_pos"]},
                        {"name": params["atom2_name"], "pos": params["atom2_pos"]}
                    ]
                
                from analysis_tools import eval_density_z
                rho_z = eval_density_z(res, z)
                
                fig_3d_slice = go.Figure()
                fig_3d_slice.add_trace(go.Scatter(x=z, y=rho_z, name='$\\rho(z)$', line=dict(color='#1f77b4', width=2.5)))
                
                for idx, at in enumerate(params["atoms"]):
                    fig_3d_slice.add_vline(
                        x=at["pos"][2], 
                        line_dash="dash", 
                        line_color="#d62728" if idx % 2 == 0 else "#2ca02c", 
                        annotation_text=f"{at['name']}#{idx+1}"
                    )
                
                fig_3d_slice.update_layout(
                    xaxis_title="Z轴坐标 (Bohr)",
                    yaxis_title="电荷密度分布 ρ(z) (e/Bohr³)",
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=20, b=40)
                )
                st.plotly_chart(fig_3d_slice, use_container_width=True)
                
            with col_d2:
                st.markdown("##### 2. 二维等密度线与原子结构重叠图 (2D 截面)")
                
                # 计算 2D 密度网格
                grid_y = np.linspace(-3.0, 3.0, 100)
                grid_z = np.linspace(-4.0, 4.0, 120)
                Z, Y = np.meshgrid(grid_z, grid_y)
                
                from analysis_tools import eval_density_2d
                rho_2d = eval_density_2d(res, Y, Z)
                
                fig_cont = go.Figure(data=go.Contour(
                    z=rho_2d,
                    x=grid_z,
                    y=grid_y,
                    colorscale='Viridis',
                    contours=dict(
                        coloring='heatmap',
                        showlabels=True,
                        labelfont=dict(size=10, color='white')
                    )
                ))
                
                # 确保 atoms 存在
                if "atoms" not in params:
                    params["atoms"] = [
                        {"name": params["atom1_name"], "pos": params["atom1_pos"]},
                        {"name": params["atom2_name"], "pos": params["atom2_pos"]}
                    ]
                
                # 绘制所有原子圆圈与元素名
                for idx, item in enumerate(params["atoms"]):
                    aname = item["name"]
                    apos = item["pos"]
                    pz = apos[2]
                    py = apos[1]
                    color = "rgba(230, 230, 230, 0.95)" if aname == "H" else "rgba(255, 200, 200, 0.95)"
                    fig_cont.add_shape(type="circle", xref="x", yref="y", x0=pz-0.3, y0=py-0.3, x1=pz+0.3, y1=py+0.3, line_color="black", fillcolor=color, line_width=1.5)
                    fig_cont.add_annotation(x=pz, y=py, text=aname, showarrow=False, font=dict(color="black", size=10, weight="bold"))
                    
                # 绘制虚线化学键
                num_atoms = len(params["atoms"])
                for i in range(num_atoms):
                    for j in range(i+1, num_atoms):
                        pos_i = np.array(params["atoms"][i]["pos"])
                        pos_j = np.array(params["atoms"][j]["pos"])
                        dist = np.linalg.norm(pos_i - pos_j)
                        if dist < 3.5:
                            fig_cont.add_shape(type="line", xref="x", yref="y", x0=pos_i[2], y0=pos_i[1], x1=pos_j[2], y1=pos_j[1], line=dict(color="white", width=2, dash="dash"))
                
                fig_cont.update_layout(
                    xaxis_title="Z轴核键方向 (Bohr)",
                    yaxis_title="Y轴径向 (Bohr)",
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=20, b=40)
                )
                st.plotly_chart(fig_cont, use_container_width=True)
                
            st.markdown("##### 3. 三维空间电子密度等值面图 (3D Isosurface Cloud)")
            st.write("在下方三维交互空间中，您可以使用鼠标任意旋转、缩放，查看三维电子云等值面在空间中的真实形态。")
            
            # 生成 3D 密度网格
            x_3d = np.linspace(-3.0, 3.0, 30)
            y_3d = np.linspace(-3.0, 3.0, 30)
            z_3d = np.linspace(-4.0, 4.0, 35)
            X_3d, Y_3d, Z_3d = np.meshgrid(x_3d, y_3d, z_3d, indexing='ij')
            
            from analysis_tools import eval_density_3d_grid
            rho_3d = eval_density_3d_grid(res, X_3d, Y_3d, Z_3d)
            
            fig_3d = go.Figure()
            
            fig_3d.add_trace(go.Isosurface(
                x=X_3d.flatten(),
                y=Y_3d.flatten(),
                z=Z_3d.flatten(),
                value=rho_3d.flatten(),
                isomin=0.03,
                isomax=0.5,
                surface_count=4,
                opacity=0.35,
                colorscale='Viridis',
                caps=dict(x_show=False, y_show=False, z_show=False),
                name='电子云等值面'
            ))
            
            # 确保 atoms 存在
            if "atoms" not in params:
                params["atoms"] = [
                    {"name": params["atom1_name"], "pos": params["atom1_pos"]},
                    {"name": params["atom2_name"], "pos": params["atom2_pos"]}
                ]
                
            # 绘制所有原子核
            for idx, item in enumerate(params["atoms"]):
                aname = item["name"]
                apos = item["pos"]
                color = 'lightgrey' if aname == 'H' else 'pink'
                fig_3d.add_trace(go.Scatter3d(
                    x=[apos[0]], y=[apos[1]], z=[apos[2]],
                    mode='markers+text',
                    marker=dict(size=14, color=color, symbol='circle'),
                    text=[aname],
                    textposition="top center",
                    textfont=dict(color="black", size=14, weight="bold"),
                    name=f"{aname} 原子核 #{idx+1}"
                ))
                
            # 绘制所有符合距离的化学键
            num_atoms = len(params["atoms"])
            for i in range(num_atoms):
                for j in range(i+1, num_atoms):
                    pos_i = np.array(params["atoms"][i]["pos"])
                    pos_j = np.array(params["atoms"][j]["pos"])
                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist < 3.5:
                        fig_3d.add_trace(go.Scatter3d(
                            x=[pos_i[0], pos_j[0]], y=[pos_i[1], pos_j[1]], z=[pos_i[2], pos_j[2]],
                            mode='lines',
                            line=dict(color='black', width=6),
                            showlegend=False
                        ))
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='X (Bohr)',
                    yaxis_title='Y (Bohr)',
                    zaxis_title='Z (Bohr)',
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                height=500
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
    with tab3:
        if solver_mode == "1d_dft":
            st.markdown("#### 被占据态科恩-沈（Kohn-Sham）轨道波函数幅值分布图")
            fig_orb = go.Figure()
            
            num_orbs = res['orbitals'].shape[1]
            for idx in range(num_orbs):
                psi = res['orbitals'][:, idx]
                energy = res['eigenvalues'][idx]
                fig_orb.add_trace(go.Scatter(
                    x=res['x'],
                    y=psi,
                    name=f"轨道 $\\psi_{idx}$ (E = {energy:.4f} Ha)",
                    line=dict(width=1.8)
                ))
                
            fig_orb.update_layout(
                xaxis_title="网格坐标位置 x (Bohr)",
                yaxis_title="波函数振幅值",
                template="plotly_white",
                margin=dict(l=40, r=40, t=20, b=40)
            )
            st.plotly_chart(fig_orb, use_container_width=True)
            
            st.markdown("#### 科恩-沈单粒子轨道本征能量能级列表")
            orb_table = {
                "本征轨道 (Orbitals)": [f"轨道 ψ_{i}" for i in range(len(res['eigenvalues']))],
                "本征值 Energy (Hartree)": [f"{val:.6f}" for val in res['eigenvalues']],
                "本征值 Energy (eV)": [f"{val * 27.2114:.4f}" for val in res['eigenvalues']],
                "填充状态 (Occupation)": ["2.0 (全充满轨道)" if (i < params["num_electrons"]//2) else ("1.0 (半充满轨道)" if (params["num_electrons"]%2==1 and i==params["num_electrons"]//2) else "0.0 (未占据轨道)") for i in range(len(res['eigenvalues']))]
            }
            st.table(orb_table)
            
        else:
            multiplicity = res.get("multiplicity", 1)
            if multiplicity > 1:
                st.markdown("#### 分子轨道线性组合系数矩阵 (LCAO Coefficient C)")
                st.write("行对应原子基组轨道 (AO)，列代表分子本征轨道 (MO)，当前为 UHF 自旋极化开壳层系统，请选择不同的自旋通道进行分析。")
                spin_ch = st.radio("选择自旋通道 (Spin Channel)", ["Alpha (自旋向上, Spin Up)", "Beta (自旋向下, Spin Down)"], key="mo_spin_ch")
                if spin_ch == "Alpha (自旋向上, Spin Up)":
                    C_mat = res['C_alpha']
                    eps_mat = res['eps_alpha']
                    ch_name = "Alpha"
                else:
                    C_mat = res['C_beta']
                    eps_mat = res['eps_beta']
                    ch_name = "Beta"
            else:
                st.markdown("#### 分子轨道线性组合系数矩阵 (LCAO Coefficient C)")
                st.write("行对应原子基组轨道 (AO)，列代表分子本征轨道 (MO)，当前为闭壳层 RHF 系统。")
                C_mat = res['C']
                eps_mat = res['eps']
                ch_name = "RHF"
                
            nbasis = C_mat.shape[0]
            
            # 动态生成原子轨道标签
            from diatomic_engine import get_element_orbitals
            ao_labels = []
            for j, at in enumerate(params["atoms"]):
                o_list, _ = get_element_orbitals(at["name"], at["pos"])
                for k in range(len(o_list)):
                    ao_labels.append(f"{at['name']}#{j+1} (AO-{k+1})")
                    
            # 轨道本征值能量标签
            mo_cols = [f"MO {i+1}\n({eps_mat[i]:.3f} Ha)" for i in range(nbasis)]
            
            import pandas as pd
            df_c = pd.DataFrame(C_mat, index=ao_labels, columns=mo_cols)
            
            # 学术级彩色渐变表格
            st.dataframe(df_c.style.format("{:.5f}").background_gradient(cmap='RdBu_r', vmin=-1.0, vmax=1.0), use_container_width=True)
            
            st.markdown(f"##### 📊 {ch_name} 自旋通道分子轨道杂化贡献组分图")
            fig_hm = go.Figure(data=go.Heatmap(
                z=C_mat,
                x=[f"MO {i+1}" for i in range(nbasis)],
                y=ao_labels,
                colorscale='RdBu_r',
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title="系数 C_μi")
            ))
            fig_hm.update_layout(
                xaxis_title="分子轨道 (Molecular Orbitals)",
                yaxis_title="原子基组轨道 (Atomic Orbitals)",
                template="plotly_white",
                margin=dict(l=40, r=40, t=20, b=40),
                height=300 + 20 * nbasis
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            
    with tab4:
        st.markdown("#### 差分电荷密度 (Charge Density Difference - CDD) 分布图 (SCI 规范)")
        st.write("差分电荷密度展示了当原子相互靠近形成分子时，电子密度的重新分布情况： $\\Delta\\rho = \\rho_{\\mathrm{molecule}} - \\sum \\rho_{\\mathrm{isolated\\_atoms}}$。")
        
        with st.spinner("计算孤立原子参考状态密度中..."):
            if solver_mode == "1d_dft":
                cdd, rho_mol, rho_a1, rho_a2 = calculate_cdd_1d(params, res)
                x_cdd = res['x']
                
                if np.all(cdd == 0):
                    st.warning("一维差分电荷密度计算仅支持双势阱中心模型（如1D H2分子预设）。当前单势阱体系无法计算差分密度。")
                else:
                    fig_cdd = go.Figure()
                    fig_cdd.add_trace(go.Scatter(x=x_cdd, y=cdd, name='差分电荷密度 $\\Delta\\rho(x)$', line=dict(color='#2ca02c', width=2.5)))
                    fig_cdd.add_trace(go.Scatter(x=x_cdd, y=rho_mol, name='分子总密度 $\\rho_{\\mathrm{mol}}$', line=dict(color='#1f77b4', dash='dash')))
                    fig_cdd.add_trace(go.Scatter(x=x_cdd, y=rho_a1 + rho_a2, name='原子叠加参考密度', line=dict(color='#7f7f7f', dash='dot')))
                    
                    fig_cdd.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
                    
                    fig_cdd.update_layout(
                        xaxis_title="网格坐标位置 x (Bohr)",
                        yaxis_title="电荷密度差 $\\Delta\\rho(x)$ (e/Bohr)",
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    st.plotly_chart(fig_cdd, use_container_width=True)
                    st.info("💡 **学术图表分析**：一维差分电荷密度在两核中心之间若为**正值**（电荷富集），代表形成了典型的**共价键合电荷聚集区**；两侧出现**负值**，代表孤立原子的电荷向键中心迁移发生极化。")
            else:
                st.markdown("#### 双原子分子差分电荷密度 (CDD) 图")
                
                col_cdd1, col_cdd2 = st.columns(2)
                
                with col_cdd1:
                    st.markdown("##### 1. 沿键轴方向的差分电荷密度投影 (1D 剖面)")
                    z_cdd = np.linspace(-4.0, 4.0, 300)
                    if "atoms" not in params:
                        params["atoms"] = [
                            {"name": params["atom1_name"], "pos": params["atom1_pos"]},
                            {"name": params["atom2_name"], "pos": params["atom2_pos"]}
                        ]
                    cdd, rho_mol, rho_atoms_sum = calculate_cdd_3d(
                        params["atoms"],
                        params["num_electrons"],
                        res,
                        z_cdd
                    )
                    
                    fig_cdd = go.Figure()
                    fig_cdd.add_trace(go.Scatter(x=z_cdd, y=cdd, name='差分电荷密度 $\\Delta\\rho(z)$', line=dict(color='#2ca02c', width=2.5)))
                    fig_cdd.add_trace(go.Scatter(x=z_cdd, y=rho_mol, name='分子总电荷密度 $\\rho_{\\mathrm{mol}}$', line=dict(color='#1f77b4', dash='dash')))
                    fig_cdd.add_trace(go.Scatter(x=z_cdd, y=rho_atoms_sum, name='原子电荷密度简单叠加', line=dict(color='#7f7f7f', dash='dot')))
                    
                    fig_cdd.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
                    
                    for idx, at in enumerate(params["atoms"]):
                        fig_cdd.add_vline(
                            x=at["pos"][2], 
                            line_dash="dash", 
                            line_color="#d62728" if idx % 2 == 0 else "#2ca02c", 
                            annotation_text=f"{at['name']}#{idx+1}"
                        )
                    
                    fig_cdd.update_layout(
                        xaxis_title="Z轴坐标 (Bohr)",
                        yaxis_title="电荷密度差 $\\Delta\\rho(z)$ (e/Bohr³)",
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    st.plotly_chart(fig_cdd, use_container_width=True)
                    
                with col_cdd2:
                    st.markdown("##### 2. 二维差分电荷密度等值线与原子结构重叠图 (2D 截面)")
                    
                    # 计算 2D 差分电荷密度网格
                    grid_y = np.linspace(-3.0, 3.0, 100)
                    grid_z = np.linspace(-4.0, 4.0, 120)
                    Z, Y = np.meshgrid(grid_z, grid_y)
                    
                    from analysis_tools import calculate_cdd_2d
                    if "atoms" not in params:
                        params["atoms"] = [
                            {"name": params["atom1_name"], "pos": params["atom1_pos"]},
                            {"name": params["atom2_name"], "pos": params["atom2_pos"]}
                        ]
                    cdd_2d, _, _ = calculate_cdd_2d(
                        params["atoms"], 
                        params["num_electrons"], res, Y, Z
                    )
                    
                    # 差分电荷密度通常使用发散色标，红色表示电子耗尽，蓝色表示电子富集
                    fig_cdd_cont = go.Figure(data=go.Contour(
                        z=cdd_2d,
                        x=grid_z,
                        y=grid_y,
                        colorscale='RdBu',
                        zmid=0,
                        contours=dict(
                            coloring='heatmap',
                            showlabels=True,
                            labelfont=dict(size=10, color='black')
                        )
                    ))
                    
                    # 绘制所有原子圆圈和名字
                    for idx, item in enumerate(params["atoms"]):
                        aname = item["name"]
                        apos = item["pos"]
                        pz = apos[2]
                        py = apos[1]
                        color = "rgba(230, 230, 230, 0.95)" if aname == "H" else "rgba(255, 200, 200, 0.95)"
                        fig_cdd_cont.add_shape(type="circle", xref="x", yref="y", x0=pz-0.3, y0=py-0.3, x1=pz+0.3, y1=py+0.3, line_color="black", fillcolor=color, line_width=1.5)
                        fig_cdd_cont.add_annotation(x=pz, y=py, text=aname, showarrow=False, font=dict(color="black", size=10, weight="bold"))
                        
                    # 绘制化学键
                    num_atoms = len(params["atoms"])
                    for i in range(num_atoms):
                        for j in range(i+1, num_atoms):
                            pos_i = np.array(params["atoms"][i]["pos"])
                            pos_j = np.array(params["atoms"][j]["pos"])
                            dist = np.linalg.norm(pos_i - pos_j)
                            if dist < 3.5:
                                fig_cdd_cont.add_shape(type="line", xref="x", yref="y", x0=pos_i[2], y0=pos_i[1], x1=pos_j[2], y1=pos_j[1], line=dict(color="black", width=2, dash="dash"))
                    
                    fig_cdd_cont.update_layout(
                        xaxis_title="Z轴核键方向 (Bohr)",
                        yaxis_title="Y轴径向 (Bohr)",
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    st.plotly_chart(fig_cdd_cont, use_container_width=True)
                
                # 3D CDD Isosurfaces
                st.markdown("##### 3. 三维空间差分电荷密度等值面 (3D CDD Isosurfaces)")
                st.write("在下方三维交互空间中，蓝色等值面代表电子**积聚区** ($\\Delta\\rho > 0$)，红色等值面代表电子**耗尽区** ($\\Delta\\rho < 0$)。")
                
                x_3d = np.linspace(-3.0, 3.0, 30)
                y_3d = np.linspace(-3.0, 3.0, 30)
                z_3d = np.linspace(-4.0, 4.0, 35)
                X_3d, Y_3d, Z_3d = np.meshgrid(x_3d, y_3d, z_3d, indexing='ij')
                
                from analysis_tools import calculate_cdd_3d_grid
                if "atoms" not in params:
                    params["atoms"] = [
                        {"name": params["atom1_name"], "pos": params["atom1_pos"]},
                        {"name": params["atom2_name"], "pos": params["atom2_pos"]}
                    ]
                cdd_3d, _, _ = calculate_cdd_3d_grid(
                    params["atoms"], 
                    params["num_electrons"], res, X_3d, Y_3d, Z_3d
                )
                
                fig_3d_cdd = go.Figure()
                
                # Accumulation (Blue)
                fig_3d_cdd.add_trace(go.Isosurface(
                    x=X_3d.flatten(),
                    y=Y_3d.flatten(),
                    z=Z_3d.flatten(),
                    value=cdd_3d.flatten(),
                    isomin=0.002,
                    isomax=0.05,
                    surface_count=2,
                    opacity=0.4,
                    colorscale=[[0, 'blue'], [1, 'blue']],
                    showscale=False,
                    name='电子富集区 (+)'
                ))
                
                # Depletion (Red)
                fig_3d_cdd.add_trace(go.Isosurface(
                    x=X_3d.flatten(),
                    y=Y_3d.flatten(),
                    z=Z_3d.flatten(),
                    value=cdd_3d.flatten(),
                    isomin=-0.05,
                    isomax=-0.002,
                    surface_count=2,
                    opacity=0.4,
                    colorscale=[[0, 'red'], [1, 'red']],
                    showscale=False,
                    name='电子耗尽区 (-)'
                ))
                
                # 绘制所有原子核
                for idx, item in enumerate(params["atoms"]):
                    aname = item["name"]
                    apos = item["pos"]
                    color = 'lightgrey' if aname == 'H' else 'pink'
                    fig_3d_cdd.add_trace(go.Scatter3d(
                        x=[apos[0]], y=[apos[1]], z=[apos[2]],
                        mode='markers+text',
                        marker=dict(size=14, color=color, symbol='circle'),
                        text=[aname],
                        textposition="top center",
                        textfont=dict(color="black", size=14, weight="bold"),
                        name=f"{aname} 原子核 #{idx+1}"
                    ))
                    
                # 绘制化学键实线
                num_atoms = len(params["atoms"])
                for i in range(num_atoms):
                    for j in range(i+1, num_atoms):
                        pos_i = np.array(params["atoms"][i]["pos"])
                        pos_j = np.array(params["atoms"][j]["pos"])
                        dist = np.linalg.norm(pos_i - pos_j)
                        if dist < 3.5:
                            fig_3d_cdd.add_trace(go.Scatter3d(
                                x=[pos_i[0], pos_j[0]], y=[pos_i[1], pos_j[1]], z=[pos_i[2], pos_j[2]],
                                mode='lines',
                                line=dict(color='black', width=6),
                                showlegend=False
                            ))
                
                fig_3d_cdd.update_layout(
                    scene=dict(
                        xaxis_title='X (Bohr)',
                        yaxis_title='Y (Bohr)',
                        zaxis_title='Z (Bohr)',
                        aspectmode='data'
                    ),
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=500
                )
                polish_3d_figure(fig_3d_cdd)
                st.plotly_chart(fig_3d_cdd, use_container_width=True)
                
                st.info("💡 **学术图表分析**：在双原子核正中间（图2蓝色区域和图3蓝色3D包络面，$\\Delta\\rho > 0$），代表电子发生了向键合中心的显著转移，形成了坚固的**化学共价键**；原子核周围和外侧（红色区域，$\\Delta\\rho < 0$），对应电子的耗尽区。")
                
    with tab5:
        st.markdown("#### 态密度 (Density of States - DOS) & 投影态密度 (PDOS) (SCI 规范)")
        st.write("态密度表征单位能量区间内的本征量子物理态个数。投影态密度 (PDOS) 则进一步将分子本征态分解映射到独立的原子轨道上。")
        
        E_min = -1.5
        E_max = 0.5
        E_grid = np.linspace(E_min, E_max, 400)
        
        sigma = st.slider("高斯展宽因子 (Gaussian Broadening σ in Hartree)", min_value=0.01, max_value=0.20, value=0.04, step=0.01)
        
        if solver_mode == "1d_dft":
            e_vals = res['eigenvalues']
            occs = []
            rem = params["num_electrons"]
            for i in range(len(e_vals)):
                occ = min(2.0, rem)
                occs.append(occ)
                rem -= occ
            
            dos = calculate_dos(e_vals, occs, E_grid, sigma)
            
            fig_dos = go.Figure()
            fig_dos.add_trace(go.Scatter(x=E_grid, y=dos, name='总态密度 (DOS)', line=dict(color='black', width=2)))
            
            for i, ev in enumerate(e_vals):
                fig_dos.add_vline(x=ev, line_dash="dash", line_color="#d62728", annotation_text=f"E_{i}")
                
            fig_dos.update_layout(
                xaxis_title="能量 E (Hartree)",
                yaxis_title="态密度 (States/Hartree)",
                template="plotly_white",
                margin=dict(l=40, r=40, t=20, b=40)
            )
            st.plotly_chart(fig_dos, use_container_width=True)
            
        else:
            e_vals = res['eps']
            occs = [2.0 if i < int(np.ceil(params["num_electrons"]/2)) else 0.0 for i in range(len(e_vals))]
            
            dos = calculate_dos(e_vals, occs, E_grid, sigma)
            
            # 兼容旧配置并使用全新的通用多中心PDOS分解
            if "atoms" not in params:
                params["atoms"] = [
                    {"name": params["atom1_name"], "pos": params["atom1_pos"]},
                    {"name": params["atom2_name"], "pos": params["atom2_pos"]}
                ]
            pdos_list = calculate_pdos_multi(e_vals, res['C'], res['S'], res['basis'], params['atoms'], E_grid, sigma)
            
            fig_dos = go.Figure()
            fig_dos.add_trace(go.Scatter(x=E_grid, y=dos, name='总态密度 (DOS)', line=dict(color='black', width=2)))
            
            colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            for j, at in enumerate(params["atoms"]):
                fig_dos.add_trace(go.Scatter(
                    x=E_grid, 
                    y=pdos_list[j], 
                    name=f'原子 {j+1} ({at["name"]}) 贡献 PDOS', 
                    line=dict(color=colors[j % len(colors)], dash='dash' if j % 2 == 0 else 'dot')
                ))
            
            homo_idx = max(0, int(np.ceil(params["num_electrons"]/2)) - 1)
            E_f = e_vals[homo_idx]
            fig_dos.add_vline(x=E_f, line_dash="solid", line_color="blue", annotation_text="费米面/HOMO")
            
            fig_dos.update_layout(
                xaxis_title="能量 E (Hartree)",
                yaxis_title="态密度 (States/Hartree)",
                template="plotly_white",
                margin=dict(l=40, r=40, t=20, b=40)
            )
            st.plotly_chart(fig_dos, use_container_width=True)
            st.info("💡 **学术图表分析**：总态密度（黑色实线）对应的波峰即代表分子的能级分布。彩色虚线代表各原子的本征投影态密度 (PDOS)，它能够直观展示每个原子在杂化成键时对特定分子轨道能级的贡献比重与轨道成分分裂。")
            
    with tab6:
        if solver_mode == "3d_diatomic":
            st.markdown("#### 二原子系统基态势能面（PES）拉伸势能曲线与分子能级演化图")
            
            # 检查当前是否为双原子分子体系
            is_diatomic = True
            if "atoms" in params:
                if len(params["atoms"]) != 2:
                    is_diatomic = False
                    
            if not is_diatomic:
                st.info("💡 **扫描功能不可用**：势能面（PES）扫描仅适用于 **双原子分子**（如 H2, LiF 等）。您当前设置的分子体系包含多于或少于两个原子，无法进行简单的核间距一维拉伸扫描。")
            else:
                # 获取双原子名称
                if "atoms" in params:
                    a1_name = params["atoms"][0]["name"]
                    a2_name = params["atoms"][1]["name"]
                else:
                    a1_name = params.get("atom1_name", "H")
                    a2_name = params.get("atom2_name", "H")
                    
                st.write(f"该扫描将自动拉伸 **{a1_name}-{a2_name}** 二原子体系的核间距 $R$，计算各位置下的总能量，并输出分子轨道 (HOMO/LUMO) 能级的杂化分裂过程。")
                
                scan_run = st.button("🚀 启动分子势能面与能级演化扫描")
                if scan_run:
                    distances = np.linspace(0.6, 5.0, 23)
                    progress_bar = st.progress(0.0)
                    
                    scan_energies = []
                    scan_homo = []
                    scan_lumo = []
                    for i, r in enumerate(distances):
                        progress_bar.progress((i + 1) / len(distances))
                        pos1 = [0.0, 0.0, -r/2.0]
                        pos2 = [0.0, 0.0, r/2.0]
                        res_scan = solve_diatomic_scf(
                            atom1_name=a1_name,
                            atom1_pos=pos1,
                            atom2_name=a2_name,
                            atom2_pos=pos2,
                            num_electrons=params["num_electrons"],
                            max_iter=50,
                            tol=1e-6
                        )
                        scan_energies.append(res_scan["E_tot"])
                        scan_homo.append(res_scan["eps"][0])
                        scan_lumo.append(res_scan["eps"][1])
                        
                    # 绘制 SCI 双面板图表
                    col_pes1, col_pes2 = st.columns(2)
                    
                    with col_pes1:
                        st.markdown("##### 1. 分子势能曲线 (PES Curve)")
                        fig_pes = go.Figure()
                        fig_pes.add_trace(go.Scatter(
                            x=distances,
                            y=scan_energies,
                            mode='lines+markers',
                            line=dict(color='black', width=1.5),
                            marker=dict(size=6, symbol='circle-open', color='black'),
                            name='PES 自洽势能'
                        ))
                        
                        min_idx = np.argmin(scan_energies)
                        eq_dist = distances[min_idx]
                        eq_energy = scan_energies[min_idx]
                        
                        fig_pes.add_annotation(
                            x=eq_dist,
                            y=eq_energy,
                            text=f"Re ≈ {eq_dist:.2f} Bohr",
                            showarrow=True,
                            arrowhead=1,
                            arrowsize=1.5,
                            arrowcolor="red",
                            ax=40,
                            ay=-40
                        )
                        fig_pes.update_layout(
                            xaxis_title="核间距 R (Bohr)",
                            yaxis_title="系统总能量 E (Hartree)",
                            template="plotly_white",
                            margin=dict(l=40, r=40, t=20, b=40)
                        )
                        st.plotly_chart(fig_pes, use_container_width=True)
                    
                    with col_pes2:
                        st.markdown("##### 2. 分子能级分裂与演化 (Orbital Energy Splitting)")
                        fig_levels = go.Figure()
                        fig_levels.add_trace(go.Scatter(x=distances, y=scan_homo, name='HOMO (成键轨道)', line=dict(color='#1f77b4', width=2)))
                        fig_levels.add_trace(go.Scatter(x=distances, y=scan_lumo, name='LUMO (反键轨道)', line=dict(color='#d62728', width=2)))
                        
                        fig_levels.update_layout(
                            xaxis_title="核间距 R (Bohr)",
                            yaxis_title="能级能量 (Hartree)",
                            template="plotly_white",
                            margin=dict(l=40, r=40, t=20, b=40)
                        )
                        st.plotly_chart(fig_levels, use_container_width=True)
                    
                    st.info(f"计算结果报告：体系的平衡共价键长约为 **{eq_dist:.2f} Bohr** (~ {eq_dist * 0.529177:.3f} Å)，对应的基态能量为 **{eq_energy:.5f} Hartree**。右图清晰显示了随着核间距缩短，孤立的 1s 能级发生强烈的杂化分裂成 bonding (HOMO) 和 anti-bonding (LUMO) 态的物理现象（Walsh 分裂模型）。")
        else:
            st.markdown("#### 一维双势阱系统自洽场势能面（PES）与能级分裂扫描")
            st.write("在该扫描中，系统将构造两个带正电的核吸引势阱： $V_{\\mathrm{ext}}(x) = -\\frac{Z_1}{\\sqrt{(x - d/2)^2 + a^2}} - \\frac{Z_2}{\\sqrt{(x + d/2)^2 + a^2}}$。并通过改变势阱间距 $d$ 来模拟化学键的拉伸，同时计算包含核排斥力 $E_{\\mathrm{nuc}} = \\frac{Z_1 Z_2}{\\sqrt{d^2 + a^2}}$ 的总能量。")
            
            # 参数设置分栏
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                scan_z1 = st.number_input("核 1 电荷 Z1", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
            with col_s2:
                scan_z2 = st.number_input("核 2 电荷 Z2", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
            with col_s3:
                scan_ne = st.number_input("系统电子数", min_value=1, max_value=6, value=2)
            with col_s4:
                scan_soft = st.slider("势场软化常数 a", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
                
            scan_run_1d = st.button("🚀 启动一维势能面与能级演化扫描")
            if scan_run_1d:
                distances = np.linspace(0.4, 6.0, 25)
                progress_bar = st.progress(0.0)
                
                scan_energies = []
                scan_homo = []
                scan_lumo = []
                
                for i, d in enumerate(distances):
                    progress_bar.progress((i + 1) / len(distances))
                    
                    Vext_fn = lambda x_arr: -scan_z1 / np.sqrt((x_arr - d/2.0)**2 + scan_soft**2) - scan_z2 / np.sqrt((x_arr + d/2.0)**2 + scan_soft**2)
                    
                    res_scan = solve_1d_dft(
                        Vext_fn=Vext_fn,
                        num_electrons=scan_ne,
                        L=params["L"],
                        N=params["N"],
                        max_iter=80,
                        tol=1e-5,
                        alpha=0.2,
                        softening=scan_soft
                    )
                    
                    E_nuc = (scan_z1 * scan_z2) / np.sqrt(d**2 + scan_soft**2)
                    E_tot = res_scan["energies"]["E_tot"] + E_nuc
                    scan_energies.append(E_tot)
                    
                    e_vals = res_scan["eigenvalues"]
                    scan_homo.append(e_vals[0])
                    if len(e_vals) > 1:
                        scan_lumo.append(e_vals[1])
                    else:
                        scan_lumo.append(e_vals[0])
                        
                # 绘制图表
                col_p1d1, col_p1d2 = st.columns(2)
                
                with col_p1d1:
                    st.markdown("##### 1. 一维分子势能面曲线 (PES)")
                    fig_pes = go.Figure()
                    fig_pes.add_trace(go.Scatter(
                        x=distances,
                        y=scan_energies,
                        mode='lines+markers',
                        line=dict(color='black', width=1.5),
                        marker=dict(size=6, symbol='circle-open', color='black'),
                        name='1D 总能量 (含核排斥)'
                    ))
                    
                    min_idx = np.argmin(scan_energies)
                    eq_dist = distances[min_idx]
                    eq_energy = scan_energies[min_idx]
                    
                    fig_pes.add_annotation(
                        x=eq_dist,
                        y=eq_energy,
                        text=f"Re ≈ {eq_dist:.2f} Bohr",
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1.5,
                        arrowcolor="red",
                        ax=40,
                        ay=-40
                    )
                    fig_pes.update_layout(
                        xaxis_title="势阱间距 d (Bohr)",
                        yaxis_title="系统总能量 E (Hartree)",
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    st.plotly_chart(fig_pes, use_container_width=True)
                    
                with col_p1d2:
                    st.markdown("##### 2. 科恩-沈单电子能级分裂演化")
                    fig_levels = go.Figure()
                    fig_levels.add_trace(go.Scatter(x=distances, y=scan_homo, name='HOMO 轨道能级', line=dict(color='#1f77b4', width=2)))
                    if scan_ne > 1:
                        fig_levels.add_trace(go.Scatter(x=distances, y=scan_lumo, name='LUMO 轨道能级', line=dict(color='#d62728', width=2)))
                    
                    fig_levels.update_layout(
                        xaxis_title="势阱间距 d (Bohr)",
                        yaxis_title="能级本征值 (Hartree)",
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    st.plotly_chart(fig_levels, use_container_width=True)
                    
                st.info(f"一维自洽扫描结果：体系在间距 **{eq_dist:.2f} Bohr** 时总能量达到基态最低值 **{eq_energy:.5f} Hartree**。这表明两个势阱在空间中发生了有效的轨道杂化与成键锁定。")

    with tab7:
        st.markdown("#### 量子化学自洽场吸附能计算面板 (Adsorption Energy Calculator)")
        st.write("吸附能是研究多体表面催化与键合的最核心热力学量，定义为： $E_{\\mathrm{ads}} = E_{AB} - (E_A + E_B)$。")
        st.write("其中 $E_{AB}$ 为吸附复合物总能量， $E_A$ 和 $E_B$ 分别为孤立吸附质与孤立基底的能量。负值 ($E_{\\mathrm{ads}} < 0$) 代表放热吸附（物理或化学键合稳定状态）。")
        
        # 初始化吸附计算会话状态
        if "ads_results" not in st.session_state:
            st.session_state.ads_results = None
            
        if solver_mode == "1d_dft":
            st.markdown("##### 一维双阱模型吸附能计算参数")
            col_ad1, col_ad2, col_ad3 = st.columns(3)
            with col_ad1:
                ads_z1 = st.number_input("基底势阱核电荷 Z_substrate", min_value=0.5, max_value=3.0, value=1.0, step=0.5, key="ads_1d_z1")
            with col_ad2:
                ads_z2 = st.number_input("吸附质核电荷 Z_adsorbate", min_value=0.5, max_value=3.0, value=1.0, step=0.5, key="ads_1d_z2")
            with col_ad3:
                ads_dist = st.slider("吸附工作距离 d (Bohr)", min_value=0.5, max_value=6.0, value=1.5, step=0.1, key="ads_1d_d")
                
            col_ad4, col_ad5, col_ad6 = st.columns(3)
            with col_ad4:
                ads_ne = st.number_input("复合物总电子数 N_total", min_value=1, max_value=100, value=2, key="ads_1d_ne")
            with col_ad5:
                ads_na = st.number_input("基底单独占据电子数 N_sub", min_value=0, max_value=50, value=1, key="ads_1d_na")
            with col_ad6:
                ads_nb = st.number_input("吸附质单独占据电子数 N_ads", min_value=0, max_value=50, value=1, key="ads_1d_nb")
                
            if ads_na + ads_nb != ads_ne:
                st.warning(f"注意：孤立基底与孤立吸附质的电子数之和 ({ads_na} + {ads_nb} = {ads_na+ads_nb}) 不等于总电子数 ({ads_ne})！这将触发非中性态吸附计算。")
                
            run_ads_1d = st.button("🚀 计算一维系统吸附能")
            
            if run_ads_1d:
                with st.spinner("自洽场 DFT 迭代计算中..."):
                    # 1. Complex AB
                    Vext_ab = lambda x_arr: -ads_z1 / np.sqrt((x_arr - ads_dist/2.0)**2 + params["softening"]**2) - ads_z2 / np.sqrt((x_arr + ads_dist/2.0)**2 + params["softening"]**2)
                    res_ab = solve_1d_dft(Vext_ab, num_electrons=ads_ne, L=params["L"], N=params["N"], max_iter=80, tol=1e-5, alpha=0.2, softening=params["softening"])
                    E_nuc = (ads_z1 * ads_z2) / np.sqrt(ads_dist**2 + params["softening"]**2)
                    E_ab = res_ab["energies"]["E_tot"] + E_nuc
                    
                    # 2. Isolated substrate B
                    Vext_b = lambda x_arr: -ads_z1 / np.sqrt((x_arr - ads_dist/2.0)**2 + params["softening"]**2)
                    if ads_na > 0:
                        res_b = solve_1d_dft(Vext_b, num_electrons=ads_na, L=params["L"], N=params["N"], max_iter=80, tol=1e-5, alpha=0.2, softening=params["softening"])
                        E_b = res_b["energies"]["E_tot"]
                    else:
                        E_b = 0.0
                        
                    # 3. Isolated adsorbate A
                    Vext_a = lambda x_arr: -ads_z2 / np.sqrt((x_arr + ads_dist/2.0)**2 + params["softening"]**2)
                    if ads_nb > 0:
                        res_a = solve_1d_dft(Vext_a, num_electrons=ads_nb, L=params["L"], N=params["N"], max_iter=80, tol=1e-5, alpha=0.2, softening=params["softening"])
                        E_a = res_a["energies"]["E_tot"]
                    else:
                        E_a = 0.0
                        
                    E_ads = E_ab - (E_a + E_b)
                    E_ads_ev = E_ads * 27.2114
                    
                    st.session_state.ads_results = {
                        "type": "1d",
                        "E_ab": E_ab,
                        "E_b": E_b,
                        "E_a": E_a,
                        "E_ads": E_ads,
                        "E_ads_ev": E_ads_ev,
                        "ne": ads_ne,
                        "na": ads_na,
                        "nb": ads_nb
                    }
                    
            if st.session_state.ads_results and st.session_state.ads_results["type"] == "1d":
                r = st.session_state.ads_results
                st.success("吸附能计算完成！")
                st.markdown("##### 🧾 热力学能级数据表格")
                ads_data = {
                    "状态系统": ["复合物总能量 (E_AB, 含核排斥)", "孤立基底能量 (E_substrate)", "孤立吸附质能量 (E_adsorbate)", "吸附能 (E_ads) = E_AB - (E_A + E_B)"],
                    "能量 (Hartree)": [f"{r['E_ab']:.6f}", f"{r['E_b']:.6f}", f"{r['E_a']:.6f}", f"{r['E_ads']:.6f}"],
                    "能量 (eV)": [f"{r['E_ab']*27.2114:.4f}", f"{r['E_b']*27.2114:.4f}", f"{r['E_a']*27.2114:.4f}", f"{r['E_ads_ev']:.4f}"],
                    "电子排布": [f"{r['ne']} e", f"{r['na']} e", f"{r['nb']} e", "N/A"]
                }
                st.table(ads_data)
                
                if r['E_ads'] < -0.05:
                    st.info(f"💡 **热力学分析**：吸附能为 **{r['E_ads_ev']:.3f} eV**。吸附过程为**放热反应**，该吸附结构在能量上是**稳定的**。由于结合能较大，该体系倾向于发生**化学吸附**。")
                elif r['E_ads'] < 0:
                    st.info(f"💡 **热力学分析**：吸附能为 **{r['E_ads_ev']:.3f} eV**。吸附为**放热反应**，体系**稳定**。由于结合能极弱，该体系主要表现为弱相互作用下的**物理吸附**（范德华相互作用）。")
                else:
                    st.warning(f"💡 **热力学分析**：吸附能为 **{r['E_ads_ev']:.3f} eV**（正值）。吸附为**吸热反应**，该几何构型在能量上是**不稳定的**，吸附质将发生自发脱附。")
                    
        else:
            st.markdown("##### 3D 双原子分子吸附能计算参数 (STO-3G)")
            col_ad3d1, col_ad3d2, col_ad3d3 = st.columns(3)
            elements_3d = ["H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG"]
            with col_ad3d1:
                ads_atom1 = st.selectbox("基底原子 Substrate", elements_3d, index=0, key="ads_3d_a1")
            with col_ad3d2:
                ads_atom2 = st.selectbox("吸附质气体原子 Adsorbate", elements_3d, index=0, key="ads_3d_a2")
            with col_ad3d3:
                ads_dist_3d = st.slider("吸附工作距离 R (Bohr)", min_value=0.5, max_value=6.0, value=1.4, step=0.1, key="ads_3d_r")
                
            run_ads_3d = st.button("🚀 计算三维分子吸附能")
            
            if run_ads_3d:
                with st.spinner("自洽场 Hartree-Fock 积分与 SCF 求解中..."):
                    # 动态确定真实原子序数（电子数）
                    def get_atomic_number(name):
                        ELEMENT_Z = {
                            'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6,
                            'N': 7, 'O': 8, 'F': 9, 'NE': 10, 'NA': 11, 'MG': 12, 'G': 0
                        }
                        return ELEMENT_Z.get(name.upper(), 0)
                        
                    ne_b = get_atomic_number(ads_atom1)
                    ne_a = get_atomic_number(ads_atom2)
                    ne_ab = ne_b + ne_a
                    
                    # 1. Complex AB
                    pos1 = [0.0, 0.0, -ads_dist_3d/2.0]
                    pos2 = [0.0, 0.0, ads_dist_3d/2.0]
                    res_ab = solve_diatomic_scf(ads_atom1, pos1, ads_atom2, pos2, num_electrons=ne_ab, max_iter=50, tol=1e-6)
                    E_ab = res_ab["E_tot"]
                    
                    # 2. Isolated substrate B (Atom 1 + Ghost at far-away pos)
                    pos_far2 = [0.0, 0.0, 50.0]
                    res_b = solve_diatomic_scf(ads_atom1, pos1, 'G', pos_far2, num_electrons=ne_b, max_iter=50, tol=1e-6)
                    E_b = res_b["E_tot"]
                    
                    # 3. Isolated adsorbate A (Ghost at far-away pos + Atom 2)
                    pos_far1 = [0.0, 0.0, -50.0]
                    res_a = solve_diatomic_scf('G', pos_far1, ads_atom2, pos2, num_electrons=ne_a, max_iter=50, tol=1e-6)
                    E_a = res_a["E_tot"]
                    
                    E_ads = E_ab - (E_a + E_b)
                    E_ads_ev = E_ads * 27.2114
                    
                    st.session_state.ads_results = {
                        "type": "3d",
                        "atom1": ads_atom1,
                        "atom2": ads_atom2,
                        "E_ab": E_ab,
                        "E_b": E_b,
                        "E_a": E_a,
                        "E_ads": E_ads,
                        "E_ads_ev": E_ads_ev
                    }
                    
            if st.session_state.ads_results and st.session_state.ads_results["type"] == "3d":
                r = st.session_state.ads_results
                st.success("吸附能计算完成！")
                st.markdown("##### 🧾 热力学能级数据表格")
                ads_data = {
                    "状态系统": [f"复合体系 ({r['atom1']}-{r['atom2']} 总能量)", f"孤立基底原子 ({r['atom1']} 能量)", f"孤立吸附原子 ({r['atom2']} 能量)", "吸附能 (E_ads) = E_AB - (E_A + E_B)"],
                    "能量 (Hartree)": [f"{r['E_ab']:.6f}", f"{r['E_b']:.6f}", f"{r['E_a']:.6f}", f"{r['E_ads']:.6f}"],
                    "能量 (eV)": [f"{r['E_ab']*27.2114:.4f}", f"{r['E_b']*27.2114:.4f}", f"{r['E_a']*27.2114:.4f}", f"{r['E_ads_ev']:.4f}"],
                    "电荷体系": ["中性分子 (2 e)", f"中性 {r['atom1']} 原子", f"中性 {r['atom2']} 原子", "N/A"]
                }
                st.table(ads_data)
                
                if r['E_ads'] < -0.05:
                    st.info(f"💡 **热力学分析**：吸附能为 **{r['E_ads_ev']:.3f} eV**。吸附过程为**放热反应**，该化学成键吸附结构在能量上是**稳定的**。表现为强烈的**化学吸附**。")
                elif r['E_ads'] < 0:
                    st.info(f"💡 **热力学分析**：吸附能为 **{r['E_ads_ev']:.3f} eV**。吸附为**放热反应**，体系**微弱稳定**。主要表现为弱相互作用下的**物理吸附**。")
                else:
                    st.warning(f"💡 **热力学分析**：吸附能为 **{r['E_ads_ev']:.3f} eV**（正值）。吸附为**吸热反应**，该核间距过于接近（排斥力主导），结构**不稳定**，分子将发生脱附。")

    with tab8:
        st.markdown("#### ⚛️ 3D 分子几何结构优化与坐标弛豫 (Geometry Optimization)")
        st.write("结构优化是通过计算各原子核在三维空间中受到的Born-Oppenheimer势能面力矢量，采用数值梯度算法迭代寻找能量极小值对应的平衡空间构型。")
        
        if solver_mode == "1d_dft":
            st.info("💡 结构优化功能目前专为 **3D 分子自洽场求解器** 设计。请在侧边栏选择 **3D STO-3G 自洽场求解器** 并选用多原子模型启动！")
        else:
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                opt_step = st.slider("结构优化步长 (Relax Step Size)", min_value=0.02, max_value=0.20, value=0.08, step=0.02, key="opt_step_val")
            with col_opt2:
                opt_max_iter = st.slider("最大弛豫迭代步数", min_value=5, max_value=40, value=20, step=1, key="opt_max_it_val")
                
            run_opt = st.button("🚀 开始分子结构优化 (Cartesian Geometry Relaxation)")
            
            if st.session_state.get("auto_optimize", False):
                st.session_state.auto_optimize = False
                run_opt = True
                
            if "opt_history" not in st.session_state:
                st.session_state.opt_history = None
                
            if run_opt:
                with st.spinner("分子结构在 Cartesian 空间中弛豫优化中，求解每一步的数值梯度受力..."):
                    import copy
                    curr_atoms = copy.deepcopy(params["atoms"])
                    num_at = len(curr_atoms)
                    opt_hist = []
                    
                    from diatomic_engine import solve_multi_atom_scf
                    
                    for it in range(opt_max_iter):
                        # 运行 SCF 计算当前能量
                        res_scf = solve_multi_atom_scf(curr_atoms, num_electrons=params["num_electrons"], max_iter=45, tol=1e-5, multiplicity=params.get("multiplicity", None))
                        E0 = res_scf["E_tot"]
                        
                        # 数值差分计算受力
                        d = 0.005
                        forces = np.zeros((num_at, 3))
                        
                        for i in range(num_at):
                            for c in range(3):
                                # Shift +d
                                curr_atoms[i]["pos"][c] += d
                                res_p = solve_multi_atom_scf(curr_atoms, num_electrons=params["num_electrons"], max_iter=30, tol=1e-4, multiplicity=params.get("multiplicity", None))
                                E_p = res_p["E_tot"]
                                
                                # Shift -d
                                curr_atoms[i]["pos"][c] -= 2*d
                                res_m = solve_multi_atom_scf(curr_atoms, num_electrons=params["num_electrons"], max_iter=30, tol=1e-4, multiplicity=params.get("multiplicity", None))
                                E_m = res_m["E_tot"]
                                
                                # Restore
                                curr_atoms[i]["pos"][c] += d
                                
                                forces[i, c] = -(E_p - E_m) / (2 * d)
                                
                        max_f = np.max(np.abs(forces))
                        opt_hist.append({
                            "step": it,
                            "energy": E0,
                            "max_force": max_f,
                            "atoms": copy.deepcopy(curr_atoms)
                        })
                        
                        # 收敛阈值
                        if max_f < 1.5e-3:
                            break
                            
                        # 更新位置
                        for i in range(num_at):
                            for c in range(3):
                                curr_atoms[i]["pos"][c] += opt_step * forces[i, c]
                                
                        # 减去质心平移，保持分子居中
                        coords = np.array([at["pos"] for at in curr_atoms])
                        mean_coord = np.mean(coords, axis=0)
                        for i in range(num_at):
                            curr_atoms[i]["pos"] = (np.array(curr_atoms[i]["pos"]) - mean_coord).tolist()
                            
                    st.session_state.opt_history = opt_hist
                    
            if st.session_state.opt_history:
                h_list = st.session_state.opt_history
                final_atoms = h_list[-1]["atoms"]
                final_E = h_list[-1]["energy"]
                
                st.success(f"结构优化收敛完成！最终收敛总能量为 **{final_E:.6f} Hartree**。")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown("##### 📉 结构优化能量与受力收敛路径")
                    fig_opt = make_subplots(specs=[[{"secondary_y": True}]])
                    steps = [h["step"] for h in h_list]
                    es = [h["energy"] for h in h_list]
                    forces = [h["max_force"] for h in h_list]
                    
                    fig_opt.add_trace(go.Scatter(x=steps, y=es, name="总能量 E (Hartree)", line=dict(color="#1f77b4", width=2.0)), secondary_y=False)
                    fig_opt.add_trace(go.Scatter(x=steps, y=forces, name="最大原子力 F_max (a.u.)", line=dict(color="#d62728", width=1.5, dash="dash")), secondary_y=True)
                    
                    fig_opt.update_layout(
                        xaxis_title="弛豫步数 (Optimization Steps)",
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    fig_opt.update_yaxes(title_text="总能量 E (Hartree)", secondary_y=False)
                    fig_opt.update_yaxes(title_text="最大受力 F_max (a.u.)", secondary_y=True)
                    st.plotly_chart(fig_opt, use_container_width=True)
                    
                with col_res2:
                    st.markdown("##### 🧾 优化后的平衡几何构型 (Bohr)")
                    opt_xyz_lines = []
                    for idx, at in enumerate(final_atoms):
                        opt_xyz_lines.append(f"{at['name']} {at['pos'][0]:.6f} {at['pos'][1]:.6f} {at['pos'][2]:.6f}")
                    opt_xyz_str = "\n".join(opt_xyz_lines)
                    st.text_area("优化后的 XYZ 坐标结构", value=opt_xyz_str, height=180)
                    
                    if st.button("💾 应用此优化后的结构至主模拟配置"):
                        st.session_state.config["params"]["atoms"] = final_atoms
                        st.session_state.calc_results = None
                        sync_config_to_sidebar_state()
                        st.success("结构更新成功！主配置及侧边栏坐标已刷新。")
                        st.rerun()

    with tab9:
        st.markdown("#### 🎵 分子简正振动分析与红外光谱 (Vibrational Analysis & IR Spectrum)")
        st.write("振动分析是在优化后的平衡结构处计算原子力常数矩阵 (Hessian)，进而通过对角化求得分子本征振动频率与红外吸收强度。")
        
        if solver_mode == "1d_dft":
            st.info("💡 振动能谱分析目前专为 **3D 分子求解器** 设计。请在侧边栏选择 **3D STO-3G 自洽场求解器**。")
        else:
            st.warning("⚠️ **温馨提示**：振动分析需要对所有原子坐标进行数值双侧微分微扰，计算约需要 10~30 秒，建议在进行结构优化收敛后再运行。")
            run_vib = st.button("🎵 开始红外振动能谱分析 (Run IR Analysis)")
            
            if "vib_results" not in st.session_state:
                st.session_state.vib_results = None
                
            if run_vib:
                with st.spinner("计算力常数 Hessian 矩阵与偶极矩数值导数中..."):
                    freqs, ints, vecs = run_vibrational_analysis(params["atoms"], params["num_electrons"], multiplicity=params.get("multiplicity", None))
                    
                    # 过滤虚频和刚体平移转动模式 (低于 10 cm^-1 的模式)
                    vib_modes = []
                    for k in range(len(freqs)):
                        if freqs[k] > 10.0:
                            vib_modes.append({
                                "mode_idx": len(vib_modes) + 1,
                                "frequency": freqs[k],
                                "intensity": ints[k],
                                "vec": vecs[:, k]
                            })
                            
                    st.session_state.vib_results = vib_modes
                    
            if st.session_state.vib_results:
                modes = st.session_state.vib_results
                st.success("振动分析计算成功！")
                
                col_vib1, col_vib2 = st.columns(2)
                with col_vib1:
                    st.markdown("##### 📈 模拟分子红外吸收光谱 (Infrared Spectrum)")
                    # 生成连续的红外吸收光谱图 (Lorentzian 展宽)
                    x_ir = np.linspace(200.0, 4000.0, 500)
                    y_ir = np.zeros_like(x_ir)
                    gamma = 30.0 # 谱线半峰宽 (cm^-1)
                    
                    for m in modes:
                        # Lorentzian line shape
                        y_ir += m["intensity"] * (gamma**2) / ((x_ir - m["frequency"])**2 + gamma**2)
                        
                    # 转换为透过率 (Transmittance) 或者吸光度 (Absorbance)
                    # 吸光度常规表示
                    fig_ir = go.Figure()
                    fig_ir.add_trace(go.Scatter(x=x_ir, y=y_ir, fill='tozeroy', line=dict(color='#d62728', width=2), name="IR Absorbance"))
                    fig_ir.update_layout(
                        xaxis_title="波数 Wavenumber (cm^-1)",
                        yaxis_title="红外吸收强度 IR Intensity (Debye^2/AMU-Bohr^2)",
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    st.plotly_chart(fig_ir, use_container_width=True)
                    
                with col_vib2:
                    st.markdown("##### 🧾 分子振动模式列表")
                    vib_table = {
                        "振动简正模式 (Mode)": [f"模式 #{m['mode_idx']}" for m in modes],
                        "本征频率 (Frequency, cm^-1)": [f"{m['frequency']:.1f}" for m in modes],
                        "红外强度 (IR Intensity)": [f"{m['intensity']:.4f}" for m in modes],
                        "简正分量性质": ["红外强活性" if m['intensity'] > 1.0 else ("弱活性" if m['intensity'] > 0.05 else "非活性 (Raman)") for m in modes]
                    }
                    st.table(vib_table)


    with tab10:
        st.markdown("#### ⚡ 分子静电势分析 (Molecular Electrostatic Potential - MEP)")
        st.write("分子静电势 (MEP) 描述了分子在周围空间产生静电势场的强弱与极性分布，是预测亲电/亲核反应位点以及分子间非共价相互作用的最核心工具。")
        
        if solver_mode == "1d_dft":
            st.info("💡 静电势分析目前专为 **3D 分子自洽场** 设计。请在侧边栏选择 **3D STO-3G 自洽场求解器** 并选用多原子模型。")
        else:
            with st.spinner("正在基于电子密度矩阵与核电荷解析求解三维网格上的精确静电势能场..."):
                # 计算 2D 截面的 MEP
                grid_y = np.linspace(-3.5, 3.5, 60)
                grid_z = np.linspace(-4.5, 4.5, 70)
                Z_mep, Y_mep = np.meshgrid(grid_z, grid_y)
                
                # 计算 MEP 矩阵
                mep_2d = calculate_mep_grid_2d(params["atoms"], res, Y_mep, Z_mep)
                
                # 绘制 2D MEP 等高线图，使用 RdBu 发散色标（红负蓝正，常用于MEP）
                fig_mep = go.Figure(data=go.Contour(
                    z=mep_2d,
                    x=grid_z,
                    y=grid_y,
                    colorscale='RdBu',
                    zmid=0.0,
                    colorbar=dict(title="静电势 V (Hartree/e)"),
                    contours=dict(
                        coloring='heatmap',
                        showlabels=True,
                        labelfont=dict(size=10, color='white')
                    )
                ))
                
                # 叠加绘制分子核结构
                for idx, at in enumerate(params["atoms"]):
                    fig_mep.add_trace(go.Scatter(
                        x=[at["pos"][2]],
                        y=[at["pos"][1]],
                        mode='markers+text',
                        marker=dict(size=14, color='black', symbol='circle'),
                        text=f"{at['name']}#{idx+1}",
                        textposition="top center",
                        textfont=dict(color='white', size=11, weight='bold'),
                        showlegend=False
                    ))
                    
                fig_mep.update_layout(
                    title="分子平面静电势能等高线图 (YZ 截面，X=0)",
                    xaxis_title="Z 轴位置 (Bohr)",
                    yaxis_title="Y 轴位置 (Bohr)",
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_mep, use_container_width=True)
                st.info("💡 **学术图表分析**：在静电势图中，**红色区域**（负值）代表富电子区，是潜在的**亲电攻击反应活性中心**（例如水分子中氧原子的孤对电子区）；**蓝色区域**（正值）代表缺电子区，是潜在的**亲核攻击反应活性中心**（例如水分子中的氢原子核外侧）。")

    with tab11:
        st.markdown("#### 🌐 周期性晶格自洽 DFT 计算与能带结构 (Periodic Band Structure)")
        st.write("能带结构（Band Structure）是固体物理和材料科学的核心，描述了电子在周期性晶格中允许的能级与准动量 $k$ 的色散关系。")
        
        if solver_mode != "1d_dft":
            st.info("💡 周期性能带结构目前专为 **1D 周期性晶格 DFT 求解器** 设计。请在左侧侧边栏切换求解器为 **1D Kohn-Sham DFT 求解器**。")
        else:
            st.warning("⚠️ **物理说明**：周期性色散计算采用 Bloch 边界条件 $\\psi(x+L) = e^{ikL} \\psi(x)$。将在第一布里渊区 $k \\in [-\\pi/L, \\pi/L]$ 采样 $15$ 个 $k$-点进行自洽阻尼迭代求解。")
            
            run_band = st.button("🌐 运行一维晶格周期性能带 SCF 计算", key="run_band_scf_btn")
            
            if "band_results" not in st.session_state:
                st.session_state.band_results = None
                
            if run_band:
                with st.spinner("正在 Brillouin 区进行 k-点自洽能带求解..."):
                    # 动态生成势函数
                    from dft_engine import solve_1d_periodic_dft
                    
                    # 定义一维外部势能
                    raw_expr = params.get("potential_expr", "-2.0 / np.sqrt(x**2 + 1.0)")
                    def v_ext_pbc(x_arr):
                        import numpy as np
                        # 局部作用域执行以支持numpy数学函数
                        local_dict = {"x": x_arr, "np": np}
                        return eval(raw_expr, {"__builtins__": None}, local_dict)
                        
                    band_res = solve_1d_periodic_dft(
                        Vext_fn=v_ext_pbc,
                        num_electrons=params["num_electrons"],
                        L=params["L"],
                        N=params["N"],
                        max_iter=params["max_iter"],
                        tol=params["tol"],
                        alpha=params["alpha"],
                        softening=params["softening"],
                        functional=params["functional"],
                        nkpoints=15
                    )
                    st.session_state.band_results = band_res
                    
            if st.session_state.band_results is not None:
                b_res = st.session_state.band_results
                st.success(f"🎉 能带自洽场在第 {b_res['iterations']} 步收敛成功！总能量: {b_res['energies']['E_tot']:.6f} Hartree")
                
                col_b1, col_b2 = st.columns([2, 1])
                with col_b1:
                    st.markdown("##### 📈 一维第一布里渊区电子能带色散图 (Electronic Band Structure)")
                    
                    # 绘制能带色散
                    fig_band = go.Figure()
                    k_pts = b_res['k_points']
                    
                    # b_res['bands'] 形状为 (N, nkpoints)
                    n_bands_to_plot = min(8, b_res['bands'].shape[0])
                    
                    # 查找 Fermi Level
                    occupied_energies = b_res['bands'][b_res['occupations'] > 0]
                    fermi_level = np.max(occupied_energies) if len(occupied_energies) > 0 else 0.0
                    
                    for ib in range(n_bands_to_plot):
                        band_y = b_res['bands'][ib, :]
                        is_occ = np.any(b_res['occupations'][ib, :] > 0)
                        line_style = dict(color="#1f77b4", width=2.0) if is_occ else dict(color="#d62728", width=1.5, dash="dash")
                        name_str = f"价带 VB {ib+1}" if is_occ else f"导带 CB {ib-np.sum(b_res['occupations'].max(axis=1) > 0)+1}"
                        fig_band.add_trace(go.Scatter(
                            x=k_pts,
                            y=band_y,
                            mode='lines+markers',
                            name=f"{name_str} ({'占' if is_occ else '空'})",
                            line=line_style
                        ))
                        
                    # 绘制费米能级虚线
                    fig_band.add_hline(y=fermi_level, line_dash="dash", line_color="black", line_width=1.5, annotation_text=f"Fermi Level (Ef={fermi_level:.4f} Ha)", annotation_position="bottom right")
                    
                    fig_band.update_layout(
                        xaxis=dict(
                            title="晶体动量 k (1/Bohr)",
                            tickmode="array",
                            tickvals=[-np.pi/params["L"], 0, np.pi/params["L"]],
                            ticktext=["-π/L", "Γ (0)", "π/L"]
                        ),
                        yaxis_title="能级能值 Energy (Hartree)",
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=30, b=40)
                    )
                    st.plotly_chart(fig_band, use_container_width=True)
                    
                with col_b2:
                    st.markdown("##### ⚛️ 晶元单胞内电子密度与周期分布")
                    fig_p_dens = go.Figure()
                    fig_p_dens.add_trace(go.Scatter(
                        x=b_res['x'],
                        y=b_res['density'],
                        name="周期晶包内电荷密度",
                        line=dict(color="#2ca02c", width=2.5)
                    ))
                    fig_p_dens.update_layout(
                        xaxis_title="晶胞内相对位置 x (Bohr)",
                        yaxis_title="电荷密度 ρ(x) (e/Bohr)",
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=30, b=40)
                    )
                    st.plotly_chart(fig_p_dens, use_container_width=True)
                    
                    # 能量分解显示
                    st.markdown("**周期晶胞能量分项：**")
                    p_comps = b_res['energies']
                    st.write(f"- 动能项 E_kin: `{p_comps['E_kin']:.4f} Ha`")
                    st.write(f"- 外部势 E_ext: `{p_comps['E_ext']:.4f} Ha`")
                    st.write(f"- 经典静电排斥 E_H: `{p_comps['E_H']:.4f} Ha`")
                    st.write(f"- 交换关联能 E_xc: `{p_comps['E_xc']:.4f} Ha`")
                    st.write(f"- **晶胞基态总能量 E_tot**: `{p_comps['E_tot']:.4f} Ha`")

# 理论支持文档面板
st.markdown("---")
st.markdown("### 📚 密度泛函理论与量子化学方法学背景 (DFT & Hartree-Fock Formulations)")
with st.expander("显示理论支持数学公式"):
    st.markdown("""
    #### 1. 密度泛函理论（DFT）与 Kohn-Sham 自洽场方程
    根据 Hohenberg-Kohn 定理，多体电子相互作用基态体系的所有物理量都可以被电子几率密度 $\\rho(\\mathbf{r})$ 唯一确定。
    
    Kohn-Sham 方法通过引入辅助无相互作用体系，将原本极其复杂的非对角电子相关联问题化归为有效势 $V_{\mathrm{eff}}(\\mathbf{r})$ 中的单粒子薛定谔方程组求解：
    $$ \\left( -\\frac{1}{2}\\nabla^2 + V_{\\mathrm{eff}}(\\mathbf{r}) \\right) \\psi_i(\\mathbf{r}) = \\epsilon_i \\psi_i(\\mathbf{r}) $$
    有效势包括外部原子核电荷势 $V_{\\mathrm{ext}}$、哈特里势 $V_{\\mathrm{H}}$ 与交换关联势 $V_{\\mathrm{xc}}$：
    $$ V_{\\mathrm{eff}}(\\mathbf{r}) = V_{\\mathrm{ext}}(\\mathbf{r}) + V_{\\mathrm{H}}(\\mathbf{r}) + V_{\\mathrm{xc}}(\\mathbf{r}) $$
    
    - **哈特里势 $V_{\\mathrm{H}}$**（经典静电屏蔽）：
      $$ V_{\\mathrm{H}}(\\mathbf{r}) = \\int \\frac{\\rho(\\mathbf{r}')}{|\\mathbf{r} - \\mathbf{r}'|} d\\mathbf{r}' $$
    - **交换关联势 $V_{\\mathrm{xc}}$**：处理量子自旋排斥（泡利原理）与关联。我们在一维中采用 Wigner LDA 局域密度近似描述：
      $$ V_{\\mathrm{xc}}(\\rho) = \\frac{\\partial (\\rho \\epsilon_{\\mathrm{xc}})}{\\partial \\rho} $$
    
    密度自洽迭代公式：
    $$ \\rho(\\mathbf{r}) = \\sum_i f_i |\\psi_i(\\mathbf{r})|^2 $$
    
    #### 2. Hartree-Fock 分子轨道自洽场与 Roothaan 方程组
    Hartree-Fock 方法使用单 Slater 行列式来近似分子波函数。将分子轨道（MO）用原子基组轨道（AO）线性组合展开（LCAO）：
    $$ \\psi_i(\\mathbf{r}) = \\sum_{\\mu} C_{\\mu i} \\phi_{\\mu}(\\mathbf{r}) $$
    通过变分法，将连续极值问题离散转化为代数矩阵求解（**Roothaan-Hall 方程组**）：
    $$ \\mathbf{F} \\mathbf{C} = \\mathbf{S} \\mathbf{C} \\boldsymbol{\\epsilon} $$
    
    其中：
    - $\\mathbf{F}$ 为 **Fock 矩阵**：包含单电子核吸引和动能部分 $H^\\text{core}$，以及双电子 Coulomb 和 Exchange 排斥相互作用贡献：
      $$ F_{\\mu\\nu} = H^\\text{core}_{\\mu\\nu} + \\sum_{\\lambda\\sigma} P_{\\lambda\\sigma} \\left[ (\\mu\\nu|\\lambda\\sigma) - \\frac{1}{2} (\\mu\\sigma|\\lambda\\nu) \\right] $$
    - $\\mathbf{S}$ 为 **重叠矩阵 (Overlap Matrix)**：$S_{\\mu\\nu} = \\langle \\phi_\\mu | \\phi_\\nu \\rangle$。
    - $\\mathbf{P}$ 为 **电子密度矩阵 (Density Matrix)**：$P_{\\mu\\nu} = 2 \\sum_i C_{\\mu i} C_{\\nu i}$。
    - $(\\mu\\nu|\\lambda\\sigma)$ 代表原子基组间的 **双电子排斥积分 (ERI)**。
    
    #### 3. STO-3G 极小高斯基组
    为实现快速解析计算，我们用 3 个原始 Gaussian Primitives 去最小二乘拟合 Slater-Type 轨道。高斯乘积定理保证了重叠、动能、核吸引以及 ERI 都可以用 Boys 误差函数解析地给出精确闭式解，无需空间网格数值求和积分，这是量子化学的核心基石。
    """)
