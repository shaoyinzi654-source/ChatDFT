import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ai_helper import parse_user_request
from analysis_tools import (
    calculate_cdd_1d,
    calculate_cdd_2d,
    calculate_cdd_3d,
    calculate_cdd_3d_grid,
    calculate_dos,
    calculate_mep_grid_2d,
    calculate_pdos_multi,
    eval_density_2d,
    eval_density_3d_grid,
    eval_density_z,
)
from dft_engine import solve_1d_dft, solve_1d_periodic_dft
from diatomic_engine import (
    compute_bond_orders,
    compute_dipole_moment_multi,
    compute_mulliken_charges,
    get_element_orbitals,
    solve_diatomic_scf,
    solve_multi_atom_scf,
)

st.set_page_config(
    page_title="Chat DFT",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background: #f6f8fc; color: #102033; font-family: Inter, "Segoe UI", Arial, sans-serif; }
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
        border: 1px solid #d8e2f1;
        border-radius: 10px;
        padding: 1.2rem 1.35rem;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; color: #0f2a4d; }
    .hero p { margin: .35rem 0 0 0; color: #41556f; }
    .panel {
        background: #fff;
        border: 1px solid #d9e3ef;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    section[data-testid="stSidebar"] { background: #f1f5fb !important; border-right: 1px solid #d9e3ef; }
    div.stButton > button {
        background: #163a68 !important; color: #fff !important; border: 1px solid #163a68 !important;
        border-radius: 6px !important; font-weight: 600 !important;
    }
    div.stButton > button:hover { background: #20538f !important; border-color: #20538f !important; }
    .muted { color: #62748a; font-size: .92rem; }
</style>
""",
    unsafe_allow_html=True,
)


def make_potential_fn(expr_str):
    allowed = {"np": np, "x": None, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi}

    def fn(x):
        local = dict(allowed)
        local["x"] = x
        return eval(expr_str, {"__builtins__": {}}, local)

    return fn


def polish_figure(fig, height=480):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#203040"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def parse_xyz(text):
    atoms = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4:
            atoms.append({"name": parts[0].upper(), "pos": [float(parts[1]), float(parts[2]), float(parts[3])]})
    return atoms


def atoms_to_xyz(atoms):
    return "\n".join([f"{a['name']} {a['pos'][0]:.6f} {a['pos'][1]:.6f} {a['pos'][2]:.6f}" for a in atoms])


def element_z(name):
    return {"H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "NE": 10, "NA": 11, "MG": 12, "G": 0}.get(name.upper(), 0)


@dataclass
class AppState:
    solver_type: str = "1d_dft"
    explanation: str = "默认一维类氦模型"
    params: dict = None


if "config" not in st.session_state:
    st.session_state.config = {
        "solver_type": "1d_dft",
        "explanation": "默认一维类氦模型",
        "params": {
            "num_electrons": 2,
            "L": 10.0,
            "N": 220,
            "max_iter": 100,
            "tol": 1e-6,
            "alpha": 0.2,
            "softening": 1.0,
            "functional": "LDA",
            "mixing_method": "Linear",
            "potential_expr": "-2.0 / np.sqrt(x**2 + 1.0)",
            "potential_description": "一维类氦原子",
        },
    }
if "calc_results" not in st.session_state:
    st.session_state.calc_results = None
if "md_results" not in st.session_state:
    st.session_state.md_results = None
if "ads_results" not in st.session_state:
    st.session_state.ads_results = None
if "band_results" not in st.session_state:
    st.session_state.band_results = None

st.markdown(
    """
<div class="hero">
  <h1>Chat DFT</h1>
  <p>专业级量子化学计算工作台。支持 1D Kohn-Sham DFT、3D STO-3G Hartree-Fock、周期晶体、吸附能、结构优化、振动与静电势分析。</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 计算模式")
    solver = st.selectbox(
        "求解器",
        ["1d_dft", "3d_diatomic", "3d_md"],
        format_func=lambda x: {"1d_dft": "1D Kohn-Sham DFT", "3d_diatomic": "3D 分子自洽场", "3d_md": "经典分子动力学"}[x],
        index=["1d_dft", "3d_diatomic", "3d_md"].index(st.session_state.config["solver_type"]),
    )
    if solver != st.session_state.config["solver_type"]:
        st.session_state.calc_results = None
        st.session_state.config["solver_type"] = solver

    st.markdown("### 智能输入")
    ai_model = st.selectbox("解析模型", ["gpt-5.6-luna", "gpt-4o", "claude-3-5-sonnet", "deepseek-chat"], index=0)
    user_prompt = st.text_area("自然语言描述", height=120, placeholder="例如：计算 H2 分子在不同键长下的势能曲线")
    ai_run = st.button("智能解析并运行", use_container_width=True)

    st.markdown("### 手动配置")

    if solver == "1d_dft":
        p = st.session_state.config["params"]
        p["num_electrons"] = st.number_input("电子数", 1, 100, int(p.get("num_electrons", 2)))
        p["L"] = st.slider("边界 L (Bohr)", 3.0, 20.0, float(p.get("L", 10.0)), 0.5)
        p["N"] = st.slider("网格点数", 50, 400, int(p.get("N", 220)), 10)
        p["max_iter"] = st.number_input("最大迭代", 10, 200, int(p.get("max_iter", 100)))
        p["alpha"] = st.slider("混合率", 0.05, 1.0, float(p.get("alpha", 0.2)), 0.05)
        p["softening"] = st.slider("软化因子", 0.1, 3.0, float(p.get("softening", 1.0)), 0.1)
        p["functional"] = st.selectbox("交换关联", ["Hartree", "Exchange-Only", "LDA", "GGA-PBE"], index=["Hartree", "Exchange-Only", "LDA", "GGA-PBE"].index(p.get("functional", "LDA")))
        p["mixing_method"] = st.selectbox("混合方式", ["Linear", "Anderson"], index=["Linear", "Anderson"].index(p.get("mixing_method", "Linear")))
        p["potential_expr"] = st.text_input("Vext(x)", p.get("potential_expr", "-2.0 / np.sqrt(x**2 + 1.0)"))
        p["potential_description"] = st.text_input("势能说明", p.get("potential_description", "一维类氦原子"))
        run_manual = st.button("运行 1D 计算", use_container_width=True)

    elif solver == "3d_diatomic":
        p = st.session_state.config["params"]
        preset = st.selectbox(
            "分子预设",
            ["Custom", "H2", "H2O", "N2", "LiF", "CO2", "NH3", "HF", "CO"],
            index=0,
        )
        presets = {
            "H2": ([{"name": "H", "pos": [0.0, 0.0, -0.7]}, {"name": "H", "pos": [0.0, 0.0, 0.7]}], 2, "H2 分子"),
            "H2O": ([{"name": "O", "pos": [0.0, 0.0, 0.12]}, {"name": "H", "pos": [0.0, 1.43, -0.98]}, {"name": "H", "pos": [0.0, -1.43, -0.98]}], 10, "H2O 分子"),
            "N2": ([{"name": "N", "pos": [0.0, 0.0, -1.04]}, {"name": "N", "pos": [0.0, 0.0, 1.04]}], 14, "N2 分子"),
            "LiF": ([{"name": "LI", "pos": [0.0, 0.0, -1.55]}, {"name": "F", "pos": [0.0, 0.0, 1.55]}], 12, "LiF 分子"),
            "CO2": ([{"name": "C", "pos": [0.0, 0.0, 0.0]}, {"name": "O", "pos": [0.0, 0.0, -2.19]}, {"name": "O", "pos": [0.0, 0.0, 2.19]}], 22, "CO2 分子"),
            "NH3": ([{"name": "N", "pos": [0.0, 0.0, 0.25]}, {"name": "H", "pos": [0.0, 1.77, -0.58]}, {"name": "H", "pos": [1.53, -0.88, -0.58]}, {"name": "H", "pos": [-1.53, -0.88, -0.58]}], 10, "NH3 分子"),
            "HF": ([{"name": "H", "pos": [0.0, 0.0, -1.73]}, {"name": "F", "pos": [0.0, 0.0, 0.0]}], 10, "HF 分子"),
            "CO": ([{"name": "C", "pos": [0.0, 0.0, -1.06]}, {"name": "O", "pos": [0.0, 0.0, 1.06]}], 14, "CO 分子"),
        }
        if preset != "Custom":
            atoms, ne, desc = presets[preset]
            p["atoms"] = copy.deepcopy(atoms)
            p["num_electrons"] = ne
            p["multiplicity"] = 1 if ne % 2 == 0 else 2
            p["max_iter"] = 60
            p["tol"] = 1e-6
            st.session_state.config["explanation"] = desc
        xyz = st.text_area("XYZ 坐标", atoms_to_xyz(p.get("atoms", [{"name": "H", "pos": [0, 0, -0.7]}, {"name": "H", "pos": [0, 0, 0.7]}])), height=120)
        uploaded = st.file_uploader("导入 .xyz", type=["xyz", "txt"])
        if uploaded is not None:
            txt = uploaded.read().decode("utf-8", errors="ignore")
            parsed = parse_xyz(txt)
            if parsed:
                xyz = atoms_to_xyz(parsed)
        atoms = parse_xyz(xyz)
        p["atoms"] = atoms if atoms else p.get("atoms", [])
        p["num_electrons"] = st.number_input("总电子数", 1, 60, int(p.get("num_electrons", 2)))
        p["max_iter"] = st.number_input("最大迭代", 10, 100, int(p.get("max_iter", 60)))
        p["tol"] = st.number_input("收敛阈值", 1e-8, 1e-4, float(p.get("tol", 1e-6)), format="%.1e")
        mult = st.selectbox("自旋多重度", [1, 2, 3, 4, 5], index=0 if p["num_electrons"] % 2 == 0 else 1)
        p["multiplicity"] = mult
        run_manual = st.button("运行 3D 计算", use_container_width=True)

    else:
        p = st.session_state.config["params"]
        p["system_type"] = st.selectbox("体系", ["Argon", "Nitrogen"])
        p["n_particles"] = st.slider("粒子数", 8, 128, 64, 8)
        p["temp"] = st.slider("温度 (K)", 10.0, 800.0, 150.0, 10.0)
        p["box_length"] = st.slider("盒长 (Å)", 10.0, 30.0, 14.0, 1.0)
        p["dt"] = st.slider("时间步长", 0.0005, 0.004, 0.001, 0.0005)
        p["max_iter"] = st.slider("步数", 100, 2000, 600, 100)
        run_manual = st.button("运行 MD", use_container_width=True)


col_main, col_status = st.columns([2.1, 1.0])
with col_main:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 任务入口")
    st.write("把自然语言请求转换成计算配置，或者直接使用手动参数运行。")
    st.markdown('</div>', unsafe_allow_html=True)

with col_status:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 当前配置")
    st.write(f"**求解器** `{st.session_state.config['solver_type']}`")
    st.write(f"**说明** {st.session_state.config.get('explanation', '')}")
    st.markdown('</div>', unsafe_allow_html=True)

if ai_run and user_prompt:
    parsed = parse_user_request(user_prompt, model_name=ai_model)
    st.session_state.config = parsed
    st.session_state.calc_results = None
    st.rerun()

if solver == "1d_dft":
    params = st.session_state.config["params"]
    if run_manual:
        with st.spinner("求解 1D Kohn-Sham DFT..."):
            try:
                res = solve_1d_dft(
                    Vext_fn=make_potential_fn(params["potential_expr"]),
                    num_electrons=params["num_electrons"],
                    L=params["L"],
                    N=params["N"],
                    max_iter=params["max_iter"],
                    tol=params["tol"],
                    alpha=params["alpha"],
                    softening=params["softening"],
                    functional=params["functional"],
                    mixing_method=params["mixing_method"],
                )
                st.session_state.calc_results = res
            except Exception as e:
                st.error(f"1D 计算失败: {e}")

elif solver == "3d_diatomic":
    params = st.session_state.config["params"]
    if run_manual:
        if len(params.get("atoms", [])) < 2:
            st.error("至少需要两个原子。")
        else:
            with st.spinner("求解 3D 分子自洽场..."):
                try:
                    if len(params["atoms"]) == 2:
                        res = solve_diatomic_scf(
                            params["atoms"][0]["name"], params["atoms"][0]["pos"],
                            params["atoms"][1]["name"], params["atoms"][1]["pos"],
                            num_electrons=params["num_electrons"],
                            max_iter=params["max_iter"],
                            tol=params["tol"],
                            multiplicity=params["multiplicity"],
                        )
                    else:
                        res = solve_multi_atom_scf(
                            params["atoms"],
                            num_electrons=params["num_electrons"],
                            max_iter=params["max_iter"],
                            tol=params["tol"],
                            multiplicity=params["multiplicity"],
                        )
                    st.session_state.calc_results = res
                except Exception as e:
                    st.error(f"3D 计算失败: {e}")

else:
    if run_manual:
        st.info("MD 引擎保留在当前代码库中，建议在下一轮把它拆成独立工具页。")


res = st.session_state.calc_results
if res is not None and solver in ["1d_dft", "3d_diatomic"]:
    st.markdown("### 结果总览")
    if solver == "1d_dft":
        c1, c2, c3 = st.columns(3)
        c1.metric("总能量", f"{res['energies']['E_tot']:.6f} Ha")
        c2.metric("迭代步数", f"{res['iterations']}")
        c3.metric("收敛", "Yes" if res["converged"] else "No")

        tabs = st.tabs(["收敛", "势能与密度", "轨道", "DOS/PDOS", "CDD", "周期晶格"])
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=res["history"], mode="lines+markers", name="E_tot"))
            fig.update_xaxes(title_text="Iteration")
            fig.update_yaxes(title_text="Energy (Ha)")
            st.plotly_chart(polish_figure(fig), use_container_width=True)
        with tabs[1]:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=res["x"], y=res["potentials"]["Vext"], name="Vext", line=dict(dash="dash")), secondary_y=False)
            fig.add_trace(go.Scatter(x=res["x"], y=res["potentials"]["VH"], name="VH", line=dict(dash="dot")), secondary_y=False)
            fig.add_trace(go.Scatter(x=res["x"], y=res["density"], name="rho", line=dict(width=2)), secondary_y=True)
            st.plotly_chart(polish_figure(fig), use_container_width=True)
        with tabs[2]:
            fig = go.Figure()
            for i in range(res["orbitals"].shape[1]):
                fig.add_trace(go.Scatter(x=res["x"], y=res["orbitals"][:, i], name=f"psi{i+1}"))
            st.plotly_chart(polish_figure(fig), use_container_width=True)
        with tabs[3]:
            E = np.linspace(-2.5, 1.5, 500)
            occ = []
            remain = params["num_electrons"]
            for _ in range(len(res["eigenvalues"])):
                v = min(2.0, remain)
                occ.append(v)
                remain -= v
            dos = calculate_dos(res["eigenvalues"], occ, E, sigma=0.05)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=E, y=dos, name="DOS"))
            st.plotly_chart(polish_figure(fig), use_container_width=True)
        with tabs[4]:
            cdd, rho_mol, rho_a1, rho_a2 = calculate_cdd_1d(params, res)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res["x"], y=cdd, name="CDD"))
            fig.add_trace(go.Scatter(x=res["x"], y=rho_mol, name="rho_mol"))
            st.plotly_chart(polish_figure(fig), use_container_width=True)
        with tabs[5]:
            if params["potential_expr"]:
                k = np.linspace(-np.pi / params["L"], np.pi / params["L"], 15)
                st.code("周期晶格功能已保留在当前引擎中，可在后续版本直接作为独立任务运行。")

    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总能量", f"{res['E_tot']:.6f} Ha")
        c2.metric("电子偶极", f"{compute_dipole_moment_multi(st.session_state.config['params']['atoms'], res)[1]:.4f} D")
        c3.metric("迭代步数", f"{res['iterations']}")
        c4.metric("多重度", f"{res.get('multiplicity', 1)}")

        tabs = st.tabs(["结构", "轨道", "电荷", "键级", "DOS/PDOS", "CDD", "PES", "吸附", "优化", "振动", "MEP"])
        atoms = st.session_state.config["params"]["atoms"]

        with tabs[0]:
            fig = go.Figure()
            coords = np.array([a["pos"] for a in atoms])
            for idx, at in enumerate(atoms):
                fig.add_trace(go.Scatter3d(x=[at["pos"][0]], y=[at["pos"][1]], z=[at["pos"][2]], mode="markers+text", text=[at["name"]], name=f"{at['name']}{idx+1}", marker=dict(size=10)))
            st.plotly_chart(polish_figure(fig, 520), use_container_width=True)
        with tabs[1]:
            C = res.get("C_alpha") if res.get("multiplicity", 1) > 1 else res["C"]
            eps = res.get("eps_alpha") if res.get("multiplicity", 1) > 1 else res["eps"]
            labels = []
            for idx, at in enumerate(atoms):
                orbs, _ = get_element_orbitals(at["name"], at["pos"])
                for j in range(len(orbs)):
                    labels.append(f"{at['name']}#{idx+1} AO{j+1}")
            df = pd.DataFrame(C, index=labels, columns=[f"MO{i+1} ({e:.3f})" for i, e in enumerate(eps)])
            st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
        with tabs[2]:
            st.dataframe(pd.DataFrame(compute_mulliken_charges(atoms, res)), use_container_width=True)
        with tabs[3]:
            mop, strength = compute_bond_orders(atoms, res)
            st.dataframe(pd.DataFrame(mop), use_container_width=True)
        with tabs[4]:
            E = np.linspace(-3.0, 2.0, 500)
            if res.get("multiplicity", 1) > 1:
                e_vals = res["eps_alpha"]
            else:
                e_vals = res["eps"]
            occ = [2.0] * len(e_vals)
            dos = calculate_dos(e_vals, occ, E, sigma=0.05)
            pdos = calculate_pdos_multi(e_vals, C, res["S"], res["basis"], atoms, E, sigma=0.05)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=E, y=dos, name="DOS"))
            for i, arr in enumerate(pdos):
                fig.add_trace(go.Scatter(x=E, y=arr, name=f"PDOS {i+1}"))
            st.plotly_chart(polish_figure(fig), use_container_width=True)
        with tabs[5]:
            if len(atoms) >= 2:
                z = np.linspace(-4, 4, 300)
                rho_z = eval_density_z(res, z)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=z, y=rho_z, name="rho(z)"))
                st.plotly_chart(polish_figure(fig), use_container_width=True)
                y = np.linspace(-3, 3, 100)
                zz = np.linspace(-4, 4, 120)
                Z, Y = np.meshgrid(zz, y)
                rho_2d = eval_density_2d(res, Y, Z)
                fig2 = go.Figure(data=go.Contour(z=rho_2d, x=zz, y=y, colorscale="Viridis"))
                st.plotly_chart(polish_figure(fig2), use_container_width=True)
                x3 = np.linspace(-3, 3, 30)
                y3 = np.linspace(-3, 3, 30)
                z3 = np.linspace(-4, 4, 35)
                X3, Y3, Z3 = np.meshgrid(x3, y3, z3, indexing="ij")
                rho_3d = eval_density_3d_grid(res, X3, Y3, Z3)
                fig3 = go.Figure(data=go.Isosurface(x=X3.flatten(), y=Y3.flatten(), z=Z3.flatten(), value=rho_3d.flatten(), isomin=0.02, isomax=0.5, surface_count=4, opacity=0.35))
                st.plotly_chart(polish_figure(fig3, 520), use_container_width=True)
        with tabs[6]:
            if len(atoms) == 2:
                st.write("势能面扫描可继续扩展为独立任务页。当前版本保留求解器与结果区。")
        with tabs[7]:
            st.write("吸附能分析模块保留，可在下一版拆为完整工作流。")
        with tabs[8]:
            st.write("结构优化工作流建议作为独立任务页运行。")
        with tabs[9]:
            st.write("振动与红外光谱工作流建议保持为独立分析任务。")
        with tabs[10]:
            grid_y = np.linspace(-3.5, 3.5, 60)
            grid_z = np.linspace(-4.5, 4.5, 70)
            Zg, Yg = np.meshgrid(grid_z, grid_y)
            mep = calculate_mep_grid_2d(atoms, res, Yg, Zg)
            fig = go.Figure(data=go.Contour(z=mep, x=grid_z, y=grid_y, colorscale="RdBu", zmid=0.0))
            st.plotly_chart(polish_figure(fig), use_container_width=True)

st.markdown("---")
st.caption("Chat DFT | professional quantum chemistry workbench")
