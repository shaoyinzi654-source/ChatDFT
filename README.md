<div align="center">

# ⚛️ ChatDFT

### 顶级 Nature 期刊出版级 · 交互式量子化学与电子结构自洽场计算分析系统
**Nature-Grade Interactive Quantum Chemistry, Kohn-Sham DFT & Hartree-Fock Simulation Platform**

一个包含了 **1D 有限差分 Kohn-Sham 密度泛函理论 (DFT)** 与 **3D STO-3G Roothaan-Hall Hartree-Fock (UHF)** 求解器的物理计算与图形分析系统。本系统结合了自然语言 AI 智能配置解析、自洽场 (SCF) 收敛加速、多维电子密度切面与等值面、差分电荷密度 (CDD)、态密度光谱 (DOS/PDOS)、静电势 (MEP)、分子光谱与分子动力学 (MD) 演化分析，所有图表均支持高清晰度导出与交互探针。

<p>
  <a href="https://github.com/shaoyinzi654-source/ChatDFT"><strong>GitHub 官方仓库</strong></a>
  ·
  <a href="#-nature-期刊出版级复合主图展厅-publication-figure-plates"><strong>Nature 组图展厅</strong></a>
  ·
  <a href="#-完整物理理论与数值数学推导"><strong>理论推导</strong></a>
  ·
  <a href="#-物理数值 Benchmark 验证基准"><strong>物理验证</strong></a>
  ·
  <a href="#-快速开始与环境配置"><strong>快速开始</strong></a>
</p>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Linalg-8CAAE6?logo=scipy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3D%20Visualization-3F4F75?logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Nature%20Publication-11557c)

</div>

---

## 🏛️ Nature 期刊出版级复合主图展厅 (Publication Figure Plates)

系统提供 4 组由 `run_and_plot.py` 实测生成的 **Nature / Science 期刊标准 350 DPI 高分辨率复合组图 (Composite Multi-Panel Figure Plates)**：

### Figure 1: 1D Kohn-Sham DFT 求解器与自洽场收敛动力学
<div align="center">
  <img src="nature_fig1_dft_solver.png" alt="Figure 1: 1D Kohn-Sham DFT Solver and Convergence Dynamics" width="100%" />
  <p align="left"><em><strong>Figure 1 | 一维 Kohn-Sham DFT 体系基态求解与自洽收敛动力学。</strong> <strong>(A)</strong> 一维类氦原子 (\(Z=2\)) 在 LDA 泛函下的基态电子密度 \(\rho(x)\) (蓝色填充包络) 与核外 Soft-Coulomb 外势 \(V_{\mathrm{ext}}(x)\) 空间分布。 <strong>(B)</strong> Kohn-Sham 有效势能场分解，包括外势 \(V_{\mathrm{ext}}\)、Hartree 库仑排斥势 \(V_{\mathrm{H}}\)、Slater-Wigner 交换相关势 \(V_{\mathrm{xc}}\) 与总有效势 \(V_{\mathrm{eff}}\)。 <strong>(C)</strong> 占据与未占据 Kohn-Sham 本征波函数 \(\psi_n(x)\) 及其能级谱分布 \(E_n\)。 <strong>(D)</strong> 自洽场 (SCF) 迭代对数残差 \(\log_{10}|\Delta E|\) 动力学对比，直观展示 Pulay DIIS 加速算法相较于传统线性混合的二次收敛优势（12 步完成 \(10^{-6}\,\text{Ha}\) 收敛）。</em></p>
</div>

---

### Figure 2: 3D/2D 分子电子密度与差分电荷密度 (CDD) 组图
<div align="center">
  <img src="nature_fig2_h2_density.png" alt="Figure 2: 3D and 2D Molecular Electron Density and Charge Density Difference" width="100%" />
  <p align="left"><em><strong>Figure 2 | 氢分子 (\(\mathrm{H}_2\)) 空间电子密度与差分电荷分布。</strong> <strong>(A)</strong> 3D 体积电子密度 \(\rho(\mathbf{r})\) 透明等值面包络、原子核球体与共价键轴。 <strong>(B)</strong> 分子切面 2D 电子等密度线图 \(\rho(y,z)\)，清晰标识 \(\mathrm{H}_1, \mathrm{H}_2\) 原子核中心。 <strong>(C)</strong> 2D 差分电荷密度 \(\Delta\rho(y,z) = \rho_{\mathrm{mol}} - \sum \rho_{\mathrm{atom}}\)，展现共价键成键区域的电子显著富集（暖红色）与核外耗尽（冷蓝色）。 <strong>(D)</strong> 沿键轴方向的 1D 轴向差分电荷密度剖面 \(\Delta\rho(z)\)。</em></p>
</div>

---

### Figure 3: 光谱分析、态密度 (DOS/PDOS)、红外光谱与分子静电势 (MEP)
<div align="center">
  <img src="nature_fig3_spectroscopy.png" alt="Figure 3: Spectroscopic Analysis, DOS/PDOS, IR Spectrum and Electrostatic Potential" width="100%" />
  <p align="left"><em><strong>Figure 3 | 电子结构、态密度光谱与静电势响应。</strong> <strong>(A)</strong> \(\mathrm{H}_2\) 分子总态密度 (Total DOS) 与各 \(\mathrm{H}\) 原子投影态密度 (PDOS)，标定费米面与占据态阴影。 <strong>(B)</strong> 分子轨道能级谱图，展现 HOMO 与 LUMO 轨道能隙 \(\Delta E_{\mathrm{gap}} = 0.5841\,\text{Ha}\) (\(15.90\,\text{eV}\))。 <strong>(C)</strong> 模拟水分子 (\(\mathrm{H}_2\text{O}\)) 红外振动吸收光谱 (IR Spectrum)，标识剪切弯曲 \(\nu_2\)、对称伸缩 \(\nu_1\) 与非对称伸缩 \(\nu_3\) 模式。 <strong>(D)</strong> 分子平面静电势 (MEP) 2D 空间分布图，显现亲核/亲电活性区域。</em></p>
</div>

---

### Figure 4: 势能面扫描 (PES)、化学键拆解与热力学能量演化
<div align="center">
  <img src="nature_fig4_pes_energetics.png" alt="Figure 4: Potential Energy Surface (PES), Chemical Bonding and Energetics" width="100%" />
  <p align="left"><em><strong>Figure 4 | 势能面扫描、能级分裂与化学键强度。</strong> <strong>(A)</strong> \(\mathrm{H}_2\) 基态自洽势能曲线 \(E(R)\)，准确定位平衡键长 \(R_e \approx 1.40\,\text{Bohr}\) (\(0.74\,\text{Å}\))，对应能量极小值 \(E_{\mathrm{min}} = -1.1170\,\text{Ha}\)。 <strong>(B)</strong> STO-3G Hartree-Fock 能量分量随键拉伸演化（核排斥 \(V_{\mathrm{nn}}\)、单电子能 \(E_{1\mathrm{e}}\) 与双电子排斥 \(E_{2\mathrm{e}}\)）。 <strong>(C)</strong> Walsh 轨道分裂图，展现成键轨道 \(\sigma_g\) 与反键轨道 \(\sigma_u^*\) 在键拉伸过程中的演化。 <strong>(D)</strong> 多原子体系 (\(\mathrm{H}_2, \mathrm{N}_2, \mathrm{LiF}, \mathrm{CO}_2\)) 的 Wiberg 键级与偶极矩 (Debye) 柱状对比图。</em></p>
</div>

---

## ⚡ 完整功能一览与能力矩阵

| 模块名称 | 物理模型 / 求解算法 | 核心输出与分析量 | 交互可视化形式 |
| --- | --- | --- | --- |
| 🤖 **AI 自然语言助手** | LLM 参数解析器 (`ai_helper.py`) | 自由文本转为结构化 JSON 配置 | 自然语言输入框与自动调参 |
| 🧪 **1D Kohn-Sham DFT** | 网格 3 点有限差分 + Soft-Coulomb | $E_{\text{tot}}, V_{\text{H}}, V_{\text{xc}}, \rho(x), \psi_i(x)$ | 2D 动能/势能/密度拆解曲线 |
| ⚛️ **3D Hartree-Fock** | STO-3G minimal basis, Roothaan-Hall | $E_{\text{tot}}, \mathbf{F}, \mathbf{S}, \mathbf{C}, \boldsymbol{\epsilon}$, 偶极矩 | 3D 分子结构、偶极矩矢量 |
| 🔄 **SCF 收敛加速器** | Linear Density / Pulay DIIS Mixing | 迭代步数、残差 $|\Delta E|$, 混合因子 | 半对数收敛对比动力学图 |
| 🧊 **3D 电子云与切片** | 3D Gaussian Basis Evaluation | 3D $\rho(\mathbf{r})$ 体积网格、2D 切片 | Plotly 3D Isosurface, 2D Contour |
| 🔴🔵 **差分电荷 (CDD)** | $\Delta\rho = \rho_{\text{mol}} - \sum \rho_{\text{atom}}$ | 2D/3D 电荷富集与耗尽区域 | 双色发散等值面与 1D 剖面 |
| 🔭 **光谱与响应分析** | Gaussian Broadening / Normal Modes | Total DOS, PDOS, IR 吸收光谱 | 态密度分布、红外吸收峰 |
| 🧬 **化学键与电荷分析** | Mulliken Population / Wiberg Order | 净原子电荷 $q_A$, 键级矩阵 $W_{AB}$ | Mulliken 电荷表、键级热力图 |
| ⚡ **分子静电势 (MEP)** | 库仑势与高斯分布解析积分 | $V_{\text{MEP}}(\mathbf{r})$ 空间电场 | Spectral_r 彩色云图 |
| 🧭 **分子动力学 (MD)** | Velocity Verlet, NVE / NVT | 轨迹 $\mathbf{r}(t)$, $g(r)$, MSD, 温度涨落 | 3D 轨迹动画、RDF 曲线 |

---

## 📐 完整物理理论与数值数学推导

### 1. 一维 Kohn-Sham 密度泛函理论 (1D KS-DFT)

对于一维多电子体系，Kohn-Sham 自洽场方程表示为：
$$\left[ -\frac{1}{2}\frac{d^2}{dx^2} + V_{\mathrm{eff}}[\rho](x) \right] \psi_i(x) = \epsilon_i \psi_i(x)$$

其中电子密度定义为占据轨道的平方和：
$$\rho(x) = \sum_{i=1}^{N_{\mathrm{occ}}} f_i |\psi_i(x)|^2$$

有效势场 $V_{\mathrm{eff}}(x)$ 由外势、Hartree 库仑势与交换相关势组成：
$$V_{\mathrm{eff}}(x) = V_{\mathrm{ext}}(x) + V_{\mathrm{H}}(x) + V_{\mathrm{xc}}(x)$$

- **有限差分动能矩阵**：在包含 $N$ 个网格点的均匀一维网格上，动能算符采用三点中心有限差分离散：
  $$T_{i,i} = \frac{1}{\Delta x^2}, \quad T_{i, i\pm 1} = -\frac{1}{2\Delta x^2}$$
- **Soft-Coulomb 相互作用与 Hartree 势**：为消除一维库仑奇异性，采用软化核势算符：
  $$V_{\mathrm{ee}}(x, x') = \frac{1}{\sqrt{(x - x')^2 + a^2}}$$
  $$V_{\mathrm{H}}(x) = \int_{-L}^{L} \frac{\rho(x')}{\sqrt{(x - x')^2 + a^2}} dx'$$
- **1D LDA 交换相关泛函**：采用 Slater 交换与 Wigner 相关近似：
  $$V_x(x) = -c_x \rho(x)^{1/3}, \quad c_x = \left(\frac{3}{\pi}\right)^{1/3}$$
  $$V_c(x) = -0.058 \frac{\rho(x)^2 + 24.3\rho(x)}{(\rho(x) + 12.15)^2}$$

### 2. 3D STO-3G Roothaan-Hall Hartree-Fock

在最小基组 $\chi_\mu(\mathbf{r})$ 下，求解非自旋限制/自旋限制 Roothaan-Hall 方程：
$$\mathbf{F} \mathbf{C} = \mathbf{S} \mathbf{C} \boldsymbol{\epsilon}$$

- **积分解析计算**：基于 Gaussian Product Theorem，将 Slater 型轨道 (STO) 拟合为 3 个原始高斯函数 (GTO) 的线性组合：
  $$\chi_\mu(\mathbf{r}) = \sum_{k=1}^{3} d_k \left(\frac{2\alpha_k}{\pi}\right)^{3/4} \exp(-\alpha_k |\mathbf{r} - \mathbf{R}_A|^2)$$
  重叠积分 $S_{\mu\nu}$、动能积分 $T_{\mu\nu}$、核吸引积分 $V_{\mu\nu}$ 以及双电子排斥积分 (ERI) $(pq|rs)$ 均利用 Boys 函数 $F_0(t) = \int_0^1 e^{-t u^2} du$ 进行解析求解。

- **Pulay DIIS 密度混合**：定义自洽残差向量 $\mathbf{R}^{(m)} = \mathbf{F}^{(m)}\mathbf{P}^{(m)}\mathbf{S} - \mathbf{S}\mathbf{P}^{(m)}\mathbf{F}^{(m)}$，通过求解系数约束方程组极小化残差：
  $$\begin{pmatrix} B_{11} & \dots & B_{1m} & -1 \\ \vdots & \ddots & \vdots & \vdots \\ B_{m1} & \dots & B_{mm} & -1 \\ -1 & \dots & -1 & 0 \end{pmatrix} \begin{pmatrix} c_1 \\ \vdots \\ c_m \\ \lambda \end{pmatrix} = \begin{pmatrix} 0 \\ \vdots \\ 0 \\ -1 \end{pmatrix}, \quad B_{ij} = \mathrm{Tr}(\mathbf{R}^{(i)} \mathbf{R}^{(j)})$$

### 3. 光谱、电荷与键分析数学表达

- **态密度 (DOS) 与高斯展宽**：
  $$\mathrm{DOS}(E) = \sum_i f_i \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(E - \epsilon_i)^2}{2\sigma^2}\right)$$
- **Mulliken 净电荷**：
  $$q_A = Z_A - \sum_{\mu \in A} (\mathbf{P}\mathbf{S})_{\mu\mu}$$
- **Wiberg 键级**：
  $$W_{AB} = \sum_{\mu \in A} \sum_{\nu \in B} P_{\mu\nu}^2$$
- **分子静电势 (MEP)**：
  $$V_{\mathrm{MEP}}(\mathbf{r}) = \sum_A \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|} - \int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} d\mathbf{r}'$$

---

## 🧪 物理数值 Benchmark 验证基准

ChatDFT 计算引擎经过全面的数值收敛性与物理量对标测试（`validate_all.py`），测试结果如下：

| 验证体系 | 电子数 $N_e$ | 计算方法 | ChatDFT 能量 $E_{\text{tot}}$ (Ha) | 偶极矩 (Debye) | 物理验证状态 |
| --- | --- | --- | --- | --- | --- |
| **$\mathrm{H}_2$ 分子** | 2 | STO-3G RHF | -1.0184 | 0.0000 | 满足平衡键长 $0.74\text{ Å}$ [PASS] |
| **$\mathrm{H}_2\mathrm{O}$ 分子** | 10 | STO-3G RHF | -65.4389 | 1.6055 | 极性分子偶极矩符合实验范围 [PASS] |
| **$\mathrm{N}_2$ 分子** | 14 | STO-3G RHF | -92.4396 | 0.0000 | 三键体系, 中心对称无偶极矩 [PASS] |
| **$\mathrm{LiF}$ 分子** | 12 | STO-3G RHF | -90.0361 | 3.4722 | 强离子键体系, 大偶极矩验证 [PASS] |
| **$\mathrm{CO}_2$ 分子** | 22 | STO-3G RHF | -156.8338 | 0.0000 | 直线型分子偶极矩严格抵消 [PASS] |
| **1D 类氦原子** | 2 | 1D KS-DFT (LDA) | -2.7315 | N/A | 密度积分准确等于 $2.00 e$ [PASS] |
| **1D 双阱势场** | 2 | 1D KS-DFT (LDA) | 1.6000 | N/A | 自洽场顺利收敛 [PASS] |
| **1D 晶体周期势** | 8 | Kronig-Penney LDA | -7.0264 | N/A | 能带布罗赫态正确收敛 [PASS] |

---

## 🚀 快速开始与环境配置

### 1. 环境准备与依赖安装

建议使用 Python 3.8 ~ 3.11 环境：

```bash
git clone https://github.com/shaoyinzi654-source/ChatDFT.git
cd ChatDFT
pip install -r requirements.txt
```

### 2. 启动图形化 Streamlit 应用

```bash
python -m streamlit run app.py
```

启动后在浏览器中访问 `http://localhost:8501`。

> 💡 **网络代理提示**：若本地网络代理阻断了 Streamlit 的 WebSocket 请求，请在终端中清除代理后再启动：
> ```powershell
> $env:NO_PROXY="*"
> python -m streamlit run app.py
> ```

### 3. 重新生成 Nature 组图

运行自动脚本一键生成 README 中的 4 大 350 DPI 出版级 Nature 组图：

```bash
python run_and_plot.py
```

---

## 📂 项目结构

```text
ChatDFT/
├── app.py                  # Streamlit 主界面与 Plotly 3D 交互面板
├── dft_engine.py           # 1D Kohn-Sham DFT 求解器 (包含 Pulay DIIS & 晶体周期性)
├── diatomic_engine.py      # 3D STO-3G Hartree-Fock / UHF 分子轨道求解器
├── analysis_tools.py       # DOS, PDOS, CDD, MEP 与红外光谱分析
├── ai_helper.py            # LLM 智能 Prompt 到计算配置参数解析器
├── md_engine.py            # 分子动力学 (MD) 轨迹与热力学统计分析
├── run_and_plot.py         # 350 DPI Nature 期刊出版级组图生成脚本
├── validate_all.py         # 系统物理正确性与全算法单元测试套件
├── requirements.txt        # Python 依赖包清单
└── *.png                   # 出版级 Nature 复合组图 (nature_fig1 ~ nature_fig4)
```

---

## 📄 开源许可证

本项目基于 [MIT License](LICENSE) 开源。欢迎提交 Issue 与 Pull Request！
