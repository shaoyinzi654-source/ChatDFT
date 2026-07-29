<div align="center">

# ⚛️ Chat DFT

### 用自然语言探索 DFT、Hartree-Fock 与分子电子结构

一个基于 Streamlit 的交互式量子化学计算与可视化工具。输入想研究的体系，配置计算参数，运行自洽场计算，并在浏览器中查看轨道、电子密度、能量和势能面结果。所有关键图表都支持缩放、旋转、悬停读数和导出。

<p>
  <a href="https://github.com/shaoyinzi654-source/ChatDFT"><strong>GitHub</strong></a>
  ·
  <a href="#快速开始"><strong>快速开始</strong></a>
  ·
  <a href="#功能一览"><strong>功能一览</strong></a>
</p>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-computation-013243?logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-interactive%20charts-3F4F75?logo=plotly&logoColor=white)

<br />

<img src="h2_molecular_orbitals_pes.png" alt="H2 molecular orbitals and potential energy surface" width="88%" />

</div>

## 功能一览

| 模块 | 能做什么 |
| --- | --- |
| 🤖 自然语言入口 | 将类似“计算一个 Z=2 的一维类氦原子”这样的描述转换为计算配置 |
| 🧪 1D Kohn-Sham DFT | 网格有限差分、Hartree 势、LDA 交换-相关泛函与密度混合 |
| ⚛️ 3D STO-3G Hartree-Fock | 支持 H、He 及双原子分子，解析计算重叠、动能、核吸引和 ERI |
| 📈 PES 扫描 | 扫描 H₂、HeH⁺ 的键长并定位最低能量平衡构型 |
| 🧊 3D 电子云 | 交互式 3D 电子密度等值面、原子核、化学键和偶极矩预览 |
| 🔴🔵 差分电荷 | 2D 等高线与 3D 正负等值面，观察成键区域的电子富集和耗尽 |
| 📊 结果可视化 | 查看轨道、电子密度、有效势、能量分解、SCF 收敛和 DOS/PDOS |
| 🧬 多原子分析 | Mulliken 电荷、键级、MO/AO 系数、几何结构和键长信息 |
| 🔭 光谱与响应 | DOS/PDOS、红外振动频率、振动强度、静电势和能带式能级图 |
| 🧭 几何与动力学 | 多原子几何优化、3D 分子动力学轨迹、温度、RDF 和 MSD 分析 |

## 计算结果预览

<div align="center">
  <img src="dft_results.png" alt="DFT calculation results" width="48%" />
  <img src="dft_orbitals.png" alt="DFT orbitals" width="48%" />
</div>

<div align="center">
  <img src="h2_density_3d.png" alt="H2 three-dimensional electron density" width="88%" />
</div>

<div align="center">
  <img src="scf_convergence.png" alt="SCF convergence" width="48%" />
  <img src="h2_density_of_states.png" alt="H2 density of states" width="48%" />
</div>

## 快速开始

### 1. 安装依赖

建议使用 Python 3.8 或更高版本：

```bash
pip install -r requirements.txt
```

### 2. 启动应用

```bash
python -m streamlit run app.py
```

启动后打开 [http://localhost:8501](http://localhost:8501)。

如果本地代理影响网络请求，可以先清除代理：

```powershell
$env:NO_PROXY="*"
python -m streamlit run app.py
```

## 使用示例

在应用的自然语言输入区域尝试：

```text
Calculate a 1D helium-like atom with Z=2
```

或：

```text
Calculate the potential energy surface scan of H2
```

也可以直接从预设模板选择体系，再调整电子数、网格范围、收敛阈值、键长和 SCF 迭代次数。

## 一次计算能看到什么

1. **体系设置**：自然语言解析、原子坐标编辑器、电子数、自旋多重度和收敛参数。
2. **结构预览**：实时 3D 分子构型，显示原子、键线和计算得到的偶极矩方向。
3. **SCF 诊断**：总能量收敛曲线、每次迭代能量变化和最终收敛状态。
4. **电子结构**：HOMO/LUMO、轨道能级、MO/AO 系数矩阵、轨道占据和能隙。
5. **空间分布**：一维剖面、二维切片、3D 电子云等值面和 3D 差分电荷等值面。
6. **化学解释**：Mulliken 原子电荷、键级/键强度、DOS/PDOS、静电势与成键分析。
7. **分子性质**：势能面、几何优化、振动频率、红外强度和分子动力学统计量。

## 项目结构

```text
ChatDFT/
├── app.py                  # Streamlit 交互界面与可视化
├── dft_engine.py           # 一维 Kohn-Sham DFT 求解器
├── diatomic_engine.py      # 3D STO-3G Hartree-Fock 求解器
├── analysis_tools.py       # DOS、PDOS、电荷密度和静电势分析
├── ai_helper.py            # 自然语言到计算配置的解析
├── md_engine.py            # 分子动力学相关计算
├── requirements.txt        # Python 依赖
└── *.png                   # 示例计算结果与可视化素材
```

## 数值方法

### 一维 DFT

使用三点有限差分离散动能算符，结合 soft-Coulomb 相互作用计算 Hartree 势，并采用 Slater exchange 与 Wigner correlation 的 LDA 近似。密度通过线性混合迭代至收敛。

### 双原子 Hartree-Fock

采用 STO-3G 最小基组和 Gaussian Product Theorem，解析计算一电子积分与四中心双电子排斥积分，通过 Roothaan-Hall 方程和对称正交化求解分子轨道。

## 运行测试

```bash
python -m pytest
```

也可以运行项目自带的验证脚本：

```bash
python validate_all.py
```

## 许可证

当前仓库尚未指定开源许可证。若要公开分发，建议根据你的使用场景补充 LICENSE 文件。

