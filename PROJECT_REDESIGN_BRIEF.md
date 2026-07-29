# Chat DFT 项目改造概要

## 项目地址

GitHub：<https://github.com/shaoyinzi654-source/ChatDFT>

本地路径：`C:\Users\frddx\Desktop\DFY`

## 项目简介

这是一个基于 Python 和 Streamlit 的量子化学计算与可视化工具，包含：

- 1D Kohn-Sham DFT
- 3D STO-3G Hartree-Fock
- H₂、HeH⁺ 等双原子分子计算
- PES 势能面扫描
- SCF 收敛分析
- 电子密度和差分电荷密度
- DOS/PDOS
- Mulliken 电荷和键分析
- 分子结构 3D 预览
- 几何优化
- 振动和红外分析
- 分子动力学模拟

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `app.py` | Streamlit 主界面和交互式图表 |
| `dft_engine.py` | 1D DFT 求解器 |
| `diatomic_engine.py` | 3D Hartree-Fock 求解器 |
| `analysis_tools.py` | 密度、差分电荷、DOS、PDOS 等分析 |
| `run_and_plot.py` | 生成示例计算图片 |
| `README.md` | 项目说明文档 |

## 当前已完成

- GitHub 仓库已经建立。
- README 已加入功能介绍、启动方式和结果图片。
- 已加入 `h2_density_3d.png` 3D 电子密度图片。
- `app.py` 已加入 `polish_3d_figure()`，用于统一 3D 图表样式。
- 当前最新提交：`df271ad`。

## 改造目标

### 1. 全面重新设计 Streamlit 界面

- 使用现代、专业、清晰的界面风格。
- 减少拥挤布局，改善信息层级、间距和可读性。
- 使用统一的颜色系统、卡片、标签页、按钮和标题。
- 保留所有计算功能，不要破坏求解器逻辑。
- 修复页面中可能存在的中文乱码。
- 删除“学术版”等不必要品牌字样。

### 2. 全面美化图表

- 所有 Plotly 图表统一字体、背景、网格线、图例、颜色和边距。
- 2D 图表必须有清晰的标题、轴标签和单位。
- 3D 图表必须使用真实计算数据，并支持：
  - 电子密度等值面
  - 差分电荷密度正负等值面
  - 原子核和化学键
  - 偶极矩箭头
  - 旋转、缩放和悬停查看
  - 合理的 camera 和 aspect ratio
- 不要制作只有装饰作用的假 3D 图。

### 3. 重新组织结果页面

建议使用以下模块：

1. **Overview**：计算摘要、体系信息、总能量和收敛状态。
2. **Structure**：3D 分子结构、原子坐标和键长。
3. **SCF**：SCF 收敛曲线和能量分解。
4. **Orbitals**：分子轨道、HOMO/LUMO 和能级图。
5. **Density**：电子密度 2D 切片和 3D 等值面。
6. **Charge Difference**：差分电荷密度 2D/3D 图。
7. **Spectra**：DOS、PDOS、振动频率和红外光谱。
8. **Analysis**：Mulliken 电荷、键级、静电势和几何优化。
9. **Molecular Dynamics**：轨迹、温度、RDF 和 MSD。

### 4. 改进 README

- 首屏包含项目名称、简介、功能徽章和真实计算图片。
- 增加 2D 和 3D 结果展示。
- 增加安装、启动、示例输入和项目结构。
- 说明每个分析模块的用途。
- 说明当前支持的元素和模型限制。
- 不使用 “academic edition” 或类似字样。

## 验证要求

完成后必须：

```bash
python -m py_compile app.py run_and_plot.py analysis_tools.py dft_engine.py diatomic_engine.py
python -m streamlit run app.py
```

同时检查：

- Streamlit 页面可以正常打开。
- 所有图片和资源路径有效。
- DFT 和 Hartree-Fock 计算功能没有被破坏。
- 不提交 `__pycache__`、日志、密钥或 `.env` 文件。
- 所有 3D 图均来自实际计算数据。
- 修改完成后提交并推送到 GitHub 的 `main` 分支。

## 开发原则

- 先阅读现有代码，再进行修改。
- 尽量只修改 UI、图表和文档层，不重写求解器。
- 保留现有计算接口和 `st.session_state` 结构。
- 不创建虚假的计算结果。
- 完成后说明修改的文件、测试结果和 Git 提交哈希。

