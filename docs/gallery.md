# ChatDFT 实机图集 / Gallery

[仓库首页](../README.md) · [English README](../README.en.md) · [全自动计算指南](workflow.md)

## 186 原子 PDC–镁显式水组装

[![球棍、空间填充与棍状视图展示同一个 PDC–镁显式水组装。](images/pdc-hydration-assembly.png)](examples/pdc-hydration-assembly/)

6 个 PDC²⁻、6 个 Mg²⁺、30 个显式水。案例包含可打开的原生项目、XYZ、建模约束和旋转演示。

[查看案例与文件](examples/pdc-hydration-assembly/)

## 视角演示

同一个 PDC–镁水合簇在双视图中旋转。左侧球棍图保留成键关系，右侧空间填充图展示相同构型的占据空间。

<details>
<summary>播放 5 秒工作台演示 / Play the 5-second demo</summary>
<br>
<img src="images/pdc-dual-view.webp" alt="在 ChatDFT 的球棍窗口和空间填充窗口中旋转同一个 PDC–镁水合簇。">
</details>

[查看静态原图](images/pdc-dual-view.png)

## 建模 / Modeling

### PDC–镁离子水合簇

[![同一个 PDC–镁离子水合簇的球棍图和空间填充图。](images/cluster-comparison.png)](images/cluster-comparison.png)

同一构型在两个视窗使用不同的显示方式。球棍图适合查看成键关系，空间填充图用于观察原子占据的空间。

### 水二聚体测距

[![工作台对水二聚体中的 O···O 距离进行测量，读数为 2.9000 Å。](images/water-dimer-measurement.png)](images/water-dimer-measurement.png)

[打开水二聚体示例与坐标](examples/water-dimer/)

### 导出的三维结构

[![根据 XYZ 坐标导出的 CO₂–单乙醇胺水合簇结构图。](images/hydrated-cluster-geometry.png)](images/hydrated-cluster-geometry.png)

结构图可和 XYZ 坐标一起保留，便于从计算输入重新绘制。

## 计算与分析

### PySCF 输入

[![计算面板显示水二聚体的气相 HF/STO-3G 单点能设置。](images/pyscf-job-setup.png)](images/pyscf-job-setup.png)

在提交窗口核对结构、方法、基组和分子状态后，再提交服务器。

### CO₂–单乙醇胺水合簇单点计算

[![工作台中的 40 原子水合簇、结果表和已完成的 PySCF 作业。](images/workbench-5.8.png)](images/workbench-5.8.png)

图中为 40 原子、总电荷 0、单重态的簇模型。使用 PySCF 在给定构型上进行 PBE0/def2-SVP 单点计算。

[在首页跟读实算案例](../README.md#实算案例)

### SCF 收敛、原子电荷与轨道

[![从单点计算输出绘制的 SCF 电子能、Mulliken 原子电荷和前线轨道能级。](images/calculation-output.png)](images/calculation-output.png)

SCF 曲线记录电子自洽迭代。Mulliken 电荷和轨道能量对应这一次计算采用的构型、方法和基组。

### 按分子片段汇总电荷

[![按分子片段对 Mulliken 原子电荷求和后的条形图。](images/fragment-charge.png)](images/fragment-charge.png)

用同一作业的 Mulliken 数据按片段求和，查看水、CO₂ 和单乙醇胺各自的净电荷。

## 项目与报告

### 项目库

[![项目库列出已完成作业和已取消作业。](images/project-library.png)](images/project-library.png)

从已归档的作业恢复结构和计算结果，也可直接打开对应目录。

### 分析报告

[![ChatDFT 中打开的水合簇补充分析报告。](images/ai-report.png)](images/ai-report.png)

补充报告与计算结果保存在一起。报告记录计算条件、几何检查和解释依据，项目目录保留相应数据。

[查看输出文件的用途](workflow.md#查看输出)
