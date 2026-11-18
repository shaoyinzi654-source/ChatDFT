# ChatDFT 实机图集 / Gallery

[仓库首页](../README.md) · [English README](../README.en.md) · [全自动计算指南](workflow.md)

## 建模 / Modeling

### PDC–镁离子水合簇

[![同一个 PDC–镁离子水合簇的球棍图和空间填充图。](images/cluster-comparison.png)](images/cluster-comparison.png)

同一构型在两个视窗使用不同的显示方式。球棍图适合查看成键关系，空间填充图用于观察原子占据的空间。

### 导出的三维结构

[![根据 XYZ 坐标导出的 CO₂–单乙醇胺水合簇结构图。](images/hydrated-cluster-geometry.png)](images/hydrated-cluster-geometry.png)

结构图可和 XYZ 坐标一起保留，便于从计算输入重新绘制。

## 计算与分析 / Calculation and analysis

### CO₂–单乙醇胺水合簇单点计算

[![工作台中的 40 原子水合簇、结果表和已完成的 PySCF 作业。](images/workbench-5.8.png)](images/workbench-5.8.png)

图中为 40 原子、总电荷 0、单重态的簇模型。使用 PySCF 在给定构型上进行 PBE0/def2-SVP 单点计算。

### SCF 收敛、原子电荷与轨道

[![从单点计算输出绘制的 SCF 电子能、Mulliken 原子电荷和前线轨道能级。](images/calculation-output.png)](images/calculation-output.png)

SCF 曲线记录电子自洽迭代。Mulliken 电荷和轨道能量对应这一次计算采用的构型、方法和基组。

### 按分子片段汇总电荷

[![按分子片段对 Mulliken 原子电荷求和后的条形图。](images/fragment-charge.png)](images/fragment-charge.png)

用同一作业的 Mulliken 数据按片段求和，查看水、CO₂ 和单乙醇胺各自的净电荷。

## 项目与报告 / Projects and reports

### 项目库

[![项目库列出已完成作业和已取消作业。](images/project-library.png)](images/project-library.png)

从已归档的作业恢复结构和计算结果，也可直接打开对应目录。

### 分析报告

[![ChatDFT 中打开的水合簇补充分析报告。](images/ai-report.png)](images/ai-report.png)

补充报告与计算结果保存在一起。报告记录计算条件、几何检查和解释依据，项目目录保留相应数据。

[查看输出文件的用途](workflow.md#查看输出)
