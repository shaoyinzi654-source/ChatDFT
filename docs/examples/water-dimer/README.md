# 水二聚体入门 / Water dimer quickstart

[打开示例 XYZ / Open example XYZ](water-dimer.xyz?raw=1)

该文件包含两个水分子和 6 个原子，坐标单位为 Å，初始 O···O 距离为 2.90 Å。坐标是为入门演示手工设定的初始构型。The file contains a hand-built six-atom starting geometry in ångströms with an O···O separation of 2.90 Å.

| 参数 / Setting | 值 / Value |
| --- | --- |
| 作业 / Task | 单点能 / Single-point energy |
| 方法与基组 / Level of theory | HF/STO-3G |
| 溶剂 / Solvent | 气相 / Gas phase |
| 总电荷 / Total charge | 0 |
| 自旋多重度 / Spin multiplicity | 1 |

1. 从 **文件 → 导入结构** 载入 XYZ。
2. 打开 **计算设置**，按上表设置作业，确认已配置 PySCF 服务器。
3. 提交后在作业监视器看状态。完成后从项目库打开文件目录，检查 SCF 收敛和最终电子能。

In ChatDFT, import the XYZ from the File menu, apply the settings above, and submit to your configured PySCF server. Use the job monitor and project library to inspect SCF convergence and the final energy.

[查看工作流 / Workflow guide](../../workflow.md) · [返回首页 / Repository home](../../../README.md)
