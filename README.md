<h1 align="center">ChatDFT</h1>

<p align="center">Windows 分子建模与量子化学工作台</p>

<p align="center">
  <strong><a href="https://github.com/shaoyinzi654-source/ChatDFT/releases/download/v5.8.0-standalone/ChatDFT-Standalone-5.8.0-Windows.zip">下载 Windows 5.8.0</a></strong>
  &nbsp;·&nbsp; <a href="#安装与上手">安装</a>
  &nbsp;·&nbsp; <a href="docs/gallery.md">图集</a>
  &nbsp;·&nbsp; <a href="README.en.md">English</a>
</p>

[![同一个 PDC–镁水合簇在 ChatDFT 的两个视窗中显示为球棍图和空间填充图。](docs/images/pdc-dual-view.png)](docs/gallery.md#视角演示)

**PDC–镁水合簇** · 101 个原子，19 个分子与离子片段。左为球棍图，右为空间填充图。 [播放视角演示](docs/gallery.md#视角演示)

## 从结构到计算结果

| 工作步骤 | ChatDFT 支持 |
| --- | --- |
| 构建结构 | 从 SMILES 建模，导入 XYZ 或非周期 XSD。组装多分子、离子和显式水。编辑原子坐标，测量距离和角度。 |
| 提交计算 | 通过 SSH 提交 PySCF 单点能、几何优化、谐振频率及垂直激发态作业。支持气相和 IEF-PCM 溶剂设置。 |
| 管理作业 | 查看队列和日志。取消或重试任务，从项目库恢复输入和结果，打开项目文件夹。 |
| 检查与导出 | 导出 XYZ、JSON、CSV、PNG、SVG 与 Markdown 报告。根据计算类型检查 SCF 收敛、振动频率或几何变化。 |
| 自动化 | 本机 Codex 或兼容 API 可从描述规划建模与计算。缺少输入时可追问，作业完成后可用实算数据补充出图和报告。 |
| 论文与周期体系 | 从论文整理建模条件。另有 [GROMACS 指南](standalone/PDC-MD-GUIDE.md)，用于 PDC 周期水溶液建盒和轨迹分析。 |

## 实算案例

**CO₂–单乙醇胺水合簇** · 40 个原子，气相 PBE0/def2-SVP 单点计算。图中展示 SCF 电子能、Mulliken 原子电荷及 α 通道前线轨道。

[![CO₂–单乙醇胺水合簇的计算结果，包括 SCF 电子能、Mulliken 原子电荷和前线轨道能级。](docs/images/calculation-output.png)](docs/gallery.md#计算与分析)

[查看同一作业的工作台截图和原子电荷分析](docs/gallery.md#计算与分析)

## 安装与上手

1. 安装 **64 位 Python 3.13**，下载并解压 Windows 版 ChatDFT。
2. 在解压目录打开 PowerShell，运行 `.Install-ChatDFT.ps1`，然后打开桌面快捷方式。
3. 选择 **构建 → 从 SMILES 建模**，输入 `O` 查看水分子。要运行计算，在 **计算 → 连接与模型设置** 填写 PySCF 服务器信息。

[用水二聚体试跑一次计算](docs/examples/water-dimer/) · [下载示例 XYZ](docs/examples/water-dimer/water-dimer.xyz?raw=1) · [工作台手册](standalone/README.md)

## 实机图集

<p>
  <a href="docs/gallery.md#导出的三维结构"><img src="docs/images/hydrated-cluster-geometry.png" alt="水合簇三维结构与坐标轴" width="38%"></a>
  &nbsp;
  <a href="docs/gallery.md#按分子片段汇总电荷"><img src="docs/images/fragment-charge.png" alt="按分子片段汇总的净电荷" width="55%"></a>
</p>

[查看完整实机图集](docs/gallery.md) · [项目库和报告](docs/gallery.md#项目与报告)

---

[源码](standalone/) · [版本发布](https://github.com/shaoyinzi654-source/ChatDFT/releases) · [问题反馈](https://github.com/shaoyinzi654-source/ChatDFT/issues) · [官网](https://chatdft.com/) · [许可证](LICENSE) · [第三方来源](standalone/OPEN_SOURCE.md) · [早期浏览器版](legacy/browser/)
