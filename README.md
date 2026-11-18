# ChatDFT

ChatDFT 是 Windows 上的分子建模和计算工作台。可以搭建分子与水合簇，将量子化学作业提交给 PySCF，再查看和导出结果。

[**下载 Windows 版 5.8.0**](https://github.com/shaoyinzi654-source/ChatDFT/releases/download/v5.8.0-standalone/ChatDFT-Standalone-5.8.0-Windows.zip) · [安装与上手](#安装与上手) · [实机图集](docs/gallery.md) · [English](README.en.md)

[![ChatDFT 5.8.0 工作台。中央为 CO₂–单乙醇胺水合簇，右侧为计算结果，底部为作业列表。](docs/images/workbench-5.8.png)](docs/images/workbench-5.8.png)

*ChatDFT 5.8.0 中的 CO₂–单乙醇胺水合簇及单点计算结果。*

## 安装与上手

1. 安装 **64 位 Python 3.13**，下载并解压 Windows 版 ChatDFT。
2. 在解压目录打开 PowerShell，运行 `.Install-ChatDFT.ps1`，然后打开桌面快捷方式。
3. 选择 **构建 → 从 SMILES 建模**，输入 `O`，在三维窗口查看水分子。要运行计算，先在 **计算 → 连接与模型设置** 填写 PySCF 服务器信息。

[用水分子试跑一次计算](docs/workflow.md#提交作业) · [完整工作台手册](standalone/README.md)

## 功能

- **建模** — 导入 SMILES、XYZ 或非周期 XSD，组装多分子、离子与显式水。可编辑原子坐标、测量距离和角度，并使用双视图对照结构。
- **量子化学计算** — 通过 SSH 提交 PySCF 作业，支持单点能、几何优化、谐振频率和垂直激发态，以及气相和 IEF-PCM 溶剂模型。
- **作业与结果** — 查看队列、进度与日志，恢复归档作业的模型和结果。按项目保存 XYZ、JSON、CSV、PNG、SVG 与 Markdown。
- **自动化** — 使用本机 Codex 或兼容 API 规划结构和计算。条件不明确时可追问，计算完成后可依据实算数据补充出图和报告。
- **论文与周期体系** — 支持从论文提取建模和计算条件。PDC 周期水溶液可按独立的 [GROMACS 指南](standalone/PDC-MD-GUIDE.md) 建盒并分析轨迹。

## 实机图集

[![CO₂–单乙醇胺水合簇的 SCF 电子能、Mulliken 原子电荷和前线轨道能级。](docs/images/calculation-output.png)](docs/images/calculation-output.png)

*同一作业导出的三项分析。横坐标为 SCF 迭代步的曲线用于检查电子自洽收敛。*

[查看全部 7 张图](docs/gallery.md) · [了解输出文件](docs/workflow.md#查看输出)

## 项目链接

[源码](standalone/) · [版本发布](https://github.com/shaoyinzi654-source/ChatDFT/releases) · [问题反馈](https://github.com/shaoyinzi654-source/ChatDFT/issues) · [项目官网](https://chatdft.com/) · [许可证](LICENSE) · [第三方来源](standalone/OPEN_SOURCE.md) · [早期浏览器版](legacy/browser/)
