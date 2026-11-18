<div align="center">
  <img src="standalone/runtime/assets/chatdft-atom.png" width="74" height="74" alt="">
  <h1>ChatDFT</h1>
  <p><strong>用自然语言建模，在 Windows 工作台完成分子计算与结果分析。</strong></p>
  <p>
    <a href="https://github.com/shaoyinzi654-source/ChatDFT/releases/download/v5.8.0-standalone/ChatDFT-Standalone-5.8.0-Windows.zip"><strong>下载 Windows 版 5.8.0</strong></a> ·
    <a href="https://github.com/shaoyinzi654-source/ChatDFT/releases">版本发布</a> ·
    <a href="https://chatdft.com/">项目官网</a> ·
    <a href="README.en.md">English</a>
  </p>
</div>

<a href="docs/images/workbench-5.8.png"><img src="docs/images/workbench-5.8.png" alt="ChatDFT 5.8.0 工作台展示完成的二氧化碳–单乙醇胺水合簇作业，左侧为项目浏览器，中间为三维模型，右侧为结果与 AI 助手"></a>

*ChatDFT 5.8.0 实机界面。图中展示已完成的 CO₂–单乙醇胺水合簇 PySCF 单点计算。*

## 工作台功能

| 工作流 | 可以做什么 |
| --- | --- |
| 三维分子建模 | 从 SMILES、XYZ、非周期 XSD 或自然语言创建结构，编辑坐标、测量距离和角度，对照两个模型。 |
| AI 结构设计 | 描述多分子、离子和显式水的组成及排布。遇到缺少的建模条件时，补充后继续。 |
| 服务器计算 | 通过 SSH 向自己的 PySCF 服务器提交单点能、几何优化、频率或垂直激发态任务，跟踪状态与日志。 |
| 分析和出图 | 全自动模式归档结果和计算输入，生成 CSV、PNG、SVG 与 Markdown 报告。AI 可根据研究问题选择补充图。 |
| 项目与作业管理 | 按作业恢复历史结构及结果、继续未提交的草稿、打开项目文件夹，并接收完成提醒。 |

## 快速开始

1. [下载 Windows ZIP](https://github.com/shaoyinzi654-source/ChatDFT/releases/download/v5.8.0-standalone/ChatDFT-Standalone-5.8.0-Windows.zip)，完整解压到要保留的目录。
2. 安装 **64 位 Python 3.13**，在解压目录的 PowerShell 运行 `.\Install-ChatDFT.ps1`。安装脚本会配置依赖并创建 **ChatDFT** 桌面快捷方式。
3. 打开 ChatDFT。从 **构建 → 从 SMILES 建模** 输入 `O`，即可在本机预览水分子。要运行 PySCF 作业，请在 **计算 → 连接与模型设置** 填写自己的 SSH 主机、用户名、可信主机指纹和服务器上的 PySCF Python 路径。

AI 建模可使用本机已登录的 Codex，或在软件里配置兼容 API。选中 **AI → 全自动计算** 后，可以从这条小任务开始验证建模、计算和导出：

> 全自动：用 SMILES O 建一个水分子，总电荷 0，多重度 1。运行气相 HF/STO-3G 单点能。计算完成后整理电子能、偶极矩、作业日志和结构图。

更多示例、结果解释及输出文件见 [全自动计算指南](docs/workflow.md) 和 [工作台使用说明](standalone/README.md)。

## 计算环境与版本

Windows 工作台负责建模、作业管理和分析。电子结构任务由用户配置的 Linux PySCF 服务器执行。周期 PDC 溶液研究使用单独的 GROMACS 工作流，所需参数和参数迁移假设见 [PDC 分子动力学指南](standalone/PDC-MD-GUIDE.md)。

| 目录 | 内容 |
| --- | --- |
| [`standalone/`](standalone/) | 当前 Windows 独立应用、启动器、安装脚本和计算工作流。 |
| [`docs/`](docs/) | 全自动计算教程、实机截图和分析结果示例。 |
| [`legacy/browser/`](legacy/browser/) | 早期浏览器版源码、示例和原始文档。也可从 [浏览器版归档分支](https://github.com/shaoyinzi654-source/ChatDFT/tree/archive/browser-workbench) 访问原目录结构。 |

提交问题时，请附上应用版本、可公开的输入示例和相关日志，并去掉密码和模型 API 密钥。 [报告问题](https://github.com/shaoyinzi654-source/ChatDFT/issues)

应用代码的许可见 [LICENSE](LICENSE)。组件与研究参数归属见 [第三方来源说明](standalone/OPEN_SOURCE.md)。
