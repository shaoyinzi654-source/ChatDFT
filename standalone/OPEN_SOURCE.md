# ChatDFT 界面来源与复用说明

ChatDFT 使用下列开源桌面框架与图标组件。建模、AI 和计算流程由本项目提供。

| 项目 | 官方源码 / 文档 | 本项目的使用范围 |
| --- | --- | --- |
| Avogadro 2 | https://github.com/OpenChemistry/avogadroapp | 参考分子编辑器的紧凑工具栏、文档和属性布局；未移植其计算或渲染引擎，未复制其源码或品牌图标。上游为 BSD-3-Clause。 |
| ParaView | https://github.com/Kitware/ParaView · https://docs.paraview.org/en/latest/UsersGuide/introduction.html | 参考对象浏览器、上下文参数、拆分视图和输出面板的组织；未复制其源码或引入 VTK。上游为 BSD-3-Clause。 |
| FreeCAD | https://github.com/FreeCAD/FreeCAD | 调研工程工作台、可停靠面板和工具栏布局；未复制其源码。上游为 LGPL-2.1。 |
| Qt Advanced Docking System | https://github.com/githubuser0xFFFF/Qt-Advanced-Docking-System | **实际依赖和运行的窗口框架**。通过 PySide6-QtAds 4.3.1.4 提供拆分、标签合并、浮动、拖动停靠及布局序列化，使用组件内置的默认主题及控件图标。LGPL-2.1。 |
| PySide6-QtAds | https://github.com/mborgerson/pyside6_qtads | **实际安装的 Python 绑定**，与 Qt 6.8.2.1 配套。通过普通 Python 模块导入，不静态链接、不修改绑定源码。LGPL-2.1。 |
| QtAwesome | https://github.com/spyder-ide/qtawesome | **实际依赖的工具图标组件**，使用其 Font Awesome 5 Solid 图标。QtAwesome 为 MIT；字体按包内随附的字体许可证提供。 |

## 文件与边界

- `runtime/workbench_docking.py` 把 ChatDFT 原有功能控件装入真实的 `CDockManager` / `CDockWidget`。结构、曲线和侧边面板均使用 ADS 容器。
- `runtime/workbench.py` 提供 ChatDFT 的操作绑定和布局预设，采用 ADS 官方示例中的中央文档和周边停靠面板组织方法。
- `runtime/workbench_theme.py` 定义本项目的 Qt 调色板、紧凑工具栏样式及 QtAwesome 图标映射。
- 原窗口控制器、分子预览及 AI / SSH 计算代码复用本项目已有实现。
- 本项目没有将 Avogadro、ParaView、FreeCAD 的完整程序或源码改名为 ChatDFT。

## 许可证与可替换依赖

随包的 `third_party/` 保留 ADS / 绑定的许可，以及 QtAwesome 和其字体的许可文件。组件通过可替换的 Python 包安装；解压后可查看并修改 ChatDFT 的 Python 源码。上游对应版本的源码可通过上述项目仓库和发行标签获取。

安装的精确依赖版本见 `requirements.txt`。PySide6 / Qt 的许可随对应 Python 包提供；未将 Qt 或 ADS 静态编译进单一可执行文件。

## 5.3 分子动力学依赖和研究数据

- GROMACS：服务器实际安装 2021.4 / 2018.8，GNU LGPL 2.1 或后续版本。官方源码：https://gitlab.com/gromacs/gromacs ，2018.8 源码下载：https://ftp.gromacs.org/gromacs/gromacs-2018.8.tar.gz 。GCC 11 构建仅在 biasparams.cpp / minimize.cpp 补充标准头 `<limits>`；安装脚本保留此修改。不随本包分发 GROMACS 二进制。
- MDTraj：用于流式读取 GROMACS 轨迹，LGPL 2.1+，https://github.com/mdtraj/mdtraj 。
- SciPy：用于稀疏图连通分量，BSD-3-Clause，https://github.com/scipy/scipy 。NumPy 用于数值运算。
- PDC 参数由使用者从原作者取得，出处 https://github.com/jlianl/PDC_Aggregation ，固定提交 fb868ea2646ecf98aa4faed389a5ef64528069a5。作者文件不以 ChatDFT 名义重新授权；新增可运行输入和修正另存。研究发表或再分发请核对上游数据许可并引用原论文。
- 文献 kappa 依据 Li / Merz DOI 10.1021/ct400751u 式 4 / 表 2 SPC/E。重建及参数迁移假设见 PDC-MD-GUIDE.md。未复制第三方分析软件源码；聚集与网格分析由本项目实现。
