from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from PySide6.QtCore import QSize, QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QActionGroup, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QFormLayout,
    QDockWidget, QFileDialog, QGroupBox, QHeaderView, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget,
    QStatusBar, QStyle, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
    QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from chatdft.ai import ask, validate_calculation
from chatdft.app import SettingsDialog, Task
from chatdft.chemistry import LIBRARY, Structure, from_smiles, validate_structure
from chatdft.editors import AtomEditor
from chatdft.gaussian import BASES, METHODS, SOLVENTS, Calculation, build_input, report
from chatdft.jobs import JobStore, Remote, SSH_AUTH_ERROR
from chatdft.storage import atomic_json, load_settings, save_settings
from chatdft.structure_io import read_structure
from chatdft.viewport import COLORS, MoleculeView, measurement
from chatdft.widgets import PlotView
from theme import ChoiceBox, ElementLegend, NumberBox, apply_theme, application_icon, symbol_icon
from workflow import result_geometry, submit


TASKS = {'单点能': 'sp', '几何优化': 'opt', '优化与频率': 'optfreq', '振动频率': 'freq', '垂直激发态': 'td'}
STATUSES = {'prepared': '已准备', 'queued': '排队中', 'running': '运行中', 'completed': '已完成',
            'succeeded': '已完成', 'failed': '失败', 'cancelled': '已取消'}
VALUE_FIELDS = (
    ('final_energy_hartree', '电子能', 'Eh'), ('gibbs_hartree', 'Gibbs 自由能', 'Eh'),
    ('enthalpy_hartree', '焓', 'Eh'), ('dipole_debye', '偶极矩', 'D'),
    ('alpha_gap_ev', 'α HOMO–LUMO 能隙', 'eV'), ('beta_gap_ev', 'β HOMO–LUMO 能隙', 'eV'),
)


def combo(items):
    widget = ChoiceBox()
    widget.addItems(items)
    return widget


def spin(low, high, value):
    widget = NumberBox()
    widget.setRange(low, high)
    widget.setValue(value)
    return widget


def label(text):
    widget = QLabel(text)
    widget.setTextFormat(Qt.PlainText)
    widget.setWordWrap(True)
    return widget


def row(layout, *actions, centered=False):
    bar = QHBoxLayout()
    if centered:
        bar.addStretch()
    buttons = []
    for title, callback in actions:
        button = QPushButton(title)
        button.clicked.connect(callback)
        bar.addWidget(button)
        buttons.append(button)
    bar.addStretch()
    layout.addLayout(bar)
    return buttons


def table(headers, stretch=0, alternating=True):
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.verticalHeader().hide()
    widget.verticalHeader().setDefaultSectionSize(23)
    widget.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    widget.horizontalHeader().setHighlightSections(False)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    widget.horizontalHeader().setSectionResizeMode(stretch, QHeaderView.Stretch)
    widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
    widget.setSelectionMode(QAbstractItemView.SingleSelection)
    widget.setSelectionBehavior(QAbstractItemView.SelectRows)
    widget.setAlternatingRowColors(alternating)
    widget.setShowGrid(False)
    return widget


def populate(widget, rows):
    widget.setRowCount(len(rows))
    for y, values in enumerate(rows):
        for x, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setToolTip(str(value))
            widget.setItem(y, x, item)


def field_group(title, parent, fields, section='quantum'):
    box = QGroupBox(title)
    box.setProperty('section', section)
    form = QFormLayout(box)
    form.setContentsMargins(10, 9, 10, 8)
    form.setVerticalSpacing(5)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    for text, widget in fields:
        form.addRow(text, widget)
    parent.addWidget(box)
    return form


class Panel(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle('ChatDFT')
        self.setWindowIcon(application_icon())
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1280, available.width() - 40), min(820, available.height() - 55))
        self.setMinimumSize(1020, 650)
        self.settings = load_settings()
        self.store = JobStore()
        self.remote = self.operation = self.model = None
        self.server_operation = None
        self.history = []
        self.proposals = []
        self.suggestion = {}
        self.jobs = {}
        self.undo_stack = []
        self.redo_stack = []
        self.project_path = None

        self.toolbar = QToolBar()
        self.toolbar.setWindowTitle('建模工具')
        self.toolbar.setObjectName('ModelTools')
        self.toolbar.setIconSize(QSize(16, 16))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)
        brand = QLabel('ChatDFT')
        brand.setObjectName('Brand')
        self.toolbar.addWidget(brand)
        self.toolbar.addSeparator()
        self.project_dock = self.dock('模型浏览器', 'ProjectDock', self.project_pane(), Qt.LeftDockWidgetArea)
        self.create_viewport()
        self.tabs = QTabWidget()
        self.tabs.setObjectName('InspectorTabs')
        self.tabs.setMinimumWidth(330)
        self.inspector = self.dock('设计与计算', 'InspectorDock', self.tabs, Qt.RightDockWidgetArea)
        self.add_ai_tab()
        self.add_calculation_tab()
        self.add_jobs_tab()
        self.status_bar = QStatusBar()
        self.status = label('就绪')
        self.status_bar.addWidget(self.status, 1)
        self.provider_status = QLabel()
        self.connection_status = QLabel('SSH 未连接')
        self.status_bar.addPermanentWidget(self.provider_status)
        self.status_bar.addPermanentWidget(self.connection_status)
        self.setStatusBar(self.status_bar)
        self.toolbar_command(QStyle.SP_DialogOpenButton, '导入', self.import_file)
        self.toolbar_command(QStyle.SP_DialogSaveButton, '保存', self.save_project)
        self.toolbar.addSeparator()
        self.undo_action = self.toolbar_command(QStyle.SP_ArrowBack, '撤销', self.undo_model)
        self.redo_action = self.toolbar_command(QStyle.SP_ArrowForward, '重做', self.redo_model)
        self.toolbar.addSeparator()
        self.fit_action = self.toolbar_command(QStyle.SP_TitleBarMaxButton, '适合窗口', self.preview.fit)
        self.display_style = combo(['球棍', '棍状', '线框', '空间填充'])
        self.display_style.currentIndexChanged.connect(
            lambda index: self.preview.set_style(['ballstick', 'stick', 'line', 'cpk'][index]))
        self.toolbar.addWidget(self.display_style)
        self.toolbar.addSeparator()
        self.create_menus()
        self.create_navigation()
        self.show_source()
        self.reload_jobs()
        self.tabs.setCurrentIndex(1 if args.mode == 'server' else 0)
        self.resizeDocks([self.project_dock, self.inspector], [215, 365], Qt.Horizontal)
        self.resizeDocks([self.activity_dock], [210], Qt.Vertical)
        if not self.jobs:
            self.activity_dock.hide()

        self.monitor = QTimer(self)
        self.monitor.setInterval(5000)
        self.monitor.timeout.connect(self.poll)
        self.monitor.start()

    def toolbar_command(self, icon, title, callback, tooltip=None):
        symbols = {QStyle.SP_DialogOpenButton: 'open', QStyle.SP_DialogSaveButton: 'save',
                   QStyle.SP_ArrowBack: 'undo', QStyle.SP_ArrowForward: 'redo',
                   QStyle.SP_TitleBarMaxButton: 'fit', QStyle.SP_ComputerIcon: 'external',
                   QStyle.SP_ArrowUp: 'send'}
        action = self.toolbar.addAction(symbol_icon(symbols.get(icon, 'file')), title)
        action.triggered.connect(callback)
        if tooltip:
            action.setToolTip(tooltip)
        return action

    def update_preview_action(self):
        enabled = self.preview.structure is not None
        self.fit_action.setEnabled(enabled)
        for action in (*self.view_modes.values(), self.labels_action, self.grid_action, *self.structure_actions):
            action.setEnabled(enabled)

    def dock(self, title, name, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        heading = QLabel(title)
        heading.setObjectName('DockHeading')
        dock.setTitleBarWidget(heading)
        self.addDockWidget(area, dock)
        return dock

    def create_navigation(self):
        navigation = QToolBar('工作区导航', self)
        navigation.setObjectName('Navigation')
        navigation.setMovable(False)
        navigation.setIconSize(QSize(23, 23))
        navigation.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.LeftToolBarArea, navigation)
        self.navigation = navigation
        self.nav_group = QActionGroup(self)
        self.nav_actions = {}
        for name, symbol, callback in (
                ('模型', 'model', self.open_model),
                ('助手', 'assistant', self.open_assistant),
                ('任务', 'tasks', self.open_tasks),
                ('设置', 'settings', self.configure)):
            action = navigation.addAction(symbol_icon(symbol), name)
            action.triggered.connect(callback)
            if name != '设置':
                action.setCheckable(True)
                self.nav_group.addAction(action)
                action.setChecked(name == '模型')
                self.nav_actions[name] = action
        self.documents.currentChanged.connect(self.sync_navigation)
        self.tabs.currentChanged.connect(self.sync_navigation)

    def sync_navigation(self):
        name = {
            0: '助手', 1: '模型', 2: '任务'}[self.tabs.currentIndex()]
        self.nav_actions[name].setChecked(True)

    def open_model(self):
        self.activity_dock.hide()
        self.documents.setCurrentIndex(0)
        self.tabs.setCurrentIndex(1)

    def open_assistant(self):
        self.activity_dock.hide()
        self.documents.setCurrentIndex(0)
        self.tabs.setCurrentIndex(0)

    def open_tasks(self):
        self.documents.setCurrentIndex(0)
        self.activity_dock.show()
        self.tabs.setCurrentIndex(2)

    def create_viewport(self):
        self.documents = QTabWidget()
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.document_title = QLabel('未打开结构')
        self.document_title.setObjectName('DocumentTitle')
        layout.addWidget(self.document_title)
        self.preview_stack = QStackedWidget()
        self.preview = MoleculeView()
        self.preview.light_background = False
        self.preview.grid = True
        self.preview.setMinimumSize(300, 210)
        self.preview.set_mode('rotate')
        self.preview.selectionChanged.connect(self.show_measurement)
        self.add_scene_tools(layout)
        self.preview_stack.addWidget(self.preview)
        self.preview_note = label('导入 XYZ / XSD，或从“构建”菜单生成分子。')
        self.preview_note.setAlignment(Qt.AlignCenter)
        self.preview_empty = QWidget()
        empty_layout = QVBoxLayout(self.preview_empty)
        empty_layout.addStretch()
        empty_layout.addWidget(self.preview_note)
        buttons = QHBoxLayout()
        buttons.addStretch()
        for title, callback in (('导入结构…', self.import_file), ('从 SMILES 建模…', self.build_smiles)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        buttons.addStretch()
        empty_layout.addLayout(buttons)
        empty_layout.addStretch()
        self.preview_stack.addWidget(self.preview_empty)
        layout.addWidget(self.preview_stack, 1)
        self.element_legend = ElementLegend(COLORS)
        layout.addWidget(self.element_legend)
        self.documents.addTab(page, '结构')
        self.atom_table = table(['原子', '元素', 'X / Å', 'Y / Å', 'Z / Å'], 2)
        self.documents.addTab(self.atom_table, '坐标')
        self.setCentralWidget(self.documents)

    def add_scene_tools(self, layout):
        tools = QToolBar()
        tools.setObjectName('SceneTools')
        tools.setIconSize(QSize(19, 19))
        tools.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.mode_group = QActionGroup(self)
        self.view_modes = {}
        self.structure_actions = []
        for title, symbol, mode, count in (
                ('旋转模型', 'rotate', 'rotate', 0), ('平移模型', 'pan', 'pan', 0),
                ('选取原子', 'model', 'select', 0), ('测量距离', 'measure', 'distance', 2),
                ('测量角度', 'angle', 'angle', 3)):
            action = tools.addAction(symbol_icon(symbol), title)
            action.setCheckable(True)
            self.mode_group.addAction(action)
            action.setChecked(mode == 'rotate')
            action.triggered.connect(
                lambda checked=False, m=mode, n=count: self.measure_mode(n) if n else self.set_model_mode(m))
            self.view_modes[mode] = action
        tools.addSeparator()
        self.labels_action = tools.addAction(symbol_icon('label'), '显示原子编号')
        self.labels_action.setCheckable(True)
        self.labels_action.toggled.connect(lambda checked: self.view_flag('labels', checked))
        self.grid_action = tools.addAction(symbol_icon('fit'), '坐标网格（1 Å）')
        self.grid_action.setCheckable(True)
        self.grid_action.setChecked(True)
        self.grid_action.toggled.connect(lambda checked: self.view_flag('grid', checked))
        tools.addSeparator()
        action = tools.addAction(symbol_icon('atoms'), '编辑原子坐标')
        action.triggered.connect(self.edit_atoms)
        self.structure_actions.append(action)
        action = tools.addAction(symbol_icon('optimize'), '设置几何优化')
        action.triggered.connect(self.setup_optimization)
        self.structure_actions.append(action)
        tools.addSeparator()
        action = tools.addAction(symbol_icon('model'), '从 SMILES 构建')
        action.triggered.connect(self.build_smiles)
        layout.addWidget(tools)

    def project_pane(self):
        pane = QSplitter(Qt.Vertical)
        pane.setChildrenCollapsible(False)
        self.project = QTreeWidget()
        self.project.setColumnCount(1)
        self.project.setHeaderHidden(True)
        self.project.setAlternatingRowColors(False)
        pane.addWidget(self.project)
        self.project.itemActivated.connect(self.activate_project_item)
        self.structure_node = QTreeWidgetItem(self.project, ['当前结构'])
        self.structure_node.setIcon(0, symbol_icon('model'))
        self.structure_node.setData(0, Qt.UserRole, 1)
        self.candidate_node = QTreeWidgetItem(self.project, ['候选结构'])
        self.candidate_node.setIcon(0, symbol_icon('folder'))
        self.candidate_node.setData(0, Qt.UserRole, 0)
        self.job_node = QTreeWidgetItem(self.project, ['远程任务'])
        self.job_node.setIcon(0, symbol_icon('tasks'))
        self.job_node.setData(0, Qt.UserRole, 2)
        self.properties = table(['结构属性', '值'], 1, False)
        self.properties.horizontalHeader().setMinimumSectionSize(64)
        pane.addWidget(self.properties)
        pane.setSizes([280, 285])
        pane.setMinimumWidth(185)
        return pane

    def activate_project_item(self, item, _column):
        self.tabs.setCurrentIndex(item.data(0, Qt.UserRole))
        if item.parent() is self.candidate_node:
            self.candidates.selectRow(self.candidate_node.indexOfChild(item))
        if item.parent() is self.job_node:
            self.job_table.selectRow(self.job_node.indexOfChild(item))
            self.activity_dock.show()

    def tab(self, title):
        page = QWidget()
        page.setObjectName('InspectorPage')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, title)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        return layout

    def add_ai_tab(self):
        layout = self.tab('建模助手')
        self.answer = QPlainTextEdit()
        self.answer.setReadOnly(True)
        self.answer.setPlaceholderText('会话记录')
        layout.addWidget(self.answer, 3)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText('输入建模要求或计算问题…')
        self.prompt.setFixedHeight(64)
        layout.addWidget(self.prompt)
        self.send_button, self.suggestion_button = row(
            layout, ('发送', self.send), ('应用建议参数', self.apply_suggestion))
        self.suggestion_button.setEnabled(False)
        box = QGroupBox('候选结构')
        candidate_layout = QVBoxLayout(box)
        candidate_layout.setContentsMargins(6, 7, 6, 6)
        self.candidates = table(['名称', 'SMILES', '多重度'], 1)
        candidate_layout.addWidget(self.candidates)
        self.candidate_button, = row(candidate_layout, ('载入结构', self.use_candidate))
        self.candidate_button.setEnabled(False)
        self.candidates.itemSelectionChanged.connect(self.update_candidate_actions)
        layout.addWidget(box, 2)

    def add_calculation_tab(self):
        page = QWidget()
        page.setObjectName('InspectorPage')
        outer = QVBoxLayout(page)
        outer.setContentsMargins(7, 7, 7, 7)
        outer.setSpacing(5)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        content.setObjectName('InspectorPage')
        layout = QVBoxLayout(content)
        layout.setContentsMargins(1, 1, 4, 1)
        layout.setSpacing(5)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        self.tabs.addTab(page, '计算设置')
        self.endpoint = label('')
        self.task = combo(TASKS)
        self.method = combo([method for method in METHODS if method != 'wB97XD'])
        self.basis = combo(BASES)
        self.solvent = combo(SOLVENTS)
        self.charge = spin(-20, 20, 0)
        self.multiplicity = spin(1, 20, 1)
        self.cores = spin(1, 16, 4)
        self.memory = spin(1, 14, 4)
        field_group('电子结构', layout, (
            ('任务', self.task), ('方法', self.method), ('基组', self.basis),
            ('溶剂 / IEF-PCM', self.solvent),
        ))
        field_group('分子状态', layout, (
            ('总电荷', self.charge),
            ('自旋多重度', self.multiplicity),
        ), 'molecule')
        field_group('计算资源', layout, (
            ('CPU 核数', self.cores), ('内存 / GB', self.memory),
        ), 'resources')
        self.charge.valueChanged.connect(self.show_source)
        self.multiplicity.valueChanged.connect(self.show_source)
        connection = QVBoxLayout()
        actions = QHBoxLayout()
        for title, callback in (('测试连接', self.test_connection), ('断开', self.disconnect)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            actions.addWidget(button)
        connection.addLayout(actions)
        layout.addLayout(connection)
        footer = QHBoxLayout()
        self.submit_button = QPushButton('提交到服务器…')
        self.submit_button.setObjectName('Submit')
        self.submit_button.clicked.connect(self.submit_job)
        footer.addWidget(self.submit_button)
        outer.addWidget(self.endpoint)
        outer.addLayout(footer)
        layout.addStretch()

    def add_jobs_tab(self):
        properties = self.tab('结果')
        self.result_grid = table(['结果', '值', '单位'], 1)
        self.result_grid.setWordWrap(False)
        self.result_grid.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.result_grid.setColumnWidth(0, 115)
        properties.addWidget(self.result_grid)
        self.explain_button, self.export_button = row(
            properties, ('解读结果', self.explain_job), ('载入结果结构', self.load_job_geometry))
        self.activity = QTabWidget()
        self.activity.setObjectName('ActivityTabs')
        task_page = QWidget()
        layout = QVBoxLayout(task_page)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        self.job_table = table(['结构', '状态', '方法 / 基组', '提交时间'])
        self.job_table.itemSelectionChanged.connect(self.show_job)
        self.refresh_button, self.cancel_button, self.directory_button = row(
            layout, ('同步状态', self.refresh_job), ('取消任务…', self.cancel_job), ('打开目录', self.open_job))
        task_splitter = QSplitter(Qt.Horizontal)
        task_splitter.setChildrenCollapsible(False)
        task_splitter.addWidget(self.job_table)
        self.job_summary_plot = PlotView()
        self.job_summary_plot.setMinimumSize(210, 95)
        task_splitter.addWidget(self.job_summary_plot)
        task_splitter.setStretchFactor(0, 3)
        task_splitter.setStretchFactor(1, 1)
        task_splitter.setSizes([800, 275])
        layout.addWidget(task_splitter)
        self.job_table.cellClicked.connect(lambda *_: self.tabs.setCurrentIndex(2))
        self.activity.addTab(task_page, '计算任务')
        self.result_view = QTextBrowser()
        self.activity.addTab(self.result_view, '报告')
        self.energy_plot = PlotView()
        self.energy_plot.setMinimumHeight(110)
        self.activity.addTab(self.energy_plot, 'SCF 能量')
        self.spectra = table(['类型 / 态', '频率 / cm⁻¹', '激发能 / eV', '振子强度'], 0)
        self.activity.addTab(self.spectra, '频率与激发态')
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.activity.addTab(self.output, '日志')
        self.activity_dock = self.dock('计算与输出', 'ActivityDock', self.activity, Qt.BottomDockWidgetArea)

    def create_menus(self):
        self.file_menu = file_menu = self.menuBar().addMenu('文件')
        self.menu_action(file_menu, '导入结构…', self.import_file, 'Ctrl+O')
        self.menu_action(file_menu, '打开 ChatDFT 项目…', self.open_project)
        self.menu_action(file_menu, '保存项目…', self.save_project, 'Ctrl+S')
        export = file_menu.addMenu('导出')
        self.menu_action(export, 'XYZ 坐标…', self.export_xyz)
        self.menu_action(export, '结构视图…', self.export_image)
        self.menu_action(export, '计算结果 CSV…', self.export_csv)
        self.menu_action(export, '计算报告 Markdown…', self.export_report)
        file_menu.addSeparator()
        self.menu_action(file_menu, '关闭', self.close, 'Ctrl+Q')
        edit = self.menuBar().addMenu('编辑')
        edit.addAction(self.undo_action)
        self.undo_action.setShortcut('Ctrl+Z')
        edit.addAction(self.redo_action)
        self.redo_action.setShortcut('Ctrl+Y')
        self.menu_action(edit, '原子坐标…', self.edit_atoms)
        build = self.menuBar().addMenu('构建')
        self.menu_action(build, '从 SMILES 建模…', self.build_smiles)
        library = build.addMenu('分子库')
        for name, smiles in LIBRARY.items():
            self.menu_action(library, name, lambda checked=False, n=name, s=smiles: self.generate(s, n))
        build.addSeparator()
        self.menu_action(build, '设置几何优化', self.setup_optimization)
        view = self.menuBar().addMenu('视图')
        for action in self.view_modes.values():
            view.addAction(action)
        dihedral = self.menu_action(view, '测量二面角', lambda: self.measure_mode(4))
        dihedral.setCheckable(True)
        self.mode_group.addAction(dihedral)
        view.addSeparator()
        for title, key in [('正视图', 'front'), ('俯视图', 'top'), ('侧视图', 'right'), ('等轴视图', 'iso')]:
            self.menu_action(view, title, lambda checked=False, k=key: self.preview.orient(k))
        self.menu_action(view, '适合窗口', self.preview.fit, 'F')
        view.addAction(self.labels_action)
        view.addAction(self.grid_action)
        background = self.menu_action(view, '浅色背景', lambda checked: self.view_flag('light_background', checked))
        background.setCheckable(True)
        background.setChecked(False)
        compute = self.menuBar().addMenu('计算')
        self.server_jobs_action = self.menu_action(compute, '服务器作业管理…', self.open_server_jobs)
        self.menu_action(compute, '计算设置', lambda: self.tabs.setCurrentIndex(1))
        self.menu_action(compute, '提交到服务器…', self.submit_job)
        self.menu_action(compute, '同步所选任务', self.refresh_job)
        self.menu_action(compute, '重试所选失败任务…', self.retry_job)
        self.menu_action(compute, '解读所选结果', self.explain_job)
        self.menu_action(compute, '连接与模型设置…', self.configure)

    def menu_action(self, menu, text, callback, shortcut=None):
        action = QAction(text, self)
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        menu.addAction(action)
        return action

    def show_measurement(self, indices):
        if not hasattr(self, 'status'):
            return
        if self.preview.structure and len(indices) >= 2:
            self.status.setText('原子 ' + '–'.join(str(i + 1) for i in indices) + '   ' +
                                measurement(self.preview.structure.atoms, indices))
        elif indices:
            atom = self.preview.structure.atoms[indices[0]]
            self.status.setText(f"原子 {indices[0] + 1}   {atom['element']}   " +
                                '   '.join(f"{key.upper()} {atom[key]:.5f} Å" for key in 'xyz'))

    def measure_mode(self, count):
        self.preview.clear_selection()
        self.preview.measure_count = count
        self.preview.set_mode('measure')
        self.status.setText(f'依次选择 {count} 个原子。')

    def set_model_mode(self, mode):
        self.preview.set_mode(mode)
        self.status.setText({'rotate': '拖动旋转模型', 'pan': '拖动平移模型',
                             'select': '点击选择原子，Ctrl 多选'}[mode])

    def view_flag(self, flag, checked):
        setattr(self.preview, flag, checked)
        self.preview.update()

    def remember(self):
        self.undo_stack.append(self.model_state())
        self.undo_stack = self.undo_stack[-30:]
        self.redo_stack.clear()

    def model_state(self):
        return dict(model=self.model.to_dict() if self.model else None,
                    charge=self.charge.value(), spin=self.multiplicity.value(),
                    calculation=self.calculation(), project_path=self.project_path)

    def restore_model_state(self, state):
        self.project_path = state['project_path']
        self.set_calculation(state['calculation'])
        self.model = Structure.from_dict(state['model']) if state['model'] else None
        for widget, value in ((self.charge,state['charge']), (self.multiplicity,state['spin'])):
            before = widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(before)
        self.show_source()
        self.documents.setCurrentIndex(0)

    def set_calculation(self, calculation):
        self.method.setCurrentText(calculation.method)
        self.basis.setCurrentText(calculation.basis)
        self.solvent.setCurrentText(calculation.solvent)
        self.task.setCurrentText(next(title for title, key in TASKS.items() if key == calculation.task))
        self.cores.setValue(calculation.cores)
        self.memory.setValue(calculation.memory_gb)

    def restore_model(self, data):
        self.model = Structure.from_dict(data)
        self.charge.blockSignals(True)
        self.multiplicity.blockSignals(True)
        self.charge.setValue(self.model.charge)
        self.multiplicity.setValue(self.model.multiplicity)
        self.charge.blockSignals(False)
        self.multiplicity.blockSignals(False)
        self.show_source()

    def undo_model(self):
        if not self.undo_stack or self.operation:
            return
        self.redo_stack.append(self.model_state())
        self.restore_model_state(self.undo_stack.pop())
        self.status.setText('已撤销结构修改。')

    def redo_model(self):
        if not self.redo_stack or self.operation:
            return
        self.undo_stack.append(self.model_state())
        self.restore_model_state(self.redo_stack.pop())
        self.status.setText('已重做结构修改。')

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, '导入结构', '', '结构文件 (*.xyz *.xsd)')
        if path:
            try:
                structure = read_structure(Path(path), self.charge.value(), self.multiplicity.value())
                self.select_candidate(structure)
                self.status.setText('已导入 ' + Path(path).name)
            except (OSError, ValueError) as exc:
                self.show_error(exc)

    def build_smiles(self):
        smiles, ok = QInputDialog.getText(self, '从 SMILES 建模', 'SMILES')
        if ok and smiles.strip():
            self.generate(smiles.strip(), 'SMILES 分子')

    def generate(self, smiles, name):
        self.run('正在生成三维结构…', lambda: from_smiles(smiles, name), self.select_candidate)

    def edit_atoms(self):
        try:
            structure = self.structure()
        except (OSError, ValueError) as exc:
            self.show_error(exc)
            return
        editor = AtomEditor(structure, self)
        editor.setWindowTitle('原子坐标 - ChatDFT')
        if editor.exec():
            self.select_candidate(editor.result_structure)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开项目', '', 'ChatDFT 项目 (*.chatdft.json)')
        if not path:
            return
        try:
            path = Path(path)
            if path.stat().st_size > 10 * 1024 * 1024:
                raise ValueError('项目文件超过 10 MB。')
            value = json.loads(path.read_text('utf-8'))
            if value.get('format') != 'ChatDFT' or value.get('version') != 1:
                raise ValueError('项目格式或版本不支持。')
            structure = Structure.from_dict(value['structure'])
            validate_structure(structure)
            calculation = Calculation(**value.get('calculation', {}))
            if (calculation.task not in TASKS.values() or calculation.method == 'wB97XD'
                    or calculation.dispersion):
                raise ValueError('项目的计算设置不适用于当前 PySCF 引擎。')
            if type(calculation.cores) is not int or type(calculation.memory_gb) is not int:
                raise ValueError('CPU 核数和内存必须是整数。')
            build_input(structure, calculation)
            self.select_candidate(structure)
            self.set_calculation(calculation)
            self.project_path = path
            self.status.setText('已打开 ' + path.name)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.show_error(exc)

    def calculation(self):
        return Calculation(method=self.method.currentText(), basis=self.basis.currentText(),
                           task=TASKS[self.task.currentText()], solvent=self.solvent.currentText(),
                           cores=self.cores.value(), memory_gb=self.memory.value())

    def save_project(self):
        try:
            structure = self.structure()
            path, _ = QFileDialog.getSaveFileName(self, '保存项目',
                str(self.project_path or Path.home() / 'Documents' / 'Untitled.chatdft.json'),
                'ChatDFT 项目 (*.chatdft.json)')
            if path:
                atomic_json(path, dict(format='ChatDFT', version=1, structure=structure.to_dict(),
                                       calculation=self.calculation().to_dict()))
                self.project_path = Path(path)
                self.status.setText('已保存 ' + Path(path).name)
        except (OSError, ValueError) as exc:
            self.show_error(exc)

    def export_xyz(self):
        try:
            structure = self.structure()
            path, _ = QFileDialog.getSaveFileName(self, '导出坐标', 'structure.xyz', 'XYZ (*.xyz)')
            if path:
                Path(path).write_text(structure.xyz(), encoding='utf-8')
                self.status.setText('已导出 XYZ 坐标。')
        except (OSError, ValueError) as exc:
            self.show_error(exc)

    def export_image(self):
        path, _ = QFileDialog.getSaveFileName(self, '导出结构视图', 'structure.png', 'PNG 图像 (*.png)')
        if path:
            if not self.preview.grab().save(path, 'PNG'):
                self.show_error('无法保存图像。')
                return
            self.status.setText('已导出结构视图。')

    def export_csv(self):
        job = self.current_job()
        if not job:
            self.show_error('请选择一个任务。')
            return
        path, _ = QFileDialog.getSaveFileName(self, '导出结果表格', 'results.csv', 'CSV (*.csv)')
        if path:
            try:
                with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['结果', '值', '单位'])
                    for row in range(self.result_grid.rowCount()):
                        writer.writerow([self.result_grid.item(row, col).text() for col in range(3)])
                self.status.setText('已导出结果表格。')
            except OSError as exc:
                self.show_error(exc)

    def export_report(self):
        try:
            job = self.current_job()
            text = report(self.read_result(job), 'ChatDFT / PySCF 计算报告')
            path, _ = QFileDialog.getSaveFileName(self, '导出报告', 'report.md', 'Markdown (*.md)')
            if path:
                Path(path).write_text(text, encoding='utf-8')
                self.status.setText('已导出计算报告。')
        except (OSError, ValueError) as exc:
            self.show_error(exc)

    def setup_optimization(self):
        self.task.setCurrentText('几何优化')
        self.tabs.setCurrentIndex(1)
        self.status.setText('已选择 PySCF 几何优化。核对方法和基组后，可提交到计算服务器。')

    def show_source(self):
        name, kind, atoms = '—', '—', '—'
        if self.model:
            name, atoms = self.model.name, len(self.model.atoms)
            pieces = self.model.component_count()
            kind = f'分子簇 · {pieces} 个片段' if pieces > 1 else '分子'
        self.structure_node.setText(0, name if name != '—' else '当前结构')
        self.structure_node.setToolTip(0, name)
        self.document_title.setText(name if name != '—' else 'ChatDFT')
        self.setWindowTitle(f'{name} - ChatDFT' if name != '—' else 'ChatDFT')
        host = self.settings['ssh_host']
        self.endpoint.setText(f"PySCF · {host}:{self.settings['ssh_port']}" if host else 'PySCF · 尚未配置服务器')
        provider = 'Codex' if self.settings['provider'] == 'codex' else 'API'
        model = self.settings.get('model') or ('默认模型' if provider == 'Codex' else '待配置')
        self.provider_status.setText(f'{provider} · {model}')
        try:
            structure = self.structure()
        except (OSError, ValueError) as exc:
            structure = None
            self.preview_note.setText(str(exc) if self.model else
                                      '导入 XYZ / XSD，或从 SMILES 创建分子。')
        if structure != self.preview.structure:
            self.preview.set_structure(structure)
        self.element_legend.set_structure(structure)
        self.preview_stack.setCurrentWidget(self.preview if structure else self.preview_empty)
        self.update_preview_action()
        self.submit_button.setEnabled(structure is not None)
        self.undo_action.setEnabled(bool(self.undo_stack))
        self.redo_action.setEnabled(bool(self.redo_stack))
        populate(self.atom_table, [(i + 1, atom['element'], *[f"{atom[k]:.6f}" for k in 'xyz'])
                                  for i, atom in enumerate(structure.atoms)] if structure else [])
        populate(self.properties, [
            ('名称', name), ('类型', kind), ('原子数', atoms),
            ('分子式', structure.formula() if structure else '—'),
            ('总电荷', self.charge.value()), ('自旋多重度', self.multiplicity.value()),
        ])

    def configure(self):
        if self.operation:
            return
        if self.server_operation:
            self.status.setText('服务器请求结束后可以修改连接设置。')
            return
        dialog = SettingsDialog(self.settings, self)
        dialog.setWindowTitle('ChatDFT 设置')
        for key in ('target', 'engine', 'local_gaussian', 'gaussian_exe', 'gaussian_profile'):
            field = dialog.fields[key]
            field.parentWidget().layout().setRowVisible(field, False)
        if dialog.exec():
            save_settings(dialog.settings)
            self.settings = dialog.settings
            self.disconnect()
            self.show_source()

    def disconnect(self):
        if not self.server_ready():
            return
        if self.remote:
            self.remote.close()
            self.remote = None
        self.connection_status.setText('SSH 未连接')
        self.connection_status.setToolTip('')
        self.status.setText('SSH 已断开。')

    def run(self, text, function, done, quiet=False):
        if self.operation:
            return
        if not quiet:
            self.status.setText(text)
        self.tabs.setEnabled(False)
        self.toolbar.setEnabled(False)
        self.menuBar().setEnabled(False)
        task = self.operation = Task(function)
        task.success.connect(done)
        task.failure.connect(lambda error: self.status.setText(error))
        task.finished.connect(self.finish_operation)
        task.start()

    def finish_operation(self):
        task, self.operation = self.operation, None
        self.tabs.setEnabled(True)
        self.toolbar.setEnabled(True)
        self.menuBar().setEnabled(True)
        task.deleteLater()

    def show_error(self, exc):
        self.status.setText(str(exc))

    def server_ready(self):
        if self.server_operation:
            self.status.setText('正在处理上一条服务器请求，可以继续使用本地工作区。')
            return False
        return True

    def clear_rejected_login(self, remote):
        if self.remote is remote:
            self.remote = None
            if not self.server_operation:
                remote.close()
            self.connection_status.setText('SSH 认证失败 · 请重新连接')
            self.connection_status.setToolTip(SSH_AUTH_ERROR)

    def run_server(self, text, remote, function, done, *, quiet=False):
        if self.server_operation:
            return
        if not quiet:
            self.status.setText(text)

        def succeeded(value):
            self.connection_status.setText('SSH 已连接')
            self.connection_status.setToolTip('')
            done(value)

        def failed(message):
            if message == SSH_AUTH_ERROR:
                self.clear_rejected_login(remote)
            else:
                self.connection_status.setText('SSH 作业同步异常' if quiet else 'SSH 请求失败')
                self.connection_status.setToolTip(message)
            if not quiet or message == SSH_AUTH_ERROR:
                self.status.setText(message)

        def finished():
            self.server_operation = None
            if self.remote is not remote:
                remote.close()
            task.deleteLater()

        task = self.server_operation = Task(function)
        task.success.connect(succeeded)
        task.failure.connect(failed)
        task.finished.connect(finished)
        task.start()
        return task

    def structure(self):
        charge, multiplicity = self.charge.value(), self.multiplicity.value()
        if self.model:
            structure = type(self.model).from_dict(self.model.to_dict())
            structure.charge = charge
            structure.multiplicity = multiplicity
            validate_structure(structure)
            return structure
        raise ValueError('请先导入结构，或使用 SMILES / AI 建模。')

    def send(self, prompt=None, result=None):
        text = prompt or self.prompt.toPlainText().strip()
        if not text:
            return
        ctx = {'application': 'ChatDFT', 'engine': 'PySCF', 'calculation': self.calculation().to_dict()}
        if self.model:
            try:
                ctx['structure'] = self.structure().to_dict()
            except ValueError as exc:
                self.show_error(exc)
                return
        if result is not None:
            ctx['result'] = result
        settings = dict(self.settings)
        self.run('正在等待 AI 回复…', lambda: ask(settings, [
            *self.history, {'role': 'user', 'content': text}
        ], ctx), lambda value: self.receive(text, value))

    def receive(self, prompt, value):
        self.history.extend([{'role': 'user', 'content': prompt},
                             {'role': 'assistant', 'content': value}])
        self.proposals = value['candidates']
        self.suggestion = value.get('calculation') or {}
        self.suggestion_button.setEnabled(isinstance(self.suggestion, dict) and bool(self.suggestion))
        populate(self.candidates, [(item['name'], item['smiles'], item['multiplicity']) for item in self.proposals])
        self.candidate_node.takeChildren()
        for item in self.proposals:
            node = QTreeWidgetItem(self.candidate_node, [item['name']])
            node.setIcon(0, symbol_icon('model'))
            node.setData(0, Qt.UserRole, 0)
        self.candidate_node.setExpanded(True)
        if self.proposals:
            self.candidates.selectRow(0)
        lines = [value['message']]
        for item in self.proposals:
            lines.append(f"{item['name']}  [{item['smiles']}]\n{item['rationale']}")
        if isinstance(self.suggestion, dict) and self.suggestion:
            lines.append('建议参数  ' + ' / '.join(str(self.suggestion.get(k, '')) for k in ('method', 'basis', 'task', 'solvent')))
        prior = self.answer.toPlainText()
        self.answer.setPlainText((prior + '\n\n' if prior else '') + prompt + '\n\n' + '\n\n'.join(lines))
        self.answer.verticalScrollBar().setValue(self.answer.verticalScrollBar().maximum())
        self.status.setText('AI 回复已收到。')

    def apply_suggestion(self):
        if not isinstance(self.suggestion, dict) or not self.suggestion:
            self.status.setText('当前没有 AI 建议参数。')
            return
        try:
            validate_calculation(self.suggestion)
            fields = {key: self.suggestion[key] for key in ('method', 'basis', 'task', 'solvent')}
            calculation = Calculation(**fields, cores=self.cores.value(), memory_gb=self.memory.value())
            build_input(self.structure(), calculation)
        except (OSError, ValueError) as exc:
            self.show_error(exc)
            return
        self.set_calculation(calculation)
        self.tabs.setCurrentIndex(1)
        self.status.setText('已填写可用的建议参数。请核对后提交。')

    def candidate(self):
        index = self.candidates.currentRow()
        if index < 0:
            raise ValueError('请先用 AI 生成并选择一个候选。')
        return self.proposals[index]

    def update_candidate_actions(self):
        selected = 0 <= self.candidates.currentRow() < len(self.proposals)
        self.candidate_button.setEnabled(selected)

    @staticmethod
    def build_candidate(proposal):
        structure = from_smiles(proposal['smiles'], proposal['name'])
        if structure.multiplicity != proposal['multiplicity']:
            raise ValueError('候选自旋与 SMILES 推断的自旋不一致，请检查 SMILES 和自旋设置。')
        return structure

    def use_candidate(self):
        try:
            proposal = self.candidate()
        except ValueError as exc:
            self.show_error(exc)
            return
        self.run('正在生成三维候选…', lambda: self.build_candidate(proposal), self.select_candidate)

    def select_candidate(self, structure, *, focus_calculation=True):
        self.remember()
        self.project_path = None
        self.restore_model(structure.to_dict())
        self.documents.setCurrentIndex(0)
        if focus_calculation:
            self.tabs.setCurrentIndex(1)
        self.status.setText('已载入 ' + structure.name)

    def connection(self):
        if self.remote:
            return self.remote
        if not (self.settings['ssh_host'] and self.settings['ssh_user'] and
                self.settings['host_fingerprint'].startswith('SHA256:')):
            self.status.setText('请在「计算 → 连接与模型设置」填写 SSH 主机、用户名和可信指纹。')
            return None
        password = ''
        if not self.settings.get('ssh_key'):
            password, ok = QInputDialog.getText(self, '连接计算服务器',
                f"{self.settings['ssh_user']}@{self.endpoint.text()}\nSSH 密码（用 SSH Agent 时可留空）", QLineEdit.Password)
            if not ok:
                return None
        self.remote = Remote({**self.settings, 'target': 'ssh', 'engine': 'pyscf'}, password)
        return self.remote

    def open_server_jobs(self):
        from server_jobs_dialog import ServerJobsDialog

        if not hasattr(self, 'server_jobs_dialog'):
            self.server_jobs_dialog = ServerJobsDialog(self)
        self.server_jobs_dialog.show()
        self.server_jobs_dialog.raise_()
        self.server_jobs_dialog.activateWindow()

    def test_connection(self):
        if not self.server_ready():
            return
        remote = self.connection()
        if remote:
            self.run_server('正在检查服务器…', remote, remote.health, self.show_health)

    def show_health(self, health):
        self.connection_status.setText('SSH 已连接')
        if health.get('engine_ready'):
            self.status.setText(f"已连接 {health['host']}，PySCF {health['engine_version']}，{health['cores']} 核。")
        else:
            self.status.setText('SSH 已连接，PySCF 尚未就绪。请检查服务器上的 Python 环境。')

    def submit_job(self):
        if not self.server_ready():
            return
        try:
            structure = self.structure()
            calculation = self.calculation()
        except (ValueError, OSError) as exc:
            self.show_error(exc)
            return
        summary = (f'{structure.name}   {structure.formula()}\n'
                   f'{self.task.currentText()}  {calculation.method}/{calculation.basis}  {calculation.solvent}\n'
                   f'总电荷 {structure.charge}   多重度 {structure.multiplicity}   '
                   f'{calculation.cores} 核 / {calculation.memory_gb} GB\n\n{self.endpoint.text()}')
        if structure.provenance.startswith('论文复刻 '):
            geometry_origin = structure.provenance.split(' / ', 2)[-1]
            summary += f'\n\n论文复刻 · {geometry_origin}\n请核对几何和计算条件与原文的差异。'
        if QMessageBox.question(self, '提交到计算服务器', summary) != QMessageBox.Yes:
            return
        if not self.server_ready():
            return
        remote = self.connection()
        if remote:
            self.run_server('正在提交任务…', remote,
                            lambda: submit(remote, self.store, structure, calculation), self.submitted)

    def submitted(self, job):
        self.connection_status.setText('SSH 已连接')
        self.reload_jobs(job['id'])
        self.tabs.setCurrentIndex(2)
        self.activity_dock.show()
        self.status.setText(f"服务器已接收任务 {job['id']}，等待可用资源。")

    def reload_jobs(self, selected=None):
        job = self.current_job()
        selected = selected or (job and job['id'])
        self.jobs = {j['id']: j for j in self.store.list() if j.get('engine') == 'pyscf' and j.get('target') == 'ssh'}
        self.job_table.blockSignals(True)
        self.job_node.takeChildren()
        rows = []
        index = -1
        for row_index, (jid, job) in enumerate(self.jobs.items()):
            configuration = job['calculation']
            rows.append((job['name'], STATUSES.get(job['status'], job['status']),
                         f"{configuration['method']} / {configuration['basis']}",
                         datetime.fromtimestamp(job['created']).strftime('%m-%d %H:%M')))
            node = QTreeWidgetItem(self.job_node, [job['name']])
            node.setData(0, Qt.UserRole, 2)
            node.setToolTip(0, jid)
            node.setIcon(0, symbol_icon('file'))
            if jid == selected:
                index = row_index
        populate(self.job_table, rows)
        for row_index, jid in enumerate(self.jobs):
            entry = self.job_table.item(row_index, 0)
            entry.setData(Qt.UserRole, jid)
            entry.setToolTip(jid)
            state = self.jobs[jid]['status']
            color = {'completed': '#2f896b', 'succeeded': '#2f896b', 'running': '#3979bb', 'queued': '#a67725',
                     'failed': '#bd5c5c', 'cancelled': '#9c7b48'}.get(state, '#7b899a')
            self.job_table.item(row_index, 1).setForeground(QColor(color))
            self.job_table.item(row_index, 1).setIcon(symbol_icon('status', color))
        if index >= 0:
            self.job_table.selectRow(index)
        else:
            self.job_table.clearSelection()
            self.job_table.setCurrentCell(-1, -1)
        self.job_table.blockSignals(False)
        self.show_job()

    def current_job(self):
        entry = self.job_table.item(self.job_table.currentRow(), 0)
        return self.jobs.get(entry.data(Qt.UserRole)) if entry else None

    def read_result(self, job):
        if not job or not (path := self.store.folder(job) / 'result.json').is_file():
            raise ValueError('当前任务还没有可读的结果。')
        return json.loads(path.read_text('utf-8'))

    def show_job(self):
        job = self.current_job()
        self.energy_plot.set_data([], 'SCF 电子能 / Eh')
        self.job_summary_plot.set_data([], 'SCF 电子能 / Eh')
        self.spectra.setRowCount(0)
        self.output.clear()
        self.refresh_button.setEnabled(bool(job and job.get('remote_dir')))
        self.cancel_button.setEnabled(bool(job and job['status'] in ('queued','running')))
        self.directory_button.setEnabled(job is not None)
        self.explain_button.setEnabled(False)
        self.export_button.setEnabled(False)
        if not job:
            self.result_grid.setRowCount(0)
            self.result_view.setPlainText('此电脑上还没有 PySCF 远程任务。')
            return
        rows = [('结构', job['name'], ''), ('状态', STATUSES.get(job['status'], job['status']), ''),
                ('任务编号', job['id'], '')]
        if job.get('effective_cores') is not None:
            rows.append(('实际 CPU 核数', job['effective_cores'], '核'))
        if job.get('retry_of'):
            rows.append(('重试的原任务', job['retry_of'], ''))
        log = self.store.folder(job) / 'output.log'
        if log.is_file():
            try:
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 150000))
                    self.output.setPlainText(stream.read().decode('utf-8', errors='replace'))
            except OSError as exc:
                self.output.setPlainText(str(exc))
        try:
            result = self.read_result(job)
            self.result_view.setMarkdown(report(result, 'PySCF 结果'))
            self.energy_plot.set_data(result.get('scf_energies_hartree', []), 'SCF 电子能 / Eh')
            self.job_summary_plot.set_data(result.get('scf_energies_hartree', []), 'SCF 电子能 / Eh')
            spectra = [('振动 ' + str(i + 1), f'{value:.2f}', '—', '—')
                       for i, value in enumerate(result.get('frequencies_cm1', []))]
            spectra.extend((f"激发态 {state['state']}", '—', f"{state['energy_ev']:.5f}",
                            f"{state['oscillator_strength']:.5f}") for state in result.get('excited_states', []))
            populate(self.spectra, spectra)
            self.explain_button.setEnabled(True)
            self.export_button.setEnabled(bool(result.get('normal_termination') and result.get('geometry')))
            rows.extend((label, f'{result[key]:.8f}', unit)
                        for key, label, unit in VALUE_FIELDS if result.get(key) is not None)
            if 'scf_converged' in result:
                rows.append(('SCF 收敛', '是' if result['scf_converged'] else '否', ''))
            if job['calculation']['task'] in ('opt', 'optfreq'):
                rows.append(('几何优化收敛', '是' if result.get('optimization_converged') else '否', ''))
            if result.get('frequencies_cm1'):
                rows.append(('频率数量', len(result['frequencies_cm1']), ''))
                rows.append(('显著虚频', len(result['imaginary_frequencies']), ''))
        except (OSError, ValueError, KeyError):
            message = job.get('message') or job.get('connection_warning') or '结果尚未同步。'
            rows.append(('消息', message, ''))
            self.result_view.setPlainText(message)
        populate(self.result_grid, rows)

    def poll(self):
        remote = self.remote
        if self.operation or self.server_operation or not remote:
            return
        jobs = [job for job in self.jobs.values() if self.needs_job_sync(job) and job.get('remote_dir')
                and all(remote.settings.get(key) == value for key, value in job['connection'].items())]
        if jobs:
            def sync():
                for job in jobs:
                    remote.sync(job, self.store.folder(job))

            self.run_server('', remote, sync, lambda _: self.reload_jobs(), quiet=True)

    def needs_job_sync(self, job):
        state = job['status']
        folder = self.store.folder(job)
        return state in ('queued', 'running') or (
            job.get('automation') and state in ('completed', 'failed', 'cancelled')
            and (not (folder / 'terminal-sync.json').is_file()
                 or (state == 'completed' and not (folder / 'result.json').is_file())))

    def refresh_job(self):
        if not self.server_ready():
            return
        self.reload_jobs()
        job = self.current_job()
        if not job or not job.get('remote_dir'):
            return
        remote = self.connection()
        if remote:
            self.run_server('正在同步任务…', remote,
                            lambda: remote.sync(job, self.store.folder(job)), self.synced)

    def synced(self, _):
        self.connection_status.setText('SSH 已连接')
        self.reload_jobs()
        self.status.setText('已同步服务器状态。')

    def cancel_job(self):
        if not self.server_ready():
            return
        job = self.current_job()
        if not job or job['status'] not in ('queued','running'):
            self.status.setText('请选择排队或运行中的任务。')
            return
        if QMessageBox.question(self, '取消服务器任务', f"取消 {job['name']}（{job['id']}）？") != QMessageBox.Yes:
            return
        if not self.server_ready():
            return
        remote = self.connection()
        if remote:
            self.run_server('正在请求取消…', remote, lambda: remote.cancel(job),
                            lambda _: self.status.setText('已发送取消请求，等待服务器更新状态。'))

    def retry_job(self):
        if not self.server_ready():
            return
        job = self.current_job()
        if not job or job['status'] != 'failed':
            self.status.setText('请选择失败的任务。')
            return
        if QMessageBox.question(self, '重试任务', f"使用原结构和计算参数重试 {job['name']}？") != QMessageBox.Yes:
            return
        if not self.server_ready():
            return
        remote = self.connection()
        if remote:
            def retry():
                remote.require_connection(job)
                return submit(remote, self.store, Structure.from_dict(job['structure']),
                              Calculation(**job['calculation']), retry_of=job['id'],
                              automation=job.get('automation'))
            self.run_server('正在重试服务器任务…', remote, retry, self.submitted)

    def open_job(self):
        if job := self.current_job():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.folder(job))))

    def explain_job(self):
        try:
            result = self.read_result(self.current_job())
        except (OSError, ValueError) as exc:
            self.show_error(exc)
            return
        self.tabs.setCurrentIndex(0)
        self.send('请解释这个 PySCF 任务的结果，指出收敛状态和可用于下一步分析的量。', result)

    def load_job_geometry(self):
        job = self.current_job()
        try:
            result = self.read_result(job)
            structure = result_geometry(job, result)
            calculation = Calculation(**job['calculation'])
            self.select_candidate(structure, focus_calculation=False)
            self.set_calculation(calculation)
            self.reload_jobs(job['id'])
            self.status.setText(f'已载入作业 {job["id"]} 的结果结构。可保存为项目或导出 XYZ。')
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.show_error(error)

    def closeEvent(self, event):
        manager = getattr(self, 'server_jobs_dialog', None)
        if self.operation or self.server_operation or (manager is not None and manager.operation):
            self.status.setText('当前操作结束后即可关闭。')
            event.ignore()
            return
        if self.remote:
            self.remote.close()
        if manager is not None:
            manager.shutdown()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['assistant', 'server'], default='assistant')
    parser.add_argument('--pdc-models', action='store_true')
    parser.add_argument('--server-jobs', action='store_true')
    args = parser.parse_args()
    app = QApplication([])
    app.setApplicationName('ChatDFT')
    app.setApplicationVersion('5.8.0')
    app.setOrganizationName('ChatDFT')
    from workbench_theme import apply_workbench_theme
    apply_workbench_theme(app)
    from desktop import InstanceServer, activate_existing
    if activate_existing(args):
        return 0
    from workbench import WorkbenchWindow
    window = WorkbenchWindow(args)
    window.instance_server = InstanceServer(window)
    window.show()
    if not (args.pdc_models or args.server_jobs):
        QTimer.singleShot(0, lambda: window.show_about(startup=True))
    if args.pdc_models:
        QTimer.singleShot(0, window.prepare_pdc_paper)
    if args.server_jobs:
        QTimer.singleShot(0, window.open_server_jobs)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
