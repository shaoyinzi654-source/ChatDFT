from __future__ import annotations

from pathlib import Path
from collections import Counter
import datetime

from PySide6.QtCore import QTimer, Qt, QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QStackedWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QToolBar, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from panel import Panel, label, populate, row, table
from chatdft.chemistry import Structure, validate_structure
from chatdft.structure_io import read_structure
from chatdft.viewport import MoleculeView
from research_widgets import DataPlot, EnergyPlot, ResearchArea, read_plot_csv
from theme import symbol_icon


class ResearchPanel(Panel):
    def __init__(self, args):
        self.reference_structure = None
        self.reference_pinned = False
        self.curve_user_loaded = False
        self.last_model_key = None
        super().__init__(args)
        self.setWindowTitle('ChatDFT — 多文档研究工作台')
        self.setMinimumSize(1100, 700)
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1440, available.width() - 40), min(900, available.height() - 60))
        self.tabs.setMinimumWidth(255)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        for action in self.toolbar.actions():
            widget = self.toolbar.widgetForAction(action)
            if widget and widget.objectName() == 'Brand':
                self.toolbar.removeAction(action)
        self.add_research_commands()
        self.arrange_command_bar()
        self.split_ai_inspector()
        self.attach_analysis_windows()
        self.project_dock.setWindowTitle('项目管理器')
        self.inspector.setWindowTitle('文档检查器')
        self.activity_dock.setWindowTitle('作业监视器')
        self.job_summary_plot.hide()
        self.resizeDocks([self.project_dock, self.inspector], [220, 278], Qt.Horizontal)
        self.resizeDocks([self.activity_dock], [138], Qt.Vertical)
        self.resizeDocks([self.inspector, self.ai_dock], [390, 220], Qt.Vertical)
        for table_widget in self.findChildren(QTableWidget):
            table_widget.verticalHeader().setDefaultSectionSize(20)
            table_widget.setShowGrid(True)
        self.job_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.job_table.setColumnWidth(0, 220)
        self.job_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.scene_toolbar.setIconSize(QSize(16, 16))
        self.preview.ball_scale = self.comparison.ball_scale = 1.12
        self.preview.stick_scale = self.comparison.stick_scale = 1.1
        self.preview.fit_padding = self.comparison.fit_padding = (64, 75)
        self.show_source()
        self.activity_dock.show()
        self.ai_dock.show()
        self.refresh_inspector()
        QTimer.singleShot(0, self.documents.tile_all)

    def dock(self, title, name, widget, area):
        dock = super().dock(title, name, widget, area)
        header = QWidget()
        header.setObjectName('DockTitle')
        bar = QHBoxLayout(header)
        bar.setContentsMargins(5, 1, 2, 1)
        bar.setSpacing(3)
        caption = QLabel(title)
        caption.setObjectName('DockCaption')
        bar.addWidget(caption, 1)
        dock.windowTitleChanged.connect(caption.setText)
        for text, tooltip, callback in (
                ('↗', '浮动 / 停靠', lambda: dock.setFloating(not dock.isFloating())),
                ('×', '隐藏面板', dock.hide)):
            button = QPushButton(text)
            button.setObjectName('WindowControl')
            button.setFixedSize(18, 18)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            bar.addWidget(button)
        dock.setTitleBarWidget(header)
        return dock

    def create_navigation(self):
        pass

    def arrange_command_bar(self):
        actions = {action.text(): action for action in self.toolbar.actions() if not action.isSeparator()}
        style_action = next((action for action in self.toolbar.actions()
                             if self.toolbar.widgetForAction(action) is self.display_style), None)
        for action in list(self.toolbar.actions()):
            self.toolbar.removeAction(action)
        action = self.toolbar.addAction(symbol_icon('open'), '打开项目')
        action.triggered.connect(self.open_project)
        groups = [
            ['导入', '保存'], ['撤销', '重做'], ['分子', '原子坐标', '几何优化'],
            ['单点能', '频率'], ['导入曲线'], ['AI 助手'],
        ]
        for index, group in enumerate(groups):
            if index:
                self.toolbar.addSeparator()
            for title in group:
                self.toolbar.addAction(actions[title])
        self.scene_toolbar.addSeparator()
        self.scene_toolbar.addAction(self.fit_action)
        if style_action:
            self.scene_toolbar.addAction(style_action)

    def sync_navigation(self):
        pass

    def open_model(self):
        self.documents.setCurrentIndex(0)
        self.tabs.setCurrentIndex(0)

    def open_assistant(self):
        self.ai_dock.show()
        self.ai_dock.raise_()

    def open_tasks(self):
        self.activity_dock.show()
        self.tabs.setCurrentIndex(2)

    def project_pane(self):
        pane = super().project_pane()
        browser = QWidget()
        layout = QVBoxLayout(browser)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        search = QLineEdit()
        search.setPlaceholderText('搜索模型、作业或结果…')
        search.setClearButtonEnabled(True)
        search.textChanged.connect(self.filter_project)
        layout.addWidget(search)
        layout.addWidget(self.project)
        pane.insertWidget(0, browser)
        self.project_search = search
        for item in (self.structure_node, self.candidate_node, self.job_node):
            self.project.takeTopLevelItem(self.project.indexOfTopLevelItem(item))
        self.project_root = QTreeWidgetItem(self.project, ['研究项目'])
        self.project_root.setIcon(0, symbol_icon('folder'))
        models = QTreeWidgetItem(self.project_root, ['模型'])
        models.setIcon(0, symbol_icon('folder'))
        models.addChild(self.structure_node)
        self.project_root.addChild(self.candidate_node)
        self.job_node.setText(0, '计算作业')
        self.project_root.addChild(self.job_node)
        results = QTreeWidgetItem(self.project_root, ['分析结果'])
        results.setIcon(0, symbol_icon('folder'))
        for title, index in (('能量收敛', 2), ('数据曲线', 3), ('原子坐标', 4)):
            item = QTreeWidgetItem(results, [title])
            item.setData(0, Qt.UserRole, ('document', index))
            item.setIcon(0, symbol_icon('file'))
        self.project.expandAll()
        return pane

    def filter_project(self, query):
        query = query.casefold().strip()

        def matches(item):
            child_visible = False
            for i in range(item.childCount()):
                child_visible = matches(item.child(i)) or child_visible
            visible = not query or query in item.text(0).casefold() or child_visible
            item.setHidden(not visible)
            return visible

        for index in range(self.project.topLevelItemCount()):
            matches(self.project.topLevelItem(index))

    def activate_project_item(self, item, column):
        data = item.data(0, Qt.UserRole)
        if isinstance(data, (tuple, list)) and data[0] == 'document':
            self.documents.setCurrentIndex(data[1])
        elif item.parent() is self.candidate_node:
            self.open_assistant()
            self.ai_tabs.setCurrentIndex(1)
            self.candidates.selectRow(self.candidate_node.indexOfChild(item))
        elif item.parent() is self.job_node:
            self.open_tasks()
            self.job_table.selectRow(self.job_node.indexOfChild(item))
        elif item is self.structure_node:
            self.documents.setCurrentIndex(0)

    def create_viewport(self):
        self.documents = ResearchArea()
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.document_title = QLabel('当前结构')
        self.document_title.hide()
        self.preview_stack = QStackedWidget()
        self.preview = MoleculeView()
        self.preview.setMinimumSize(160, 120)
        self.preview.light_background = False
        self.preview.background_colors = ('#151820', '#303642')
        self.preview.selectionChanged.connect(self.show_measurement)
        self.preview_stack.addWidget(self.preview)
        self.preview_empty = QWidget()
        empty = QVBoxLayout(self.preview_empty)
        self.preview_note = label('导入 XYZ / XSD，或从分子库创建结构。')
        self.preview_note.setAlignment(Qt.AlignCenter)
        empty.addStretch()
        empty.addWidget(self.preview_note)
        row(empty, ('导入结构…', self.import_file), ('SMILES 建模…', self.build_smiles), centered=True)
        empty.addStretch()
        self.preview_stack.addWidget(self.preview_empty)
        self.add_scene_tools(layout)
        layout.addWidget(self.preview_stack)
        from theme import ElementLegend
        from chatdft.viewport import COLORS
        self.element_legend = ElementLegend(COLORS)
        self.element_legend.setParent(page)
        self.element_legend.hide()
        self.primary_window = self.documents.add_document(page, '当前结构')
        reference = QWidget()
        v = QVBoxLayout(reference)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        tools = QToolBar()
        tools.setIconSize(QSize(15, 15))
        for text, icon, callback in (
                ('载入', 'open', self.import_reference), ('固定当前结构', 'model', self.pin_reference),
                ('用于计算', 'send', self.use_reference)):
            action = tools.addAction(symbol_icon(icon), text)
            action.triggered.connect(callback)
        v.addWidget(tools)
        self.comparison = MoleculeView()
        self.comparison.setMinimumSize(160, 120)
        self.comparison.light_background = True
        self.comparison.background_colors = ('#e2e3e9', '#b9bdca')
        v.addWidget(self.comparison, 1)
        self.comparison_window = self.documents.add_document(reference, '结构对照 · 未载入')
        energy = QWidget()
        self.energy_layout = QVBoxLayout(energy)
        self.energy_layout.setContentsMargins(0, 0, 0, 0)
        self.energy_window = self.documents.add_document(energy, '能量收敛 · 等待计算结果')
        data = QWidget()
        v = QVBoxLayout(data)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        tools = QToolBar()
        action = tools.addAction(symbol_icon('open'), '导入 CSV')
        action.triggered.connect(self.import_curve)
        action = tools.addAction(symbol_icon('save'), '导出图像')
        action.triggered.connect(self.export_curve_image)
        v.addWidget(tools)
        self.data_plot = DataPlot()
        v.addWidget(self.data_plot, 1)
        self.curve_window = self.documents.add_document(data, '数据分析 · 外部曲线')
        self.atom_table = table(['原子', '元素', 'X / Å', 'Y / Å', 'Z / Å'], 2)
        self.documents.add_document(self.atom_table, '原子坐标').hide()
        self.documents.setCurrentIndex(0)
        self.setCentralWidget(self.documents)

    def add_scene_tools(self, layout):
        super().add_scene_tools(layout)
        tools = layout.itemAt(layout.count() - 1).widget()
        layout.removeWidget(tools)
        self.addToolBarBreak()
        self.addToolBar(tools)
        self.scene_toolbar = tools
        tools.addSeparator()
        self.element_buttons = {}
        from chatdft.viewport import COLORS
        for element in ('H', 'C', 'N', 'O', 'S'):
            button = QToolButton()
            button.setObjectName('ElementButton')
            button.setText(element)
            button.setToolTip('选择所有 ' + element + ' 原子')
            color = QColor(COLORS[element]).lighter(170).name()
            button.setStyleSheet('QToolButton { background: '+color+'; color: #283044; border: 1px solid #b6bdcc; }')
            button.clicked.connect(lambda checked=False, e=element: self.select_element(e))
            tools.addWidget(button)
            self.element_buttons[element] = button

    def select_element(self, element):
        if self.preview.structure:
            self.preview.select_indices([i for i, atom in enumerate(self.preview.structure.atoms)
                                         if atom['element'] == element])

    def split_ai_inspector(self):
        ai_page = self.tabs.widget(0)
        self.tabs.removeTab(0)
        self.inspector_table = QTreeWidget()
        self.inspector_table.setColumnCount(2)
        self.inspector_table.setHeaderLabels(['属性', '值'])
        self.inspector_table.setAlternatingRowColors(True)
        self.inspector_table.setRootIsDecorated(True)
        self.inspector_table.setIndentation(12)
        self.inspector_table.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.inspector_table.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tabs.insertTab(0, self.inspector_table, '属性')
        self.tabs.setTabText(1, '计算')
        self.tabs.setCurrentIndex(0)
        ai_tabs = QTabWidget()
        chat = QWidget()
        layout = QVBoxLayout(chat)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        layout.addWidget(self.answer, 1)
        self.prompt.setFixedHeight(40)
        layout.addWidget(self.prompt)
        actions = QHBoxLayout()
        actions.addWidget(self.send_button)
        actions.addWidget(self.suggestion_button)
        actions.addStretch()
        layout.addLayout(actions)
        ai_tabs.addTab(chat, '操作记录')
        candidates = QWidget()
        layout = QVBoxLayout(candidates)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.candidates)
        actions = QHBoxLayout()
        actions.addWidget(self.candidate_button)
        layout.addLayout(actions)
        ai_tabs.addTab(candidates, '候选结构')
        ai_page.deleteLater()
        self.ai_tabs = ai_tabs
        self.ai_dock = self.dock('AI 操作记录', 'AiDock', ai_tabs, Qt.RightDockWidgetArea)
        self.splitDockWidget(self.inspector, self.ai_dock, Qt.Vertical)
        self.window_menu.addAction(self.ai_dock.toggleViewAction())

    def attach_analysis_windows(self):
        index = self.activity.indexOf(self.energy_plot)
        if index >= 0:
            self.activity.removeTab(index)
        old_plot = self.energy_plot
        self.energy_plot = EnergyPlot()
        self.energy_layout.addWidget(self.energy_plot)
        self.energy_plot.set_data(old_plot.series, old_plot.caption)
        old_plot.deleteLater()
        self.energy_plot.setMinimumSize(160, 110)
        self.energy_plot.show()
        self.activity.setTabText(0, '作业监视器')

    def add_research_commands(self):
        self.toolbar.addSeparator()
        for text, icon, callback in (
                ('分子', 'model', self.build_smiles), ('原子坐标', 'atoms', self.edit_atoms),
                ('几何优化', 'optimize', self.setup_optimization),
                ('单点能', 'tasks', lambda: self.choose_task('单点能')),
                ('频率', 'angle', lambda: self.choose_task('振动频率')),
                ('导入曲线', 'open', self.import_curve), ('AI 助手', 'assistant', self.open_assistant)):
            action = self.toolbar.addAction(symbol_icon(icon), text)
            action.triggered.connect(callback)
        windows = self.menuBar().addMenu('窗口')
        self.window_menu = windows
        self.menu_action(windows, '四窗口平铺', self.documents.tile_all)
        self.menu_action(windows, '层叠窗口', self.documents.cascadeSubWindows)
        self.menu_action(windows, '结构对照', lambda: self.documents.setCurrentIndex(1))
        self.menu_action(windows, '原子坐标', lambda: self.documents.setCurrentIndex(4))
        for dock in (self.project_dock, self.inspector, self.activity_dock):
            windows.addAction(dock.toggleViewAction())
        analysis = self.menuBar().addMenu('数据分析')
        self.menu_action(analysis, '导入 CSV 曲线…', self.import_curve)
        self.menu_action(analysis, '导出曲线图像…', self.export_curve_image)
        modules = analysis.addMenu('扩展计算模块')
        for title in ('DOS 计算 · 尚未接入', '能带计算 · 尚未接入', 'NEB · 尚未接入', '分子动力学 · 尚未接入'):
            action = modules.addAction(title)
            action.setEnabled(False)

    def choose_task(self, title):
        self.task.setCurrentText(title)
        self.tabs.setCurrentIndex(1)

    def pin_reference(self):
        try:
            self.set_reference(self.structure())
        except (OSError, ValueError) as exc:
            self.show_error(exc)

    def set_reference(self, structure):
        validate_structure(structure)
        self.reference_pinned = True
        self.update_reference(structure)
        self.comparison_window.show()
        self.status.setText('已固定独立对照结构。')
        self.record_action('固定对照结构', structure.name)

    def update_reference(self, structure):
        if structure != self.reference_structure:
            self.reference_structure = Structure.from_dict(structure.to_dict()) if structure else None
            self.comparison.set_structure(self.reference_structure)
        mode = '固定' if self.reference_pinned else '跟随主模型'
        name = structure.name if structure else '未载入'
        self.comparison_window.setWindowTitle(f'结构对照（{mode}） · {name}')

    def import_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, '载入对照结构', '', '结构 (*.xyz *.xsd)')
        if path:
            try:
                self.set_reference(read_structure(Path(path), 0, 1))
            except (OSError, ValueError) as exc:
                self.show_error(exc)

    def reset_reference(self):
        self.reference_pinned = False
        self.update_reference(self.preview.structure)
        self.refresh_inspector()
        self.status.setText('结构对照已恢复跟随主模型。')

    def use_reference(self):
        if self.reference_structure:
            self.select_candidate(Structure.from_dict(self.reference_structure.to_dict()))

    def import_curve(self):
        path, _ = QFileDialog.getOpenFileName(self, '导入曲线', '', 'CSV 曲线 (*.csv)')
        if path:
            try:
                headers, rows = read_plot_csv(path)
                self.curve_user_loaded = True
                self.data_plot.set_data(headers, rows, Path(path).name)
                self.curve_window.setWindowTitle('数据分析 · ' + Path(path).name)
                self.curve_window.show()
                self.status.setText(f'已导入 {len(rows)} 个数据点，{len(headers)-1} 条曲线。')
            except (OSError, ValueError, UnicodeError) as exc:
                self.show_error(exc)

    def export_curve_image(self):
        path, _ = QFileDialog.getSaveFileName(self, '导出曲线图像', 'curve.png', 'PNG (*.png)')
        if path and not self.data_plot.grab().save(path, 'PNG'):
            self.show_error('无法保存曲线图像。')

    def refresh_inspector(self):
        if not hasattr(self, 'inspector_table'):
            return
        self.inspector_table.clear()
        try:
            structure = self.structure()
            from rdkit import Chem
            masses = Chem.GetPeriodicTable()
            molecular_mass = sum(masses.GetAtomicWeight(atom['element']) for atom in structure.atoms)
            self.inspector_group('分子属性', [
                ('文档', structure.name), ('分子式', structure.formula()),
                ('原子数', len(structure.atoms)), ('化学键数', len(structure.bonds)),
                ('分子 / 离子片段', structure.component_count()),
                ('分子质量 / u', f'{molecular_mass:.3f}'),
                ('总电荷', structure.charge), ('自旋多重度', structure.multiplicity),
            ])
            self.inspector_group('坐标范围', [
                (key.upper() + ' / Å', f"{min(a[key] for a in structure.atoms):.3f} … {max(a[key] for a in structure.atoms):.3f}")
                for key in 'xyz'
            ])
            if structure.components and structure.component_count() > 1:
                counts = Counter((component['name'], component['charge']) for component in structure.components)
                self.inspector_group('组分清单', [(name, f'{count} 个 · 每个电荷 {charge:+d}')
                                               for (name, charge), count in counts.items()])
        except (OSError, ValueError):
            pass
        if self.preview.structure and self.preview.picks:
            index = self.preview.picks[-1]
            atom = self.preview.structure.atoms[index]
            self.inspector_group('选中原子', [
                ('数量', len(self.preview.picks)), ('原子编号', index + 1), ('元素', atom['element']),
                *[(key.upper() + ' / Å', f'{atom[key]:.6f}') for key in 'xyz'],
            ])

    def inspector_group(self, name, rows):
        group = QTreeWidgetItem(self.inspector_table, [name, ''])
        for column in (0, 1):
            group.setBackground(column, QColor('#e0e4f1'))
            group.setForeground(column, QColor('#3c4771'))
        for title, value in rows:
            item = QTreeWidgetItem(group, [title, str(value)])
            item.setToolTip(1, str(value))
        group.setExpanded(True)

    def show_measurement(self, indices):
        super().show_measurement(indices)
        self.refresh_inspector()

    def show_source(self):
        previous = self.preview.structure if hasattr(self, 'preview') else None
        previous_key = previous.to_dict() if previous else None
        camera = (self.preview.yaw, self.preview.pitch, self.preview.zoom, self.preview.pan) if previous else None
        super().show_source()
        self.element_legend.hide()
        self.primary_window.setWindowTitle('结构模型 · ' + self.document_title.text())
        self.preview.zoom = 1.1
        if self.preview.structure:
            key = self.preview.structure.to_dict()
            if camera and key == previous_key:
                self.preview.yaw, self.preview.pitch, self.preview.zoom, self.preview.pan = camera
            if key != self.last_model_key:
                self.last_model_key = key
                self.record_action('载入结构', self.preview.structure.name +
                                   f' · {len(self.preview.structure.atoms)} 原子')
        elements = {atom['element'] for atom in self.preview.structure.atoms} if self.preview.structure else set()
        for element, button in self.element_buttons.items():
            button.setEnabled(element in elements)
        if not self.reference_pinned:
            self.update_reference(self.preview.structure)
        self.refresh_inspector()

    def record_action(self, title, detail):
        stamp = datetime.datetime.now().strftime('%H:%M:%S')
        self.answer.appendPlainText(f'[{stamp}] {title}\n{detail}\n')

    def show_job(self):
        super().show_job()
        job = self.current_job()
        self.energy_window.setWindowTitle('SCF 能量 · ' + (job['name'] if job else '等待计算结果'))
        if not self.curve_user_loaded:
            self.data_plot.set_data([], [], '')
            self.curve_window.setWindowTitle('数据分析 · 外部曲线')
            if job:
                try:
                    charges = self.read_result(job).get('mulliken_charges', [])
                    if len(charges) >= 2:
                        rows = [[item['index'], item['charge']] for item in charges]
                        self.data_plot.set_data(
                            ['原子', 'Mulliken 电荷 / e'], rows, job['name'], kind='bar',
                            categories=[item['element'] + str(item['index']) for item in charges])
                        self.curve_window.setWindowTitle('原子电荷 · ' + job['name'])
                except (OSError, ValueError, KeyError):
                    pass

    def send(self, prompt=None, result=None):
        self.ai_dock.show()
        super().send(prompt, result)

    def run(self, text, function, done, quiet=False):
        if self.operation:
            return
        self.scene_toolbar.setEnabled(False)
        if hasattr(self, 'ai_dock'):
            self.ai_dock.widget().setEnabled(False)
        super().run(text, function, done, quiet)

    def finish_operation(self):
        super().finish_operation()
        self.scene_toolbar.setEnabled(True)
        self.ai_dock.widget().setEnabled(True)

    def reload_jobs(self, selected=None):
        super().reload_jobs(selected)
        self.job_table.setColumnCount(6)
        self.job_table.setHorizontalHeaderLabels(['模型', '状态', '方法 / 基组', '提交时间', '电子能 / Eh', '作业编号'])
        for row_index, job in enumerate(self.jobs.values()):
            energy = '—'
            try:
                result = self.read_result(job)
                if result.get('final_energy_hartree') is not None:
                    energy = f"{result['final_energy_hartree']:.8f}"
            except (OSError, ValueError):
                pass
            self.job_table.setItem(row_index, 4, QTableWidgetItem(energy))
            self.job_table.setItem(row_index, 5, QTableWidgetItem(job['id']))
        self.filter_project(self.project_search.text())

