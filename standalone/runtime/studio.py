from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QSizePolicy, QStackedWidget, QTabBar,
    QToolBar, QToolButton, QVBoxLayout, QWidget, QTableWidget,
)

from research import ResearchPanel
from chatdft.viewport import measurement
from theme import ChoiceBox, symbol_icon
from studio_theme import studio_icon


class StudioWindow(ResearchPanel):
    def __init__(self, args):
        self.shell_ready = False
        super().__init__(args)
        self.setWindowTitle('ChatDFT')
        self.setWindowIcon(studio_icon())
        self.setMinimumSize(1100, 720)
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1440, available.width() - 40), min(900, available.height() - 60))
        self.actions_by_name = {action.text(): action for action in self.toolbar.actions()
                                if not action.isSeparator()}
        self.toolbar.hide()
        self.scene_toolbar.hide()
        self.create_workspace_bar()
        self.route_view_commands()
        self.create_command_ribbon()
        self.create_document_tools()
        self.documents.subWindowActivated.connect(self.document_activated)
        self.project_dock.setWindowTitle('研究项目')
        self.inspector.setWindowTitle('属性与参数')
        self.ai_dock.setWindowTitle('命令助手')
        self.ai_dock.hide()
        self.activity_dock.setWindowTitle('作业与输出')
        self.resizeDocks([self.project_dock, self.inspector], [220, 285], Qt.Horizontal)
        self.resizeDocks([self.activity_dock], [165], Qt.Vertical)
        self.preview.background_colors = ('#282e38', '#424b58')
        self.comparison.background_colors = ('#edf0f4', '#c5ccd7')
        self.preview.ball_scale = self.comparison.ball_scale = 1.04
        self.preview.stick_scale = self.comparison.stick_scale = .95
        self.preview.zoom = self.comparison.zoom = 1.02
        for table_widget in self.findChildren(QTableWidget):
            table_widget.setShowGrid(False)
            table_widget.verticalHeader().setDefaultSectionSize(23)
        self.inspector_table.setAlternatingRowColors(False)
        self.project.setIndentation(16)
        self.project_root.setText(0, '未命名项目')
        self.project.collapseItem(self.candidate_node)
        self.compute_input = QLabel()
        self.compute_input.setObjectName('ConnectionLabel')
        self.tabs.widget(1).layout().insertWidget(0, self.compute_input)
        self.compute_input.setText('计算输入 · ' + self.document_title.text())
        self.tabs.setTabText(0, '属性')
        self.tabs.setTabText(1, '计算')
        self.tabs.setTabText(2, '结果')
        self.shell_ready = True
        self.tabs.currentChanged.connect(self.inspector_tab_changed)
        self._changing_workspace = False
        phase = 1 if args.mode == 'server' else 0
        self.workspace_tabs.setCurrentIndex(phase)
        self.change_workspace(phase)
        QTimer.singleShot(0, lambda: self.change_workspace(phase))
        self.documents.setCurrentIndex(0)
        self.document_activated(self.primary_window)

    def dock(self, title, name, widget, area):
        dock = super().dock(title, name, widget, area)
        header = dock.titleBarWidget()
        label = header.findChild(QLabel, 'DockCaption')
        if label:
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return dock

    def create_workspace_bar(self):
        self.workspace_bar = QToolBar('工作区', self)
        self.workspace_bar.setObjectName('WorkspaceBar')
        self.workspace_bar.setMovable(False)
        self.addToolBar(self.workspace_bar)
        wordmark = QLabel('ChatDFT')
        wordmark.setObjectName('ApplicationWordmark')
        self.workspace_bar.addWidget(wordmark)
        self.workspace_tabs = QTabBar()
        self.workspace_tabs.setObjectName('WorkspaceTabs')
        self.workspace_tabs.setExpanding(False)
        for title in ('建模', '计算', '分析'):
            self.workspace_tabs.addTab(title)
        self.workspace_bar.addWidget(self.workspace_tabs)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.workspace_bar.addWidget(spacer)
        self.layout_choice = ChoiceBox()
        self.layout_choice.addItems(['双结构视图', '四窗口平铺'])
        self.layout_choice.setMinimumWidth(110)
        self.workspace_bar.addWidget(self.layout_choice)
        self.layout_choice.currentIndexChanged.connect(self.change_layout)
        self.documents.layoutChanged.connect(self.reflect_layout)
        self.workspace_tabs.currentChanged.connect(self.change_workspace)
        self.addToolBarBreak()

    def command(self, title, symbol, callback):
        action = QAction(symbol_icon(symbol), title, self)
        action.triggered.connect(callback)
        return action

    def command_group(self, parent, title, actions):
        box = QWidget()
        box.setObjectName('CommandGroup')
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 4, 10, 3)
        layout.setSpacing(1)
        buttons = QHBoxLayout()
        buttons.setSpacing(2)
        for action in actions:
            button = QToolButton()
            button.setObjectName('RibbonButton')
            button.setIconSize(QSize(20, 20))
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setDefaultAction(action)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        label = QLabel(title)
        label.setObjectName('CommandGroupTitle')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        parent.addWidget(box)

    def create_command_ribbon(self):
        self.ribbon = QToolBar('工作区命令', self)
        self.ribbon.setObjectName('CommandRibbon')
        self.ribbon.setMovable(False)
        self.addToolBar(self.ribbon)
        self.command_pages = QStackedWidget()
        self.command_pages.setFixedHeight(70)
        self.ribbon.addWidget(self.command_pages)
        self.actions_new = {
            '建模': self.command('SMILES', 'model', self.build_smiles),
            '原子': self.command('坐标编辑', 'atoms', self.edit_atoms),
            '优化': self.command('几何优化', 'optimize', self.setup_optimization),
            '参考': self.command('固定对照', 'model', self.pin_reference),
            '设置': self.command('计算设置', 'settings', lambda: self.set_phase(1)),
            '提交': self.command('提交任务', 'send', self.submit_job),
            '连接': self.command('测试连接', 'external', self.test_connection),
            '同步': self.command('同步状态', 'tasks', self.refresh_job),
            '取消': self.command('取消任务', 'tasks', self.cancel_job),
            '目录': self.command('任务目录', 'folder', self.open_job),
            '图像': self.command('结构图像', 'fit', self.export_image),
            '表格': self.command('结果表格', 'tasks', self.export_csv),
            '报告': self.command('计算报告', 'file', self.export_report),
            '曲线': self.command('导入曲线', 'open', self.import_curve),
            '出图': self.command('导出曲线', 'save', self.export_curve_image),
            '助手': self.command('命令助手', 'assistant', self.open_assistant),
            '解释': self.command('解读结果', 'assistant', self.explain_job),
        }
        files = [self.actions_by_name[key] for key in ('打开项目', '导入', '保存')]
        definitions = [
            [('文件', files), ('结构构建', [self.actions_new[k] for k in ('建模', '原子')]),
             ('主模型处理', [self.actions_new[k] for k in ('优化', '参考')]),
             ('计算', [self.actions_new['设置']]),
             ('助手', [self.actions_new['助手']])],
            [('任务', [self.actions_new[k] for k in ('设置', '提交')]),
             ('服务器', [self.actions_new[k] for k in ('连接', '同步')]),
             ('作业控制', [self.actions_new[k] for k in ('取消', '目录')]),
             ('结果交付', [self.actions_new[k] for k in ('表格', '报告')]),
             ('助手', [self.actions_new['解释']])],
            [('分析数据', [self.actions_new[k] for k in ('曲线', '出图')]),
             ('导出', [self.actions_new[k] for k in ('图像', '表格', '报告')]),
             ('比较', [self.actions_new['参考']]),
             ('解释', [self.actions_new['解释']])],
        ]
        for groups in definitions:
            page = QWidget()
            page.setObjectName('CommandPage')
            layout = QHBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            for title, actions in groups:
                self.command_group(layout, title, actions)
            layout.addStretch()
            self.command_pages.addWidget(page)

    def active_view(self):
        return self.comparison if self.documents.activeSubWindow() is self.comparison_window else self.preview

    def route_view_commands(self):
        for key, action in self.view_modes.items():
            action.triggered.disconnect()
            action.triggered.connect(lambda checked=False, mode=key: self.activate_view_mode(mode))
        self.fit_action.triggered.disconnect()
        self.fit_action.triggered.connect(lambda: self.active_view().fit())
        self.display_style.currentIndexChanged.disconnect()
        self.display_style.currentIndexChanged.connect(
            lambda index: self.active_view().set_style(['ballstick', 'stick', 'line', 'cpk'][index]))
        self.labels_action.toggled.disconnect()
        self.labels_action.toggled.connect(lambda checked: self.active_view_flag('labels', checked))
        self.grid_action.toggled.disconnect()
        self.grid_action.toggled.connect(lambda checked: self.active_view_flag('grid', checked))
        self.comparison.selectionChanged.connect(self.comparison_selection)
        self.comparison.selected.connect(self.status.setText)

    def activate_view_mode(self, mode):
        view = self.active_view()
        if mode in ('distance', 'angle'):
            view.clear_selection()
            view.measure_count = 2 if mode == 'distance' else 3
            view.set_mode('measure')
        else:
            view.set_mode(mode)

    def active_view_flag(self, key, checked):
        view = self.active_view()
        setattr(view, key, checked)
        view.update()

    def comparison_selection(self, indices):
        if self.documents.activeSubWindow() is self.comparison_window:
            self.refresh_inspector()
            if len(indices) >= 2 and self.comparison.structure:
                self.status.setText('对照模型 · ' + measurement(self.comparison.structure.atoms, indices))

    def create_document_tools(self):
        # Each view owns a compact contextual toolbar.
        self.view_toolbars = []
        for subwindow in (self.primary_window, self.comparison_window):
            container = subwindow.widget()
            panel = container.layout().itemAt(1).widget()
            layout = panel.layout()
            if subwindow is self.comparison_window:
                old = layout.takeAt(0).widget()
                old.hide()
            tools = QToolBar('显示与测量')
            tools.setIconSize(QSize(15, 15))
            tools.setToolButtonStyle(Qt.ToolButtonIconOnly)
            for key in ('rotate', 'pan', 'select'):
                tools.addAction(self.view_modes[key])
            tools.addSeparator()
            tools.addAction(self.fit_action)
            for key in ('distance', 'angle'):
                tools.addAction(self.view_modes[key])
            tools.addSeparator()
            tools.addAction(self.labels_action)
            tools.addAction(self.grid_action)
            layout.insertWidget(0, tools)
            self.view_toolbars.append(tools)
            if subwindow is self.comparison_window:
                tools.addSeparator()
                action = tools.addAction(symbol_icon('open'), '载入对照结构')
                action.triggered.connect(self.import_reference)
                action = tools.addAction(symbol_icon('send'), '设为主模型')
                action.triggered.connect(self.use_reference)

    def set_phase(self, phase):
        self.workspace_tabs.setCurrentIndex(phase)
        self.change_workspace(phase)

    def change_workspace(self, index):
        if not self.shell_ready:
            return
        self._changing_workspace = True
        self.command_pages.setCurrentIndex(index)
        self.ai_dock.hide()
        self.tabs.setCurrentIndex((0, 1, 2)[index])
        self.activity_dock.setVisible(index != 0)
        if index == 0:
            self.documents.focus_models()
        else:
            self.documents.tile_all()
        self.layout_choice.blockSignals(True)
        self.layout_choice.setCurrentIndex(0 if index == 0 else 1)
        self.layout_choice.blockSignals(False)
        self.status.setText(('建模工作区', '计算工作区', '数据分析工作区')[index])
        self._changing_workspace = False

    def change_layout(self, index):
        self.documents.focus_models() if index == 0 else self.documents.tile_all()

    def reflect_layout(self, mode):
        self.layout_choice.blockSignals(True)
        self.layout_choice.setCurrentIndex(0 if mode == 'models' else 1)
        self.layout_choice.blockSignals(False)

    def inspector_tab_changed(self, index):
        if self.shell_ready and not self._changing_workspace and index in (1, 2):
            self.set_phase(index)

    def document_activated(self, window):
        for subwindow in self.documents.windows:
            title = subwindow.widget().findChild(QWidget, 'MdiTitle')
            if title:
                title.setProperty('active', subwindow is window)
                title.style().unpolish(title)
                title.style().polish(title)
                title.update()
        view = self.active_view()
        for action in (*self.view_modes.values(), self.fit_action, self.labels_action, self.grid_action):
            action.setEnabled(view.structure is not None)
        self.display_style.setEnabled(view.structure is not None)
        styles = ['ballstick', 'stick', 'line', 'cpk']
        self.display_style.blockSignals(True)
        self.display_style.setCurrentIndex(styles.index(view.style))
        self.display_style.blockSignals(False)
        for action, flag in ((self.labels_action, 'labels'), (self.grid_action, 'grid')):
            action.blockSignals(True)
            action.setChecked(getattr(view, flag))
            action.blockSignals(False)
        mode = view.mode
        key = 'distance' if mode == 'measure' and view.measure_count == 2 else (
            'angle' if mode == 'measure' else mode)
        if key in self.view_modes:
            self.view_modes[key].setChecked(True)
        self.refresh_inspector()
        self.update_command_states()

    def update_command_states(self):
        if not self.shell_ready:
            return
        job = self.current_job()
        for key in ('同步', '目录', '表格'):
            self.actions_new[key].setEnabled(job is not None)
        self.actions_new['取消'].setEnabled(bool(job and job['status'] in ('queued','running')))
        self.actions_new['报告'].setEnabled(self.explain_button.isEnabled())
        self.actions_new['解释'].setEnabled(self.explain_button.isEnabled())
        for key in ('原子', '优化', '参考', '提交', '图像'):
            self.actions_new[key].setEnabled(self.submit_button.isEnabled())

    def open_model(self):
        self.set_phase(0)
        self.documents.setCurrentIndex(0)

    def open_tasks(self):
        self.set_phase(1)

    def open_assistant(self):
        self.ai_dock.show()
        self.ai_dock.raise_()

    def refresh_inspector(self):
        if not getattr(self, 'shell_ready', False):
            return super().refresh_inspector()
        view = self.active_view()
        if view is self.preview:
            return super().refresh_inspector()
        structure = view.structure
        self.inspector_table.clear()
        if not structure:
            return
        self.inspector_group('对照文档', [
            ('名称', structure.name), ('分子式', structure.formula()),
            ('原子数', len(structure.atoms)), ('总电荷', structure.charge),
            ('自旋多重度', structure.multiplicity),
        ])
        if view.picks:
            index = view.picks[-1]
            atom = structure.atoms[index]
            self.inspector_group('选中原子', [
                ('编号', index + 1), ('元素', atom['element']),
                *[(key.upper() + ' / Å', f'{atom[key]:.6f}') for key in 'xyz'],
            ])

    def show_source(self):
        super().show_source()
        if getattr(self, 'shell_ready', False):
            self.compute_input.setText('计算输入 · ' + self.document_title.text())
            self.update_command_states()

    def show_job(self):
        super().show_job()
        self.update_command_states()

    def inspector_group(self, name, rows):
        super().inspector_group(name, rows)
        group = self.inspector_table.topLevelItem(self.inspector_table.topLevelItemCount() - 1)
        for column in (0, 1):
            group.setBackground(column, QColor('#eef2f7'))
            group.setForeground(column, QColor('#65758c'))

    def run(self, text, function, done, quiet=False):
        if getattr(self, 'shell_ready', False) and not self.operation:
            self.ribbon.setEnabled(False)
            self.workspace_bar.setEnabled(False)
        super().run(text, function, done, quiet)

    def finish_operation(self):
        super().finish_operation()
        self.ribbon.setEnabled(True)
        self.workspace_bar.setEnabled(True)

    def submitted(self, job):
        super().submitted(job)
        self.set_phase(1)

    def explain_job(self):
        self.open_assistant()
        super().explain_job()

