from pathlib import Path
import json

from PySide6.QtCore import QByteArray, QEvent, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QDesktopServices, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QHeaderView, QLabel, QTabBar, QToolBar,
    QTableWidget, QToolButton, QMessageBox, QSizePolicy, QVBoxLayout, QWidget,
)

from studio import StudioWindow
from workbench_theme import tool_icon
from chatdft.storage import data_dir, atomic_json
from workbench_docking import install_docking
from workbench_ai import WorkbenchAI


class WorkbenchWindow(WorkbenchAI, StudioWindow):
    """Scientific Qt workbench using real ADS containers and QtAwesome icons."""
    def __init__(self, args):
        self.docking_ready = False
        super().__init__(args)
        self.shell_ready = False
        install_docking(self)
        self.docking_ready = True
        self.shell_ready = True
        self.setWindowTitle('ChatDFT — 材料与分子计算工作台')
        self.preview.background_colors = ('#33475c', '#71869b')
        self.comparison.background_colors = ('#e7edf3', '#bccbd9')
        self.preview.zoom = self.comparison.zoom = 1.05
        self._view_toolbar_buttons = {}
        for index, toolbar in enumerate(self.view_toolbars):
            toolbar.setIconSize(QSize(16, 16))
            for action in toolbar.actions():
                button = toolbar.widgetForAction(action)
                if isinstance(button, QToolButton):
                    self._view_toolbar_buttons[button] = index
                    button.installEventFilter(self)
        for table in self.findChildren(QTableWidget):
            table.setShowGrid(True)
            table.verticalHeader().setDefaultSectionSize(23)
        self.job_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.job_table.setColumnWidth(0, 210)
        self.project.setIndentation(15)
        self.project.itemClicked.connect(self.activate_project_item)
        self.project_root.setText(0, '研究项目')
        self.project_search.setPlaceholderText('筛选项目对象…')
        self.rebuild_window_menu()
        self.add_source_information()
        self._applying_workspace = False
        self._layout_path = data_dir() / 'workbench5-layout.bin'
        self._geometry_path = data_dir() / 'workbench5-geometry.bin'
        self._preferences_path = data_dir() / 'workbench5-layout.json'
        self.install_ai_tools()
        self.directory_button.setText('打开项目文件夹')
        self.file_menu.addAction('项目库…', self.open_project_library)
        phase = 1 if args.mode == 'server' else 0
        QTimer.singleShot(0, lambda: self.initialize_layout(phase))

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.MouseButtonPress, QEvent.FocusIn):
            index = getattr(self, '_view_toolbar_buttons', {}).get(watched)
            if index is not None:
                self.documents.activate(self.documents.windows[index])
        return super().eventFilter(watched, event)

    def create_workspace_bar(self):
        self.workspace_tabs = QTabBar(self)
        self.workspace_tabs.hide()
        for name in ('建模', '计算', '分析'):
            self.workspace_tabs.addTab(name)
        self.workspace_tabs.currentChanged.connect(self.change_workspace)
        bar = QToolBar('标准工具', self)
        bar.setObjectName('StandardToolbar')
        bar.setMovable(False)
        bar.setIconSize(QSize(20, 20))
        bar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.workspace_bar = bar
        self.addToolBar(bar)
        for title, icon in [('打开项目', 'open'), ('导入', 'file'), ('保存', 'save'),
                            ('撤销', 'undo'), ('重做', 'redo')]:
            action = self.actions_by_name[title]
            action.setIcon(tool_icon(icon))
            if title == '撤销':
                bar.addSeparator()
            bar.addAction(action)
        bar.addSeparator()
        bar.addWidget(QLabel(' 工作台 '))
        self.workbench_choice = QComboBox()
        self.workbench_choice.addItems(['分子建模', '量子计算', '结果分析'])
        self.workbench_choice.setMinimumWidth(115)
        self.workbench_choice.currentIndexChanged.connect(self.set_phase)
        bar.addWidget(self.workbench_choice)
        bar.addSeparator()
        for title, symbol in [('分子', 'model'), ('原子坐标', 'atoms'), ('几何优化', 'optimize')]:
            action = self.actions_by_name[title]
            action.setIcon(tool_icon(symbol))
            self.text_action(bar, action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self.layout_choice = QComboBox()
        self.layout_choice.addItems(['单文档', '双结构对照', '四视图分析', '文档标签'])
        self.layout_choice.setMinimumWidth(108)
        self.layout_choice.currentIndexChanged.connect(self.change_layout)
        bar.addWidget(self.layout_choice)
        self.addToolBarBreak()

    def text_action(self, bar, action):
        bar.addAction(action)
        button = bar.widgetForAction(action)
        if isinstance(button, QToolButton):
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

    def create_command_ribbon(self):
        # Reuse all existing domain actions; replace the ribbon with Qt toolbars.
        super().create_command_ribbon()
        self.ribbon.hide()
        self._retired_ribbon = self.ribbon
        bar = QToolBar('视图与计算', self)
        bar.setObjectName('ViewCalculationToolbar')
        bar.setMovable(False)
        bar.setIconSize(QSize(17, 17))
        bar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.addToolBar(bar)
        self.ribbon = bar
        for key, symbol in [('rotate', 'rotate'), ('pan', 'pan'), ('select', 'model'),
                             ('distance', 'measure'), ('angle', 'angle')]:
            action = self.view_modes[key]
            action.setIcon(tool_icon(symbol))
            bar.addAction(action)
        self.fit_action.setIcon(tool_icon('fit'))
        bar.addAction(self.fit_action)
        bar.addSeparator()
        previous_style = self.display_style
        self.display_style = QComboBox()
        self.display_style.addItems([previous_style.itemText(i) for i in range(previous_style.count())])
        self.display_style.setMinimumWidth(90)
        self.display_style.currentIndexChanged.connect(
            lambda index: self.active_view().set_style(['ballstick', 'stick', 'line', 'cpk'][index]))
        bar.addWidget(self.display_style)
        self.display_style.show()
        for action, symbol in [(self.labels_action, 'label'), (self.grid_action, 'grid')]:
            action.setIcon(tool_icon(symbol))
            bar.addAction(action)
        bar.addSeparator()
        for key, symbol in [('设置', 'settings'), ('提交', 'send'), ('同步', 'rotate'), ('取消', 'cancel')]:
            action = self.actions_new[key]
            action.setIcon(tool_icon(symbol))
            self.text_action(bar, action)
        self.server_jobs_action.setIcon(tool_icon('tasks'))
        self.text_action(bar, self.server_jobs_action)
        bar.addSeparator()
        for key, symbol in [('曲线', 'chart'), ('报告', 'file'), ('助手', 'assistant')]:
            action = self.actions_new[key]
            action.setIcon(tool_icon(symbol))
            self.text_action(bar, action)

    def create_document_tools(self):
        super().create_document_tools()
        # Contextual buttons remain next to the document they operate on.
        symbols = ['rotate', 'pan', 'model', None, 'fit', 'measure', 'angle', None, 'label', 'grid']
        for toolbar in self.view_toolbars:
            for action, symbol in zip(toolbar.actions(), symbols):
                if symbol and not action.isSeparator():
                    action.setIcon(tool_icon(symbol))

    def initialize_layout(self, phase):
        self.change_workspace(phase)
        try:
            if self._geometry_path.exists():
                self.restoreGeometry(QByteArray(self._geometry_path.read_bytes()))
            if self._layout_path.exists():
                saved = self._layout_path.read_bytes()
                if self.dock_manager.restoreState(QByteArray(saved)):
                    preferences = json.loads(self._preferences_path.read_text('utf-8')) if self._preferences_path.exists() else {}
                    mode = preferences.get('mode', 'four')
                    self.documents.layout_mode = mode
                    self.reflect_layout(mode)
                else:
                    self.reset_layout()
            else:
                self.size_default_panels()
        except (OSError, ValueError):
            self.reset_layout()
        for panel in self.documents.windows:
            if not panel.isClosed():
                panel.tabWidget().show()
        if not self._layout_path.exists():
            self.size_default_panels()
        self.documents.setCurrentIndex(0)
        # Show the AI entry on the first launch of this revision; later respect saved visibility.
        try:
            preferences = json.loads(self._preferences_path.read_text('utf-8'))
        except (OSError, ValueError):
            preferences = {}
        if preferences.get('ai_visible', True):
            self.ai_dock.show()

    def size_default_panels(self):
        manager = self.dock_manager
        manager.setSplitterSizes(self.project_dock.dockAreaWidget(), [220, 950, 285])
        manager.setSplitterSizes(self.activity_dock.dockAreaWidget(), [690, 160])

    def change_workspace(self, index):
        if not getattr(self, 'docking_ready', False):
            if getattr(self, 'shell_ready', False):
                return super().change_workspace(index)
            return
        if getattr(self, '_applying_workspace', False):
            return
        self._applying_workspace = True
        self._changing_workspace = True
        for control in (self.workspace_tabs, self.workbench_choice):
            control.blockSignals(True)
            control.setCurrentIndex(index)
            control.blockSignals(False)
        self.tabs.setCurrentIndex(index)
        self.inspector.show()
        self.activity_dock.show()
        self.documents.focus_single() if index == 0 else self.documents.tile_all()
        self.status.setText(('分子建模', '量子计算', '结果分析')[index])
        self._changing_workspace = False
        self._applying_workspace = False

    def change_layout(self, index):
        if not getattr(self, 'docking_ready', False):
            return
        self.documents.set_layout(('single', 'models', 'four', 'tabs')[index])

    def reflect_layout(self, mode):
        self.layout_choice.blockSignals(True)
        self.layout_choice.setCurrentIndex({'single': 0, 'models': 1, 'four': 2, 'tabs': 3}.get(mode, 2))
        self.layout_choice.blockSignals(False)

    def rebuild_window_menu(self):
        self.window_menu.clear()
        for name, callback in [('单文档', self.documents.focus_single),
                               ('双结构对照', self.documents.focus_models),
                               ('四视图分析', self.documents.tile_all),
                               ('文档标签', self.documents.cascadeSubWindows),
                               ('恢复自动对照', self.reset_reference),
                               ('恢复默认布局', self.reset_layout)]:
            self.window_menu.addAction(name, callback)
        self.window_menu.addSeparator()
        for panel in [self.project_dock, self.inspector, self.activity_dock, self.ai_dock,
                      *self.documents.windows]:
            self.window_menu.addAction(panel.toggleViewAction())

    def reset_layout(self):
        if not self.docking_ready:
            return
        import PySide6QtAds as ads
        manager = self.dock_manager
        self.documents.tile_all()
        manager.addDockWidget(ads.LeftDockWidgetArea, self.project_dock)
        manager.addDockWidget(ads.RightDockWidgetArea, self.inspector)
        manager.addDockWidget(ads.BottomDockWidgetArea, self.activity_dock)
        manager.addDockWidget(ads.BottomDockWidgetArea, self.ai_dock, self.inspector.dockAreaWidget())
        for panel in (self.project_dock, self.inspector, self.activity_dock):
            panel.show()
        self.ai_dock.hide()
        self.size_default_panels()

    def add_source_information(self):
        menu = self.menuBar().addMenu('帮助')
        menu.addAction('界面来源与开源许可', self.show_sources)
        menu.addAction('使用说明', lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(
            str(Path(__file__).resolve().parent.parent / 'README.md'))))
        menu.addSeparator()
        menu.addAction('关于 ChatDFT', self.show_about)

    def show_about(self, startup=False):
        dialog = QDialog(self)
        dialog.setWindowTitle('关于 ChatDFT')
        dialog.setMinimumWidth(390)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(14)

        logo = QLabel()
        logo.setAlignment(Qt.AlignCenter)
        logo.setAccessibleName('ChatDFT')
        logo.setStyleSheet('background: white; border: 1px solid #ced3d9;')
        asset = Path(__file__).resolve().parent / 'assets' / 'chatdft-wordmark.png'
        ratio = dialog.devicePixelRatioF()
        pixmap = QPixmap(str(asset)).scaledToWidth(round(320 * ratio), Qt.SmoothTransformation)
        pixmap.setDevicePixelRatio(ratio)
        logo.setPixmap(pixmap)
        logo.setMinimumHeight(120)
        layout.addWidget(logo)

        version = QApplication.applicationVersion() or '5.1'
        layout.addWidget(QLabel(f'材料与分子计算工作台 · {version}'))
        layout.addWidget(QLabel('分子建模、PySCF 服务器计算与 AI 结果分析'))
        layout.addWidget(QLabel('© 2026 Yinzi Shao · 保留全部权利'))
        layout.addWidget(QLabel('CC 4.0 · 具体授权以随附许可声明为准'))
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText('关闭')
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if startup:
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            dialog.show()
            QTimer.singleShot(3000, dialog, dialog.close)
        else:
            dialog.exec()

    def show_sources(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(
            str(Path(__file__).resolve().parent.parent / 'OPEN_SOURCE.md')))

    def document_activated(self, window):
        if not getattr(self, 'docking_ready', False):
            return super().document_activated(window)
        view = self.active_view()
        for action in (*self.view_modes.values(), self.fit_action, self.labels_action, self.grid_action):
            action.setEnabled(view.structure is not None)
        self.display_style.setEnabled(view.structure is not None)
        self.display_style.blockSignals(True)
        self.display_style.setCurrentIndex(['ballstick', 'stick', 'line', 'cpk'].index(view.style))
        self.display_style.blockSignals(False)
        key = ('distance' if view.measure_count == 2 else 'angle') if view.mode == 'measure' else view.mode
        if key in self.view_modes:
            self.view_modes[key].setChecked(True)
        for action, key in [(self.labels_action, 'labels'), (self.grid_action, 'grid')]:
            action.blockSignals(True)
            action.setChecked(getattr(view, key))
            action.blockSignals(False)
        self.refresh_inspector()
        self.update_command_states()
        self.sync_ai_target()

    def inspector_group(self, name, rows):
        super().inspector_group(name, rows)
        group = self.inspector_table.topLevelItem(self.inspector_table.topLevelItemCount() - 1)
        for column in (0, 1):
            group.setBackground(column, QColor('#e8edf2'))
            group.setForeground(column, QColor('#31465a'))

    def closeEvent(self, event):
        if getattr(self, '_analysis_task', None):
            self._analysis_cancel.set()
            self.status.setText('正在停止本次 AI 分析。重新打开后可继续，请稍后关闭窗口。')
            event.ignore()
            return
        if getattr(self, '_auto_export_task', None):
            self.status.setText('正在生成计算图件和报告，完成后可关闭。')
            event.ignore()
            return
        if not self.operation and self.docking_ready:
            try:
                self._layout_path.write_bytes(bytes(self.dock_manager.saveState()))
                self._geometry_path.write_bytes(bytes(self.saveGeometry()))
                atomic_json(self._preferences_path, {'mode': self.documents.layout_mode,
                                                     'ai_visible': not self.ai_dock.isClosed()})
            except OSError:
                pass
        super().closeEvent(event)
