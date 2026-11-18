"""ADS document adapter. Uses the upstream simple example's CDockManager pattern.

Qt Advanced Docking System / PySide6-QtAds: LGPL-2.1.
Sources and license copies are recorded in third_party/ and OPEN_SOURCE.md.
"""
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget
import PySide6QtAds as ads


class DockPanel(ads.CDockWidget):
    def show(self):
        self.toggleView(True)
        self.tabWidget().show()

    def hide(self):
        self.toggleView(False)

    def raise_(self):
        self.toggleView(True)
        self.setAsCurrentTab()


class DockDocuments(QObject):
    subWindowActivated = Signal(object)
    layoutChanged = Signal(str)

    def __init__(self, owner, manager, windows):
        super().__init__(owner)
        self.owner, self.manager, self.windows = owner, manager, windows
        self.layout_mode = 'four'
        self._active = windows[0]
        for window in windows:
            window.topLevelChanged.connect(lambda floating, w=window: self.activate(w))
            window.visibilityChanged.connect(lambda visible, w=window: self.visible(w, visible))
        manager.focusedDockWidgetChanged.connect(self.focus_changed)

    def focus_changed(self, old, current):
        if current in self.windows:
            self.activate(current)

    def visible(self, window, visible):
        if visible and window.isCurrentTab():
            self.activate(window)
        elif window is self._active and window.isClosed():
            self.activate(self.windows[0])

    def activate(self, window):
        if window.isClosed():
            return
        self._active = window
        self.subWindowActivated.emit(window)

    def activeSubWindow(self):
        return self._active

    def currentIndex(self):
        return self.windows.index(self._active)

    def setCurrentIndex(self, index):
        if 0 <= index < len(self.windows):
            window = self.windows[index]
            window.show()
            window.setAsCurrentTab()
            self.activate(window)

    def arrange(self):
        self.set_layout(self.layout_mode)

    def set_layout(self, mode):
        self.layout_mode = mode
        main, reference, energy, curve, *extra = self.windows
        for window in [reference, energy, curve, *extra]:
            window.hide()
        main.show()
        if mode in ('models', 'four'):
            self.manager.addDockWidget(ads.RightDockWidgetArea, reference, main.dockAreaWidget())
            reference.show()
        if mode == 'four':
            self.manager.addDockWidget(ads.BottomDockWidgetArea, energy, main.dockAreaWidget())
            self.manager.addDockWidget(ads.BottomDockWidgetArea, curve, reference.dockAreaWidget())
            energy.show()
            curve.show()
            self.manager.setSplitterSizes(main.dockAreaWidget(), [380, 260])
            self.manager.setSplitterSizes(reference.dockAreaWidget(), [380, 260])
        elif mode == 'tabs':
            for window in self.windows[1:4]:
                self.manager.addDockWidget(ads.CenterDockWidgetArea, window, main.dockAreaWidget())
                window.show()
        main.setAsCurrentTab()
        self.activate(main)
        self.layoutChanged.emit(mode)

    def focus_single(self):
        self.set_layout('single')

    def focus_models(self):
        self.set_layout('models')

    def tile_all(self):
        self.set_layout('four')

    def cascadeSubWindows(self):
        self.set_layout('tabs')


def install_docking(owner):
    """Move existing functional widgets into actual ADS dock containers."""
    ads.CDockManager.setConfigFlags(ads.CDockManager.DefaultOpaqueConfig)
    ads.CDockManager.setConfigFlag(ads.CDockManager.FocusHighlighting, True)
    ads.CDockManager.setConfigFlag(ads.CDockManager.AlwaysShowTabs, True)
    old_documents = owner.takeCentralWidget()
    old_docks = [getattr(owner, name) for name in ('project_dock', 'inspector', 'activity_dock', 'ai_dock')]
    contents = []
    for dock in old_docks:
        content = dock.widget()
        content.setParent(owner)
        dock.setWidget(None)
        owner.removeDockWidget(dock)
        dock.hide()
        contents.append(content)
    manager = ads.CDockManager(owner)
    manager.setObjectName('ScientificDockManager')
    documents = []
    for index, window in enumerate(old_documents.windows):
        shell = window.widget()
        content = shell.layout().itemAt(1).widget()
        content.setParent(owner)
        panel = DockPanel(window.windowTitle())
        panel.setObjectName(f'Document{index}')
        panel.setWidget(content)
        documents.append(panel)
    # The upstream example's central widget provides the stable document anchor.
    manager.setCentralWidget(documents[0])
    for panel in documents[1:]:
        manager.addDockWidget(ads.CenterDockWidgetArea, panel, documents[0].dockAreaWidget())
        panel.hide()
    owner.primary_window, owner.comparison_window, owner.energy_window, owner.curve_window = documents[:4]
    for name, title, content in zip(
            ('project_dock', 'inspector', 'activity_dock', 'ai_dock'),
            ('项目浏览器', '属性 / 计算', '作业与输出', '命令控制台'), contents):
        panel = DockPanel(title)
        panel.setObjectName(name)
        panel.setWidget(content)
        setattr(owner, name, panel)
    manager.addDockWidget(ads.LeftDockWidgetArea, owner.project_dock)
    manager.addDockWidget(ads.RightDockWidgetArea, owner.inspector)
    manager.addDockWidget(ads.BottomDockWidgetArea, owner.activity_dock)
    manager.addDockWidget(ads.BottomDockWidgetArea, owner.ai_dock, owner.inspector.dockAreaWidget())
    owner.inspector.setMinimumSize(280, 140)
    owner.inspector.setMinimumSizeHintMode(
        ads.CDockWidget.eMinimumSizeHintMode.MinimumSizeHintFromDockWidgetMinimumSize)
    owner.inspector.dockAreaWidget().setMinimumWidth(280)
    owner.project_dock.setMinimumWidth(165)
    owner.ai_dock.hide()
    owner.dock_manager = manager
    owner.documents = DockDocuments(owner, manager, documents)
    owner.documents.subWindowActivated.connect(owner.document_activated)
    owner.documents.layoutChanged.connect(owner.reflect_layout)
    old_documents.hide()
    owner._retired_windows = [old_documents, *old_docks]
    return manager
