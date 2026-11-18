from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QIcon


def studio_icon():
    return QIcon(str(Path(__file__).resolve().parent.parent / 'chatdft.ico'))


def apply_studio_theme(app):
    app.setStyle('Fusion')
    font = QFont('Microsoft YaHei UI', 9)
    app.setFont(font)
    app.setWindowIcon(studio_icon())
    app.setStyleSheet("""
        QWidget { font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 12px; color: #334155; }
        QMainWindow { background: #dce0e5; }
        QMenuBar { background: #f7f7f8; color: #344050; padding: 1px 8px; }
        QMenuBar::item { padding: 4px 11px; }
        QMenuBar::item:selected { background: #e3eaf4; }
        QMenu { background: #ffffff; border: 1px solid #bfc7d1; padding: 4px; }
        QMenu::item { padding: 6px 24px; }
        QMenu::item:selected { background: #e6edf7; }
        QToolBar#WorkspaceBar { background: #f7f7f8; border: 0; spacing: 0; padding: 0 10px; }
        QToolBar#CommandRibbon { background: #f3f4f6; border: 0;
                                border-bottom: 1px solid #c0c7d1; padding: 0; }
        QToolBar#StudioQuickAccess { background: #f7f7f8; border: 0; padding: 0 6px; }
        QToolBar { spacing: 3px; padding: 3px; border: 0; background: #f4f5f7; }
        QTabBar#WorkspaceTabs::tab { color: #52647b; background: transparent;
                        padding: 6px 21px; border-bottom: 3px solid transparent; }
        QTabBar#WorkspaceTabs::tab:selected { color: #2e5c97; border-bottom-color: #497db8; }
        QWidget#CommandPage { background: #f3f4f6; }
        QWidget#CommandGroup { background: #f3f4f6; border-right: 1px solid #d2d7de; }
        QLabel#CommandGroupTitle { color: #87909c; font-size: 10px; padding: 2px 0 0; }
        QLabel#ApplicationWordmark { font-family: 'Segoe UI'; color: #344e72;
                                     font-size: 19px; font-weight: 600; padding: 0 28px 0 2px; }
        QLabel#ConnectionLabel { color: #7d8796; font-size: 11px; padding: 0 10px; }
        QToolButton { border: 1px solid transparent; border-radius: 2px; padding: 5px 8px; color: #455367; }
        QToolButton:hover { background: #e5ebf3; border-color: #bdccde; }
        QToolButton:checked { background: #e0e9f6; border-color: #96b2d8; color: #315d92; }
        QToolButton:disabled { color: #a7adb5; }
        QToolButton#RibbonButton { padding: 2px 10px; min-width: 39px; }
        QWidget#DockTitle { background: #e7eaee; border-bottom: 1px solid #c5ccd6; }
        QLabel#DockCaption { color: #4e6078; font-size: 11px; padding: 3px 2px; font-weight: 600; }
        QWidget#MdiTitle { background: #f1f3f6; border: 0; border-bottom: 1px solid #c6ced9; }
        QWidget#MdiTitle[active="true"] { background: #e7eef8; border-bottom: 2px solid #5883ba; }
        QLabel#MdiCaption { color: #44566f; font-size: 11px; font-weight: 500; }
        QWidget#DocumentShell { border: 1px solid #b9c3d1; background: #ffffff; }
        QPushButton#WindowControl { color: #778398; background: transparent; border: 0;
                                    padding: 0; min-height: 0; font-size: 13px; }
        QPushButton#WindowControl:hover { color: #253f63; background: #d5e0ef; }
        QDockWidget { background: #fafbfc; }
        QTabWidget::pane { background: #fafbfc; border: 0; }
        QTabBar::tab { background: #edf0f4; color: #7a8698; padding: 6px 11px; border: 0; }
        QTabBar::tab:selected { background: #ffffff; color: #3b6394; border-bottom: 2px solid #698ab6; }
        QTableView, QTreeView { background: #ffffff; alternate-background-color: #f6f8fb;
                    border: 0; gridline-color: #e6eaf0; outline: 0;
                    selection-background-color: #e1eaf7; selection-color: #2e4e78; }
        QTreeView::item { padding: 3px 2px; min-height: 19px; }
        QTableView::item { padding: 3px 5px; }
        QHeaderView::section { background: #f2f4f7; color: #8a94a3; border: 0;
                    border-bottom: 1px solid #d7dee8; padding: 6px 5px; font-size: 11px; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    background: #ffffff; color: #46576c; border: 1px solid #cbd3df;
                    border-radius: 2px; padding: 4px; min-height: 18px; }
        QComboBox::drop-down { border: 0; background: #f0f3f8; width: 20px; }
        QSpinBox::up-button, QSpinBox::down-button { background: #f0f3f8; border: 0; width: 19px; }
        QLineEdit:focus, QComboBox:focus { border-color: #7d9fc9; }
        QWidget#InspectorPage, QScrollArea { background: #fafbfc; border: 0; }
        QGroupBox { background: #fafbfc; border: 0; border-top: 1px solid #dfe4ec;
                    margin-top: 14px; padding: 9px 2px 4px; }
        QGroupBox::title { color: #7c899a; left: 2px; padding: 0 5px 0 0; font-size: 11px; }
        QPlainTextEdit, QTextBrowser { background: #ffffff; color: #46566d;
                    border: 1px solid #d3dbe6; border-radius: 2px; padding: 6px; }
        QPushButton { background: #f5f7fa; color: #5c6e86; border: 1px solid #c6d2e1;
                    border-radius: 2px; padding: 6px 9px; min-height: 18px; }
        QPushButton:hover { background: #e8eff8; border-color: #9bb2d0; }
        QPushButton:disabled { color: #b6bdc7; border-color: #dce2e9; background: #f4f6f8; }
        QPushButton#Submit { background: #466b9e; color: #ffffff; border-color: #3a6095; }
        QStatusBar { background: #f5f6f8; border-top: 1px solid #ccd4df; padding: 2px 7px; color: #8a95a5; }
        QStatusBar::item { border: 0; }
        QSplitter::handle { background: #e0e5ec; width: 3px; height: 3px; }
        QScrollBar:vertical { background: #f5f7fa; width: 9px; }
        QScrollBar::handle:vertical { background: #c5cedb; border-radius: 3px; margin: 2px; min-height: 24px; }
        QToolTip { background: #fffef3; color: #3e5069; border: 1px solid #c4cdd9; padding: 6px; }
    """)

