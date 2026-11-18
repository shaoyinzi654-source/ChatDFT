"""Native Qt workbench chrome; docking uses the unmodified upstream ADS theme."""
from PySide6.QtGui import QColor, QFont, QPalette
from studio_theme import studio_icon
import qtawesome as qta


ICONS = {
    'open': ('fa5s.folder-open', '#c38b24'), 'folder': ('fa5s.folder', '#c38b24'),
    'save': ('fa5s.save', '#3976ad'), 'undo': ('fa5s.undo', '#5082af'),
    'redo': ('fa5s.redo', '#5082af'), 'model': ('fa5s.project-diagram', '#4382b4'),
    'atoms': ('fa5s.atom', '#378c82'), 'optimize': ('fa5s.magic', '#a47436'),
    'send': ('fa5s.play', '#3e9956'), 'external': ('fa5s.external-link-alt', '#4d79aa'),
    'tasks': ('fa5s.list-alt', '#4d79aa'), 'settings': ('fa5s.sliders-h', '#627180'),
    'assistant': ('fa5s.terminal', '#52667e'), 'file': ('fa5s.file-alt', '#5f80a1'),
    'rotate': ('fa5s.sync-alt', '#347aa7'), 'pan': ('fa5s.arrows-alt', '#347aa7'),
    'fit': ('fa5s.expand-arrows-alt', '#347aa7'), 'measure': ('fa5s.ruler', '#ad842e'),
    'angle': ('fa5s.drafting-compass', '#917045'), 'label': ('fa5s.font', '#52667e'),
    'cancel': ('fa5s.stop', '#be5550'), 'chart': ('fa5s.chart-line', '#557eb4'),
    'grid': ('fa5s.border-all', '#647b91'),
}


def tool_icon(name, color=None):
    glyph, default = ICONS.get(name, ICONS['file'])
    return qta.icon(glyph, color=color or default, color_disabled='#a4a8ae')


def apply_workbench_theme(app):
    app.setStyle('Fusion')
    app.setProperty('nativeControls', True)
    # Keep font selection in Qt; a CSS font-family overrides CJK font fallback.
    font = QFont('Microsoft YaHei', 9)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)
    app.setWindowIcon(studio_icon())
    palette = QPalette()
    for role, value in {
        QPalette.Window: '#ededed', QPalette.WindowText: '#20252a',
        QPalette.Base: '#ffffff', QPalette.AlternateBase: '#f3f6f9',
        QPalette.Text: '#20252a', QPalette.Button: '#ededed',
        QPalette.ButtonText: '#20252a', QPalette.Highlight: '#3479b5',
        QPalette.HighlightedText: '#ffffff', QPalette.Light: '#ffffff',
        QPalette.Midlight: '#f5f5f5', QPalette.Mid: '#b8bdc4',
        QPalette.Dark: '#8b929a', QPalette.ToolTipBase: '#fffde7',
        QPalette.ToolTipText: '#20252a',
    }.items():
        palette.setColor(role, QColor(value))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor('#969ba1'))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor('#969ba1'))
    app.setPalette(palette)
    app.setStyleSheet('''
        QMenuBar { padding: 1px 5px; background: #f5f5f5; }
        QMenuBar::item { padding: 4px 9px; }
        QMenuBar::item:selected { background: #dce9f6; }
        QToolBar { border: 0; border-bottom: 1px solid #bcc2c9;
                   background: #ededed; spacing: 2px; padding: 2px 5px; }
        QToolBar::separator { background: #bbc0c6; width: 1px; margin: 3px 5px; }
        QToolButton { padding: 3px; border: 1px solid transparent; border-radius: 2px; }
        QToolButton:hover { background: #dfebf7; border-color: #a6bfd7; }
        QToolButton:checked { background: #cfe1f3; border-color: #82a7ca; }
        QLabel#ToolSection { color: #555f69; padding: 0 5px; }
        QComboBox, QLineEdit, QSpinBox { min-height: 21px; padding: 1px 4px;
                    background: white; border: 1px solid #adb6c0; border-radius: 1px; }
        QComboBox:disabled, QSpinBox:disabled { background: #f0f0f0; }
        QComboBox { combobox-popup: 0; }
        QComboBox QAbstractItemView { selection-background-color: #d8e8f8;
                    selection-color: #182a3b; }
        QSpinBox::up-button, QSpinBox::down-button { width: 18px; background: #edf0f3; border: 0; }
        QSpinBox::up-arrow, QSpinBox::down-arrow { image: none; }
        QTreeView, QTableView { border: 0; background: white; gridline-color: #dee3e8;
                   alternate-background-color: #f4f6f8; outline: 0; }
        QTreeView::item { min-height: 20px; padding: 0 2px; }
        QTreeView::item:selected, QTableView::item:selected { background: #d8e8f8; color: #182a3b; }
        QHeaderView::section { background: #edf0f3; padding: 3px 5px;
                   border: 0; border-right: 1px solid #d1d7dd; border-bottom: 1px solid #bfc8d1; }
        QTabWidget::pane { border: 0; background: white; }
        QTabBar::tab { background: #e5e8ed; padding: 5px 11px;
                      border: 1px solid #c3c9d1; border-bottom: 0; }
        QTabBar::tab:selected { background: white; color: #1a507c; }
        QGroupBox { border: 1px solid #ced3d9; margin-top: 11px; padding: 7px 5px 4px; }
        QGroupBox::title { left: 7px; padding: 0 4px; color: #465564; }
        QPushButton { padding: 4px 8px; min-height: 19px; }
        QPushButton#Submit { background: #3d7cae; color: white; border: 1px solid #326a99; padding: 6px; }
        QPushButton#Submit:disabled { background: #e3e7ec; color: #9098a1; border-color: #c8d0d9; }
        QTextBrowser, QPlainTextEdit { background: white; border: 1px solid #c7ced7; }
        QStatusBar { background: #e9edf1; border-top: 1px solid #bdc6d0; padding: 1px 4px; }
        QStatusBar::item { border: 0; }
        QWidget#InspectorPage, QScrollArea { border: 0; background: #f7f8fa; }
        QLabel#ConnectionLabel { color: #526474; padding: 3px; }
        ads--CDockWidgetTab[activeTab="true"] { background: #ffffff; border-top: 2px solid #447eae; }
        ads--CDockWidgetTab[activeTab="true"] QLabel { color: #1c3e5c; }
    ''')
