from collections import Counter

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox, QWidget


TOOL_COLORS = {
    'open': '#ba8528', 'folder': '#c08c30', 'save': '#376eae', 'undo': '#b98331',
    'redo': '#408b72', 'fit': '#348da1', 'external': '#7160b2', 'send': '#337bac',
    'model': '#4b91bf', 'assistant': '#9566be', 'tasks': '#42a080', 'settings': '#8799b0',
    'rotate': '#238c9e', 'pan': '#4e8ba9', 'measure': '#ca8a37', 'angle': '#ad6d9b',
    'optimize': '#7c6ab8', 'atoms': '#568f8d', 'file': '#5d8fb3', 'label': '#5073ac',
}


def chevron(painter, x, y, direction=1, enabled=True):
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor('#566e8b' if enabled else '#b7c3d1'), 1.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawPolyline(QPolygonF([
        QPointF(x - 3, y - direction * 1.5),
        QPointF(x, y + direction * 1.5),
        QPointF(x + 3, y - direction * 1.5),
    ]))


class ChoiceBox(QComboBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        if QApplication.instance().property('nativeControls'):
            return
        painter = QPainter(self)
        chevron(painter, self.width() - 12, self.height() / 2, enabled=self.isEnabled())
        painter.end()


class NumberBox(QSpinBox):
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        chevron(painter, self.width() - 11, self.height() * .29, -1, self.isEnabled())
        chevron(painter, self.width() - 11, self.height() * .71, enabled=self.isEnabled())
        painter.end()


def symbol_icon(name, color=None):
    pixmap = QPixmap(96, 96)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(4, 4)
    color = color or TOOL_COLORS.get(name, '#5d7b9b')
    painter.setPen(QPen(QColor(color), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)
    if name in ('open', 'folder'):
        painter.setBrush(QColor('#f5cf77'))
        path = QPainterPath(QPointF(3, 7))
        for x, y in ((10, 7), (12, 9), (21, 9), (20, 19), (3, 19), (3, 7)):
            path.lineTo(x, y)
        painter.drawPath(path)
        painter.drawLine(3, 12, 20, 12)
    elif name == 'save':
        painter.setBrush(QColor('#d2e2f6'))
        painter.drawRoundedRect(QRectF(4, 3, 16, 18), 2, 2)
        painter.setBrush(QColor('#ffffff'))
        painter.drawRect(8, 3, 8, 6)
        painter.drawRect(8, 14, 8, 7)
    elif name in ('undo', 'redo'):
        direction = -1 if name == 'undo' else 1
        origin = 12
        painter.drawArc(QRectF(5, 7, 14, 13), 0 if direction < 0 else 90 * 16, 220 * 16)
        x = origin + direction * 7
        painter.drawLine(x, 6, x, 12)
        painter.drawLine(x, 12, x - direction * 6, 12)
    elif name == 'fit':
        for x, y, dx, dy in ((4, 4, 1, 1), (20, 4, -1, 1), (4, 20, 1, -1), (20, 20, -1, -1)):
            painter.drawLine(x, y, x + dx * 5, y)
            painter.drawLine(x, y, x, y + dy * 5)
    elif name in ('external', 'send'):
        painter.setBrush(QColor('#e3ddf5' if name == 'external' else '#d6eaf8'))
        painter.drawRoundedRect(QRectF(4, 9, 12, 12), 2, 2)
        painter.drawLine(11, 13, 20, 4)
        painter.drawLine(14, 4, 20, 4)
        painter.drawLine(20, 4, 20, 10)
    elif name in ('model', 'atoms'):
        points = [QPointF(5, 7), QPointF(18, 5), QPointF(19, 18), QPointF(7, 19)]
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)):
            painter.drawLine(points[a], points[b])
        for point, atom_color in zip(points, ('#5da1d5', '#e08357', '#8d79c1', '#62b8a0')):
            painter.setBrush(QColor(atom_color))
            painter.drawEllipse(point, 2.4, 2.4)
    elif name == 'assistant':
        painter.setBrush(QColor('#eee6f8'))
        painter.drawRoundedRect(QRectF(3, 4, 18, 14), 4, 4)
        painter.drawLine(8, 18, 6, 22)
        painter.drawLine(6, 22, 13, 18)
        painter.drawLine(8, 9, 16, 9)
        painter.drawLine(8, 13, 13, 13)
    elif name == 'tasks':
        painter.setBrush(QColor('#e1f0e9'))
        painter.drawRoundedRect(QRectF(4, 3, 16, 18), 2, 2)
        for y in (8, 12, 16):
            painter.drawEllipse(QPointF(8, y), .6, .6)
            painter.drawLine(12, y, 17, y)
    elif name == 'settings':
        for (x, y), tone in zip(((6, 8), (12, 16), (18, 10)), ('#e7bc69', '#76b7aa', '#8999d4')):
            painter.drawLine(x, 3, x, 21)
            painter.setBrush(QColor(tone))
            painter.drawEllipse(QPointF(x, y), 2.5, 2.5)
    elif name == 'rotate':
        painter.drawArc(QRectF(4, 4, 16, 16), 30 * 16, 290 * 16)
        painter.drawLine(19, 4, 20, 10)
        painter.drawLine(20, 10, 14, 9)
    elif name == 'pan':
        painter.drawLine(12, 3, 12, 21)
        painter.drawLine(3, 12, 21, 12)
        for a, b in (((12, 3), (9, 6)), ((12, 3), (15, 6)), ((21, 12), (18, 9)),
                     ((21, 12), (18, 15)), ((12, 21), (9, 18)), ((12, 21), (15, 18)),
                     ((3, 12), (6, 9)), ((3, 12), (6, 15))):
            painter.drawLine(*a, *b)
    elif name == 'measure':
        painter.setBrush(QColor('#f8e4bb'))
        painter.drawRoundedRect(QRectF(3, 7, 18, 11), 1, 1)
        for x in (6, 10, 14, 18):
            painter.drawLine(x, 7, x, 12 if x in (6, 14) else 10)
    elif name == 'angle':
        painter.drawLine(4, 19, 21, 19)
        painter.drawLine(4, 19, 16, 4)
        painter.drawArc(QRectF(-5, 10, 18, 18), 0, 52 * 16)
    elif name == 'optimize':
        path = QPainterPath(QPointF(3, 4))
        path.cubicTo(7, 6, 5, 20, 12, 20)
        path.cubicTo(17, 20, 18, 7, 21, 4)
        painter.drawPath(path)
        painter.setBrush(QColor('#b8a0dd'))
        painter.drawEllipse(QPointF(12, 19), 2.6, 2.6)
    elif name == 'label':
        painter.setFont(QFont('Segoe UI', 16, QFont.Bold))
        painter.drawText(QRectF(2, 0, 20, 24), Qt.AlignCenter, 'A')
    elif name == 'status':
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(12, 12), 5, 5)
    else:
        painter.drawRoundedRect(QRectF(5, 3, 14, 18), 2, 2)
        painter.drawLine(9, 9, 16, 9)
        painter.drawLine(9, 13, 16, 13)
    painter.end()
    return QIcon(pixmap)


def application_icon():
    image = QPixmap(128, 128)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor('#243b5d'))
    painter.drawRoundedRect(4, 4, 120, 120, 18, 18)
    painter.drawPixmap(20, 20, 88, 88, symbol_icon('model', '#b8d8f3').pixmap(88, 88))
    painter.end()
    return QIcon(image)


class ElementLegend(QWidget):
    def __init__(self, colors):
        super().__init__()
        self.colors = colors
        self.counts = {}
        self.setMinimumHeight(29)

    def set_structure(self, structure):
        self.counts = Counter(atom['element'] for atom in structure.atoms) if structure else {}
        self.setToolTip(' · '.join(f'{element}  {count}' for element, count in self.counts.items()))
        self.setVisible(bool(self.counts))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#233851'))
        painter.setPen(QColor('#a8bed6'))
        painter.drawText(12, 19, '元素')
        x = 51
        for element, count in self.counts.items():
            text = f'{element}  {count}'
            width = painter.fontMetrics().horizontalAdvance(text) + 34
            if x + width > self.width():
                painter.setPen(QColor('#b9cce0'))
                painter.drawText(x, 19, '…')
                break
            painter.setBrush(QColor(self.colors.get(element, '#b49ed4')))
            painter.setPen(QPen(QColor('#a6b9cf'), .5))
            painter.drawEllipse(QPointF(x + 5, 14), 4, 4)
            painter.setPen(QColor('#dce8f5'))
            painter.drawText(x + 17, 19, text)
            x += width


def apply_theme(application):
    application.setStyle('Fusion')
    application.setFont(QFont('Microsoft YaHei UI', 9))
    application.setWindowIcon(application_icon())
    application.setStyleSheet("""
        QWidget { color: #27384d; }
        QMainWindow, QDialog { background: #e8edf4; }
        QWidget#InspectorPage { background: #f4f6fa; }
        QMenuBar { background: #273c58; color: #e3edf8; padding: 2px 12px; }
        QMenuBar::item { padding: 5px 11px; background: transparent; }
        QMenuBar::item:selected { background: #426183; }
        QMenu { background: #ffffff; border: 1px solid #b9c7d9; padding: 4px; }
        QMenu::item { padding: 6px 24px; }
        QMenu::item:selected { background: #e0ebfa; }
        QToolBar#ModelTools { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #fafbfd,stop:1 #e8eef6); border: 0; border-bottom: 1px solid #bac9dc;
                    spacing: 4px; padding: 7px 10px; }
        QToolBar::separator { background: #bcc9d9; width: 1px; margin: 4px 5px; }
        QToolButton { padding: 5px 7px; border: 1px solid transparent; border-radius: 2px; }
        QToolButton:hover { background: #e1ebfa; border-color: #9fb8dc; }
        QToolButton:checked { background: #d5e4f8; border-color: #7f9fc9; color: #244978; }
        QToolBar#Navigation { background: #2b3e58; border: 0; border-right: 1px solid #1d304b;
                    spacing: 7px; padding: 14px 6px; }
        QToolBar#Navigation QToolButton { color: #d6e3f3; min-width: 36px; padding: 9px 5px; }
        QToolBar#Navigation QToolButton:hover { background: #3b526f; }
        QToolBar#Navigation QToolButton:checked { background: #456182; color: #ffffff; border-color: #5d7fa9; }
        QLabel#Brand { color: #284669; font-family: 'Segoe UI'; font-size: 20px;
                      font-weight: 700; padding: 0 14px 0 0; }
        QLabel#DockHeading { background: #dae4f0; color: #284765; padding: 8px 10px;
                           border-bottom: 1px solid #b9c8da; font-weight: 600; }
        QDockWidget#InspectorDock QLabel#DockHeading { background: #e3def0; color: #594473;
                           border-bottom-color: #c7bcdd; }
        QDockWidget#ActivityDock QLabel#DockHeading { background: #dcebe7; color: #345f59;
                           border-bottom-color: #b4cdc7; }
        QLabel#PageTitle { font-size: 24px; font-weight: 600; color: #375b8b; padding: 0 0 12px; }
        QDockWidget { background: #f8fafc; }
        QTabWidget::pane { border: 1px solid #c2cede; background: #f7f9fc; }
        QTabBar { background: #e7edf5; }
        QTabBar::tab { background: #e3eaf3; padding: 9px 12px; margin: 0;
                      border: 1px solid #c6d1df; border-bottom: 0; color: #647b97; }
        QTabBar::tab:selected { color: #294e82; background: #ffffff; border-top: 3px solid #5383c3; }
        QTabBar::tab:hover { background: #f3f7fc; }
        QTabWidget#InspectorTabs QTabBar::tab:selected { border-top-color: #9278bd; color: #665182; }
        QTabWidget#ActivityTabs QTabBar::tab:selected { border-top-color: #4a9a93; color: #326962; }
        QGroupBox { border: 1px solid #c6d4e8; border-radius: 2px; margin-top: 12px;
                    padding: 8px 4px 4px; background: #f2f6fc; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 5px;
                         color: #446ba2; font-size: 12px; font-weight: 600; }
        QGroupBox[section="molecule"] { border-color: #cfc5e3; background: #f7f4fb; }
        QGroupBox[section="molecule"]::title { color: #8362a0; }
        QGroupBox[section="resources"] { border-color: #decca9; background: #fcf8ef; }
        QGroupBox[section="resources"]::title { color: #aa7b35; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #ffffff; color: #324e6d;
                    border: 1px solid #b9c9dd; border-radius: 2px; padding: 4px; min-height: 17px; }
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border-color: #5988c5; background: #fbfdff; }
        QComboBox::drop-down { border: 0; width: 22px; background: #edf2f9; }
        QSpinBox::up-button, QSpinBox::down-button { border: 0; background: #edf2f8; width: 20px; }
        QPlainTextEdit, QTextBrowser { background: #ffffff; border: 1px solid #c2cfe0;
                                     border-radius: 2px; padding: 7px; selection-background-color: #d2e3fa; }
        QPlainTextEdit:focus { border-color: #7c94bf; }
        QTableView, QTreeView { background: #ffffff; alternate-background-color: #f0f4f9;
                    border: 0; gridline-color: #dce5f0; selection-background-color: #dce9fa;
                    selection-color: #234c7b; outline: 0; }
        QTreeView::item { min-height: 26px; padding: 2px; }
        QTableView::item { padding: 4px; }
        QHeaderView::section { background: #e9eff7; color: #5c7493; padding: 6px 7px;
                    border: 0; border-right: 1px solid #d3ddec; border-bottom: 1px solid #c0cede; font-size: 11px; }
        QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #ffffff,stop:1 #eaf0f8);
                    color: #436180; border: 1px solid #b8c9de; border-radius: 2px;
                    padding: 6px 10px; min-height: 18px; }
        QPushButton:hover { background: #e4edfc; border-color: #88a7d0; }
        QPushButton:pressed { background: #ceddf3; }
        QPushButton:disabled { color: #a5b3c5; border-color: #d8e1eb; background: #f2f5f9; }
        QPushButton#Submit { background: #416eaf; color: #ffffff; border-color: #315e9e; }
        QPushButton#Submit:hover { background: #315e9d; }
        QPushButton#Submit:disabled { background: #afc2de; border-color: #afc2de; }
        QStatusBar { background: #dce5f0; color: #57708e; border-top: 1px solid #b7c7db;
                    padding: 3px 7px; font-size: 11px; }
        QStatusBar::item { border: 0; }
        QLabel#DocumentTitle { padding: 9px 12px; font-size: 13px; font-weight: 600;
                             color: #d1e0f3; background: #253b58; }
        QToolBar#SceneTools { background: #edf2f8; border: 0; border-bottom: 1px solid #b9c9dd;
                             spacing: 3px; padding: 3px 5px; }
        QSplitter::handle { background: #d8e1ed; }
        QScrollArea { border: 0; background: #f4f6fa; }
        QScrollBar:vertical { width: 10px; background: #edf2f8; }
        QScrollBar::handle:vertical { background: #b5c6dc; min-height: 26px; border-radius: 2px; margin: 2px; }
        QToolTip { background: #fffff0; color: #324b67; border: 1px solid #b5bfd0; padding: 6px; }
    
    """)



def apply_research_theme(application):
    apply_theme(application)
    font = QFont('Microsoft YaHei UI')
    font.setPixelSize(11)
    application.setFont(font)
    application.setStyleSheet("""
        QWidget { color: #202737; font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 11px; }
        QMainWindow, QDialog { background: #dadde5; }
        QMenuBar { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #f9f9fb,stop:1 #e4e6ed);
                   border-bottom: 1px solid #b6bac8; padding: 1px 5px; }
        QMenuBar::item { padding: 3px 9px; background: transparent; }
        QMenuBar::item:selected { background: #d4e1f9; }
        QMenu { background: #f8f9fc; border: 1px solid #aeb7cb; padding: 3px; }
        QMenu::item { padding: 5px 24px; }
        QMenu::item:selected { background: #d7e5ff; }
        QToolBar { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #ffffff,stop:1 #e8eaf2);
                   border: 0; border-bottom: 1px solid #b6bdcf; spacing: 2px; padding: 3px; }
        QToolBar#ModelTools { spacing: 2px; padding: 1px 4px; }
        QToolBar#SceneTools { padding: 0 4px; spacing: 2px; }
        QToolBar::separator { background: #b4bccd; width: 1px; margin: 3px 5px; }
        QToolButton { border: 1px solid transparent; padding: 2px 6px; border-radius: 1px; }
        QToolButton#ElementButton { min-width: 17px; padding: 1px 2px; margin: 0 1px; }
        QToolButton:hover { background: #e0e9fa; border-color: #a2b5d5; }
        QToolButton:checked { background: #cedff7; border-color: #7d9ecb; }
        QDockWidget { background: #f4f5f9; }
        QWidget#DockTitle, QWidget#MdiTitle {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #626898,stop:.65 #454d86,stop:1 #333c7c);
            border-bottom: 1px solid #2f386b;
        }
        QLabel#DockCaption, QLabel#MdiCaption { color: #ffffff; font-weight: 600; font-size: 11px; }
        QWidget#DocumentShell { background: #eef0f6; border: 1px solid #b1b7cb; }
        QPushButton#WindowControl { background: transparent; color: #ffffff; border: 0; padding: 0;
                                    min-height: 0; font-size: 14px; }
        QPushButton#WindowControl:hover { background: #8a94c4; }
        QTabWidget::pane { background: #f5f6fa; border: 1px solid #b8bece; }
        QTabBar::tab { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #fafbfc,stop:1 #dfe2eb);
                      color: #4a5268; border: 1px solid #b7bece; border-bottom: 0;
                      padding: 3px 9px; margin: 0 1px 0 0; }
        QTabBar::tab:selected { background: #ffffff; color: #294c8d; border-top: 2px solid #5a77bb; }
        QTabWidget#InspectorTabs QTabBar::tab:selected,
        QTabWidget#ActivityTabs QTabBar::tab:selected { color: #294c8d; border-top-color: #5a77bb; }
        QWidget#InspectorPage { background: #f5f6fa; }
        QScrollArea { border: 0; background: #f5f6fa; }
        QTableView, QTreeView { background: #ffffff; alternate-background-color: #eef0f5;
                              gridline-color: #cdd2df; selection-background-color: #c4d9fd;
                              selection-color: #1c315d; border: 0; outline: 0; }
        QTreeView::item { min-height: 18px; padding: 0 2px; }
        QTableView::item { padding: 1px 4px; }
        QHeaderView::section { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #f7f8fc,stop:1 #d7dbe8);
                              border: 0; border-right: 1px solid #bac2d4;
                              border-bottom: 1px solid #abb7cf; padding: 3px; color: #333d54; }
        QGroupBox, QGroupBox[section="molecule"], QGroupBox[section="resources"] {
            border: 1px solid #c0c6d5; border-radius: 0; margin-top: 10px;
            background: #f2f3f8; padding: 6px 2px 2px;
        }
        QGroupBox::title, QGroupBox[section="molecule"]::title, QGroupBox[section="resources"]::title {
            color: #4c5578; left: 5px; padding: 0 4px; font-size: 11px;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background: #ffffff; border: 1px solid #b8c0d2; border-radius: 1px; padding: 3px;
            min-height: 16px; color: #2e3850;
        }
        QComboBox::drop-down { border: 0; width: 19px; background: #e8ecf5; }
        QSpinBox::up-button, QSpinBox::down-button { border: 0; width: 18px; background: #e8ecf5; }
        QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #ffffff,stop:1 #dfe5f2);
                      border: 1px solid #aebbd0; border-radius: 2px; padding: 3px 7px; min-height: 16px; }
        QPushButton:hover { background: #dce7fa; border-color: #7f9bc2; }
        QPushButton:disabled { background: #eceff5; color: #a6acbd; border-color: #d2d8e2; }
        QPushButton#Submit { background: #426fc0; color: #ffffff; border-color: #325ba9; }
        QPlainTextEdit, QTextBrowser { background: #ffffff; border: 1px solid #c0c7d8;
                                     border-radius: 1px; padding: 4px; }
        QSplitter::handle { background: #d7dbe6; width: 3px; height: 3px; }
        QStatusBar { background: #eef0f6; border-top: 1px solid #b5bdcd; padding: 1px 4px; color: #505b73; }
        QStatusBar::item { border: 0; }
        QScrollBar:vertical { background: #eff1f6; width: 9px; }
        QScrollBar::handle:vertical { background: #c3c9d9; border-radius: 3px; min-height: 24px; margin: 1px; }
        QToolTip { background: #fffef1; color: #37435b; border: 1px solid #bfc3ce; padding: 5px; }
    """)

