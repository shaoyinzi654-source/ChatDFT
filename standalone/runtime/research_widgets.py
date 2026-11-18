from __future__ import annotations

import csv
import math
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMdiArea, QMdiSubWindow, QPushButton, QVBoxLayout, QWidget,
)

from theme import symbol_icon


def read_plot_csv(path):
    path = Path(path)
    if path.stat().st_size > 5 * 1024 * 1024:
        raise ValueError('曲线文件超过 5 MB。')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError('曲线文件为空。')
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        headers = rows.pop(0)
    else:
        headers = ['X', *[f'Y{i}' for i in range(1, len(rows[0]))]]
    if not 2 <= len(headers) <= 5:
        raise ValueError('CSV 需要 2–5 列，第一列为 X，其余列为 Y。')
    parsed = []
    for row in rows:
        if not row:
            continue
        if len(row) != len(headers):
            raise ValueError('曲线文件的列数不一致。')
        values = [float(value) for value in row]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('曲线数据必须是有限数值。')
        parsed.append(values)
    if not 2 <= len(parsed) <= 50000:
        raise ValueError('曲线需要 2–50000 个数据点。')
    parsed.sort(key=lambda row: row[0])
    return headers, parsed


class DataPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.headers = []
        self.rows = []
        self.caption = ''
        self.kind = 'line'
        self.category_labels = []
        self.series_colors = ['#3c68d7', '#be3cb6', '#26998b', '#d98e31']
        self.setMinimumSize(180, 100)

    def set_data(self, headers, rows, caption, kind='line', categories=None):
        self.headers, self.rows, self.caption = headers, rows, caption
        self.kind = kind
        self.category_labels = categories or []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#ffffff'))
        font = painter.font()
        font.setPixelSize(10)
        painter.setFont(font)
        painter.setPen(QColor('#5a6270'))
        if not self.rows:
            painter.drawText(self.rect().adjusted(14, 14, -14, -14), Qt.AlignCenter,
                             '导入外部 CSV 数据\n支持 DOS、光谱等分析曲线')
            return
        frame = QRectF(79, 33, max(30, self.width() - 96), max(25, self.height() - 69))
        x0, x1 = self.rows[0][0], self.rows[-1][0]
        values = [value for row in self.rows for value in row[1:]]
        y0, y1 = min(values), max(values)
        if self.kind == 'bar':
            y0, y1 = min(0, y0), max(0, y1)
            x0, x1 = x0 - .6, x1 + .6
        span = max(y1 - y0, max(abs(y0), abs(y1), 1) * 1e-7)
        y0, y1 = y0 - span * .08, y1 + span * .08
        dx, dy = max(x1 - x0, 1e-12), y1 - y0
        ticks = max(2, min(5, int(frame.height() // 25)))
        for step in range(ticks):
            fraction = step / (ticks - 1)
            y = frame.bottom() - fraction * frame.height()
            painter.setPen(QPen(QColor('#c9cfd9'), .6, Qt.DotLine))
            painter.drawLine(QPointF(frame.left(), y), QPointF(frame.right(), y))
            painter.setPen(QColor('#4c5665'))
            value = y0 + fraction * dy
            painter.drawText(QRectF(24, y - 7, 48, 15), Qt.AlignRight | Qt.AlignVCenter, f'{value:.5g}')
        xticks = [row[0] for row in self.rows] if len(self.rows) <= 8 else [
            x0 + dx * i / 4 for i in range(5)]
        for i, value in enumerate(xticks):
            x = frame.left() + (value - x0) / dx * frame.width()
            painter.setPen(QPen(QColor('#d5dae3'), .6, Qt.DotLine))
            painter.drawLine(QPointF(x, frame.top()), QPointF(x, frame.bottom()))
            painter.setPen(QColor('#4c5665'))
            label = self.category_labels[i] if i < len(self.category_labels) else f'{value:.4g}'
            painter.drawText(QRectF(x - 26, frame.bottom() + 4, 52, 15), Qt.AlignCenter, label)
        painter.setPen(QPen(QColor('#6d7482'), .8))
        painter.drawRect(frame)
        painter.save()
        painter.setClipRect(frame.adjusted(-1, -1, 1, 1))
        if self.kind == 'bar':
            zero = frame.bottom() - (0 - y0) / dy * frame.height()
            painter.setPen(QPen(QColor('#8c94a3'), .8))
            painter.drawLine(QPointF(frame.left(), zero), QPointF(frame.right(), zero))
            width = min(42, frame.width() / max(2, len(self.rows)) * .45)
            for row in self.rows:
                x = frame.left() + (row[0] - x0) / dx * frame.width()
                y = frame.bottom() - (row[1] - y0) / dy * frame.height()
                color = QColor('#5580cf' if row[1] < 0 else '#aa72bf')
                painter.setPen(QPen(color.darker(115), .7))
                painter.setBrush(color)
                painter.drawRect(QRectF(x - width / 2, min(y, zero), width, abs(y - zero)))
        else:
            for column in range(1, len(self.headers)):
                path = QPainterPath()
                stride = max(1, len(self.rows) // max(100, self.width() * 2))
                sampled = self.rows[::stride]
                if sampled[-1] is not self.rows[-1]:
                    sampled = [*sampled, self.rows[-1]]
                points = []
                for i, row in enumerate(sampled):
                    point = QPointF(frame.left() + (row[0] - x0) / dx * frame.width(),
                                    frame.bottom() - (row[column] - y0) / dy * frame.height())
                    points.append(point)
                    path.moveTo(point) if i == 0 else path.lineTo(point)
                color = QColor(self.series_colors[column - 1])
                painter.setPen(QPen(color, 1.5))
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)
                if len(points) <= 60:
                    painter.setBrush(color)
                    for point in points:
                        painter.drawEllipse(point, 2, 2)
        painter.restore()
        title = self.caption if len(self.headers) <= 2 else ' / '.join(self.headers[1:])
        title = painter.fontMetrics().elidedText(title, Qt.ElideRight, int(frame.width()))
        painter.setPen(QColor('#505969'))
        painter.drawText(QRectF(frame.left(), 5, frame.width(), 20), Qt.AlignLeft | Qt.AlignVCenter, title)
        painter.drawText(QRectF(frame.left(), self.height() - 20, frame.width(), 16), Qt.AlignCenter,
                         self.headers[0])
        painter.save()
        painter.translate(13, frame.center().y())
        painter.rotate(-90)
        ylabel = self.headers[1] if len(self.headers) == 2 else 'Y'
        painter.drawText(QRectF(-frame.height() / 2, -9, frame.height(), 17), Qt.AlignCenter, ylabel)
        painter.restore()


class EnergyPlot(DataPlot):
    def __init__(self):
        super().__init__()
        self.series = []
        self.series_colors[0] = '#df8a31'

    def set_data(self, values, caption):
        self.series = list(values)
        rows = [[i + 1, value] for i, value in enumerate(values)]
        super().set_data(['SCF 迭代步', 'E / Eh'], rows, caption)


class DocumentCaption(QLabel):
    def __init__(self, text):
        super().__init__()
        self.full_text = text
        self.setMinimumWidth(0)
        self.setText(text)

    def setText(self, text):
        self.full_text = text
        self.setToolTip(text)
        super().setText(self.fontMetrics().elidedText(text, Qt.ElideMiddle, max(1, self.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setText(self.full_text)


class DocumentTitleBar(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.drag_origin = None
        self.setObjectName('MdiTitle')
        self.setAttribute(Qt.WA_StyledBackground, True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_origin = (event.globalPosition(), self.window.pos())
            self.window.mdiArea().setActiveSubWindow(self.window)

    def mouseMoveEvent(self, event):
        if self.drag_origin and event.buttons() & Qt.LeftButton and not self.window.isMaximized():
            origin, position = self.drag_origin
            self.window.mdiArea().arranged = False
            self.window.move(position + (event.globalPosition() - origin).toPoint())

    def mouseReleaseEvent(self, event):
        self.drag_origin = None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.window.toggle_maximize()


class ResearchWindow(QMdiSubWindow):
    def __init__(self, title, widget, parent):
        super().__init__(parent, Qt.FramelessWindowHint | Qt.SubWindow)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setMinimumSize(190, 140)
        shell = QWidget()
        shell.setObjectName('DocumentShell')
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        titlebar = DocumentTitleBar(self)
        row = QHBoxLayout(titlebar)
        row.setContentsMargins(6, 2, 2, 2)
        row.setSpacing(3)
        icon = QLabel()
        icon.setPixmap(symbol_icon('model', '#d5e5ff').pixmap(15, 15))
        row.addWidget(icon)
        self.title_label = DocumentCaption(title)
        self.title_label.setObjectName('MdiCaption')
        row.addWidget(self.title_label, 1)
        for text, tooltip, callback in (
                ('−', '最小化', self.showMinimized), ('□', '最大化 / 还原', self.toggle_maximize),
                ('×', '关闭窗口', self.close)):
            button = QPushButton(text)
            button.setObjectName('WindowControl')
            button.setFixedSize(19, 18)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addWidget(titlebar)
        layout.addWidget(widget, 1)
        self.setWidget(shell)
        self.windowTitleChanged.connect(self.title_label.setText)

    def toggle_maximize(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()


class ResearchArea(QMdiArea):
    layoutChanged = Signal(str)

    def __init__(self):
        super().__init__()
        self.setBackground(QColor('#b7bdc9'))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.windows = []
        self.arranged = True
        self.layout_mode = 'four'

    def add_document(self, widget, title):
        window = ResearchWindow(title, widget, self)
        self.addSubWindow(window)
        self.windows.append(window)
        window.show()
        return window

    def currentIndex(self):
        return self.windows.index(self.activeSubWindow()) if self.activeSubWindow() in self.windows else 0

    def setCurrentIndex(self, index):
        if 0 <= index < len(self.windows):
            self.windows[index].show()
            self.setActiveSubWindow(self.windows[index])

    def arrange(self):
        width, height = self.viewport().width(), self.viewport().height()
        left, top = width // 2, int(height * .58)
        if self.layout_mode == 'models':
            geometries = [(0, 0, left, height), (left, 0, width - left, height)]
        else:
            geometries = [(0, 0, left, top), (left, 0, width - left, top),
                          (0, top, left, height - top), (left, top, width - left, height - top)]
        for window, geometry in zip(self.windows[:4], geometries):
            if not window.isHidden():
                window.showNormal()
                window.setGeometry(*geometry)

    def cascadeSubWindows(self):
        self.arranged = False
        super().cascadeSubWindows()

    def tile_all(self):
        self.arranged = True
        self.layout_mode = 'four'
        for window in self.windows[:4]:
            window.showNormal()
        self.arrange()
        self.layoutChanged.emit('four')

    def focus_models(self):
        self.arranged = True
        self.layout_mode = 'models'
        for window in self.windows[:2]:
            window.showNormal()
        for window in self.windows[2:]:
            window.hide()
        self.arrange()
        self.layoutChanged.emit('models')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.arranged and not any(w.isMaximized() for w in self.windows):
            self.arrange()

