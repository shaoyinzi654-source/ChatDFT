"""The plot template picker shared by the job and curve views."""
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListView,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from chart_templates import THEMES, curve_templates, make_figure, result_templates
from research_widgets import read_plot_csv


def _example_result():
    return result_templates({
        'scf_energies_hartree': [-76.11, -76.21, -76.232, -76.239, -76.2397, -76.23998, -76.24],
        'mulliken_charges': [{'index': i, 'charge': q} for i, q in enumerate(
            [-.61, .31, .30, -.38, .20, .18, -.45, .25, .20, -.25, .11, .14], 1)],
        'alpha_homo_ev': -6.2, 'alpha_lumo_ev': -1.4, 'beta_homo_ev': -5.8, 'beta_lumo_ev': -1.8,
        'frequencies_cm1': [-32, 112, 243, 572, 1032, 1490, 1803, 3048, 3281],
        'ir_intensities_km_mol': [0, 2, 34, 10, 80, 12, 105, 61, 96],
        'excited_states': [{'energy_ev': ev, 'wavelength_nm': 1239.841984/ev, 'oscillator_strength': strength}
                           for ev, strength in [(2.14, .05), (2.72, .30), (3.25, .08), (3.8, .18), (4.3, .12)]],
    })


def _example_curves():
    x = np.linspace(0, 10, 19)
    y = .18 + .9 * np.exp(-((x - 3.8) / 1.55)**2) + .5 * np.exp(-((x - 7.6) / 1.35)**2)
    z = .19 + .55 * np.exp(-((x - 4.7) / 1.9)**2) + .73 * np.exp(-((x - 8.05) / .9)**2)
    return curve_templates(['X', 'Series A', 'Series B'], list(zip(x, y, z)))


def _thumbnail(kind, theme):
    width, height = 126, 63
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(theme.paper))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor(theme.grid), 1))
    for y in (15, 28, 41, 53):
        painter.drawLine(14, y, 113, y)
    painter.setPen(QPen(QColor(theme.ink), 1))
    painter.drawLine(14, 9, 14, 53)
    painter.drawLine(14, 53, 113, 53)
    primary, contrast = QColor(theme.colors[0]), QColor(theme.colors[1])
    painter.setPen(QPen(primary, 2))
    if kind == 'panels':
        for index, points in enumerate(((22, 46, 31, 32, 45, 35, 56, 20), (73, 43, 86, 28, 97, 39, 108, 18))):
            painter.setPen(QPen((primary, contrast)[index], 2))
            for n in range(0, 6, 2):
                painter.drawLine(*points[n:n+4])
    elif kind.startswith('hist') or kind in ('bar', 'stem'):
        for i, h in enumerate((11, 24, 34, 20, 27, 15)):
            painter.fillRect(21 + 15*i, 53-h, 3 if kind == 'stem' else 10, h, primary if i < 4 else contrast)
    elif kind == 'levels':
        for x in (30, 77):
            painter.setPen(QPen(contrast, 3))
            painter.drawLine(x, 16, x+23, 16)
            painter.setPen(QPen(primary, 3))
            painter.drawLine(x, 40, x+23, 40)
    elif kind == 'rank':
        for i, length in enumerate((54, 36, 27, 19, 14)):
            y = 13 + i*8
            painter.drawLine(25, y, 25 + length, y)
            painter.drawEllipse(23 + length, y-2, 4, 4)
    else:
        points = ((22, 45), (37, 37), (51, 17), (65, 28), (79, 41), (94, 24), (108, 31))
        path = QPainterPath()
        path.moveTo(*points[0])
        for index, (x,y) in enumerate(points[1:], 1):
            if kind == 'step':
                path.lineTo(x, points[index-1][1])
            path.lineTo(x,y)
        if kind == 'area':
            path.lineTo(points[-1][0], 53)
            path.lineTo(points[0][0], 53)
            fill = QColor(primary)
            fill.setAlpha(55)
            painter.fillPath(path, fill)
        if kind == 'scatter':
            for x,y in points:
                painter.drawEllipse(x-2, y-2, 4, 4)
        else:
            painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)


SIZES = {'自适应': None, '标准 8 × 5 英寸': (8, 5), '宽图 11 × 6 英寸': (11, 6), '方图 7 × 7 英寸': (7, 7)}
EXPORT_FORMATS = {'PNG 图片 (*.png)': 'png', 'SVG 矢量图 (*.svg)': 'svg', 'PDF 矢量图 (*.pdf)': 'pdf'}

GALLERY_STYLE = '''
QDialog#plotGallery { background-color: #edf2f7; }
QDialog#plotGallery QLabel { color: #334959; }
QDialog#plotGallery QLabel#galleryTitle { color: #172938; font-size: 19px; font-weight: bold; }
QDialog#plotGallery QComboBox, QDialog#plotGallery QLineEdit {
    background: #fff; color: #1a2b39; border: 1px solid #bccbd7;
    border-radius: 4px; min-height: 26px; padding-left: 7px;
}
QDialog#plotGallery QComboBox:focus, QDialog#plotGallery QLineEdit:focus { border-color: #23779a; }
QDialog#plotGallery QListWidget {
    background: #f8fafc; color: #22384a; border: 1px solid #c8d4de;
    border-radius: 4px; padding: 3px; outline: none;
}
QDialog#plotGallery QListWidget::item:selected { background: #d3e6f2; color: #0e3853; border-radius: 3px; }
QDialog#plotGallery QListWidget::item:hover:!selected { background: #e8f0f6; }
QDialog#plotGallery QPushButton {
    background: #fdfefe; color: #274050; border: 1px solid #bfced9;
    border-radius: 4px; padding: 6px 11px;
}
QDialog#plotGallery QPushButton:hover { background: #e4eff7; }
QDialog#plotGallery QPushButton#exportFigure { background: #17678e; color: #fff; border-color: #17678e; }
QDialog#plotGallery QPushButton#exportFigure:hover { background: #0f5477; }
QDialog#plotGallery QPushButton#exportFigure:disabled { background: #adbcc7; border-color: #adbcc7; }
'''


class PlotGallery(QDialog):
    def __init__(self, parent, result=None, curve=None):
        super().__init__(parent)
        self.setWindowTitle('ChatDFT 绘图模板库')
        self.setObjectName('plotGallery')
        self.setStyleSheet(GALLERY_STYLE)
        self.resize(1180, 750)
        self.setMinimumSize(850, 560)
        self.catalogs = {}
        self.templates = {}

        page = QVBoxLayout(self)
        page.setContentsMargins(18, 16, 18, 18)
        page.setSpacing(12)
        title = QLabel('绘图模板库')
        title.setObjectName('galleryTitle')
        page.addWidget(title)
        layout = QHBoxLayout()
        page.addLayout(layout, 1)
        layout.setSpacing(20)
        sidebar = QWidget()
        sidebar.setMinimumWidth(318)
        sidebar.setMaximumWidth(334)
        select = QVBoxLayout(sidebar)
        select.setContentsMargins(0, 0, 0, 0)
        select.addWidget(QLabel('数据源'))
        self.source = QComboBox()
        select.addWidget(self.source)
        imports = QHBoxLayout()
        csv = QPushButton('CSV…')
        csv.clicked.connect(self.import_csv)
        imports.addWidget(csv)
        results = QPushButton('结果 JSON…')
        results.clicked.connect(self.import_result)
        imports.addWidget(results)
        select.addLayout(imports)
        select.addSpacing(8)
        select.addWidget(QLabel('图形模板'))
        self.items = QListWidget()
        self.items.setIconSize(QSize(126,63))
        self.items.setGridSize(QSize(138, 88))
        self.items.setViewMode(QListView.IconMode)
        self.items.setResizeMode(QListView.Adjust)
        self.items.setMovement(QListView.Static)
        self.items.setWordWrap(True)
        self.items.setUniformItemSizes(True)
        select.addWidget(self.items, 1)
        self.count = QLabel()
        select.addWidget(self.count)
        layout.addWidget(sidebar)

        surface = QVBoxLayout()
        layout.addLayout(surface, 1)
        controls = QHBoxLayout()
        surface.addLayout(controls)
        controls.addWidget(QLabel('主题'))
        self.theme = QComboBox()
        for key, theme in THEMES.items():
            swatch = QPixmap(18, 10)
            swatch.fill(QColor(theme.colors[0]))
            self.theme.addItem(QIcon(swatch), theme.title, key)
        controls.addWidget(self.theme, 1)
        controls.addWidget(QLabel('导出尺寸'))
        self.dimensions = QComboBox()
        self.dimensions.addItems(SIZES)
        controls.addWidget(self.dimensions, 1)

        self.heading = QLineEdit()
        self.heading.setPlaceholderText('可选总标题，例如 PDC–镁水合簇')
        self.heading.setAccessibleName('自定义图件总标题')
        surface.addWidget(self.heading)
        self.preview = Figure(figsize=(8, 5), dpi=96, layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.preview)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(470, 350)
        surface.addWidget(self.canvas, 1)
        bottom = QHBoxLayout()
        surface.addLayout(bottom)
        self.message = QLabel()
        self.message.setWordWrap(True)
        bottom.addWidget(self.message, 1)
        self.export_button = QPushButton('导出图件…')
        self.export_button.setObjectName('exportFigure')
        self.export_button.setDefault(True)
        self.export_button.clicked.connect(self.export)
        bottom.addWidget(self.export_button)
        close = QPushButton('关闭')
        close.clicked.connect(self.accept)
        bottom.addWidget(close)

        catalogs = result_templates(result) if result else {}
        if catalogs:
            self.add_source('当前作业 · PySCF 结果', catalogs)
        if curve:
            headers, rows = curve
            self.add_source('工作台曲线 · 当前数据', curve_templates(headers, rows))
        self.add_source('演示数据 · PySCF 图型', _example_result(), demo=True)
        self.add_source('演示数据 · CSV 图型', _example_curves(), demo=True)
        self.source.currentIndexChanged.connect(self.update_templates)
        self.theme.currentIndexChanged.connect(self.update_icons)
        self.theme.currentIndexChanged.connect(self.redraw)
        self.items.currentItemChanged.connect(self.redraw)
        self.timer = QTimer(self, singleShot=True, interval=240)
        self.timer.timeout.connect(self.redraw)
        self.heading.textChanged.connect(lambda: self.timer.start())
        self.update_templates()

    def add_source(self, label, templates, demo=False):
        self.catalogs[label] = templates, demo
        if self.source.findText(label) < 0:
            self.source.addItem(label)

    def update_templates(self):
        self.templates, demo = self.catalogs[self.source.currentText()]
        self.items.blockSignals(True)
        self.items.clear()
        for key, template in self.templates.items():
            item = QListWidgetItem(template.name)
            item.setData(Qt.UserRole, key)
            item.setTextAlignment(Qt.AlignHCenter)
            item.setToolTip(template.category + ' · ' + template.name)
            self.items.addItem(item)
        self.update_icons()
        self.items.setCurrentRow(0)
        self.items.blockSignals(False)
        self.count.setText(f'{len(self.templates)} 个适用模板')
        self.message.setText('演示数据 · 仅用于预览图型' if demo else '使用所选数据的原始数值')
        self.redraw()

    def update_icons(self):
        theme = THEMES[self.theme.currentData()]
        for row in range(self.items.count()):
            item = self.items.item(row)
            charts = self.templates[item.data(Qt.UserRole)].charts
            item.setIcon(_thumbnail('panels' if len(charts) > 1 else charts[0].kind, theme))

    def selected_template(self):
        item = self.items.currentItem()
        return self.templates[item.data(Qt.UserRole)] if item else None

    def redraw(self):
        template = self.selected_template()
        self.export_button.setEnabled(template is not None)
        if template:
            make_figure(template, self.theme.currentData(), self.heading.text(), figure=self.preview)
        else:
            self.preview.clear()
        self.canvas.draw_idle()

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, '导入绘图数据', '', 'CSV (*.csv)')
        if not path:
            return
        try:
            headers, rows = read_plot_csv(path)
        except (OSError, ValueError, UnicodeError) as exc:
            QMessageBox.warning(self, 'CSV 导入失败', str(exc))
            return
        label = f'CSV · {Path(path).name}'
        self.add_source(label, curve_templates(headers, rows))
        self.source.setCurrentText(label)
        self.update_templates()

    def import_result(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开 PySCF 结果', '', 'JSON (*.json)')
        if not path:
            return
        file = Path(path)
        try:
            if file.stat().st_size > 20 * 1024 * 1024:
                raise ValueError('结果文件超过 20 MB。')
            result = json.loads(file.read_text(encoding='utf-8-sig'))
            templates = result_templates(result) if isinstance(result, dict) else {}
            if not templates:
                raise ValueError('文件中没有可用于绘图的 PySCF 结果字段。')
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            QMessageBox.warning(self, '结果导入失败', str(exc))
            return
        label = f'结果 · {file.name}'
        self.add_source(label, templates)
        self.source.setCurrentText(label)
        self.update_templates()

    def export(self):
        template = self.selected_template()
        if template is None:
            return
        path, filter_name = QFileDialog.getSaveFileName(self, '保存图件', 'ChatDFT-figure',
                                                       ';;'.join(EXPORT_FORMATS))
        if not path:
            return
        output = Path(path)
        suffix = output.suffix.lstrip('.').lower()
        if not suffix:
            suffix = EXPORT_FORMATS.get(filter_name, 'png')
            output = output.with_suffix('.' + suffix)
        if suffix not in EXPORT_FORMATS.values():
            QMessageBox.warning(self, '不支持的格式', '请选择 PNG、SVG 或 PDF。')
            return
        try:
            title = self.heading.text()
            if self.catalogs[self.source.currentText()][1]:
                title = f'{title.strip()} · Demo data' if title.strip() else 'Demo data'
            buffer = BytesIO()
            figure = make_figure(template, self.theme.currentData(), title, SIZES[self.dimensions.currentText()])
            figure.savefig(buffer, format=suffix, dpi=300, facecolor=figure.get_facecolor())
            output.write_bytes(buffer.getvalue())
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, '图件导出失败', str(exc))
            return
        self.message.setText('已导出 ' + output.name)
