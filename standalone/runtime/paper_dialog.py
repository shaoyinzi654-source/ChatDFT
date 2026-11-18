from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from chatdft.paper import (
    FIELDS, analyze_paper, calculation_fields, compare_energy, fact, prepare_task,
    read_paper, source_pages,
)
from chatdft.storage import atomic_json
from chatdft.md_inputs import identifies_pdc_paper
from panel import TASKS


class PaperDialog(QDialog):
    def __init__(self, workbench):
        super().__init__(workbench)
        self.workbench = workbench
        self.plan = None
        self.prepared = {}
        self.comparisons = {}
        self.loaded_pages = None
        self.source_name = '粘贴的论文文本'
        self.source_first = 1
        self.geometry = None
        self.setWindowTitle('论文复刻')
        self.resize(810, 630)
        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.status = QLabel('导入论文或粘贴方法、结果与补充材料。')
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.create_source_tab()
        self.create_plan_tab()

    def create_source_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        self.first_page = QSpinBox()
        self.first_page.setRange(1, 9999)
        self.last_page = QSpinBox()
        self.last_page.setRange(0, 9999)
        self.last_page.setSpecialValueText('末页')
        self.button(toolbar, '导入 PDF / 文本…', self.import_source)
        toolbar.addWidget(QLabel('PDF 页码'))
        toolbar.addWidget(self.first_page)
        toolbar.addWidget(QLabel('至'))
        toolbar.addWidget(self.last_page)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.source = QPlainTextEdit()
        self.source.setPlaceholderText('也可直接粘贴论文文字。扫描 PDF 请先 OCR。')
        layout.addWidget(self.source, 1)
        self.source.textChanged.connect(self.source_edited)
        self.focus = QLineEdit()
        self.focus.setPlaceholderText('可选：指定要复现的表格、分子或计算步骤')
        layout.addWidget(self.focus)
        footer = QHBoxLayout()
        self.provider = QLabel()
        footer.addWidget(self.provider, 1)
        self.reproduce_button = self.button(footer, '提取复刻方案', self.analyze)
        layout.addLayout(footer)
        self.tabs.addTab(page, '论文原文')

    def create_plan_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.tasks = QComboBox()
        self.tasks.currentIndexChanged.connect(self.show_task)
        layout.addWidget(self.tasks)
        self.details = QPlainTextEdit(readOnly=True)
        layout.addWidget(self.details, 1)
        geometry = QHBoxLayout()
        self.button(geometry, '选择原始 XYZ / XSD…', self.select_geometry)
        self.button(geometry, '清除', self.clear_geometry)
        self.geometry_label = QLabel('使用原文坐标（如有）')
        geometry.addWidget(self.geometry_label, 1)
        layout.addLayout(geometry)
        self.approximate = QCheckBox('允许用论文中的 SMILES 生成近似初始几何')
        layout.addWidget(self.approximate)
        footer = QHBoxLayout()
        self.prepare_button = self.button(footer, '载入计算面板', lambda: self.prepare(False))
        self.submit_button = self.button(footer, '准备并提交…', lambda: self.prepare(True))
        self.compare_button = self.button(footer, '对照所选作业', self.compare)
        self.export_button = self.button(footer, '导出方案…', self.export_plan)
        footer.addStretch()
        layout.addLayout(footer)
        self.tabs.addTab(page, '复刻方案')
        self.tabs.setTabEnabled(1, False)

    @staticmethod
    def button(layout, title, callback):
        button = QPushButton(title)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def showEvent(self, event):
        self.reflect_paper_workflow()
        super().showEvent(event)

    def source_edited(self):
        self.loaded_pages = None
        self.source_name = '粘贴的论文文本'
        self.source_first = 1
        self.plan = None
        self.prepared.clear()
        self.comparisons.clear()
        self.clear_geometry()
        self.tabs.setTabEnabled(1, False)
        self.reflect_paper_workflow()

    def reflect_paper_workflow(self):
        pdc = identifies_pdc_paper(self.source.toPlainText())
        self.reproduce_button.setText('复刻并查看 21 套周期模型' if pdc else '提取复刻方案')
        self.provider.setText('Li 2024 · PDC 周期溶液' if pdc else
                              self.workbench.ai_provider_name() + ' · 分析会发送上述文字')

    def import_source(self):
        filename, _ = QFileDialog.getOpenFileName(self, '导入论文', '', '论文 (*.pdf *.txt *.md)')
        if not filename:
            return
        path = Path(filename)
        first = self.first_page.value() if path.suffix.lower() == '.pdf' else 1
        last = (self.last_page.value() or None) if path.suffix.lower() == '.pdf' else None
        if last is not None and last < first:
            self.status.setText('结束页不能小于起始页。')
            return
        self.run('正在提取论文原文…', lambda: read_paper(path, first, last),
                 lambda pages: self.source_loaded(path, first, pages))

    def source_loaded(self, path, first, pages):
        self.source.setPlainText('\n\n'.join(pages))
        self.loaded_pages = pages
        self.source_name = path.name
        self.source_first = first
        self.status.setText(f'{path.name} · 第 {first}–{first + len(pages) - 1} 页 · {sum(map(len, pages)):,} 字符')

    def run(self, message, function, done):
        if self.workbench.operation:
            self.status.setText('工作台正在执行任务，请稍后重试。')
            return
        self.setEnabled(False)
        self.status.setText(message)
        self.workbench.run(message, function, done)
        operation = self.workbench.operation
        operation.failure.connect(self.status.setText)
        operation.finished.connect(lambda: self.setEnabled(True))

    def analyze(self):
        if identifies_pdc_paper(self.source.toPlainText()):
            self.workbench.prepare_pdc_paper()
            return
        try:
            pages = self.loaded_pages or source_pages(self.source.toPlainText())
        except ValueError as error:
            self.status.setText(str(error))
            return
        settings = dict(self.workbench.settings)
        first, name, focus = self.source_first, self.source_name, self.focus.text().strip()
        self.run('正在提取论文条件和原文依据…',
                 lambda: analyze_paper(settings, pages, first, name, focus), self.receive)

    def receive(self, plan):
        self.plan = plan
        self.prepared.clear()
        self.comparisons.clear()
        self.tasks.blockSignals(True)
        self.tasks.clear()
        for index, task in enumerate(plan['tasks'], 1):
            self.tasks.addItem(f"{index}. {task['name'] or '论文计算'}")
        self.tasks.blockSignals(False)
        self.show_task()
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)
        self.status.setText(f"已提取 {self.tasks.count()} 项计算。请核对参数及原文摘录。")

    def show_task(self, *_):
        index = self.tasks.currentIndex()
        if not self.plan or index < 0:
            self.details.setPlainText('所选内容没有可提取的计算任务。请补充计算方法或补充材料。')
            for button in (self.prepare_button, self.submit_button, self.compare_button):
                button.setEnabled(False)
            return
        task = self.plan['tasks'][index]
        lines = [self.plan['title'], self.plan['source'], '']
        try:
            calculation_fields(task)
            ready = True
            lines.append('参数可载入 PySCF，提交前请核对几何来源和引擎差异。')
        except ValueError as error:
            ready = False
            lines.append('需补充或调整：' + str(error))
        for key, title in FIELDS.items():
            item = task['facts'][key]
            value = item['value'] if key != 'coordinates' or not item['value'] else '见下方坐标摘录'
            lines.extend(['', f'{title}  {value or "未报告"}'])
            if item['verified']:
                lines.append(f"第 {item['page']} 页  {item['quote']}")
            elif value:
                lines.append('找不到对应原文，需核对。')
        lines.extend(['', task['notes']])
        if fact(task, 'engine').casefold() != 'pyscf':
            lines.append('当前运行引擎为 PySCF。请核对与原文软件的算法、网格和收敛设置差异。')
        if index in self.comparisons:
            result = self.comparisons[index]
            lines.extend(['', f"作业 {result['job_id']} · {result['geometry_origin']}",
                          f"原文 {result['reference_eh']:.12g} Eh  计算 {result['calculated_eh']:.12g} Eh",
                          f"差值（计算 − 原文）{result['delta_eh']:+.6g} Eh"])
        self.details.setPlainText('\n'.join(lines))
        self.prepare_button.setEnabled(ready)
        self.submit_button.setEnabled(ready)
        self.compare_button.setEnabled(index in self.prepared)

    def select_geometry(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择补充材料坐标', '', '结构 (*.xyz *.xsd)')
        if path:
            self.geometry = Path(path)
            self.geometry_label.setText(self.geometry.name)

    def clear_geometry(self):
        self.geometry = None
        self.geometry_label.setText('使用原文坐标（如有）')

    def prepare(self, submit):
        index = self.tasks.currentIndex()
        if not self.plan or index < 0:
            return
        plan, geometry, approximate = self.plan, self.geometry, self.approximate.isChecked()
        self.run('正在校验论文参数和几何…', lambda: prepare_task(plan, index, geometry, approximate),
                 lambda prepared: self.apply_prepared(index, prepared, submit))

    def apply_prepared(self, index, prepared, submit):
        structure, values, origin = prepared
        w = self.workbench
        w.remember()
        w.restore_model(structure.to_dict())
        w.method.setCurrentText(values['method'])
        w.basis.setCurrentText(values['basis'])
        w.solvent.setCurrentText(values['solvent'])
        w.task.setCurrentText(next(title for title, key in TASKS.items() if key == values['task']))
        w.set_phase(1)
        w.tabs.setCurrentIndex(1)
        w.documents.setCurrentIndex(0)
        self.prepared[index] = prepared
        self.comparisons.pop(index, None)
        self.show_task()
        summary = f'已载入论文任务 · {origin} · {values["method"]}/{values["basis"]}'
        self.status.setText(summary)
        w.status.setText(summary)
        self.hide()
        if submit:
            QTimer.singleShot(0, self, self.submit_when_ready)

    def submit_when_ready(self):
        if self.workbench.operation:
            QTimer.singleShot(50, self, self.submit_when_ready)
        else:
            self.workbench.submit_job()

    def compare(self):
        index = self.tasks.currentIndex()
        try:
            job = self.workbench.current_job()
            self.comparisons[index] = compare_energy(self.plan, index, job, self.workbench.read_result(job),
                                                     self.prepared[index])
        except (ValueError, OSError) as error:
            self.status.setText(str(error))
            return
        self.show_task()
        self.status.setText('电子能对照已更新。')

    def export_plan(self):
        if not self.plan:
            return
        target, _ = QFileDialog.getSaveFileName(self, '导出复刻方案', 'paper-reproduction.json', 'JSON (*.json)')
        if target:
            try:
                atomic_json(target, {**self.plan, 'prepared': {
                    str(index + 1): dict(structure=s.to_dict(), calculation=c, geometry_origin=o)
                    for index, (s, c, o) in self.prepared.items()
                }, 'comparisons': {str(index + 1): row for index, row in self.comparisons.items()}})
                self.status.setText(f'已导出 {Path(target).name}。')
            except OSError as error:
                self.status.setText(str(error))
