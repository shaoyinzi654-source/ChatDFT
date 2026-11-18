import json
from dataclasses import asdict
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PySide6.QtCore import QItemSelectionModel, QPointF, QSignalBlocker, Qt
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from chatdft.md_analysis import analyze, compare_distribution, spatial_density
from chatdft.md_inputs import ASSETS, MDConfig, catalog, c4_pairs, literature_c4, paper_config, system_state
from chatdft.md_jobs import MDJobs, MDRemote
from chatdft.storage import atomic_json
from md_preview import PeriodicBoxView, load_box
from research_widgets import DataPlot
from theme import NumberBox


class DensityView(QWidget):
    """Interactive boundary-voxel surface of the saved density, without invented data."""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(250, 180)
        self.grid = None
        self.points = np.empty((0, 3))
        self.angle = .4
        self.pitch = .3
        self.last = None

    def set_grid(self, grid, threshold):
        self.grid = grid
        mask = grid >= threshold
        interior = mask.copy()
        for axis in range(3):
            interior &= np.roll(mask, 1, axis) & np.roll(mask, -1, axis)
        points = np.argwhere(mask & ~interior).astype(float)
        if len(points) > 8000:
            points = points[::int(np.ceil(len(points)/8000))]
        self.points = (points - np.asarray(grid.shape)/2) / max(grid.shape)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#223548'))
        painter.setRenderHint(QPainter.Antialiasing)
        if not len(self.points):
            painter.setPen(QColor('#dbe6ef'))
            painter.drawText(self.rect(), Qt.AlignCenter, '选择空间分布文件并调整等密度阈值')
            return
        a, b = self.angle, self.pitch
        rotation = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        tilt = np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
        points = self.points @ rotation @ tilt
        scale = min(self.width(),self.height())*.83
        painter.setPen(Qt.NoPen)
        for x,y,z in points[np.argsort(points[:,2])]:
            color = QColor('#76b9df')
            color = color.lighter(int(90+45*(z+.5)))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(self.width()/2+x*scale,self.height()/2-y*scale),2.0,2.0)

    def mousePressEvent(self, event):
        self.last = event.position()

    def mouseMoveEvent(self, event):
        if self.last is not None and event.buttons() & Qt.LeftButton:
            delta = event.position() - self.last
            self.angle += delta.x()*.01
            self.pitch += delta.y()*.01
            self.last = event.position()
            self.update()


class MDDialog(QDialog):
    def __init__(self, workbench):
        super().__init__(workbench)
        self.workbench = workbench
        self.store = MDJobs()
        self.c4 = None
        self.setWindowTitle('论文复刻 · PDC 溶液分子动力学')
        self.resize(1030, 730)
        self.setMinimumSize(850, 620)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.status = QLabel('作者参数已固定版本。选择体系、检查差异，然后准备或提交。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.create_setup()
        self.create_jobs()
        self.create_preview()
        self.create_results()
        self.create_sources()
        self.reload_jobs()

    def button(self, layout, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def spin(self, minimum, maximum, value, decimals=0):
        widget = QDoubleSpinBox() if decimals else NumberBox()
        widget.setRange(minimum,maximum)
        if decimals:
            widget.setDecimals(decimals)
        widget.setValue(value)
        return widget

    def create_setup(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        left = QVBoxLayout()
        left.addWidget(QLabel('论文体系 · Ctrl / Shift 可多选'))
        self.systems = QListWidget()
        systems = catalog()
        self.systems.addItems(systems)
        self.systems.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if 'PDC_neutral' in systems:
            self.systems.setCurrentRow(systems.index('PDC_neutral'))
        elif systems:
            self.systems.setCurrentRow(0)
        left.addWidget(self.systems)
        if not systems:
            notice = QLabel('请按包内 PDC-MD-GUIDE.md 获取原始论文参数，放入 runtime/assets/pdc2024 并重启。')
            notice.setWordWrap(True)
            left.addWidget(notice)
        self.button(left, '载入论文的 21 套周期模型', self.prepare_paper)
        self.c4_mode = QComboBox()
        self.c4_mode.addItems(['文献重建 12-6-4（有参数迁移假设）', '使用导入的逐类型 kappa 参数'])
        left.addWidget(self.c4_mode)
        left.addWidget(QLabel('重建值取自 Li & Merz 表 2，原作者表格势未公开。'))
        self.button(left,'导入 kappa 参数 JSON…',self.import_c4)
        self.button(left,'导出 kappa 参数模板…',self.export_c4_template)
        layout.addLayout(left,1)
        right = QVBoxLayout()
        self.preset = QComboBox()
        self.preset.addItems(['论文生产参数','短程功能验证（不用于论文结论）'])
        right.addWidget(self.preset)
        form = QFormLayout()
        self.fields = {}
        definitions = [
            ('molecules','PDC 分子数',1,1000,100,0),('box_nm','初始盒长 / nm',2.6,30,6,2),
            ('temperature_k','温度 / K',1,1000,293.15,2),('pressure_bar','压力 / bar',.1,1000,1,2),
            ('timestep_fs','时间步 / fs',.1,2,2,1),('equilibration_ns','平衡 / ns',.0001,100,2,4),
            ('production_ns','生产 / ns',.0001,1000,100,4),('output_ps','采样间隔 / ps',.01,100000,1000,2),
            ('stable_after_ns','分析起点 / ns',0,1000,20,4),('cores','CPU 核数',1,16,8,0),
            ('seed','随机种子',1,99999999,20261118,0),('distance_nm','聚集距离 / nm',.05,1,.31,3),
            ('angle_deg','平行夹角 / °',1,90,20,1)]
        for key,title,minimum,maximum,value,decimals in definitions:
            field=self.spin(minimum,maximum,value,decimals)
            self.fields[key]=field
            form.addRow(title,field)
        right.addLayout(form)
        analysis_note = QLabel('论文分析起点：中性和单价体系 20 ns 后，二价体系 30 ns 后。')
        analysis_note.setWordWrap(True)
        right.addWidget(analysis_note)
        self.preset.currentIndexChanged.connect(self.apply_preset)
        controls=QHBoxLayout()
        self.button(controls,'检查服务器',self.health)
        self.button(controls,'准备输入',lambda:self.prepare(False))
        self.button(controls,'准备并提交…',lambda:self.prepare(True))
        right.addLayout(controls)
        layout.addLayout(right,1)
        self.tabs.addTab(page,'建模与运行')

    def create_jobs(self):
        page=QWidget()
        layout=QVBoxLayout(page)
        controls=QHBoxLayout()
        for title,callback in [('刷新本地',self.reload_jobs),('同步状态',self.sync),
                               ('提交已准备任务…',self.submit_prepared),
                               ('取消任务',self.cancel),('继续任务',self.resume),
                               ('下载结果',lambda:self.download(False)),('下载轨迹',lambda:self.download(True)),
                               ('打开目录',self.open_folder),('导出运行包…',self.export_bundle)]:
            self.button(controls,title,callback)
        layout.addLayout(controls)
        self.jobs=QTableWidget(0,6)
        self.jobs.setHorizontalHeaderLabels(['任务','体系','状态','阶段','模拟时间 / ps','生产时长 / ns'])
        self.jobs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.jobs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.jobs.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.jobs.horizontalHeader().setStretchLastSection(True)
        self.jobs.itemSelectionChanged.connect(self.show_job)
        self.jobs.cellDoubleClicked.connect(lambda *_: self.open_preview())
        layout.addWidget(self.jobs,2)
        self.log=QPlainTextEdit(readOnly=True)
        layout.addWidget(self.log,1)
        self.tabs.addTab(page,'作业监视器')

    def create_preview(self):
        self.preview_page = QWidget()
        layout = QVBoxLayout(self.preview_page)
        tools = QHBoxLayout()
        tools.addWidget(QLabel('体系'))
        self.preview_choice = QComboBox()
        self.preview_choice.currentIndexChanged.connect(self.choose_preview)
        tools.addWidget(self.preview_choice, 1)
        self.button(tools, '打开作业目录', self.open_folder)
        self.button(tools, '作业监视器', lambda: self.tabs.setCurrentIndex(1))
        layout.addLayout(tools)

        self.box_view = PeriodicBoxView(self.workbench.preview.background_colors)
        display = QHBoxLayout()
        display.addWidget(QLabel('显示'))
        style = QComboBox()
        for name, value in (('球棍', 'ballstick'), ('棍状', 'stick'), ('线框', 'line'), ('空间填充', 'cpk')):
            style.addItem(name, value)
        style.currentIndexChanged.connect(lambda _: self.box_view.set_style(style.currentData()))

        def reflect_style():
            index = style.findData(self.box_view.style)
            if index != style.currentIndex():
                with QSignalBlocker(style):
                    style.setCurrentIndex(index)

        self.box_view.viewChanged.connect(reflect_style)
        display.addWidget(style)
        for name, default, callback in (
                ('周期边框', True, self.box_view.toggle_cell),
                ('显示水氧', False, self.box_view.toggle_water),
                ('显示氢', False, self.box_view.toggle_hydrogens)):
            control = QCheckBox(name)
            control.setChecked(default)
            control.toggled.connect(callback)
            display.addWidget(control)
        display.addStretch()
        self.button(display, '等轴视图', lambda: self.box_view.orient('iso'))
        self.button(display, '适合窗口', self.box_view.fit)
        layout.addLayout(display)
        self.box_info = QLabel('')
        self.box_info.setWordWrap(True)
        layout.addWidget(self.box_info)
        layout.addWidget(self.box_view, 1)
        layout.addWidget(QLabel('金色为反离子，蓝色为水氧　·　拖动旋转，Shift + 拖动平移，滚轮缩放'))
        self.tabs.addTab(self.preview_page, '周期结构')
        self.tabs.currentChanged.connect(self.preview_tab_changed)

    def preview_tab_changed(self, index):
        if self.tabs.widget(index) is self.preview_page:
            self.refresh_preview()

    def open_preview(self):
        if self.tabs.currentWidget() is self.preview_page:
            self.refresh_preview()
        else:
            self.tabs.setCurrentWidget(self.preview_page)

    def choose_preview(self, index):
        if index < 0:
            return
        job_id = self.preview_choice.itemData(index)
        row = next(i for i, job in enumerate(self.job_entries) if job['id'] == job_id)
        self.jobs.setCurrentCell(row, 0)

    def refresh_preview(self):
        try:
            job = self.selected_job()
            folder = self.store.folder(job)
            model = load_box(folder)
            self.box_view.set_model(model)
            self.box_info.setText(model.summary)
            with QSignalBlocker(self.preview_choice):
                self.preview_choice.setCurrentIndex(self.preview_choice.findData(job['id']))
        except (OSError, ValueError, IndexError, KeyError) as error:
            self.box_view.set_model(None)
            self.box_info.setText(str(error))

    def create_results(self):
        page=QWidget()
        layout=QVBoxLayout(page)
        bar=QHBoxLayout()
        self.observable=QComboBox()
        self.observable.addItems(['聚集时间序列','Rc 分布','Mc 分布','温度','压力','势能','密度'])
        self.observable.currentIndexChanged.connect(self.show_plot)
        bar.addWidget(self.observable)
        self.button(bar,'导入轨迹…',self.import_trajectory)
        self.button(bar,'重算聚集与 SDF',self.reanalyze)
        self.button(bar,'导入论文参考曲线…',self.compare)
        self.button(bar,'AI 审查结果',self.ai_review)
        layout.addLayout(bar)
        split=QSplitter()
        self.plot=DataPlot()
        split.addWidget(self.plot)
        spatial=QWidget()
        side=QVBoxLayout(spatial)
        self.density=DensityView()
        side.addWidget(self.density,1)
        controls=QHBoxLayout()
        self.component=QComboBox()
        self.component.addItems(['PDC','Water','Ions'])
        self.component.currentIndexChanged.connect(self.show_density)
        controls.addWidget(self.component)
        self.level=self.spin(.0001,100000,8,4)
        self.level.valueChanged.connect(self.show_density)
        controls.addWidget(self.level)
        side.addLayout(controls)
        side.addWidget(QLabel('等密度体素表面 · 拖动旋转 · 阈值单位 particles/nm³'))
        split.addWidget(spatial)
        layout.addWidget(split,1)
        self.result_text=QPlainTextEdit(readOnly=True)
        self.result_text.setMaximumHeight(170)
        layout.addWidget(self.result_text)
        self.tabs.addTab(page,'轨迹分析与对照')

    def create_sources(self):
        text=QPlainTextEdit(readOnly=True)
        text.setPlainText(
            'Li & Van Lehn, ACS Omega 2024, DOI 10.1021/acsomega.4c05431\n'
            '参数来源 https://github.com/jlianl/PDC_Aggregation\n'
            '固定提交 fb868ea2646ecf98aa4faed389a5ef64528069a5\n\n'
            '正文参数：100 PDC，初始 6 nm 立方盒，SPC/E，293.15 K，1 bar，2 ns 平衡 + 100 ns 生产。\n'
            '二价离子分析稳定区间按补充材料为 30 ns；当前字段默认 20 ns，多选时自动取至少 30 ns。\n'
            '平衡 V-rescale/Berendsen，生产 Nose-Hoover/Parrinello-Rahman，PME，LINCS。\n\n'
            '已知重建差异\n'
            '• 坐标由作者键连接和固定随机种子重新生成，不是作者原始坐标。\n'
            '• 作者部分水文件不符合正文 SPC/E。生成输入使用独立 OW/HW 的刚性 SPC/E，修正写入 manifest。\n'
            '• 作者部分离子类型大小写已统一，重新整理 atomtypes 与 include。\n'
            '• 二价离子支持 Li & Merz（DOI:10.1021/ct400751u）SPC/E 表 2 重建参数，或导入逐类型 kappa JSON。\n'
            '• 文献重建将每种离子的 kappa 均匀应用到 PDC/水类型，属于明确的参数迁移假设，不是作者原始表格势。\n'
            '• 保留正文 tau-t=tau-p=2 ps，并记录 GROMACS 共振警告；仍需检查生产阶段稳定性。\n'
            '• GROMACS 2018 的用户表格势仅支持 group，正文 Verlet 与此不兼容。二价重建使用 group，差异保留。\n'
            '• Rc/Mc 使用周期最短距离和无向连通分量。无平行多分子簇的聚集体计一个零值记录。\n'
            '• 原始直方图无高斯过程拟合。与论文比较前需匹配分箱、采样区间和归一化。\n'
            '• SDF 使用同一参考分子的环平面对齐，密度为 particles/nm³；需核对作者 gmx spatial 的选择与归一化。\n\n'
            '每项任务保存 manifest、解析后的 mdp、命令日志、引擎版本和实际体系数量。导出运行包可保留这些证据。')
        self.tabs.addTab(text,'来源与差异')

    def apply_preset(self,index):
        values = asdict(paper_config('PDC_neutral'))
        if index:
            values.update(molecules=4, box_nm=3, equilibration_ns=.001, production_ns=.002,
                          stable_after_ns=0, output_ps=.2, cores=2)
        for key, field in self.fields.items():
            field.setValue(values[key])

    def prepare_paper(self):
        if not (ASSETS.parent / 'pdc-paper-models.zip').is_file():
            self.status.setText('当前安装未配置预建周期模型包。安装原始参数后，可在本页选择体系生成输入。')
            return
        if self.workbench.operation:
            self.status.setText('工作台正在执行任务。')
            return
        self.preset.setCurrentIndex(0)
        self.apply_preset(0)
        self.systems.selectAll()
        settings = dict(self.workbench.settings)

        def complete(jobs):
            self.reload_jobs()
            self.select_jobs(jobs, preferred='PDC_neutral')
            self.open_preview()
            message = f'已载入 {len(jobs)} 套周期模型，可切换查看。运行模拟请到作业监视器。'
            self.status.setText(message)
            self.workbench.status.setText(message)

        self.perform('正在载入 21 套经验证的论文周期模型…',
                     lambda: self.store.prepare_paper_boxes(settings), complete)

    def perform(self,message,function,done=lambda value:None):
        if self.workbench.operation:
            self.status.setText('工作台当前正在执行任务。')
            return
        self.setEnabled(False)
        self.status.setText(message)
        self.workbench.run(message,function,done)
        operation=self.workbench.operation
        operation.failure.connect(self.status.setText)
        operation.finished.connect(lambda:self.setEnabled(True))

    def connection(self):
        if not self.workbench.server_ready():
            self.status.setText('服务器请求正在处理，请稍后重试。')
            return None
        remote=self.workbench.connection()
        return MDRemote(remote) if remote else None

    def health(self):
        remote=self.connection()
        if remote:
            self.perform('正在检查 MD 引擎…',remote.health,
                         lambda value:self.status.setText(
                             f"GROMACS {value['version']} · 表格势引擎 "
                             f"{value['legacy']['version'] if value['legacy'] else '未安装'} · "
                             f"{value['cores']} 核 · 空闲 {value['free_disk_gb']} GB"))

    def import_c4(self):
        name,_=QFileDialog.getOpenFileName(self,'选择带出处的 kappa 参数','','JSON (*.json)')
        if name:
            try:
                self.c4=json.loads(Path(name).read_text('utf-8'))
                self.c4_mode.setCurrentIndex(1)
                self.status.setText('已载入 kappa 参数，准备任务时逐项校验。')
            except (OSError,ValueError) as error:
                self.status.setText(str(error))

    def export_c4_template(self):
        selected=[item.text() for item in self.systems.selectedItems()]
        names=[name for name in selected if system_state(name)[1] in ('Be','Mg','Ca','Sr','Ba')]
        if not names:
            self.status.setText('请先选择二价离子体系。')
            return
        filename,_=QFileDialog.getSaveFileName(self,'保存 kappa 模板','kappa-template.json','JSON (*.json)')
        if filename:
            atomic_json(filename,dict(systems={name:dict(source='',units='nm^-2',kappa={group:None for _,group in c4_pairs(name)}) for name in names}))
            self.status.setText('已导出空参数模板。请按文献或原始表格填写数值与出处。')

    def prepare(self,submit):
        selected=[item.text() for item in self.systems.selectedItems()]
        if not selected:
            return
        values={key:field.value() for key,field in self.fields.items()}
        configs=[]
        try:
            for name in selected:
                params=dict(values)
                if self.preset.currentIndex() == 0:
                    params['stable_after_ns'] = max(paper_config(name).stable_after_ns,
                                                    params['stable_after_ns'])
                config=MDConfig(system=name,**params)
                config.validate()
                configs.append(config)
        except ValueError as error:
            self.status.setText(str(error))
            return
        remote=None
        if submit:
            summary=f'{len(configs)} 个体系，每个 {values["molecules"]} PDC\n'
            summary+=f'平衡 {values["equilibration_ns"]} ns + 生产 {values["production_ns"]} ns\n'
            summary+='任务将排队执行。“来源与差异”列出了重建条件。'
            if QMessageBox.question(self,'提交 MD 任务',summary)!=QMessageBox.Yes:
                return
            remote=self.connection()
            if remote is None:
                return
        use_literature = self.c4_mode.currentIndex()==0
        imported = self.c4
        settings = dict(self.workbench.settings)
        def work():
            created=[]
            for config in configs:
                c4 = literature_c4(config.system) if use_literature else (
                    imported.get('systems',{}).get(config.system,imported) if imported else None)
                job=self.store.create(config,settings,c4)
                if remote:
                    remote.submit(job,self.store.folder(job))
                created.append(job)
            return created
        def complete(jobs):
            self.reload_jobs()
            self.select_jobs(jobs)
            if submit:
                self.tabs.setCurrentIndex(1)
            else:
                self.open_preview()
            detail = '已加入服务器队列。' if submit else '输入已保存，可提交选中的已准备任务。'
            self.status.setText(f'{len(jobs)} 个体系。{detail}')
        self.perform('正在校验参数、生成输入…',work,complete)

    def select_jobs(self, jobs, preferred=None):
        ids = {job['id'] for job in jobs}
        rows = [i for i, job in enumerate(self.job_entries) if job['id'] in ids]
        if not rows:
            return
        current = next((i for i in rows if self.job_entries[i]['system'] == preferred), rows[0])
        with QSignalBlocker(self.jobs):
            self.jobs.setCurrentCell(current, 0)
            selection = self.jobs.selectionModel()
            selection.clearSelection()
            for row in rows:
                selection.select(self.jobs.model().index(row, 0),
                                 QItemSelectionModel.Select | QItemSelectionModel.Rows)
        self.show_job()

    def submit_prepared(self):
        selected = [self.job_entries[index.row()] for index in self.jobs.selectionModel().selectedRows()]
        if not selected or any(job['status'] != 'prepared' for job in selected):
            self.status.setText('请选择一个或多个状态为 prepared（已准备）的任务。')
            return
        total_ns = sum(job['production_ns'] for job in selected)
        message = f'提交 {len(selected)} 个已准备体系，合计生产时长 {total_ns:g} ns，另含预平衡。'
        if QMessageBox.question(self, '提交已准备的 MD 任务', message) != QMessageBox.Yes:
            return
        remote = self.connection()
        if remote is None:
            return

        def submit():
            for job in selected:
                remote.directory(job)
            for job in selected:
                remote.submit(job, self.store.folder(job))

        def done(_):
            self.reload_jobs()
            self.status.setText(f'已将 {len(selected)} 个任务加入服务器队列。')

        self.perform('正在上传已准备的输入…', submit, done)

    def selected_job(self):
        row=self.jobs.currentRow()
        if row<0:
            raise ValueError('请选择一项 MD 任务。')
        return self.job_entries[row]

    def reload_jobs(self):
        selected=None
        try:
            selected=self.selected_job()['id']
        except (ValueError,AttributeError,IndexError):
            pass
        self.job_entries=self.store.list()
        with QSignalBlocker(self.preview_choice):
            self.preview_choice.clear()
            for job in self.job_entries:
                self.preview_choice.addItem(f"{job['system']}  ·  {job['id'][:8]}", job['id'])
        self.jobs.blockSignals(True)
        self.jobs.setRowCount(len(self.job_entries))
        current=0
        for row,job in enumerate(self.job_entries):
            if job['id']==selected:
                current=row
            for column,value in enumerate((job['id'],job['system'],job['status'],job.get('stage',''),
                                            job.get('simulation_ps',''),job.get('production_ns',''))):
                self.jobs.setItem(row,column,QTableWidgetItem(str(value)))
        if self.job_entries:
            self.jobs.selectRow(current)
        self.jobs.blockSignals(False)
        self.show_job()

    def show_job(self):
        try:
            job = self.selected_job()
            folder=self.store.folder(job)
        except ValueError:
            self.box_view.set_model(None)
            self.box_info.clear()
            return
        logs=[]
        names = ('system.json', 'preflight.json', 'manifest.json') if job['status'] == 'prepared' else (
            'state.json','system.json','output.log','worker.log')
        for name in names:
            path=folder/name
            if path.exists():
                logs.append(name+'\n'+path.read_text('utf-8',errors='replace')[-40000:])
        self.log.setPlainText('\n\n'.join(logs))
        self.show_plot()
        self.show_density()
        if self.tabs.currentWidget() is self.preview_page:
            self.refresh_preview()

    def remote_action(self,method,*args):
        try:
            job=self.selected_job()
        except ValueError as error:
            self.status.setText(str(error));return
        remote=self.connection()
        if remote:
            folder=self.store.folder(job)
            self.perform('正在更新 MD 任务…',lambda:getattr(remote,method)(job,folder,*args),lambda value:self.reload_jobs())

    def sync(self):
        self.remote_action('sync')

    def download(self,trajectory):
        self.remote_action('download',trajectory)

    def cancel(self):
        try:
            job=self.selected_job()
        except ValueError as error:
            self.status.setText(str(error));return
        if QMessageBox.question(self,'取消 MD','停止所选任务并保留已有检查点？')!=QMessageBox.Yes:
            return
        remote=self.connection()
        if remote:
            self.perform('正在请求停止…',lambda:remote.cancel(job),lambda _:self.status.setText('已发送停止请求，请同步状态。'))

    def resume(self):
        self.remote_action('resume')

    def open_folder(self):
        try:
            path=self.store.folder(self.selected_job())
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except ValueError as error:
            self.status.setText(str(error))

    def export_bundle(self):
        try:
            folder=self.store.folder(self.selected_job())
        except ValueError as error:
            self.status.setText(str(error));return
        name,_=QFileDialog.getSaveFileName(self,'导出运行包',folder.name+'.zip','ZIP (*.zip)')
        if name:
            with zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED) as bundle:
                for source in folder.iterdir():
                    if source.is_file() and source.suffix!='.download':
                        bundle.write(source,source.name)
            self.status.setText('运行包已导出。')

    def show_plot(self):
        self.plot.set_data([],[],'')
        self.result_text.clear()
        try:
            folder=self.store.folder(self.selected_job())
            index = self.observable.currentIndex()
            name=('Temperature','Pressure','Potential','Density')[index-3] if index>=3 else ''
            path=folder/(('cluster_timeseries.csv','Rc_distribution.csv','Mc_distribution.csv')[index]
                         if index<3 else 'thermodynamics.csv')
            if path.exists():
                from research_widgets import read_plot_csv
                if index>=3:
                    import csv
                    with path.open(encoding='utf-8') as stream:
                        reader = csv.reader(stream)
                        source_header = next(reader)
                        column = source_header.index(name)
                        values = [[float(row[0]),float(row[column])] for row in reader if row]
                    header = [source_header[0],name]
                else:
                    header,values=read_plot_csv(path)
                self.plot.set_data(header,values,path.stem)
            if (folder/'analysis.json').exists():
                info=json.loads((folder/'analysis.json').read_text('utf-8'))
                interval=info['analyzed_interval_ns']
                lines=[f"体系：{info['system']}",
                       f"轨迹：{info['total_frames']} 帧；分析 {info['sampled_frames']} 帧；区间 {interval[0]:g}–{interval[1]:g} ns",
                       f"平均聚集体数：{info['average_aggregates']:.3f}；距离阈值 {info['distance_nm']:g} nm；平行夹角 {info['angle_deg']:g}°",
                       f"SDF 参考分子：{info['reference_molecule']}；处于平行簇的帧数：{info['reference_parallel_frames']}",
                       'Rc / Mc 为未平滑的概率质量；零值约定和分箱须与作者分析核对。']
                if interval[-1]<1:
                    lines.append('当前为短程计算，不能据此判断论文聚集结论或平衡。')
                thermodynamic=folder/'thermodynamics.json'
                if index>=3 and thermodynamic.exists():
                    report=json.loads(thermodynamic.read_text('utf-8'))
                    stats=report['columns'].get(name)
                    if stats:
                        lines.append(f"{name}：均值 {stats['mean']:.6g}；标准差 {stats['standard_deviation']:.6g}；{report['samples']} 个能量采样点")
                self.result_text.setPlainText('\n'.join(lines))
        except (ValueError,OSError) as error:
            self.status.setText(str(error))

    def show_density(self):
        try:
            folder=self.store.folder(self.selected_job())
            path=folder/f'SDF_{self.component.currentText()}.npy'
            self.density.set_grid(np.load(path,allow_pickle=False) if path.exists() else np.zeros((8,8,8)),self.level.value())
        except (ValueError,OSError) as error:
            self.status.setText(str(error))

    def import_trajectory(self):
        try:
            job=self.selected_job()
            if job['status'] in ('queued','running'):
                raise ValueError('请先停止任务，再导入外部轨迹。')
            folder=self.store.folder(job)
        except ValueError as error:
            self.status.setText(str(error));return
        xtc,_=QFileDialog.getOpenFileName(self,'导入与当前任务原子顺序匹配的轨迹','','轨迹 (*.xtc)')
        if not xtc:
            return
        gro,_=QFileDialog.getOpenFileName(self,'选择对应坐标文件','','GROMACS (*.gro)')
        if gro:
            from chatdft.md_analysis import read_manifest, validate_topology
            import mdtraj as md
            import tempfile
            def work():
                frame=md.load_frame(xtc,0,top=gro)
                validate_topology(frame,read_manifest(folder))
                with tempfile.TemporaryDirectory(dir=folder) as temp:
                    staging=Path(temp)
                    shutil.copyfile(xtc,staging/'production.xtc')
                    shutil.copyfile(gro,staging/'system.gro')
                    from datetime import datetime
                    backup=folder/('before-import-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
                    backup.mkdir()
                    names=['production.xtc','system.gro','analysis.json','spatial.json','ai-review.json',
                           'thermodynamics.csv','thermodynamics.json','thermodynamics.xvg']
                    names += [p.name for pattern in ('cluster_*.csv','*_distribution.csv','SDF_*','*_comparison.*')
                              for p in folder.glob(pattern) if p.is_file()]
                    for name in names:
                        if (folder/name).exists():
                            shutil.move(str(folder/name),str(backup/name))
                    for name in ('production.xtc','system.gro'):
                        (staging/name).replace(folder/name)
                atomic_json(folder/'trajectory-import.json',dict(trajectory=Path(xtc).name,
                            topology=Path(gro).name,backup=backup.name,
                            note='External trajectory; remote engine outputs do not describe this imported trajectory.'))
                atomic_json(folder/'state.json',dict(status='imported',stage='awaiting local analysis'))
            self.perform('正在校验并导入轨迹…',work,
                         lambda _:self.status.setText('轨迹已校验并导入，原文件已备份。请重算分析；服务器原任务的温压能量不代表导入轨迹。'))

    def reanalyze(self):
        try:
            folder=self.store.folder(self.selected_job())
        except ValueError as error:
            self.status.setText(str(error));return
        start=self.fields['stable_after_ns'].value()
        distance=self.fields['distance_nm'].value()
        angle=self.fields['angle_deg'].value()
        def work():
            result=analyze(folder,start_ns=start,distance=distance,angle=angle)
            spatial_density(folder,start_ns=start)
            if (folder/'trajectory-import.json').exists():
                atomic_json(folder/'state.json',dict(status='imported',stage='local analysis completed'))
            return result
        self.perform('正在重新计算聚集和空间分布…',work,lambda _:self.show_job())

    def compare(self):
        if self.observable.currentIndex() not in (1,2):
            self.status.setText('请选择 Rc 或 Mc 分布。');return
        try:
            folder=self.store.folder(self.selected_job())
        except ValueError as error:
            self.status.setText(str(error));return
        filename,_=QFileDialog.getOpenFileName(self,'选择带表头的论文参考分布','','CSV (*.csv)')
        if filename:
            from PySide6.QtWidgets import QInputDialog
            source,ok=QInputDialog.getText(self,'参考出处','论文图号、数据来源或数字化方式')
            if ok:
                try:
                    result=compare_distribution(folder,filename,('Rc','Mc')[self.observable.currentIndex()-1],source)
                    from research_widgets import read_plot_csv
                    header,values=read_plot_csv(folder/f"{result['observable']}_comparison.csv")
                    self.plot.set_data(header,values,result['observable']+' · 论文对照')
                    self.result_text.setPlainText(
                        f"参考来源：{result['source']}\n对照点：{result['points']}\n"
                        f"RMSE：{result['rmse']:.6g}；最大绝对偏差：{result['maximum_absolute_error']:.6g}\n"
                        '未缩放或平滑。比较前需确认相同分箱和概率定义。')
                except (ValueError,OSError) as error:
                    self.status.setText(str(error))

    def ai_review(self):
        try:
            folder = self.store.folder(self.selected_job())
        except ValueError as error:
            self.status.setText(str(error))
            return
        from chatdft.md_ai import interpret
        settings = dict(self.workbench.settings)
        def display(result):
            titles = {'summary':'结果概况', 'quality_checks':'质量检查',
                      'paper_comparison':'论文对照', 'next_steps':'下一步'}
            self.result_text.setPlainText('\n\n'.join(titles[k]+'\n'+v for k,v in result.items()))
            self.status.setText('AI 审查已完成并保存，使用当前选择的模型。')
        self.perform('正在用当前模型审查实际 MD 结果…',lambda:interpret(settings,folder),display)
