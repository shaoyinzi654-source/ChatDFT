"""Unified Qt monitor for PySCF and GROMACS jobs on the calculation server."""
from datetime import datetime
import math

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QHeaderView, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from chatdft.app import Task
from chatdft.jobs import Remote, SSH_AUTH_ERROR
from chatdft.server_jobs import ServerJobs


ENGINES = {'pyscf': 'PySCF', 'gromacs': 'GROMACS'}
STATES = {'queued': '排队中', 'running': '运行中', 'completed': '已完成',
          'failed': '失败', 'cancelled': '已取消', 'unknown': '等待状态'}
STAGES = {'waiting': '等待资源', 'build': '建盒', 'minimize': '能量最小化',
          'equilibrate': '预平衡', 'production': '生产模拟', 'analysis': '轨迹分析',
          'completed': '已完成', 'resume': '续跑'}


def key(job):
    return job['engine'], job['id']


class ServerJobsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.workbench = parent
        self.backend = None
        self.login = None
        self.live = False
        self.operation = None
        self.entries = []
        self.by_key = {}
        self.preferred = None
        self.log_key = None
        self.setWindowTitle('服务器作业管理 · ChatDFT')
        self.resize(1100, 730)
        self.setMinimumSize(840, 570)
        self.build_ui()
        self.timer = QTimer(self, interval=8000)
        self.timer.timeout.connect(self.refresh)
        self.log_timer = QTimer(self, singleShot=True, interval=220)
        self.log_timer.timeout.connect(self.read_selected_log)
        self.update_actions()

    def build_ui(self):
        layout = QVBoxLayout(self)
        connection = QHBoxLayout()
        self.endpoint = QLabel('连接计算服务器后查看实时作业。')
        connection.addWidget(self.endpoint, 1)
        self.auto = QCheckBox('自动刷新')
        self.auto.setChecked(True)
        connection.addWidget(self.auto)
        self.connect_button = QPushButton('连接服务器')
        self.connect_button.clicked.connect(self.connect_and_refresh)
        connection.addWidget(self.connect_button)
        layout.addLayout(connection)

        meter_row = QHBoxLayout()
        self.meters = {}
        for name in ('CPU', '内存', '磁盘'):
            meter_row.addWidget(QLabel(name))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat('—')
            meter_row.addWidget(bar, 1)
            self.meters[name] = bar
        layout.addLayout(meter_row)

        filters = QHBoxLayout()
        self.engine = QComboBox()
        self.engine.addItem('全部引擎', '')
        for value, title in ENGINES.items():
            self.engine.addItem(title, value)
        self.state = QComboBox()
        self.state.addItem('全部状态', '')
        self.state.addItem('活动作业', 'active')
        for value, title in STATES.items():
            if value != 'unknown':
                self.state.addItem(title, value)
        self.search = QLineEdit()
        self.search.setPlaceholderText('搜索结构、体系或作业编号')
        self.search.setClearButtonEnabled(True)
        self.engine.currentIndexChanged.connect(self.fill_table)
        self.state.currentIndexChanged.connect(self.fill_table)
        self.search.textChanged.connect(self.fill_table)
        filters.addWidget(self.engine)
        filters.addWidget(self.state)
        filters.addWidget(self.search, 1)
        self.counts = QLabel('')
        filters.addWidget(self.counts)
        layout.addLayout(filters)

        split = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['引擎', '结构 / 体系', '状态', '进度', '提交时间', '核数', '作业编号'])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        for index in (0, 2, 4, 5, 6):
            self.table.horizontalHeader().setSectionResizeMode(index, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setColumnWidth(3, 175)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        split.addWidget(self.table)
        pane = QWidget()
        details = QVBoxLayout(pane)
        details.setContentsMargins(0, 3, 0, 0)
        self.detail = QLabel('选择作业查看详情')
        self.detail.setWordWrap(True)
        details.addWidget(self.detail)
        self.log_text = QPlainTextEdit(readOnly=True)
        self.log_text.setPlaceholderText('显示服务器日志末尾')
        details.addWidget(self.log_text)
        split.addWidget(pane)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)

        actions = QHBoxLayout()
        self.cancel_button = self.button(actions, '取消作业…', self.cancel_job)
        self.continue_button = self.button(actions, '续跑 / 重试…', self.continue_job)
        self.download_button = self.button(actions, '下载结果', lambda: self.download(False))
        self.trajectory_button = self.button(actions, '下载轨迹…', lambda: self.download(True))
        self.folder_button = self.button(actions, '打开本地目录', self.open_folder)
        actions.addStretch()
        self.button(actions, '准备 MD 作业', self.workbench.open_md_reproduction)
        layout.addLayout(actions)
        self.message = QLabel('')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

    @staticmethod
    def button(layout, text, callback):
        control = QPushButton(text)
        control.clicked.connect(callback)
        layout.addWidget(control)
        return control

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()
        if self.workbench.remote is not None and self.operation is None:
            QTimer.singleShot(0, self.connect_and_refresh)
        elif self.workbench.remote is None:
            self.endpoint.setText('点击连接服务器，使用 ChatDFT 的 SSH 设置。')

    def hideEvent(self, event):
        self.timer.stop()
        self.log_timer.stop()
        super().hideEvent(event)

    def shutdown(self):
        self.timer.stop()
        if self.backend is not None:
            self.backend.remote.close()

    def connect_and_refresh(self):
        if self.operation:
            return
        login = self.workbench.connection()
        if login is None:
            return
        if self.login is not login:
            if self.backend:
                self.backend.remote.close()
            try:
                self.backend = ServerJobs(Remote(login.settings, login.password))
            except ValueError as error:
                self.message.setText(str(error))
                self.backend = None
                self.login = None
                return
            self.login = login
            self.live = False
            self.entries = []
            self.fill_table()
        self.refresh(force=True)

    def ready(self):
        return self.backend is not None and self.login is self.workbench.remote

    def perform(self, label, function, done):
        if self.operation or not self.ready():
            return
        self.message.setText(label)
        task = self.operation = Task(function)
        task.success.connect(done)
        task.failure.connect(self.failed)
        task.finished.connect(self.task_finished)
        self.update_actions()
        task.start()

    def failed(self, message):
        self.live = False
        self.log_key = self.selected_key()
        self.message.setText(str(message))
        if message == SSH_AUTH_ERROR:
            self.workbench.clear_rejected_login(self.login)
            self.endpoint.setText('SSH 认证失败。请点击连接服务器重新登录。')
            return
        self.endpoint.setText('上次操作失败。请核对状态，或重新刷新。')

    def task_finished(self):
        task, self.operation = self.operation, None
        task.deleteLater()
        self.update_actions()
        if self.isVisible() and self.log_key != self.selected_key():
            self.log_timer.start()

    def refresh(self, force=False):
        if not self.isVisible() or not self.ready() or (not force and not self.auto.isChecked()):
            if not self.ready():
                self.live = False
                self.endpoint.setText('点击连接服务器，使用 ChatDFT 的 SSH 设置。')
                self.connect_button.setText('连接服务器')
                self.update_actions()
            return
        backend = self.backend
        selected = self.selected()

        def read():
            snapshot = backend.snapshot()
            if selected and any(key(entry) == key(selected) for entry in snapshot['jobs']):
                snapshot['log'] = key(selected), backend.log(selected)
            return snapshot

        self.perform('正在同步服务器作业…', read, self.apply_snapshot)

    def apply_snapshot(self, snapshot):
        if not self.ready():
            return
        self.live = True
        self.entries = snapshot['jobs']
        self.fill_table()
        host = snapshot['host']
        address = f"{self.login.settings['ssh_host']}:{self.login.settings['ssh_port']}"
        self.endpoint.setText(f'已连接 {address}  ·  {host}  ·  {snapshot["cores"]} 核')
        self.connect_button.setText('刷新服务器')
        bar = self.meters['CPU']
        bar.setValue(round(snapshot['cpu_percent']))
        bar.setFormat(f"{snapshot['cpu_percent']:g}%")
        for name, field in [('内存', 'mem'), ('磁盘', 'disk')]:
            used, total = snapshot[field+'_used'], snapshot[field+'_total']
            self.meters[name].setValue(round(100*used/total))
            self.meters[name].setFormat(f'{used/1024**3:.1f} / {total/1024**3:.1f} GiB')
        if 'log' in snapshot:
            self.apply_log(*snapshot['log'])
        self.message.setText('最后同步  ' + datetime.fromtimestamp(snapshot['time']).strftime('%H:%M:%S'))

    def fill_table(self):
        selected = self.preferred or self.selected_key()
        engine, state = self.engine.currentData(), self.state.currentData()
        query = self.search.text().strip().casefold()
        states = {'running', 'queued'} if state == 'active' else {state} if state else set()
        filtered = [job for job in self.entries if (not engine or job['engine'] == engine)
                    and (not states or job['status'] in states)
                    and (not query or query in f"{job['name']} {job['id']}".casefold())]
        rank = {'running': 0, 'queued': 1, 'failed': 2, 'completed': 3, 'cancelled': 4}
        filtered.sort(key=lambda job: (rank.get(job['status'], 5), -job['created']))
        self.by_key = {key(job): job for job in self.entries}
        self.table.blockSignals(True)
        self.table.setRowCount(len(filtered))
        current = 0
        for row, job in enumerate(filtered):
            state_name = STATES[job['status']]
            if job.get('stale'):
                state_name += '（心跳过期）'
            if job.get('cancel_requested'):
                state_name = '取消中'
            values = (ENGINES[job['engine']], job['name'], state_name, self.progress(job),
                      datetime.fromtimestamp(job['created']).strftime('%m-%d %H:%M'),
                      job['cores'] or '—', job['id'])
            for index, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                if index == 0:
                    item.setData(Qt.UserRole, key(job))
                if index == 2:
                    item.setForeground(QColor({'running':'#2672aa','queued':'#946722',
                                               'failed':'#ae4545','completed':'#2a8064'}.get(job['status'],'#596878')))
                self.table.setItem(row, index, item)
            if key(job) == selected:
                current = row
                self.preferred = None
        if filtered:
            self.table.selectRow(current)
        self.table.blockSignals(False)
        running = sum(job['status']=='running' for job in self.entries)
        queued = sum(job['status']=='queued' for job in self.entries)
        self.counts.setText(f'{len(filtered)}/{len(self.entries)} 项 · {running} 运行 · {queued} 排队')
        self.selection_changed()

    @staticmethod
    def progress(job):
        stage = STAGES.get(job['stage'], '')
        if job['engine'] == 'gromacs':
            position, target = job.get('position_ps'), job.get('target_ps')
            if isinstance(position, (int, float)) and math.isfinite(position) and target:
                return f'{stage} · {position/1000:.3f}/{target/1000:g} ns'
            return stage or '—'
        if job.get('iteration') is not None:
            return f"几何优化 · 第 {job['iteration']} 步"
        return {'opt':'几何优化', 'optfreq':'优化与频率', 'freq':'频率', 'sp':'单点', 'td':'激发态'}.get(job['task'], '—')

    def selected_key(self):
        item = self.table.item(self.table.currentRow(), 0)
        return tuple(item.data(Qt.UserRole)) if item else None

    def selected(self):
        return self.by_key.get(self.selected_key())

    def selection_changed(self):
        job = self.selected()
        if not job:
            self.detail.setText('没有符合条件的服务器作业。')
        else:
            method = ' / '.join(part for part in (job['method'], job['basis']) if part)
            checkpoint = ' · 有检查点' if job['has_checkpoint'] else ''
            status = job['message'] or STATES[job['status']]
            self.detail.setText(f"{job['name']}  ·  {job['id']} {method}{checkpoint}\n{status}")
        if self.log_key != self.selected_key():
            self.log_text.clear()
            self.log_timer.start()
        self.update_actions()

    def read_selected_log(self):
        job = self.selected()
        if job is None or not self.isVisible() or not self.ready():
            return
        def done(text):
            self.apply_log(key(job), text)
            self.message.setText('已读取选中作业的服务器日志。')

        self.perform('正在读取选中作业的日志…', lambda: self.backend.log(job), done)

    def apply_log(self, job_key, text):
        if job_key != self.selected_key():
            return
        document = text or '服务器尚无日志。'
        if document != self.log_text.toPlainText():
            scroll = self.log_text.verticalScrollBar()
            at_end, old = self.log_key != job_key or scroll.value()==scroll.maximum(), scroll.value()
            self.log_text.setPlainText(document)
            scroll.setValue(scroll.maximum() if at_end else old)
        self.log_key = job_key

    def update_actions(self):
        job = self.selected()
        enabled = self.ready() and self.live and self.operation is None and job is not None
        state = job['status'] if job else ''
        active = state in ('queued', 'running')
        stopped = state in ('failed', 'cancelled')
        self.connect_button.setEnabled(self.operation is None)
        self.cancel_button.setEnabled(bool(enabled and active and not job['cancel_requested']))
        self.continue_button.setText('从检查点继续…' if job and job['engine']=='gromacs' and job['has_checkpoint'] else
                                     '继续 MD…' if job and job['engine']=='gromacs' else '重试量子作业…')
        self.continue_button.setEnabled(bool(enabled and stopped))
        self.download_button.setEnabled(bool(enabled and (stopped or state=='completed')))
        self.trajectory_button.setEnabled(bool(enabled and (stopped or state=='completed') and job['has_trajectory']))
        folder = self.backend.job_folder(job) if self.backend and job is not None else None
        self.folder_button.setEnabled(bool(folder and folder.is_dir()))

    def cancel_job(self):
        job = self.selected()
        if not job or QMessageBox.question(self, '取消服务器作业', f"取消 {job['name']}（{job['id']}）？") != QMessageBox.Yes:
            return
        self.perform('正在发送停止请求…', lambda: self.backend.cancel(job),
                     lambda _: self.refresh_after('停止请求已发送。'))

    def continue_job(self):
        job = self.selected()
        if job is None:
            return
        md = job['engine'] == 'gromacs'
        text = ('保留原任务和可用检查点，继续该 MD 作业。' if md else '沿用原结构和计算设置，创建一项新的量子作业。')
        if QMessageBox.question(self, '继续服务器作业', f"{job['name']}（{job['id']}）\n{text}") != QMessageBox.Yes:
            return

        def done(new_id):
            self.preferred = job['engine'], new_id
            self.workbench.reload_jobs()
            self.refresh_after('作业已重新提交，正在等待服务器调度。')

        self.perform('正在提交续跑或重试…', lambda: self.backend.continue_job(job), done)

    def download(self, trajectory):
        job = self.selected()
        if job is None:
            return
        if trajectory and QMessageBox.question(self, '下载轨迹', '下载完整 XTC 轨迹和结果？轨迹可能较大。') != QMessageBox.Yes:
            return

        def done(folder):
            self.workbench.reload_jobs()
            if hasattr(self.workbench, 'md_dialog'):
                self.workbench.md_dialog.reload_jobs()
            self.message.setText(f'已下载到 {folder}')

        self.perform('正在下载服务器文件…', lambda: self.backend.download(job, trajectory), done)

    def open_folder(self):
        folder = self.backend.job_folder(self.selected())
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def refresh_after(self, message):
        self.message.setText(message)
        QTimer.singleShot(650, self, lambda: self.refresh(force=True))
