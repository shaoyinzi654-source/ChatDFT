"""Saved model revisions and a single place to reopen their calculation outputs."""
from datetime import datetime
import json
from pathlib import Path
import re
import time
import uuid

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QHeaderView, QHBoxLayout, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout)

from automatic import save_structure
from chatdft.storage import atomic_json, data_dir
from panel import STATUSES


REVISIONS = ('初始结构', '计算结构', '优化结构')


def archive_model(model, calculation, *, prompt='', description='', pending=False):
    key = uuid.uuid4().hex
    folder = data_dir() / 'projects' / key
    folder.mkdir(parents=True)
    save_structure(folder, REVISIONS[0], model, calculation.to_dict())
    atomic_json(folder / 'project.json', dict(id=key, name=model.name, created=time.time(),
                prompt=prompt, description=description, pending=pending))
    return key


def model_projects():
    for path in (data_dir()/'projects').glob('*/project.json'):
        try:
            value = json.loads(path.read_text('utf-8'))
            if isinstance(value, dict) and value.get('id') == path.parent.name and (path.parent/'初始结构.chatdft.json').is_file():
                yield value, path.parent
        except (OSError, ValueError):
            continue


def stop_draft(project_id):
    if not isinstance(project_id, str) or not re.fullmatch('[0-9a-f]{32}',project_id):
        return
    path = data_dir()/'projects'/project_id/'project.json'
    metadata = json.loads(path.read_text('utf-8'))
    atomic_json(path, {**metadata, 'pending': False})


class ProjectLibrary(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle('项目库 · ChatDFT')
        self.resize(905, 440)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['项目', '状态', '最近记录', '编号'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 4):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        self.table.itemDoubleClicked.connect(self.reopen_final)
        layout.addWidget(self.table)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        bar = QHBoxLayout()
        self.buttons = []
        for label, action in [('恢复项目', self.reopen_final), ('载入初始结构', self.reopen_initial),
                              ('打开项目文件夹', self.open_folder), ('生成 AI 分析', self.analyze),
                              ('查看 AI 报告', self.open_report), ('继续未提交的草稿', self.resume)]:
            button = QPushButton(label)
            button.clicked.connect(action)
            bar.addWidget(button)
            self.buttons.append(button)
        layout.addLayout(bar)
        self.entries = []

    def reload(self, selected=None):
        current = self.selected()
        selected = selected or (current['id'] if current else None)
        entries = []
        linked = set()
        for job in self.owner.jobs.values():
            folder = self.owner.store.folder(job)
            meta = job.get('automation') or {}
            linked.add(meta.get('project_id'))
            entries.append(dict(id=job['id'], name=job['name'], created=job['created'],
                                status=STATUSES.get(job['status'],job['status']), folder=folder,
                                job=job, pending=False))
        for model, folder in model_projects():
            if model['id'] not in linked:
                entries.append(dict(**model, status='等待提交' if model.get('pending') else '已保存模型',
                                    folder=folder, job=None))
        self.entries = sorted(entries,key=lambda item:item['created'],reverse=True)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.entries))
        selected_row = 0
        for row, entry in enumerate(self.entries):
            if entry['id'] == selected:
                selected_row = row
            values = (entry['name'],entry['status'],
                      datetime.fromtimestamp(entry['created']).strftime('%Y-%m-%d %H:%M'),entry['id'])
            for col, value in enumerate(values):
                self.table.setItem(row,col,QTableWidgetItem(value))
        if self.entries:
            self.table.selectRow(selected_row)
        self.table.blockSignals(False)
        self.selection_changed()

    def selected(self):
        row = self.table.currentRow()
        return self.entries[row] if 0 <= row < len(self.entries) else None

    def selection_changed(self):
        entry = self.selected()
        for button in self.buttons[:3]:
            button.setEnabled(entry is not None)
        job = entry.get('job') if entry else None
        self.buttons[3].setEnabled(bool(job and job.get('automation') and
                                       job['status'] in ('completed','succeeded')))
        self.buttons[4].setEnabled(self.report_path(entry) is not None)
        self.buttons[5].setEnabled(bool(entry and entry.get('pending')))
        self.detail.setText(str(entry['folder']) if entry else '提交计算或保存模型后，可在此恢复项目。')

    def report_path(self, entry):
        if not entry or not entry.get('job'):
            return None
        output = (entry['folder']/'输出').resolve()
        try:
            status = json.loads((output/'AI分析状态.json').read_text('utf-8'))
            path = (output/status['report']).resolve()
            if path.is_relative_to(output) and path.is_file():
                return path
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return None

    def open_report(self):
        if path := self.report_path(self.selected()):
            dialog = QDialog(self)
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            dialog.setWindowTitle('AI 报告 · ' + self.selected()['name'])
            dialog.resize(900, 700)
            layout = QVBoxLayout(dialog)
            content = QTextBrowser()
            content.setOpenLinks(False)
            content.setSearchPaths([str(path.parent)])
            content.document().setBaseUrl(QUrl.fromLocalFile(str(path.parent) + '/'))
            content.setMarkdown(path.read_text('utf-8'))
            layout.addWidget(content)
            dialog.show()

    def reopen_final(self, *_):
        self.reopen(True)

    def reopen_initial(self):
        self.reopen(False)

    def reopen(self, prefer_result):
        if entry := self.selected():
            self.owner.restore_library_project(entry,prefer_result)

    def open_folder(self):
        if entry := self.selected():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(entry['folder'])))

    def analyze(self):
        if entry := self.selected():
            self.owner.retry_ai_analysis(entry['job'])

    def resume(self):
        if entry := self.selected():
            self.owner.resume_library_draft(entry)
