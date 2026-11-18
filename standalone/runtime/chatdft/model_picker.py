import json
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .ai import find_codex


class ModelPicker(QWidget):
    DEFAULT = 'Codex 默认模型'

    def __init__(self, model, provider, executable, parent=None, preferences=None):
        super().__init__(parent)
        self.executable = executable
        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.NoInsert)
        self.refresh = QPushButton('刷新模型')
        self.refresh.clicked.connect(self.load_catalog)
        self.status = QLabel()
        self.status.setWordWrap(True)

        row = QHBoxLayout()
        row.addWidget(self.combo, 1)
        row.addWidget(self.refresh)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.status)

        self.process = QProcess(self)
        self.process.finished.connect(self.catalog_finished)
        self.process.errorOccurred.connect(self.catalog_error)
        self.timer = QTimer(self, singleShot=True, interval=20000)
        self.timer.timeout.connect(self.process.kill)
        self.models = []
        self.mode = provider.currentText()
        self.preferences = {**(preferences or {}), self.mode: model}
        self.populate(model)
        provider.currentTextChanged.connect(self.change_provider)
        self.executable.editingFinished.connect(self.load_catalog)
        if self.mode == 'codex':
            QTimer.singleShot(0, self, self.load_catalog)

    def model_id(self):
        text = self.combo.currentText().strip()
        index = self.combo.currentIndex()
        if index >= 0 and text == self.combo.itemText(index):
            return self.combo.itemData(index)
        return '' if self.mode == 'codex' and text == self.DEFAULT else text

    def populate(self, selection):
        self.combo.clear()
        if self.mode == 'codex':
            self.combo.addItem(self.DEFAULT, '')
            for model_id, name in self.models:
                self.combo.addItem(name, model_id)
        index = self.combo.findData(selection)
        if index >= 0:
            self.combo.setCurrentIndex(index)
        else:
            self.combo.setEditText(selection)
        self.refresh.setVisible(self.mode == 'codex')
        self.status.setVisible(self.mode == 'codex')

    def change_provider(self, provider):
        self.preferences[self.mode] = self.model_id()
        self.mode = provider
        self.populate(self.preferences.get(provider, ''))
        if provider == 'codex':
            self.load_catalog()

    def load_catalog(self):
        if self.mode != 'codex' or self.process.state() != QProcess.NotRunning:
            return
        exe = find_codex(self.executable.text().strip())
        if not exe:
            self.status.setText('未找到 Codex。可填写程序路径后刷新。')
            return
        env = QProcessEnvironment.systemEnvironment()
        for key in ('OPENAI_API_KEY', 'CHATDFT_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL'):
            env.remove(key)
        self.process.setProcessEnvironment(env)
        self.process.setProperty('codexExecutable', exe)
        self.refresh.setEnabled(False)
        self.status.setText('正在读取 Codex 模型列表…')
        self.timer.start()
        self.process.start(exe, ['debug', 'models', '-c', 'forced_login_method="chatgpt"'])

    def catalog_finished(self, code, status):
        self.timer.stop()
        self.refresh.setEnabled(True)
        raw = bytes(self.process.readAllStandardOutput())
        self.process.readAllStandardError()
        if code != 0 or status != QProcess.NormalExit:
            self.status.setText('读取模型列表失败。可以使用 Codex 默认模型、手填模型 ID 或刷新重试。')
            return
        try:
            models = json.loads(raw)['models']
            visible = {}
            for model in models:
                if model.get('visibility') != 'list':
                    continue
                slug = model['slug']
                if isinstance(slug, str) and slug:
                    name = model.get('display_name')
                    visible.setdefault(slug, name if isinstance(name, str) and name else slug)
        except (ValueError, KeyError, TypeError, AttributeError):
            self.status.setText('Codex 模型列表格式无法读取。可手填模型 ID。')
            return
        self.models = list(visible.items())
        if self.mode == 'codex':
            self.populate(self.model_id())
        exe = self.process.property('codexExecutable')
        self.status.setText(f'已刷新 {len(self.models)} 个 Codex 模型，可选择或输入模型 ID。')
        self.status.setToolTip(str(Path(exe)) if exe else '')

    def catalog_error(self, error):
        if error == QProcess.FailedToStart:
            self.timer.stop()
            self.refresh.setEnabled(True)
            self.status.setText('无法启动 Codex。请核对程序路径。')

    def stop(self):
        self.timer.stop()
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.process.waitForFinished(300)
