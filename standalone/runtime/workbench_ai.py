"""Visible AI controls, request feedback, and failure recovery for the workbench."""
from collections import Counter
from datetime import datetime
import json
from threading import Event

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget,
                               QToolBar, QComboBox, QSystemTrayIcon)

from chatdft.ai import ask
from chatdft.app import Task
from chatdft.gaussian import Calculation, build_input
from automatic import automatic_calculation, export_job, needs_export
from workflow import submit
from projects import ProjectLibrary, archive_model, stop_draft
from chatdft.chemistry import Structure

MODELING_EXAMPLE = ('新建一个非周期簇：2 个苯分子平行堆叠，环平面相距 3.5 Å，'
                    '在周围摆放 6 个水分子、1 个 Na⁺、1 个 Cl⁻。'
                    '总电荷 0，单重态。保留每个分子的内部键，直接生成三维结构。')


class WorkbenchAI:
    def install_ai_tools(self):
        self.ai_ui_ready = True
        self._ai_request_pending = False
        self._ai_last_ok = ''
        self._ai_input = self.preview.structure
        self._modeling = None
        self._model_cancel = None
        self._auto_ready = None
        self._auto_last_job = None
        self._auto_export_errors = {}
        self._auto_export_task = None
        self._auto_export_again = False
        self._analysis_task = None
        self._analysis_cancel = None
        self._analysis_queue = {}
        self._notification_folder = None
        self._active_project_folder = None
        self._tray = QSystemTrayIcon(self.windowIcon(), self)
        self._tray.setToolTip('ChatDFT 作业提醒')
        self._tray.messageClicked.connect(self.open_notification_folder)
        self._auto_timer = QTimer(self, interval=300, singleShot=True)
        self._auto_timer.timeout.connect(self.resume_automatic)
        self.ai_dock.setWindowTitle('AI 助手')
        self.actions_new['助手'].setText('AI 助手')
        self.actions_new['助手'].setToolTip('打开 AI 助手：结构设计、结构修改、参数建议和结果解读')
        self.ai_tabs.setTabText(0, 'AI 对话')
        self.prompt.setPlaceholderText('输入结构设计或计算要求；Ctrl+Enter 发送')
        self.answer.setPlaceholderText('AI 对话与操作记录')
        self.answer.setMinimumHeight(55)
        self.ai_dock.takeWidget()
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(3)
        row = QHBoxLayout()
        self.ai_state = QLabel(self.ai_provider_name() + ' · 尚未测试')
        self.ai_state.setWordWrap(True)
        row.addWidget(self.ai_state, 1)
        self.ai_test_button = QPushButton('测试 AI')
        self.ai_test_button.clicked.connect(self.test_ai_connection)
        row.addWidget(self.ai_test_button)
        self.stop_model_button = QPushButton('结束建模')
        self.stop_model_button.clicked.connect(self.stop_modeling)
        self.stop_model_button.hide()
        row.addWidget(self.stop_model_button)
        layout.addLayout(row)
        self.ai_target = QLabel()
        self.ai_target.setWordWrap(True)
        layout.addWidget(self.ai_target)
        shortcuts = QToolBar()
        shortcuts.setToolButtonStyle(Qt.ToolButtonTextOnly)
        shortcuts.addAction('建模示例', self.prefill_modeling_example)
        for title, prompt in [
            ('设计结构', '请根据以下研究目标设计候选分子，说明建模假设并建议可执行计算参数：'),
            ('修改结构', '请在当前分子结构基础上修改，保留未要求改变的骨架。修改目标：'),
            ('计算方案', '请根据当前分子提出 PySCF 可执行的计算方案，解释方法、基组和任务的选择。'),
        ]:
            action = shortcuts.addAction(title)
            action.triggered.connect(lambda checked=False, text=prompt: self.prefill_ai(text))
        explain = shortcuts.addAction('解读结果')
        explain.triggered.connect(self.explain_job)
        shortcuts.addAction('打开输出', self.open_automatic_output)
        layout.addWidget(shortcuts)
        layout.addWidget(self.ai_tabs, 1)
        self._modeling_controls = (shortcuts, self.ai_tabs, self.ai_test_button)
        self.ai_mode = QComboBox()
        self.ai_mode.addItems(['直接建模', '对话与候选', '全自动 · 建模到数据'])
        self.ai_mode.currentIndexChanged.connect(self.update_ai_mode)
        # Route the console button and keyboard shortcut through the visible mode.
        self.send_button.clicked.disconnect()
        self.send_button.clicked.connect(self.submit_prompt)
        console = self.ai_tabs.widget(0).layout()
        console.insertWidget(console.indexOf(self.prompt), self.ai_mode)
        self.update_ai_mode()
        import PySide6QtAds as ads
        self.ai_dock.setWidget(wrapper, ads.CDockWidget.eInsertMode.ForceNoScrollArea)
        self.ai_dock.setMinimumSize(290, 270)
        self.ai_dock.setMinimumSizeHintMode(
            ads.CDockWidget.eMinimumSizeHintMode.MinimumSizeHintFromDockWidgetMinimumSize)
        self.ai_send_shortcut = QShortcut(QKeySequence('Ctrl+Return'), self.prompt)
        self.ai_send_shortcut.setContext(Qt.WidgetShortcut)
        self.ai_send_shortcut.activated.connect(self.submit_prompt)
        menu = self.menuBar().addMenu('AI')
        menu.addAction(self.actions_new['助手'])
        menu.addAction('测试 AI 连接', self.test_ai_connection)
        menu.addAction('选择模型与设置…', self.configure)
        menu.addAction('论文复刻…', self.open_paper_reproduction)
        menu.addAction('PDC 溶液 MD 复刻…', self.prepare_pdc_paper)
        menu.addSeparator()
        menu.addAction('一句话建模', self.open_modeling)
        menu.addAction('全自动计算', self.open_automatic)
        menu.addAction('连接 / 继续全自动任务', self.resume_automatic)
        menu.addAction('打开全自动输出', self.open_automatic_output)
        menu.addAction('项目库…', self.open_project_library)
        menu.addAction('保存当前模型到项目库', self.archive_current_project)
        menu.addAction('打开当前项目文件夹', self.open_active_project_folder)
        from workflow_examples import EXAMPLES
        examples = menu.addMenu('全自动案例')
        for title, prompt in EXAMPLES:
            examples.addAction(title, lambda checked=False, text=prompt: self.open_automatic(text))
        menu.addAction('填入多分子示例', self.prefill_modeling_example)
        menu.addAction('填入 PDC 论文复刻提示词', self.prefill_pdc_paper)
        menu.addAction('设计候选结构', lambda: self.prefill_ai('请为以下研究目标设计候选分子：'))
        menu.addAction('修改当前结构', lambda: self.prefill_ai('请修改当前结构，要求：'))
        menu.addAction('解读所选计算结果', self.explain_job)
        self.sync_ai_target()
        self.refresh_automatic_outputs()
        self.queue_completed_analysis()

    def update_ai_mode(self):
        automatic = self.ai_mode.currentIndex() == 2
        direct = self.ai_mode.currentIndex() != 1
        following_up = direct and self._modeling is not None and bool(self._modeling['turns'])
        self.send_button.setText('回答并继续' if following_up else '全自动运行' if automatic else
                                 '生成三维结构' if direct else '发送')
        self.suggestion_button.setVisible(not direct)
        self.prompt.setPlaceholderText('补充上面的问题，Ctrl+Enter 继续。' if following_up else
                                       '输入结构和计算要求，自动提交并导出数据。Ctrl+Enter 运行。' if automatic else
                                       '输入分子、数量、排布或原子间距。Ctrl+Enter 生成。' if direct
                                       else '询问结构、计算方法或结果。Ctrl+Enter 发送。')
        self.sync_ai_target()

    def sync_ai_target(self):
        if not hasattr(self, 'ai_mode'):
            return
        main = self.preview.structure
        direct = self.ai_mode.currentIndex() != 1
        if direct and self._modeling is not None:
            active = self._modeling['base']
        else:
            active = self.active_view().structure if direct else main
        if active:
            self.ai_target.setText(f'上下文 · {active.name} / {len(active.atoms)} 原子')
        else:
            self.ai_target.setText('上下文 · 新建结构')

    def show_source(self):
        super().show_source()
        if not getattr(self, 'ai_ui_ready', False):
            return
        if self._ai_input != self.preview.structure:
            self.suggestion = {}
            self.suggestion_button.setEnabled(False)
            self._ai_input = self.preview.structure
            self._active_project_folder = None
            self.job_table.setCurrentCell(-1, -1)
            self.job_table.clearSelection()
            self.show_job()
        self.sync_ai_target()
        if self.docking_ready:
            self.document_activated(self.documents.activeSubWindow())

    def open_modeling(self):
        self.open_assistant()
        self.ai_mode.setCurrentIndex(0)

    def prefill_modeling_example(self):
        if self._modeling is not None:
            self.stop_modeling()
        self.open_modeling()
        self.prompt.setPlainText(MODELING_EXAMPLE)

    def prefill_pdc_paper(self):
        from chatdft.md_inputs import PDC_PAPER_PROMPT

        if self._modeling is not None:
            self.stop_modeling()
        self.open_modeling()
        self.prompt.setPlainText(PDC_PAPER_PROMPT)

    def submit_prompt(self):
        if self.prompt.toPlainText().lstrip().startswith('全自动'):
            self.ai_mode.setCurrentIndex(2)
        if self.ai_mode.currentIndex() != 1:
            self.build_from_prompt(automatic=self.ai_mode.currentIndex() == 2)
        else:
            self.send()

    def build_from_prompt(self, automatic=False):
        if self.operation:
            return
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            self.status.setText('请输入建模要求，或在 AI 菜单填入多分子示例。')
            return
        if (self._modeling is not None or self._auto_ready is not None) and prompt.rstrip('。！!') in ('结束', '停止', '取消', '结束建模', '停止建模', '取消建模'):
            self.stop_modeling()
            self.prompt.clear()
            return
        if self._auto_ready is not None:
            self.status.setText('已有验证通过的结构。请从 AI 菜单继续提交，或结束本次任务。')
            return
        from chatdft.md_inputs import identifies_pdc_paper

        normalized = ''.join(prompt.casefold().split()).rstrip('。.！!')
        if not automatic and self._modeling is None and (identifies_pdc_paper(prompt) or (
                'pdc' in normalized and '论文' in normalized and
                any(word in normalized for word in ('复刻', '复现', 'gromacs')))):
            self.prepare_pdc_paper()
            return
        from chatdft.assembly import request_model

        if self._modeling is None or self._modeling['automatic'] != automatic:
            self._modeling = {'base': self.active_view().structure, 'turns': [], 'automatic': automatic}
        self.sync_ai_target()
        session = self._modeling
        cancel = self._model_cancel = Event()
        settings = dict(self.settings)
        resources = dict(cores=self.cores.value(), memory_gb=self.memory.value())
        provider = self.ai_provider_name()
        self.ai_state.setText(provider + ' · 正在生成配方与三维结构…')
        self.stop_model_button.show()

        def build():
            recipe, model = request_model(settings, prompt, session['base'],
                                          history=tuple(session['turns']), cancel=cancel)
            calc = None
            if model is not None and (automatic or recipe.get('calculation') is not None):
                try:
                    if automatic:
                        calc = automatic_calculation(model, recipe.get('calculation'), resources)
                    else:
                        calc = Calculation(**recipe['calculation'], **resources)
                        build_input(model, calc)
                except (ValueError, TypeError) as error:
                    return {**recipe, 'question': f'计算设置无法使用：{error}。请指定有效的计算设置，或结束这次建模。'}, None, None
            return recipe, model, calc

        def load(value):
            if self._modeling is not session or cancel.is_set():
                return
            recipe, model, calc = value
            if model is None:
                question = recipe['question']
                session['turns'].append({'user': prompt, 'question': question})
                self.answer.appendPlainText(f'\n你：{prompt}\n\nAI：{question}')
                self.prompt.clear()
                self.ai_state.setText(provider + ' · 等待你的回答')
                self.status.setText('建模需要补充条件。可回答并继续，或结束建模。')
                self.update_ai_mode()
                return
            self._modeling = None
            self.stop_model_button.hide()
            self.update_ai_mode()
            self.select_candidate(model, focus_calculation=calc is not None)
            if calc is not None:
                self.set_calculation(calc)
            else:
                self.tabs.setCurrentIndex(0)
            self.proposals = []
            self.candidates.setRowCount(0)
            self.candidate_node.takeChildren()
            self.update_candidate_actions()
            summary = (f'{model.name} · {model.component_count()} 个分子 / 离子片段 · '
                       f'{len(model.atoms)} 个原子 · 电荷 {model.charge:+d} · 多重度 {model.multiplicity}')
            self.answer.appendPlainText(f'\n{prompt}\n\n{summary}\n{recipe["description"]}')
            counts = Counter(part['name'] for part in model.components)
            self.answer.appendPlainText('组分 · ' + ' / '.join(f'{name} × {count}' for name, count in counts.items()))
            if model.warnings:
                self.answer.appendPlainText('\n'.join(model.warnings))
            if calc is not None:
                self.answer.appendPlainText(f'已填写 PySCF 参数 · {calc.method}/{calc.basis} · {calc.task} · {calc.solvent}')
            self.status.setText('已生成 ' + summary)
            self.ai_state.setText(provider + ' · 三维结构已载入')
            request = '\n'.join([turn['user'] for turn in session['turns']] + [prompt])
            try:
                project_id = archive_model(model, calc or self.calculation(), prompt=request,
                                           description=recipe['description'], pending=automatic)
            except OSError as error:
                self.show_error(f'无法保存项目：{error}')
                self.ai_state.setText('模型已载入 · 项目保存失败')
                return
            self._active_project_folder = self.managed_project_folder(project_id)
            if automatic:
                metadata = dict(description=recipe['description'], prompt=request,
                                project_id=project_id, ai_analysis=True)
                self._auto_ready = (model, calc, metadata)
                self.prompt.clear()
                self.stop_model_button.show()
                self._auto_timer.start()

        def failed(message):
            if self._modeling is session and not cancel.is_set():
                self.answer.appendPlainText(f'\n建模暂时中断：{message}')
                self.ai_state.setText(provider + ' · 可重试、补充条件或结束')

        self.run('正在解析描述并构建三维结构…', build, load)
        self.operation.failure.connect(failed)
        # Other tasks lock the entire assistant dock. During modeling keep Stop available.
        self.ai_dock.widget().setEnabled(True)
        for control in self._modeling_controls:
            control.setEnabled(False)

    def stop_modeling(self):
        if self._modeling is None and self._auto_ready is None:
            return
        if self._model_cancel is not None:
            self._model_cancel.set()
        if self._auto_ready is not None:
            try:
                stop_draft(self._auto_ready[2].get('project_id'))
            except (OSError, ValueError) as error:
                self.show_error(error)
                return
        self._modeling = None
        self._auto_ready = None
        self._auto_timer.stop()
        self.stop_model_button.hide()
        self.answer.appendPlainText('\n已结束这次建模。')
        self.ai_state.setText(self.ai_provider_name() + ' · 建模已结束')
        self.status.setText('已结束这次建模。')
        self.update_ai_mode()

    def open_automatic(self, prompt=None):
        self.open_assistant()
        self.ai_mode.setCurrentIndex(2)
        if prompt:
            if self._auto_ready is not None or self.operation:
                return
            if self._modeling is not None:
                self.stop_modeling()
            self.prompt.setPlainText(prompt)

    def resume_automatic(self):
        if self._auto_ready is None:
            self.test_connection()
            return
        if self.operation or self.server_operation:
            self._auto_timer.start()
            return
        prepared = self._auto_ready
        remote = self.connection()
        if self._auto_ready is not prepared:
            return
        if remote is None:
            self.ai_state.setText('结构已载入 · 等待连接服务器')
            self.answer.appendPlainText('\n请使用「AI → 连接 / 继续全自动任务」连接并提交已建好的结构。')
            return
        if self.server_operation:
            self._auto_timer.start()
            return

        model, calculation, metadata = prepared
        self._auto_ready = None
        self.stop_model_button.hide()
        self.ai_state.setText('全自动计算 · 正在提交')

        def submitted(job):
            self._auto_last_job = job['id']
            if self.preview.structure == model:
                self._active_project_folder = self.store.folder(job)
                self.submitted(job)
            else:
                self.reload_jobs()
            if job['status'] in ('queued', 'running'):
                self.ai_state.setText(f'全自动计算 · 正在跟踪 {job["id"]}')
                self.answer.appendPlainText(f'\n已提交作业 {job["id"]}，结束后自动导出实际数据。')

        def failed(message):
            self.reload_jobs()
            self.ai_state.setText('全自动提交失败 · 请核对服务器状态')
            self.answer.appendPlainText(f'\n提交失败：{message}\n已保存本地模型和输入。通过作业列表查看，再决定是否重试。')

        task = self.run_server('正在提交全自动计算…', remote,
                               lambda: submit(remote, self.store, model, calculation, automation=metadata), submitted)
        task.failure.connect(failed)

    def reload_jobs(self, selected=None):
        super().reload_jobs(selected)
        if hasattr(self, '_auto_export_errors'):
            self.refresh_automatic_outputs()
            self.queue_completed_analysis()
            if hasattr(self, '_project_library') and self._project_library.isVisible():
                self._project_library.reload()

    def refresh_automatic_outputs(self):
        if self._auto_export_task:
            self._auto_export_again = True
            return
        jobs = [(self.store.folder(job), job) for job in self.jobs.values() if job.get('automation')]
        pending = [(folder, job) for folder, job in jobs if needs_export(folder, job)]
        if not pending:
            return

        def export():
            outputs = []
            for folder, job in pending:
                try:
                    outputs.append((job['id'], job['status'], export_job(folder, job), None))
                except Exception as error:
                    outputs.append((job['id'], job['status'], None, str(error)))
            return outputs

        def finished():
            self._auto_export_task = None
            task.deleteLater()
            if self._auto_export_again:
                self._auto_export_again = False
                self.refresh_automatic_outputs()

        task = self._auto_export_task = Task(export)
        task.success.connect(self.show_automatic_outputs)
        task.finished.connect(finished)
        task.start()

    def show_automatic_outputs(self, outputs):
        for job_id, state, path, error in outputs:
            if error is not None:
                if self._auto_export_errors.get(job_id) != error:
                    self._auto_export_errors[job_id] = error
                    self.answer.appendPlainText(f'\n作业 {job_id} 输出失败：{error}')
                continue
            self._auto_export_errors.pop(job_id, None)
            if path is not None:
                self._auto_last_job = job_id
                self.answer.appendPlainText(f'\n作业 {job_id} · {state} · 已写入分析报告、图件和可用数据。\n{path.parent}')
                self.status.setText('分析报告、图件和数据已就绪。')
                job = self.jobs.get(job_id)
                if job and (state not in ('completed','succeeded') or not job['automation'].get('ai_analysis')):
                    name = {'failed':'计算失败','cancelled':'作业已取消'}.get(state, '计算输出已就绪')
                    self.notify_project(job, name)
        self.queue_completed_analysis()

    def notify_project(self, job, title):
        self._notification_folder = self.store.folder(job)
        self.status.setText(f'{title} · {job["name"]} · {job["id"]}')
        if self._tray.isSystemTrayAvailable() and self._tray.supportsMessages():
            if not self._tray.isVisible():
                self._tray.show()
            self._tray.showMessage('ChatDFT · ' + title, f'{job["name"]}\n{job["id"]}',
                                   QSystemTrayIcon.Information, 10000)
        else:
            QApplication.alert(self, 5000)

    def open_notification_folder(self):
        if self._notification_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._notification_folder)))

    def queue_completed_analysis(self):
        for job in self.jobs.values():
            metadata = job.get('automation') or {}
            folder = self.store.folder(job) / '输出'
            if (metadata.get('ai_analysis') and job['status'] in ('completed','succeeded')
                    and (folder/'计算报告.md').is_file() and (folder/'分析.json').is_file()
                    and not (folder/'AI分析状态.json').is_file()):
                if self._analysis_task is None or self._analysis_job_id != job['id']:
                    self._analysis_queue[job['id']] = job
        self.start_ai_analysis()

    def retry_ai_analysis(self, job):
        if not job or job['status'] not in ('completed','succeeded'):
            return
        if self._analysis_task is not None and self._analysis_job_id == job['id']:
            self.status.setText('这个项目正在生成 AI 报告。')
            return
        if not (self.store.folder(job)/'输出'/'分析.json').is_file():
            self.status.setText('请先同步计算结果，等基础分析文件生成后再试。')
            return
        self._analysis_queue[job['id']] = job
        self.start_ai_analysis()

    def start_ai_analysis(self):
        if self._analysis_task is not None or not self._analysis_queue:
            return
        from automatic_ai import generate_ai_analysis, record_ai_status

        job_id = next(iter(self._analysis_queue))
        job = self._analysis_queue.pop(job_id)
        self._analysis_job_id = job_id
        folder = self.store.folder(job)
        cancel = self._analysis_cancel = Event()
        settings = dict(self.settings)
        self.status.setText(f'正在生成 AI 分析和补充图 · {job["name"]}')

        def generate():
            try:
                figures, rejected, report_path = generate_ai_analysis(settings, folder, job, cancel=cancel)
                record_ai_status(folder, status='completed', figures=figures, rejected=rejected, report=report_path)
                return figures, rejected
            except Exception as error:
                if not cancel.is_set():
                    record_ai_status(folder, status='failed', message=str(error))
                raise

        def completed(counts):
            figures, rejected = counts
            missing = f'，{rejected} 个方案引用的数据不可用' if rejected else ''
            self.answer.appendPlainText(f'\n{job["name"]} · {job_id} · AI 报告和 {figures} 张补充图已保存{missing}。')
            self.notify_project(job, '计算、AI 报告和图件已就绪')
            if hasattr(self, '_project_library'):
                self._project_library.selection_changed()

        def failed(message):
            if not cancel.is_set():
                self.answer.appendPlainText(f'\n{job_id} 的 AI 分析失败：{message}。基础数据可用，可在项目库重试。')
                self.notify_project(job, '计算完成 · AI 分析可重试')

        def finished():
            self._analysis_task = self._analysis_cancel = None
            task.deleteLater()
            if not cancel.is_set():
                self.start_ai_analysis()

        task = self._analysis_task = Task(generate)
        task.success.connect(completed)
        task.failure.connect(failed)
        task.finished.connect(finished)
        task.start()

    def open_project_library(self, *_):
        if not hasattr(self, '_project_library'):
            self._project_library = ProjectLibrary(self)
        self.reload_jobs()
        self._project_library.reload()
        self._project_library.show()
        self._project_library.raise_()

    def managed_project_folder(self, project_id):
        from chatdft.storage import data_dir
        return data_dir() / 'projects' / project_id

    def archive_current_project(self):
        try:
            key = archive_model(self.structure(), self.calculation())
            self._active_project_folder = self.managed_project_folder(key)
            self.status.setText(f'已保存到项目库 · {key[:12]}')
        except (OSError, ValueError) as error:
            self.show_error(error)

    def open_active_project_folder(self):
        folder = self._active_project_folder
        if folder is None and self.project_path:
            folder = self.project_path.parent
        if folder is None:
            self.status.setText('请先打开项目库选择项目，或保存当前模型到项目库。')
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def restore_library_project(self, entry, prefer_result=True):
        if self.operation:
            self.status.setText('当前建模完成后即可切换项目。')
            return None
        folder, job = entry['folder'], entry['job']
        try:
            if job:
                value = job
            else:
                value = json.loads((folder/'初始结构.chatdft.json').read_text('utf-8'))
            original = Structure.from_dict(value['structure'])
            model = original
            if job and prefer_result and job['status'] in ('completed','succeeded'):
                from workflow import result_geometry
                try:
                    model = result_geometry(job, self.read_result(job))
                except (OSError, ValueError, KeyError):
                    pass
            calculation = Calculation(**value['calculation'])
            build_input(model, calculation)
            if (self.preview.structure and self.preview.structure != model
                    and self._active_project_folder is None and self.project_path is None):
                archive_model(self.preview.structure, self.calculation())
            self.select_candidate(model, focus_calculation=False)
            self.set_calculation(calculation)
            if model.atoms != original.atoms:
                self.set_reference(original)
            else:
                self.reset_reference()
            self._active_project_folder = folder
            if job:
                self.reload_jobs(job['id'])
                self.tabs.setCurrentIndex(2)
            else:
                self.tabs.setCurrentIndex(0)
            self.status.setText(f'已恢复项目 · {model.name} · {entry["id"][:16]}')
            return original, calculation
        except (OSError, TypeError, ValueError, KeyError) as error:
            self.show_error(error)
            return None

    def resume_library_draft(self, entry):
        if self._auto_ready is not None:
            self.status.setText('请先继续或结束当前待提交项目。')
            return
        if not entry.get('pending'):
            return
        loaded = self.restore_library_project(entry, prefer_result=False)
        if loaded is None:
            return
        model, calculation = loaded
        self._auto_ready = (model, calculation, dict(project_id=entry['id'], ai_analysis=True,
                            description=entry.get('description',''), prompt=entry.get('prompt','')))
        self.open_automatic()
        self.stop_model_button.show()
        self.resume_automatic()

    def open_automatic_output(self):
        job = self.current_job()
        if not job or not job.get('automation'):
            job = self.jobs.get(self._auto_last_job) or next(
                (entry for entry in self.jobs.values() if entry.get('automation')), None)
        if job is None:
            self.status.setText('当前还没有全自动作业。')
            return
        folder = self.store.folder(job) / '输出'
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        else:
            self.status.setText('输出目录尚未生成，请同步作业。')

    def finish_operation(self):
        super().finish_operation()
        if self._model_cancel is not None:
            self._model_cancel = None
            for control in self._modeling_controls:
                control.setEnabled(True)
            if self._modeling is not None:
                self.prompt.setFocus()
        self.update_command_states()

    def use_reference(self):
        if not self.operation:
            super().use_reference()

    def open_paper_reproduction(self):
        from paper_dialog import PaperDialog

        if not hasattr(self, 'paper_dialog'):
            self.paper_dialog = PaperDialog(self)
        self.paper_dialog.show()
        self.paper_dialog.raise_()
        self.paper_dialog.activateWindow()

    def open_md_reproduction(self):
        from md_dialog import MDDialog

        if not hasattr(self, 'md_dialog'):
            self.md_dialog = MDDialog(self)
            boxed = [job for job in self.md_dialog.job_entries
                     if (self.md_dialog.store.folder(job)/'preflight.json').is_file()]
            if boxed:
                self.md_dialog.select_jobs(boxed, preferred='PDC_neutral')
                self.md_dialog.open_preview()
        self.md_dialog.show()
        self.md_dialog.raise_()
        self.md_dialog.activateWindow()

    def prepare_pdc_paper(self):
        self.open_md_reproduction()
        self.md_dialog.prepare_paper()

    def ai_provider_name(self):
        model = self.settings.get('model')
        if self.settings.get('provider') == 'codex':
            return 'Codex · ' + (model or '默认模型')
        return 'API · ' + (model or '待配置')

    def open_assistant(self):
        super().open_assistant()
        if getattr(self, 'ai_ui_ready', False):
            self.ai_tabs.setCurrentIndex(0)
            self.prompt.setFocus()

    def prefill_ai(self, text):
        self.open_assistant()
        self.ai_mode.setCurrentIndex(1)
        self.prompt.setPlainText(text)
        self.prompt.setFocus()

    def send(self, prompt=None, result=None):
        if not getattr(self, 'ai_ui_ready', False):
            return super().send(prompt, result)
        if self.operation:
            self.status.setText('当前操作正在进行，请等待完成后发送。')
            return
        text = prompt if isinstance(prompt, str) and prompt.strip() else self.prompt.toPlainText().strip()
        if not text:
            self.open_assistant()
            self.ai_state.setText('请先输入结构设计或计算问题。')
            return
        self.open_assistant()
        self.ai_mode.setCurrentIndex(1)
        super().send(text, result)
        if not self.operation:
            self.ai_state.setText(self.status.text())
            return
        self._ai_request_pending = True
        self.ai_state.setText(self.ai_provider_name() + ' · 正在生成，请稍候…')
        # A failed request must not leave old candidates or parameters available as new advice.
        self.proposals = []
        self.suggestion = {}
        self.candidates.setRowCount(0)
        self.candidate_node.takeChildren()
        self.suggestion_button.setEnabled(False)
        self.update_candidate_actions()
        self.operation.failure.connect(self.ai_request_failed)

    def receive(self, prompt, value):
        super().receive(prompt, value)
        if getattr(self, 'ai_ui_ready', False):
            self._ai_request_pending = False
            self._ai_last_ok = datetime.now().strftime('%H:%M:%S')
            self.ai_state.setText(f'{self.ai_provider_name()} · 已回复 {self._ai_last_ok} · {len(self.proposals)} 个候选')
            if self.proposals:
                self.candidate_node.setExpanded(True)
                self.status.setText('AI 已回复；在“候选结构”页选择“用于计算”生成三维模型。')

    def ai_request_failed(self, message):
        self._ai_request_pending = False
        self.ai_state.setText(self.ai_provider_name() + ' · 请求失败，可重试')
        self.answer.appendPlainText('\nAI 请求失败：' + str(message))
        self.status.setText(str(message))

    def use_candidate(self):
        if self.operation:
            return
        try:
            candidate = self.candidate()
        except ValueError as exc:
            self.show_error(exc)
            return
        suggestion = self.suggestion.copy()

        def load(model):
            self.select_candidate(model)
            self.suggestion = suggestion
            self.suggestion_button.setEnabled(bool(suggestion))

        self.run('正在生成三维候选…', lambda: self.build_candidate(candidate), load)
        self.operation.failure.connect(lambda message: self.ai_local_failed('三维建模', message))

    def ai_local_failed(self, operation, message):
        self.ai_state.setText(operation + '失败，请检查候选结构')
        self.answer.appendPlainText('\n' + operation + '失败：' + str(message))

    def test_ai_connection(self):
        if self.operation:
            self.status.setText('请等待当前操作完成后测试 AI。')
            return
        self.open_assistant()
        self.ai_state.setText(self.ai_provider_name() + ' · 正在测试实际请求…')
        self.run('正在测试 AI 实际响应…', lambda: ask(self.settings, [
            {'role': 'user', 'content': '连接测试。请用中文简短确认可以接收请求，不生成候选，candidates为空；calculation给HF/STO-3G/sp/Gas。'}
        ], {'application': 'ChatDFT', 'purpose': 'connection_test'}), self.ai_connection_ok)
        self.operation.failure.connect(self.ai_request_failed)

    def ai_connection_ok(self, value):
        self._ai_last_ok = datetime.now().strftime('%H:%M:%S')
        self.ai_state.setText(self.ai_provider_name() + ' · 实测通过 ' + self._ai_last_ok)
        self.answer.appendPlainText('\nAI 连接测试通过：' + value['message'])
        self.status.setText('AI 实际请求成功，已收到有效结构化回复。')

    def configure(self):
        previous = self.settings
        super().configure()
        if getattr(self, 'ai_ui_ready', False) and self.settings is not previous:
            self._ai_last_ok = ''
            self.ai_state.setText(self.ai_provider_name() + ' · 设置已更新，请测试连接')
