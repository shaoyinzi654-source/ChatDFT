from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl
from PySide6.QtGui import QFont, QIcon, QPixmap, QPainter, QColor, QDesktopServices
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,
    QLabel,QPushButton,QLineEdit,QPlainTextEdit,QTextBrowser,QComboBox,QSpinBox,QCheckBox,
    QSplitter,QTabWidget,QFrame,QDialog,QDialogButtonBox,QFileDialog,QMessageBox,QInputDialog,
    QTableWidget,QTableWidgetItem,QHeaderView,QScrollArea,QProgressBar)
from .chemistry import Structure,LIBRARY,from_smiles,from_xyz,combine,validate_structure
from .gaussian import Calculation,METHODS,BASES,TASKS,SOLVENTS,build_input,parse_log,report
from .storage import data_dir,atomic_json,load_settings,save_settings,protect_secret
from .jobs import JobStore,Remote
from .ai import ask,find_codex
from .model_picker import ModelPicker
from .widgets import MoleculeView,PlotView





class Task(QThread):
    success=Signal(object)
    failure=Signal(str)
    def __init__(self,fn):super().__init__();self.fn=fn
    def run(self):
        try:self.success.emit(self.fn())
        except Exception as e:self.failure.emit(str(e))


def button(text,fn=None,primary=False):
    b=QPushButton(text)
    if primary:b.setObjectName('Primary')
    if fn:b.clicked.connect(fn)
    return b


def label(text,kind='Muted'):
    w=QLabel(text);w.setObjectName(kind);w.setWordWrap(True);return w


class SettingsDialog(QDialog):
    def __init__(self,settings,parent):
        super().__init__(parent);self.setWindowTitle('设置 · 模型与计算环境');self.resize(720,720);self.settings=dict(settings);self.fields={}
        outer=QVBoxLayout(self);tabs=QTabWidget();outer.addWidget(tabs)
        ai=QWidget();f=QFormLayout(ai);f.setSpacing(13)
        tabs.addTab(ai,'AI 模型')
        self.combo(f,'provider','接入方式',['codex','api'])
        f.addRow(label('Codex 使用本机已有登录；API 支持 OpenAI Responses 与兼容 Chat Completions。'))
        self.edit(f,'codex_path','Codex 程序路径（可留空）')
        self.model_picker = ModelPicker(settings.get('model', ''), self.fields['provider'], self.fields['codex_path'],
                                        preferences=settings.get('models_by_provider'))
        f.insertRow(2, '模型', self.model_picker)
        self.edit(f,'api_base','API Base URL')
        self.combo(f,'api_protocol','API 协议',['responses','chat_completions'])
        self.secret=QLineEdit();self.secret.setEchoMode(QLineEdit.Password);self.secret.setPlaceholderText('留空保留已有密钥；新密钥通过 Windows 凭据加密保存')
        f.addRow('API Key',self.secret)
        self.clear_secret=QCheckBox('删除已保存的 API Key');f.addRow(self.clear_secret)
        self.spin(f,'timeout','AI 超时 / 秒',30,900)
        remote=QWidget();f=QFormLayout(remote);f.setSpacing(12);tabs.addTab(remote,'计算服务器')
        self.combo(f,'target','执行位置',['ssh','local'])
        for key,text in [('ssh_host','SSH 主机'),('ssh_user','用户名')]:self.edit(f,key,text)
        self.spin(f,'ssh_port','SSH 端口',1,65535)
        self.edit(f,'ssh_key','私钥文件路径（可选）')
        self.edit(f,'host_fingerprint','可信 SSH 指纹')
        self.edit(f,'remote_root','服务器工作目录')
        self.combo(f,'engine','计算引擎',['pyscf','gaussian'])
        self.edit(f,'remote_python','PySCF Python 环境')
        self.edit(f,'gaussian_exe','Gaussian 程序（可选）')
        self.edit(f,'gaussian_profile','Gaussian 环境脚本')
        self.edit(f,'local_gaussian','本机计算环境 程序')
        f.addRow(label('SSH 密码在连接时输入，仅保存在本次会话内。默认使用服务器上的开源 PySCF。'))
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);outer.addWidget(buttons)

    def edit(self,f,key,text):
        w=QLineEdit(str(self.settings.get(key,'')));self.fields[key]=w;f.addRow(text,w)
    def combo(self,f,key,text,items):
        w=QComboBox();w.addItems(items);w.setCurrentText(str(self.settings.get(key,items[0])));self.fields[key]=w;f.addRow(text,w)
    def spin(self,f,key,text,low,high):
        w=QSpinBox();w.setRange(low,high);w.setValue(int(self.settings.get(key,low)));self.fields[key]=w;f.addRow(text,w)
    def accept(self):
        for key,w in self.fields.items():
            self.settings[key]=w.value() if isinstance(w,QSpinBox) else w.currentText() if isinstance(w,QComboBox) else w.text().strip()
        self.settings['model'] = self.model_picker.model_id()
        self.settings['models_by_provider'] = {**self.model_picker.preferences,
                                               self.settings['provider']: self.settings['model']}
        try:
            if self.clear_secret.isChecked():self.settings['api_secret']=''
            if self.secret.text():self.settings['api_secret']=protect_secret(self.secret.text())
            if self.settings['ssh_host'] and (not self.settings['ssh_user'] or
                    not self.settings['host_fingerprint'].startswith('SHA256:')):
                raise ValueError('请填写 SSH 用户名和可信的 SHA256 主机指纹。')
        except Exception as e:
            QMessageBox.warning(self,'设置未保存',str(e));return
        super().accept()

    def done(self, result):
        self.model_picker.stop()
        super().done(result)


class ApplicationActions(QMainWindow):
    def update_badges(self):
        self.ai_badge.setText('本机 Codex' if self.settings['provider']=='codex' else 'API · '+(self.settings['model'] or '待配置'))
        self.environment.setText(('SSH · '+self.settings['ssh_host']+':'+str(self.settings['ssh_port'])) if self.settings['target']=='ssh' else '本机计算环境')

    def background(self,fn,done,buttons=(),quiet=False):
        t=Task(fn);self.threads.append(t)
        for b in buttons:b.setEnabled(False)
        if not quiet:self.progress.setRange(0,0)
        def finished():
            for b in buttons:b.setEnabled(True)
            if not quiet:self.progress.setRange(0,1);self.progress.setValue(0)
            self.threads.remove(t);t.deleteLater()
        t.success.connect(done);t.failure.connect(lambda e:self.status.setText(e) if quiet else self.error(e));t.finished.connect(finished);t.start()

    def error(self,text):
        self.status.setText(text);QMessageBox.warning(self,'操作未完成',text)

    def set_structure(self,s):
        self.structure=s;self.viewer.set_structure(s);self.structure_name.setText(s.name);self.smiles.setText(s.smiles)
        self.charge.setValue(s.charge);self.multiplicity.setValue(s.multiplicity)
        self.structure_meta.setText(f'{s.formula()}  |  原子: {len(s.atoms)}  键: {len(s.bonds)}')
        self.structure_meta.setToolTip(s.provenance)
        self.structure_warning.setText('\n'.join(s.warnings));self.persist_session()
        self.update_model_panels()

    def load_library(self):
        name=self.library.currentText();self.background(lambda:from_smiles(LIBRARY[name],name),self.set_structure,buttons=(self.build_btn,))

    def build_smiles(self):
        smiles=self.smiles.text().strip()
        if not smiles:return
        self.background(lambda:from_smiles(smiles,'自定义分子'),self.set_structure,buttons=(self.build_btn,))

    def current_structure(self):
        s=copy.deepcopy(self.structure);s.charge=self.charge.value();s.multiplicity=self.multiplicity.value();validate_structure(s);return s

    def calculation(self):
        return Calculation(self.method.currentText(),self.basis.currentText(),self.task.currentData(),self.solvent.currentText(),self.cores.value(),self.memory.value(),self.dispersion.isChecked())

    def apply_calc(self,c):
        for w,key in [(self.method,'method'),(self.basis,'basis'),(self.solvent,'solvent')]:
            if w.findText(c.get(key,''))>=0:w.setCurrentText(c[key])
        i=self.task.findData(c.get('task'))
        if i>=0:self.task.setCurrentIndex(i)
        self.dispersion.setChecked(False)

    def send_ai(self):
        prompt=self.prompt.toPlainText().strip()
        if not prompt:return
        self.messages.append(dict(role='user',content=prompt));self.prompt.clear();self.render_chat();self.persist_session()
        context=dict(structure=dict(self.structure.to_dict(),charge=self.charge.value(),multiplicity=self.multiplicity.value()),calculation=self.calculation().to_dict(),results=self.result)
        settings=dict(self.settings);messages=copy.deepcopy(self.messages)
        self.status.setText('AI 正在分析研究目标与当前结构…')
        def done(reply):
            self.messages.append(dict(role='assistant',content=reply['message']));self.candidates=reply['candidates'];self.ai_calc=reply.get('calculation',{})
            self.candidate_box.clear()
            for c in self.candidates:self.candidate_box.addItem(c['name'])
            self.apply_candidate_btn.setEnabled(bool(self.candidates));self.render_chat();self.persist_session();self.status.setText('AI 建议已生成 · 选择候选结构后应用到工作区')
        self.background(lambda:ask(settings,messages,context),done,buttons=(self.send,))

    def render_chat(self):
        if not self.messages:
            self.chat.setPlainText('输入建模目标或结构修改要求。生成的候选模型将在下方列出。');return
        self.chat.setMarkdown('\n\n---\n\n'.join(('**用户**' if m['role']=='user' else '**建模助手**')+'\n\n'+m['content'] for m in self.messages))
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    def apply_candidate(self):
        i=self.candidate_box.currentIndex()
        if i<0:return
        c=copy.deepcopy(self.candidates[i])
        def build():
            s=from_smiles(c['smiles'],c['name']);s.multiplicity=c['multiplicity'];validate_structure(s);s.warnings.append('AI 设计理由：'+c['rationale']);return s
        def done(s):self.set_structure(s);self.apply_calc(self.ai_calc);self.status.setText('候选结构已通过几何与电子数检查；请检查三维模型后提交。')
        self.background(build,done,buttons=(self.apply_candidate_btn,))

    def persist_session(self):
        if not hasattr(self,'structure'):return
        atomic_json(data_dir()/'session.json',dict(structure=self.structure.to_dict(),messages=self.messages))

    def load_session(self):
        try:
            saved=json.loads((data_dir()/'session.json').read_text('utf-8'));s=Structure.from_dict(saved['structure']);validate_structure(s);self.messages=saved.get('messages',[])
        except (OSError,ValueError,KeyError,TypeError):s=from_smiles('Oc1ccccc1','苯酚 · Phenol')
        self.set_structure(s);self.render_chat()

    def import_xyz(self):
        path,_=QFileDialog.getOpenFileName(self,'导入 XYZ（单位 Å）','','XYZ (*.xyz);;所有文件 (*)')
        if not path:return
        try:self.set_structure(from_xyz(Path(path).read_text('utf-8'),Path(path).stem,self.charge.value(),self.multiplicity.value()))
        except Exception as e:self.error(str(e))

    def export_xyz(self):
        path,_=QFileDialog.getSaveFileName(self,'导出 XYZ','structure.xyz','XYZ (*.xyz)')
        if path:Path(path).write_text(self.structure.xyz(),encoding='utf-8')

    def edit_coordinates(self):
        d=QDialog(self);d.setWindowTitle('编辑坐标 · 单位 Å');d.resize(650,600);v=QVBoxLayout(d);name=QLineEdit(self.structure.name);v.addWidget(name)
        v.addWidget(label('可编辑 XYZ 原子坐标。导入坐标的连接线按距离推断，不代表经过确认的键级。'))
        text=QPlainTextEdit(self.structure.xyz());text.setFont(QFont('Consolas',11));v.addWidget(text)
        bb=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);v.addWidget(bb);bb.accepted.connect(d.accept);bb.rejected.connect(d.reject)
        if d.exec()==QDialog.Accepted:
            try:self.set_structure(from_xyz(text.toPlainText(),name.text(),self.charge.value(),self.multiplicity.value()))
            except Exception as e:self.error(str(e))

    def add_fragment(self):
        text,ok=QInputDialog.getText(self,'组合片段模型','要添加片段的 SMILES，例如 O、CCO：')
        if not ok or not text:return
        gap,ok=QInputDialog.getDouble(self,'设置片段间距','沿 X 方向的包围盒间距 / Å',2.5,.5,20,2)
        if not ok:return
        s=self.current_structure();self.background(lambda:combine(s,from_smiles(text,'片段'),gap),self.set_structure,buttons=(self.build_btn,))

    def save_project(self):
        path,_=QFileDialog.getSaveFileName(self,'保存项目','research.chatdft.json','ChatDFT (*.chatdft.json)')
        if path:
            try:atomic_json(path,dict(version=2,structure=self.current_structure().to_dict(),calculation=self.calculation().to_dict(),messages=self.messages));self.status.setText('项目已保存，包含结构、方案和对话。')
            except Exception as e:self.error(str(e))

    def open_project(self):
        path,_=QFileDialog.getOpenFileName(self,'打开项目','','ChatDFT (*.chatdft.json)')
        if not path:return
        try:
            value=json.loads(Path(path).read_text('utf-8'));s=Structure.from_dict(value['structure']);validate_structure(s)
            self.messages=value.get('messages',[]);self.set_structure(s);self.apply_calc(value.get('calculation',{}));self.cores.setValue(value.get('calculation',{}).get('cores',8));self.memory.setValue(value.get('calculation',{}).get('memory_gb',8));self.dispersion.setChecked(value.get('calculation',{}).get('dispersion',False));self.render_chat()
        except Exception as e:self.error(str(e))

    def preview_input(self):
        try:
            structure=self.current_structure();calculation=self.calculation()
            gjf=build_input(structure,calculation)
            text=json.dumps(dict(engine=self.settings.get('engine','pyscf'),structure=structure.to_dict(),calculation=calculation.to_dict(),solvent_model='IEF-PCM' if calculation.solvent!='Gas' else 'Gas'),ensure_ascii=False,indent=2) if self.settings.get('engine','pyscf')=='pyscf' else gjf
        except Exception as e:self.error(str(e));return
        d=QDialog(self);d.setWindowTitle('计算输入预览');d.resize(730,650);v=QVBoxLayout(d);v.addWidget(label('PySCF 实际使用结构化 JSON；Gaussian .gjf 为可选交换模板，其溶剂设置为 SMD，与 PySCF 的 IEF-PCM 不同。'));edit=QPlainTextEdit(text);edit.setReadOnly(True);edit.setFont(QFont('Consolas',11));v.addWidget(edit)
        def save():
            path,_=QFileDialog.getSaveFileName(d,'导出 Gaussian 输入','calculation.gjf','Gaussian (*.gjf *.com)')
            if path:Path(path).write_text(gjf,encoding='ascii',newline='\n')
        v.addWidget(button('导出 .gjf',save,True));v.addWidget(button('关闭',d.accept));d.exec()

    def open_settings(self):
        d=SettingsDialog(self.settings,self)
        if d.exec()==QDialog.Accepted:
            self.settings=d.settings;save_settings(self.settings)
            if self.remote:self.remote.close();self.remote=None
            self.engine_badge.setText('引擎待检测');self.update_badges();self.status.setText('设置已保存；请重新连接服务器。')

    def connect_server(self):
        if self.poll_busy:return
        password=''
        if not self.settings.get('ssh_key'):
            password,ok=QInputDialog.getText(self,'连接计算服务器','SSH 密码（仅本次会话使用）：',QLineEdit.Password)
            if not ok:return
        remote=Remote(self.settings,password)
        def done(health):
            if self.remote:self.remote.close()
            self.remote=remote;installed=health['gaussian_installed'];self.engine_badge.setText(health.get('engine','')+' '+health.get('engine_version','')+' · 就绪' if installed else '引擎待安装');self.environment.setText(f"{health['host']} · {health['cores']} 核\n磁盘可用 {health['free_disk_gb']} GB\n"+('计算引擎已就绪' if installed else '任务端已就绪，等待引擎安装'))
            self.status.setText('服务器任务端连接成功 · '+health['note'])
        self.background(remote.install_worker,done)

    def submit_job(self):
        try:s=self.current_structure();calc=self.calculation();gjf=build_input(s,calc)
        except Exception as e:self.error(str(e));return
        active=[j for j in self.store.list() if j['status']=='running']
        if active:self.error('已有运行中的任务。本版本限制一次运行一个任务，避免 16 GB 服务器资源超配。');return
        if self.settings['target']=='ssh' and not self.remote:self.error('请先连接计算服务器，检查引擎状态。');return
        if calc.task=='ts':
            QMessageBox.information(self,'过渡态初始结构','当前结构必须是过渡态附近的合理初猜。一个显著虚频及 IRC 连接验证后，才能确认过渡态。')
        settings=dict(self.settings);remote=self.remote
        def launch():
            health=remote.health() if settings['target']=='ssh' else __import__('chatdft.worker',fromlist=['probe']).probe(settings['local_gaussian'],'',settings.get('engine','pyscf'))
            if not health['gaussian_installed']:raise ValueError('计算引擎尚未就绪。请先完成服务器开源引擎安装。')
            j=self.store.create(s,calc,gjf,settings);return self.store.launch(j,remote)
        def done(job):self.refresh_jobs();self.show_jobs();self.status.setText('任务已提交：'+job['id'])
        self.background(launch,done,buttons=(self.submit_btn,))

    def refresh_jobs(self):
        self.job_values=self.store.list();selected=self.selected_job['id'] if self.selected_job else None
        self.jobs_table.blockSignals(True);self.jobs_table.setRowCount(len(self.job_values))
        states={'prepared':'准备完成','running':'运行中','completed':'已完成','failed':'失败','cancelled':'已取消'}
        for row,j in enumerate(self.job_values):
            c=j['calculation'];values=[j['id'],j['name'],c['method']+'/'+c['basis']+' · '+c['task'],states.get(j['status'],j['status']),j['target'],time.strftime('%m-%d %H:%M',time.localtime(j['created']))]
            for col,value in enumerate(values):self.jobs_table.setItem(row,col,QTableWidgetItem(str(value)))
            if j['id']==selected:self.jobs_table.selectRow(row)
        self.jobs_table.blockSignals(False)
        if selected:self.select_job()
        self.update_project_tree()

    def select_job(self):
        row=self.jobs_table.currentRow()
        if row<0 or row>=len(self.job_values):return
        j=self.job_values[row];self.selected_job=j
        self.job_summary.setText(f"{j['id']} · {j.get('message','')} "+j.get('connection_warning',''))
        log=self.store.folder(j)/'output.log'
        if log.exists():
            with log.open('rb') as f:f.seek(max(0,f.seek(0,2)-150000));self.log.setPlainText(f.read().decode('utf-8',errors='replace'))
        else:self.log.setPlainText('等待计算输出日志。')

    def poll_jobs(self):
        if self.poll_busy:return
        jobs=[j for j in self.store.list() if j['status']=='running' and j['target']=='ssh' and 'remote_dir' in j]
        if not jobs or not self.remote:self.refresh_jobs();return
        self.poll_busy=True;remote=self.remote
        def sync():
            for j in jobs:
                expected=j.get('connection',{})
                if any(remote.settings.get(k)!=expected.get(k) for k in ['ssh_host','ssh_port','ssh_user','remote_root','host_fingerprint']):continue
                remote.sync(j,self.store.folder(j))
        t=Task(sync);self.threads.append(t)
        t.success.connect(lambda _:self.refresh_jobs());t.failure.connect(lambda e:self.status.setText('服务器连接中断，任务不会因此自动取消：'+e))
        def finish():self.poll_busy=False;self.threads.remove(t);t.deleteLater()
        t.finished.connect(finish);t.start()

    def cancel_job(self):
        j=self.selected_job
        if not j or j['status']!='running':return
        if j['target']=='ssh':
            if not self.remote:self.error('请先连接该任务的服务器。');return
            self.background(lambda:self.remote.cancel(j),lambda _:self.status.setText('已发送取消请求，等待任务端确认。'))
        else:(self.store.folder(j)/'cancel').write_text('cancel');self.status.setText('已发送取消请求。')

    def download_job(self):
        j=self.selected_job
        if not j:return
        target=QFileDialog.getExistingDirectory(self,'选择导出计算文件的目录')
        if not target:return
        def transfer():
            folder=self.store.folder(j)
            if j['target']=='ssh':
                if not self.remote:raise ValueError('请先连接计算服务器。')
                self.remote.download_artifacts(j,folder)
            output=Path(target)/j['id'];output.mkdir(exist_ok=True)
            for name in ['input.gjf','input.json','output.log','result.json','calculation.chk','state.json','job.json']:
                if (folder/name).exists():shutil.copy2(folder/name,output/name)
            return str(output)
        self.background(transfer,lambda p:self.status.setText('计算文件已导出：'+p))

    def open_job_folder(self):
        if self.selected_job:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.folder(self.selected_job))))

    def analyze_job(self):
        if not self.selected_job:return
        result_path=self.store.folder(self.selected_job)/'result.json'
        if self.selected_job.get('engine')=='pyscf':
            if not result_path.exists():self.error('任务还没有生成已完成的量子计算结果，请先查看日志。');return
            r=json.loads(result_path.read_text('utf-8'));r['source_file']=str(result_path);r['title']=self.selected_job['name'];self.show_result(r);return
        path=self.store.folder(self.selected_job)/'output.log'
        if not path.exists():self.error('该任务还没有输出日志。');return
        self.display_result(path,self.selected_job['calculation']['task'],self.selected_job['name'])

    def import_log(self):
        path,_=QFileDialog.getOpenFileName(self,'导入 Gaussian 日志','','Gaussian 日志 (*.log *.out);;所有文件 (*)')
        if path:self.display_result(Path(path),'',Path(path).stem)

    def display_result(self,path,task,title):
        def parse():
            if path.stat().st_size>200*1024*1024:raise ValueError('日志超过 200 MB，请先提取所需计算段。')
            value=parse_log(path.read_text('utf-8',errors='replace'),task);value['source_file']=str(path);value['title']=title;return value
        def done(r):
            self.show_result(r)
        self.background(parse,done)

    def show_result(self,r):
        self.result=r;self.result_text.setMarkdown(report(r,r.get('title','量子计算结果')));self.energy_plot.set_data(r['scf_energies_hartree'],'SCF 迭代电子能 / Hartree')
        self.freq_plot.set_data(r['frequencies_cm1'],'谐振频率 / cm⁻¹ · 横轴为模式序号');self.show_results()
        self.status.setText('结果已载入 · '+('正常完成' if r['normal_termination'] else '未正常完成，请检查日志'))

    def explain_results(self):
        if not self.result:self.error('请先载入一份计算结果。');return
        self.prompt.setPlainText('请根据当前结果解释计算是否可靠、结构是否收敛，以及下一步应检查什么。只能使用已有数值，不要编造。');self.activate_model();self.show_assistant();self.send_ai()

    def load_result_geometry(self):
        if not self.result or not self.result.get('geometry'):self.error('日志中没有找到可读取的笛卡尔几何。');return
        r=self.result
        if r.get('charge') is None or r.get('multiplicity') is None:self.error('日志缺少电荷和自旋多重度，不能安全地自动载入；请手动导入 XYZ 并指定电荷与自旋。');return
        xyz=str(len(r['geometry']))+'\nFinal calculated geometry\n'+'\n'.join(f"{a['element']} {a['x']} {a['y']} {a['z']}" for a in r['geometry'])
        try:
            s=from_xyz(xyz,r.get('title','Final geometry'),r['charge'],r['multiplicity']);s.provenance=r.get('engine','Gaussian')+' 计算输出中的最终几何 / '+('存在优化收敛标记' if r['optimization_converged'] else '未检出优化收敛标记')
            s.warnings=list(r['warnings']);self.set_structure(s);self.activate_model()
        except Exception as e:self.error(str(e))

    def export_report(self):
        if not self.result:self.error('还没有可导出的计算结果。');return
        path,_=QFileDialog.getSaveFileName(self,'导出报告','gaussian-report.md','Markdown (*.md)')
        if path:Path(path).write_text(report(self.result,self.result.get('title','量子计算结果')),encoding='utf-8');atomic_json(Path(path).with_suffix('.json'),self.result);self.status.setText('Markdown 报告和 JSON 数据已导出。')

    def closeEvent(self,e):
        if self.threads:
            self.status.setText('正在完成网络或建模操作，请稍后再关闭。');e.ignore();return
        self.persist_session()
        if self.remote:self.remote.close()
        super().closeEvent(e)


def main():
    app=QApplication(sys.argv);app.setStyle('Fusion');app.setApplicationName('ChatDFT Studio');app.setOrganizationName('ChatDFT');app.setFont(QFont('Segoe UI',9))
    from .workbench import Window,icon
    app.setWindowIcon(icon('ballstick'))
    win=Window();win.show();return app.exec()
