from __future__ import annotations
import copy
import json
from pathlib import Path
import math
from PySide6.QtCore import Qt,QSize,QTimer,QByteArray,QPointF
from PySide6.QtGui import QAction,QActionGroup,QIcon,QPixmap,QPainter,QPen,QColor,QFont,QKeySequence
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,
    QDockWidget,QTreeWidget,QTreeWidgetItem,QTableWidget,QTableWidgetItem,QHeaderView,QAbstractItemView,
    QTabWidget,QDialog,QDialogButtonBox,QLineEdit,QPlainTextEdit,QTextBrowser,QComboBox,QSpinBox,QCheckBox,
    QToolBar,QStyle,QProgressBar,QSplitter,QFileDialog,QMessageBox,QMenu,QInputDialog,QTreeWidgetItemIterator,QAbstractItemView)
from .app import ApplicationActions,button,label
from .storage import data_dir,atomic_json,load_settings
from .jobs import JobStore
from .chemistry import LIBRARY,Structure,from_smiles,validate_structure
from .gaussian import METHODS,BASES,SOLVENTS
from .structure_io import read_structure
from .widgets import PlotView
from .viewport import MoleculeView,measurement
from .documents import ModelDocument,DocumentArea
from .editors import AtomEditor,DisplayEditor
from rdkit import Chem


STYLE='''
QMainWindow,QDialog {background:#f0f0f0;}
QWidget {font-family:"Segoe UI","Microsoft YaHei UI";font-size:11px;color:#222;}
QMenuBar {background:#f6f6f6;border-bottom:1px solid #c8c8c8;padding:1px;}
QMenuBar::item {padding:4px 10px;}
QMenuBar::item:selected,QMenu::item:selected {background:#d9e9fb;color:#111;}
QMenu {background:#f7f7f7;border:1px solid #b0b0b0;padding:3px;}
QMenu::item {padding:5px 27px 5px 23px;}
QToolBar {background:#f0f0f0;border:0;border-bottom:1px solid #c6c6c6;spacing:2px;padding:2px;}
QToolBar::separator {background:#c1c1c1;width:1px;margin:3px 6px;}
QToolButton {border:1px solid transparent;padding:3px;border-radius:2px;}
QToolButton:hover,QToolButton:checked {background:#dce9fa;border:1px solid #95b5df;}
QDockWidget {font-weight:400;}
QDockWidget::title {background:#c0cdd9;border:1px solid #aebac5;padding:3px;text-align:left;}
QDockWidget::close-button,QDockWidget::float-button {padding:2px;}
QMainWindow::separator {background:#d6d8da;width:4px;height:4px;}
QMainWindow::separator:hover {background:#93afd2;}
QTreeWidget,QTableWidget,QTextBrowser,QPlainTextEdit {background:#fff;border:1px solid #c7c7c7;selection-background-color:#d5e6fb;selection-color:#111;border-radius:0;}
QTreeWidget::item {height:20px;}
QTreeWidget::item:selected {background:#d5e6fb;color:#111;}
QHeaderView::section {background:#eeeeee;padding:2px 5px;border:0;border-right:1px solid #ccc;border-bottom:1px solid #bbb;font-weight:400;}
QTableWidget {gridline-color:#e4e4e4;}
QTabWidget::pane {border:1px solid #bfc3c9;background:#fff;}
QTabBar::tab {background:#e5e6e8;border:1px solid #bfc3c9;padding:5px 16px;margin-right:1px;}
QTabBar::tab:selected {background:#fff;border-top:2px solid #4a79b8;padding-top:4px;}
QLineEdit,QComboBox,QSpinBox {background:#fff;border:1px solid #b6bbc1;border-radius:1px;padding:3px;min-height:19px;}
QLineEdit:focus,QComboBox:focus,QSpinBox:focus {border:1px solid #4a79b8;}
QPushButton {background:#f8f8f8;border:1px solid #aeb4bc;border-radius:2px;padding:5px 12px;min-height:17px;}
QPushButton:hover {background:#e5effc;border-color:#7c9fc9;}
QPushButton:disabled {color:#999;background:#ededed;border-color:#ccc;}
QPushButton#Primary {background:#e2ebf8;color:#1b3964;border:1px solid #88a6cc;}
QLabel#Muted {color:#60656b;font-size:11px;}
QLabel#Title {font-weight:600;}
QStatusBar {background:#f0f0f0;border-top:1px solid #c4c4c4;}
QStatusBar::item {border:0;border-right:1px solid #d2d2d2;}
QProgressBar {border:1px solid #b8c1cc;background:#fafafa;max-height:12px;}
QProgressBar::chunk {background:#6b96ca;}
QToolTip {background:#fffde4;color:#222;border:1px solid #777;padding:4px;}
'''


def icon(kind):
    pm=QPixmap(24,24);pm.fill(Qt.transparent);p=QPainter(pm);p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor('#3e526a'),1.5));p.setBrush(Qt.NoBrush)
    if kind=='new':
        p.setBrush(QColor('white'));p.drawRect(5,3,13,18);p.drawLine(8,8,15,8);p.drawLine(8,12,15,12)
    elif kind=='open':
        p.setBrush(QColor('#e7bc62'));p.drawRoundedRect(2,7,20,13,1,1);p.drawRect(4,4,7,5);p.drawLine(3,10,21,10)
    elif kind=='save':
        p.setBrush(QColor('#6793c8'));p.drawRect(3,3,18,18);p.setBrush(QColor('#fff'));p.drawRect(6,3,11,7);p.drawRect(7,14,10,7)
    elif kind in ['ballstick','cpk','stick','line','fragment']:
        p.setPen(QPen(QColor('#425c77'),3 if kind=='stick' else 1.5));p.drawLine(5,17,12,5);p.drawLine(12,5,20,16)
        for x,y,c in [(5,17,'#747474'),(12,5,'#e4544b'),(20,16,'#8c9ba6')]:
            p.setBrush(Qt.NoBrush if kind=='line' else QColor(c));r=5 if kind=='cpk' else 1.5 if kind in ['line','stick'] else 3;p.drawEllipse(QPointF(x,y),r,r)
    elif kind=='run':
        p.setPen(QPen(QColor('#387738'),1.5));p.setBrush(QColor('#65a35b'));p.drawPath(_triangle())
    elif kind=='stop':p.setBrush(QColor('#c9574a'));p.drawRect(6,6,12,12)
    elif kind in ['tile','cascade']:
        p.setBrush(QColor('#f6f8fc'))
        if kind=='tile':
            for x,y in [(2,2),(13,2),(2,13),(13,13)]:p.drawRect(x,y,9,9)
        else:
            for x,y in [(2,2),(6,6),(10,10)]:p.drawRect(x,y,12,11)
    elif kind=='bond':
        p.setPen(QPen(QColor('#485d73'),2));p.drawLine(3,9,21,9);p.drawLine(3,15,21,15)
    elif kind=='display':
        p.setBrush(QColor('#9fb4d2'));p.drawRect(2,3,20,15);p.drawLine(7,22,17,22);p.drawLine(12,18,12,22)
    elif kind=='fit':
        for x,y,sx,sy in [(3,3,1,1),(21,3,-1,1),(3,21,1,-1),(21,21,-1,-1)]:p.drawLine(x,y,x+sx*5,y);p.drawLine(x,y,x,y+sy*5)
        p.drawEllipse(QPointF(12,12),3,3)
    elif kind in ['rotate','undo','redo']:
        if kind=='redo':p.translate(24,0);p.scale(-1,1)
        p.drawArc(4,4,16,16,35*16,285*16);p.drawLine(20,7,20,2);p.drawLine(20,7,15,6)
    elif kind=='pan':
        p.drawLine(12,2,12,22);p.drawLine(2,12,22,12)
        for x,y,dx,dy in [(12,2,-3,3),(12,2,3,3),(22,12,-3,-3),(22,12,-3,3),(12,22,-3,-3),(12,22,3,-3),(2,12,3,-3),(2,12,3,3)]:p.drawLine(x,y,x+dx,y+dy)
    elif kind=='zoom':p.drawEllipse(3,3,12,12);p.drawLine(14,14,22,22);p.drawLine(6,9,12,9);p.drawLine(9,6,9,12)
    elif kind=='select':
        from PySide6.QtGui import QPolygonF
        p.setBrush(QColor('#fafafa'));p.drawPolygon(QPolygonF([QPointF(5,2),QPointF(5,20),QPointF(10,15),QPointF(15,21),QPointF(18,19),QPointF(13,13),QPointF(20,12)]))
    elif kind=='measure':
        p.setBrush(QColor('#eac96e'));p.drawRect(2,7,20,10)
        for x in range(5,22,4):p.drawLine(x,7,x,12)
    elif kind=='settings':
        for x,y in [(5,7),(12,15),(19,9)]:p.drawLine(x,3,x,21);p.setBrush(QColor('#8bb0d8'));p.drawRect(x-3,y-2,6,4)
    elif kind=='server':
        for y in [4,13]:p.setBrush(QColor('#b5c5d6'));p.drawRoundedRect(2,y,20,7,1,1);p.setBrush(QColor('#599361'));p.drawEllipse(QPointF(6,y+3.5),1.3,1.3);p.drawLine(11,y+3,19,y+3)
    elif kind=='labels':p.setFont(QFont('Segoe UI',14));p.drawText(3,19,'A₁')
    elif kind=='plot':
        p.drawLine(3,2,3,21);p.drawLine(3,21,22,21);p.setPen(QPen(QColor('#3f76bb'),2));p.drawLine(4,17,9,11);p.drawLine(9,11,14,15);p.drawLine(14,15,21,5)
    else:p.setFont(QFont('Segoe UI',11));p.drawText(3,18,{'ai':'AI','front':'Z','top':'Y','right':'X','edit':'✎'}.get(kind,'•'))
    p.end();return QIcon(pm)


def _triangle():
    from PySide6.QtGui import QPainterPath
    p=QPainterPath();p.moveTo(7,3);p.lineTo(21,12);p.lineTo(7,21);p.closeSubpath();return p


class Window(ApplicationActions):
    def __init__(self):
        QMainWindow.__init__(self);self.resize(1500,960);self.setMinimumSize(1050,700);self.setWindowTitle('ChatDFT Studio — 未命名项目')
        self.setStyleSheet(STYLE);self.setDockOptions(QMainWindow.AllowNestedDocks|QMainWindow.AllowTabbedDocks|QMainWindow.AnimatedDocks)
        self.setCorner(Qt.BottomLeftCorner,Qt.LeftDockWidgetArea)
        self.settings=load_settings();self.remote=None;self.threads=[];self.store=JobStore();self.messages=[];self.candidates=[];self.result=None;self.poll_busy=False;self.selected_job=None;self.last_job_signature=''
        self.undo_stack=[];self.redo_stack=[];self.history_replay=False;self._atom_sync=False
        self.documents=[];self.current_document=None;self.document_pages={};self._switching_document=False;self.new_document_pending=False;self.project_name='Molecular Study';self._loading_project=True
        self.make_workspace();self.make_jobs();self.make_results();self.make_project_docks();self.make_assistant();self.make_calculation_dialog();self.make_actions();self.make_menus();self.make_toolbars();self.make_statusbar()
        self.load_session();self.refresh_jobs();self.update_badges();self.assistant_dock.hide()
        self.output_tabs.setCurrentIndex(0);self.restore_project_session()
        self.resizeDocks([self.project_dock,self.properties_dock],[380,365],Qt.Vertical);self.resizeDocks([self.project_dock],[265],Qt.Horizontal);self.resizeDocks([self.output_dock],[142],Qt.Vertical)
        self.restore_layout()
        self.timer=QTimer(self);self.timer.timeout.connect(self.poll_jobs);self.timer.start(5000)

    def dock(self,title,name,widget,area):
        d=QDockWidget(title,self);d.setObjectName(name);d.setWidget(widget);d.setMinimumWidth(190);self.addDockWidget(area,d);return d

    def make_workspace(self):
        self.tabs=DocumentArea(self);self.setCentralWidget(self.tabs);self.tabs.subWindowActivated.connect(self.active_document_changed)
        self.viewer=MoleculeView();self.structure_name=QLabel('');self.structure_meta=QLabel('');self.structure_warning=QLabel('');self.structure_warning.hide();self.smiles=QLineEdit()
        self.style_combo=QComboBox()
        for text,data in [('Ball and stick','ballstick'),('Stick','stick'),('Line','line'),('CPK','cpk')]:self.style_combo.addItem(text,data)
        self.style_combo.currentIndexChanged.connect(lambda:self.viewer.set_style(self.style_combo.currentData()))

    def make_project_docks(self):
        page=QWidget();v=QVBoxLayout(page);v.setContentsMargins(0,0,0,0);v.setSpacing(0)
        bar=QToolBar();bar.setIconSize(QSize(16,16));bar.setMovable(False)
        bar.addAction(icon('new'),'New model',self.new_structure);bar.addAction(icon('open'),'Open project',self.open_project);bar.addAction(icon('save'),'Save project',self.save_project);v.addWidget(bar)
        self.project_search=QLineEdit();self.project_search.setPlaceholderText('Filter project files…');self.project_search.textChanged.connect(self.filter_project);v.addWidget(self.project_search)
        self.project_tree=QTreeWidget();self.project_tree.setHeaderHidden(True);self.project_tree.setIndentation(15);self.project_tree.itemDoubleClicked.connect(self.activate_project_item);self.project_tree.setContextMenuPolicy(Qt.CustomContextMenu);self.project_tree.customContextMenuRequested.connect(self.project_context_menu);v.addWidget(self.project_tree)
        self.project_dock=self.dock('Project','projectExplorer',page,Qt.LeftDockWidgetArea)
        pane=QWidget();l=QVBoxLayout(pane);l.setContentsMargins(0,0,0,0);l.setSpacing(0);filterrow=QHBoxLayout();filterrow.setContentsMargins(3,2,3,2);filterrow.addWidget(QLabel('Filter:'))
        self.property_filter=QComboBox();self.property_filter.addItems(['Physical System','Selection','Display']);self.property_filter.currentIndexChanged.connect(lambda:self.update_selection(self.viewer.picks));filterrow.addWidget(self.property_filter,1);l.addLayout(filterrow)
        self.properties=QTableWidget(0,2);self.properties.setHorizontalHeaderLabels(['Property','Value']);self.properties.verticalHeader().hide();self.properties.verticalHeader().setDefaultSectionSize(21);self.properties.setEditTriggers(QAbstractItemView.NoEditTriggers);self.properties.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents);self.properties.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch);self.properties.setSelectionBehavior(QAbstractItemView.SelectRows);self.properties.cellDoubleClicked.connect(self.edit_property);l.addWidget(self.properties)
        self.properties_dock=self.dock('Properties','propertiesExplorer',pane,Qt.LeftDockWidgetArea);self.splitDockWidget(self.project_dock,self.properties_dock,Qt.Vertical)

    def make_jobs(self):
        self.output_tabs=QTabWidget();self.output_tabs.setDocumentMode(True);self.output_tabs.setTabPosition(QTabWidget.South)
        jobs=QWidget();v=QVBoxLayout(jobs);v.setContentsMargins(3,3,3,3);v.setSpacing(3)
        self.jobs_table=QTableWidget(0,6);self.jobs_table.setHorizontalHeaderLabels(['任务 ID','模型','方法 / 基组','状态','位置','创建时间']);self.jobs_table.setSelectionBehavior(QAbstractItemView.SelectRows);self.jobs_table.setEditTriggers(QAbstractItemView.NoEditTriggers);self.jobs_table.verticalHeader().hide();self.jobs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.jobs_table.itemSelectionChanged.connect(self.select_job);self.jobs_table.itemDoubleClicked.connect(lambda:self.analyze_job());v.addWidget(self.jobs_table)
        row=QHBoxLayout();self.job_summary=QLabel('');row.addWidget(self.job_summary,1)
        for t,f in [('刷新',self.poll_jobs),('取消',self.cancel_job),('结果',self.analyze_job),('下载',self.download_job)]:row.addWidget(button(t,f))
        v.addLayout(row);self.output_tabs.addTab(jobs,'任务管理')
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setFont(QFont('Consolas',9));self.output_tabs.addTab(self.log,'输出日志')
        self.atom_table=QTableWidget(0,5);self.atom_table.setHorizontalHeaderLabels(['编号','元素','X / Å','Y / Å','Z / Å']);self.atom_table.verticalHeader().hide();self.atom_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.atom_table.setSelectionBehavior(QAbstractItemView.SelectRows);self.atom_table.setEditTriggers(QAbstractItemView.NoEditTriggers);self.atom_table.itemSelectionChanged.connect(self.atom_table_selected);self.atom_table.itemDoubleClicked.connect(lambda:self.edit_coordinates());self.output_tabs.addTab(self.atom_table,'原子坐标')
        self.output_dock=self.dock('Jobs','outputPanel',self.output_tabs,Qt.BottomDockWidgetArea);self.output_tabs.setCurrentIndex(0)

    def make_results(self):
        self.result_page=QWidget();v=QVBoxLayout(self.result_page);v.setContentsMargins(5,5,5,5);v.setSpacing(4)
        bar=QHBoxLayout()
        for t,f in [('导入日志',self.import_log),('载入最终结构',self.load_result_geometry),('AI 解读',self.explain_results),('导出报告',self.export_report)]:bar.addWidget(button(t,f))
        bar.addStretch();v.addLayout(bar);split=QSplitter(Qt.Vertical);self.result_text=QTextBrowser();self.result_text.setPlainText('尚未载入计算结果。请在任务管理器中打开已完成的任务。');split.addWidget(self.result_text)
        charts=QWidget();row=QHBoxLayout(charts);row.setContentsMargins(0,0,0,0);self.energy_plot=PlotView();self.freq_plot=PlotView();row.addWidget(self.energy_plot);row.addWidget(self.freq_plot);split.addWidget(charts);split.setSizes([450,230]);v.addWidget(split)

    def make_assistant(self):
        panel=QWidget();v=QVBoxLayout(panel);v.setContentsMargins(6,6,6,6);v.setSpacing(5)
        self.ai_badge=QLabel('Codex');v.addWidget(self.ai_badge);self.chat=QTextBrowser();self.chat.setOpenExternalLinks(False);v.addWidget(self.chat,1)
        self.prompt=QPlainTextEdit();self.prompt.setPlaceholderText('建模目标、结构修改或结果问题…');self.prompt.setFixedHeight(85);v.addWidget(self.prompt)
        self.send=button('发送',self.send_ai);v.addWidget(self.send);self.candidate_box=QComboBox();self.candidate_box.setPlaceholderText('候选结构');v.addWidget(self.candidate_box)
        self.apply_candidate_btn=button('载入候选结构',self.apply_candidate);self.apply_candidate_btn.setEnabled(False);v.addWidget(self.apply_candidate_btn)
        self.assistant_dock=self.dock('Modeling Assistant','aiAssistant',panel,Qt.RightDockWidgetArea);self.assistant_dock.setMinimumWidth(290)

    def make_calculation_dialog(self):
        self.calc_dialog=QDialog(self)
        self.calc_dialog.setWindowTitle('PySCF Calculation')
        self.calc_dialog.resize(405,355)
        layout=QVBoxLayout(self.calc_dialog)
        tabs=QTabWidget();layout.addWidget(tabs,1)
        def tab(name):
            page=QWidget();form=QFormLayout(page)
            form.setContentsMargins(12,13,12,12);form.setSpacing(9)
            tabs.addTab(page,name)
            return form
        setup=tab('Setup')
        self.task=QComboBox()
        for key,title in [('sp','Energy'),('opt','Geometry Optimization'),('optfreq','Optimization + Frequencies'),('freq','Frequencies'),('td','Excited States')]:
            self.task.addItem(title,key)
        self.task.setCurrentIndex(self.task.findData('optfreq'));setup.addRow('Task:',self.task)
        self.method=QComboBox();self.method.addItems([m for m in METHODS if m!='wB97XD']);setup.addRow('Method:',self.method)
        self.charge=QSpinBox();self.charge.setRange(-20,20);setup.addRow('Charge:',self.charge)
        self.multiplicity=QSpinBox();self.multiplicity.setRange(1,20);setup.addRow('Multiplicity:',self.multiplicity)
        electronic=tab('Electronic')
        self.basis=QComboBox();self.basis.addItems(BASES);electronic.addRow('Basis set:',self.basis)
        self.solvent=QComboBox();self.solvent.addItems(SOLVENTS);electronic.addRow('Solvent:',self.solvent)
        self.dispersion=QCheckBox()
        resources=tab('Job Control')
        self.cores=QSpinBox();self.cores.setRange(1,16);self.cores.setValue(8);resources.addRow('CPU cores:',self.cores)
        self.memory=QSpinBox();self.memory.setRange(1,14);self.memory.setValue(8);self.memory.setSuffix(' GB');resources.addRow('Memory:',self.memory)
        self.environment=QLabel('');self.environment.setWordWrap(True);resources.addRow('Gateway:',self.environment)
        resources.addRow(button('Connect…',self.connect_server))
        row=QHBoxLayout();row.addStretch();row.addWidget(button('Run',self.run_from_dialog));row.addWidget(button('Files…',self.preview_input));row.addWidget(button('Close',self.calc_dialog.accept));layout.addLayout(row)
        self.charge.valueChanged.connect(lambda:self.update_selection(self.viewer.picks))
        self.multiplicity.valueChanged.connect(lambda:self.update_selection(self.viewer.picks))

    def action(self,text,fn,kind=None,shortcut=None,check=False):
        a=QAction(icon(kind) if kind else QIcon(),text,self);a.setCheckable(check)
        if shortcut:a.setShortcut(shortcut)
        if fn:a.triggered.connect(fn)
        return a

    def make_actions(self):
        self.new_action=self.action('新建结构…',self.new_structure,'new','Ctrl+N');self.open_action=self.action('打开项目…',self.open_project,'open','Ctrl+O');self.save_action=self.action('保存项目…',self.save_project,'save','Ctrl+S')
        self.import_action=self.action('Import Structure (XSD / XYZ)…',self.import_structure,'open','Ctrl+I');self.export_action=self.action('Export XYZ…',self.export_xyz)
        self.undo_action=self.action('撤销',self.undo,'undo','Ctrl+Z');self.redo_action=self.action('重做',self.redo,'redo','Ctrl+Y')
        self.build_btn=self.action('SMILES 建模…',self.smiles_dialog,'ballstick');self.edit_action=self.action('编辑原子坐标…',self.edit_coordinates,'edit');self.fragment_action=self.action('插入分子片段…',self.add_fragment,'fragment')
        self.submit_btn=self.action('运行计算',self.submit_job,'run','F5');self.calc_action=self.action('计算设置…',self.show_calculation,'settings','F6');self.connect_action=self.action('连接服务器…',self.connect_server,'server');self.ai_action=self.action('AI 建模助手',self.show_assistant,'ai','F9')
        self.fit_action=self.action('Fit to View',lambda:self.viewer.fit(),'fit','F');self.reset_action=self.action('Reset View',lambda:self.viewer.reset_view(),'rotate');self.png_action=self.action('导出模型图片…',self.export_view)
        self.label_action=self.action('原子标签',lambda v:self.set_view_option('labels',v),'labels',check=True);self.axes_action=self.action('坐标轴',lambda v:self.set_view_option('axes',v),check=True);self.axes_action.setChecked(True)
        self.perspective_action=self.action('透视投影',lambda v:self.set_view_option('perspective',v),check=True);self.background_action=self.action('浅色背景',lambda v:self.set_view_option('light_background',v),check=True)
        self.mode_group=QActionGroup(self);self.mode_actions={}
        for key,text,shortcut in [('select','选择','1'),('rotate','旋转','2'),('pan','平移','3'),('zoom','缩放','4')]:
            a=self.action(text,lambda checked,k=key:self.viewer.set_mode(k),key,shortcut,True);self.mode_group.addAction(a);self.mode_actions[key]=a
        self.mode_actions['rotate'].setChecked(True);self.viewer.set_mode('rotate')
        self.style_actions={};self.style_group=QActionGroup(self)
        for key,text in [('ballstick','球棍'),('stick','棍状'),('line','线框'),('cpk','空间填充')]:
            a=self.action(text,lambda checked,k=key:self.viewer.set_style(k),key,check=True);self.style_group.addAction(a);self.style_actions[key]=a
        self.style_actions['ballstick'].setChecked(True)
        self.undo_action.setEnabled(False);self.redo_action.setEnabled(False)

    def make_menus(self):
        bar=self.menuBar();file=bar.addMenu('File')
        for a in [self.new_action,self.open_action,self.save_action]:file.addAction(a)
        file.addSeparator();file.addAction(self.import_action);file.addAction(self.export_action);file.addAction(self.png_action);file.addSeparator();file.addAction('Exit',self.close,'Alt+F4')
        edit=bar.addMenu('Edit');edit.addAction(self.undo_action);edit.addAction(self.redo_action);edit.addSeparator();edit.addAction('Duplicate Structure',self.duplicate_document);edit.addAction('Clear Selection',lambda:self.viewer.clear_selection(),'Esc')
        view=bar.addMenu('View');view.addAction(self.fit_action);view.addAction(self.reset_action);view.addAction('Display Style…',self.display_style,'Ctrl+D')
        style=view.addMenu('Display')
        for a in self.style_actions.values():style.addAction(a)
        for a in [self.label_action,self.axes_action,self.perspective_action,self.background_action]:view.addAction(a)
        orient=view.addMenu('Orientation')
        for t,k in [('Front','front'),('Right','right'),('Top','top'),('Isometric','iso')]:orient.addAction(t,lambda checked=False,key=k:self.viewer.orient(key))
        panels=view.addMenu('Explorers')
        for d in [self.project_dock,self.properties_dock,self.output_dock,self.assistant_dock]:panels.addAction(d.toggleViewAction())
        view.addAction('Reset Layout',self.reset_layout)
        modify=bar.addMenu('Modify');modify.addAction(self.edit_action);modify.addAction('Bond Order…',self.edit_bond);modify.addSeparator();modify.addAction('Translate Atoms…',lambda:self.transform_atoms(False));modify.addAction('Rotate Atoms…',lambda:self.transform_atoms(True));modify.addAction('Set Distance…',self.set_distance)
        build=bar.addMenu('Build');build.addAction(self.build_btn);build.addAction(self.fragment_action);build.addAction('Recalculate Bonds',self.recalculate_bonds);library=build.addMenu('Fragment Library')
        for name in LIBRARY:library.addAction(name,lambda checked=False,n=name:self.insert_library(n))
        tools=bar.addMenu('Tools')
        for t,n in [('Measure Distance',2),('Measure Angle',3),('Measure Torsion',4)]:tools.addAction(t,lambda checked=False,count=n:self.measure_mode(count))
        tools.addSeparator();tools.addAction(self.ai_action);tools.addAction('Model and Server Settings…',self.open_settings)
        statistics=bar.addMenu('Statistics');statistics.addAction('Element Counts…',self.element_statistics)
        modules=bar.addMenu('Modules');pyscf=modules.addMenu('PySCF')
        for a in [self.calc_action,self.submit_btn,self.connect_action]:pyscf.addAction(a)
        modules.addAction('Job Explorer',self.show_jobs);modules.addAction('Open Results',self.analyze_job);modules.addAction('Import Gaussian Output…',self.import_log)
        self.window_menu=bar.addMenu('Window');self.window_menu.aboutToShow.connect(self.rebuild_window_menu)
        helpmenu=bar.addMenu('Help');helpmenu.addAction('操作说明 / Keyboard Shortcuts',self.show_help);helpmenu.addAction('About ChatDFT Studio',lambda:QMessageBox.about(self,'ChatDFT Studio','ChatDFT Studio 3.0\n\nMolecular Modeling Workspace\nQuantum Engine: PySCF\nStructure Toolkit: RDKit'))

    def make_toolbars(self):
        standard=self.addToolBar('Standard');standard.setObjectName('standardToolbar');standard.setIconSize(QSize(18,18))
        for a in [self.new_action,self.open_action,self.save_action]:standard.addAction(a)
        standard.addSeparator();standard.addAction(self.undo_action);standard.addAction(self.redo_action);standard.addSeparator()
        for a in self.mode_actions.values():standard.addAction(a)
        standard.addSeparator();standard.addAction(self.fit_action);standard.addAction(self.reset_action)
        for t,k in [('Front','front'),('Right','right'),('Top','top')]:standard.addAction(self.action(t,lambda checked=False,key=k:self.viewer.orient(key),k))
        standard.addSeparator();standard.addAction(self.action('Tile Windows',self.tabs.tileSubWindows,'tile'));standard.addAction(self.action('Cascade Windows',self.tabs.cascadeSubWindows,'cascade'))
        self.addToolBarBreak();model=self.addToolBar('Modeling');model.setObjectName('viewToolbar');model.setIconSize(QSize(18,18))
        for a in [self.build_btn,self.fragment_action,self.edit_action]:model.addAction(a)
        model.addAction(self.action('Bond Order',self.edit_bond,'bond'));model.addSeparator()
        for a in self.style_actions.values():model.addAction(a)
        model.addAction(self.action('Display Style',self.display_style,'display'));model.addAction(self.label_action);model.addSeparator()
        for name,n in [('Distance',2),('Angle',3),('Torsion',4)]:model.addAction(self.action(name,lambda checked=False,count=n:self.measure_mode(count),'measure'))
        model.addSeparator();model.addAction(self.calc_action);model.addAction(self.submit_btn);model.addAction(self.connect_action);model.addSeparator();model.addAction(self.ai_action)
        self.library=QComboBox();self.library.addItems(LIBRARY);self.library.hide()
        sketch=QToolBar('Manipulation',self);sketch.setObjectName('manipulationToolbar');sketch.setIconSize(QSize(18,18));sketch.setOrientation(Qt.Vertical);self.addToolBar(Qt.LeftToolBarArea,sketch)
        for a in self.mode_actions.values():sketch.addAction(a)
        sketch.addSeparator();sketch.addAction(self.action('Translate Atoms',lambda:self.transform_atoms(False),'pan'));sketch.addAction(self.action('Rotate Atoms',lambda:self.transform_atoms(True),'rotate'));sketch.addSeparator();sketch.addAction(self.edit_action);sketch.addAction(self.fragment_action);sketch.addAction(self.action('Bond Order',self.edit_bond,'bond'));sketch.addSeparator();sketch.addAction(self.fit_action);sketch.addAction(self.label_action)

    def make_statusbar(self):
        sb=self.statusBar();self.status=QLabel('就绪');self.status.setMinimumWidth(200);sb.addWidget(self.status,1);self.progress=QProgressBar();self.progress.setRange(0,1);self.progress.setValue(0);self.progress.setFixedSize(95,12);self.progress.setTextVisible(False);sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.structure_meta);self.engine_badge=QLabel('PySCF / 未连接');sb.addPermanentWidget(self.engine_badge);sb.addPermanentWidget(QLabel('Å'))

    def create_document(self,s,calculation=None):
        document=ModelDocument(copy.deepcopy(s),calculation or self.calculation().to_dict());self.documents.append(document)
        view=MoleculeView();view.set_structure(document.structure);view.selected.connect(lambda t:self.status.setText(t));view.selectionChanged.connect(self.update_selection);view.viewChanged.connect(self.sync_view_controls)
        self.document_pages[view]=document
        self._switching_document=True
        try:self.tabs.addTab(view,icon('ballstick'),s.name+'.xyz')
        finally:self._switching_document=False
        self.switch_document(view);return document

    def stash_document(self):
        if self.current_document and hasattr(self,'structure'):
            self.current_document.structure=copy.deepcopy(self.structure);self.current_document.structure.charge=self.charge.value();self.current_document.structure.multiplicity=self.multiplicity.value();self.current_document.calculation=self.calculation().to_dict();self.current_document.undo=self.undo_stack;self.current_document.redo=self.redo_stack

    def active_document_changed(self,sub):
        if self._switching_document or sub is None:return
        page=sub.widget()
        if page in self.document_pages:self.switch_document(page)

    def switch_document(self,page):
        if page not in self.document_pages:return
        if self.current_document is self.document_pages[page] and self.viewer is page:return
        self.stash_document();self.current_document=self.document_pages[page];self.viewer=page;self.undo_stack=self.current_document.undo;self.redo_stack=self.current_document.redo
        camera=(page.yaw,page.pitch,page.zoom,copy.copy(page.pan))
        self.history_replay=True
        try:
            self.apply_calc(self.current_document.calculation)
            self.cores.setValue(self.current_document.calculation.get('cores',8));self.memory.setValue(self.current_document.calculation.get('memory_gb',8))
            ApplicationActions.set_structure(self,copy.deepcopy(self.current_document.structure))
            page.yaw,page.pitch,page.zoom,page.pan=camera;page.update()
        finally:self.history_replay=False
        self.undo_action.setEnabled(bool(self.undo_stack));self.redo_action.setEnabled(bool(self.redo_stack));self.sync_view_controls()

    def set_structure(self,s):
        if self.current_document is None:self.create_document(s);return
        if hasattr(self,'structure') and not self.history_replay:
            self.undo_stack.append(copy.deepcopy(self.structure));self.undo_stack=self.undo_stack[-30:];self.redo_stack=[];self.current_document.modified=True
        self.current_document.structure=copy.deepcopy(s);ApplicationActions.set_structure(self,s)
        self.undo_action.setEnabled(bool(self.undo_stack));self.redo_action.setEnabled(bool(self.redo_stack))

    def sync_document_title(self):
        if self.current_document:
            for sub in self.tabs.subWindowList():
                if self.document_pages.get(sub.widget()) is self.current_document:sub.setWindowTitle(self.structure.name+'.xyz'+(' *' if self.current_document.modified else ''))

    def activate_model(self):
        self.tabs.setCurrentWidget(self.viewer)

    def duplicate_document(self):
        s=self.current_structure();s.name+=' — Copy';self.create_document(s)

    def import_structure(self):
        filename,_=QFileDialog.getOpenFileName(self,'Import Structure','','Molecular structures (*.xsd *.xyz);;Materials Studio (*.xsd);;XYZ (*.xyz)')
        if not filename:return
        fields=QDialog(self);fields.setWindowTitle('Import Structure — Charge and Spin')
        form=QFormLayout(fields)
        charge=QSpinBox();charge.setRange(-20,20);form.addRow('Total charge:',charge)
        spin=QSpinBox();spin.setRange(1,20);form.addRow('Spin multiplicity:',spin)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(fields.accept);buttons.rejected.connect(fields.reject);form.addRow(buttons)
        if fields.exec()!=QDialog.Accepted:return
        try:
            self.create_document(read_structure(Path(filename),charge.value(),spin.value()))
            self.tabs.tileSubWindows()
        except Exception as exc:self.error(str(exc))

    def load_library(self):
        name=self.library.currentText();self.background(lambda:from_smiles(LIBRARY[name],name),self.create_document,buttons=(self.build_btn,))

    def project_payload(self):
        self.stash_document()
        return dict(version=3,name=self.project_name,documents=[dict(structure=d.structure.to_dict(),calculation=d.calculation) for d in self.documents],active=self.documents.index(self.current_document) if self.current_document in self.documents else 0,messages=self.messages)

    def persist_session(self):
        ApplicationActions.persist_session(self)
        if self.documents and not self._loading_project:atomic_json(data_dir()/'project-session-v3.json',self.project_payload())

    def replace_project(self,payload):
        entries=payload.get('documents') or [dict(structure=payload['structure'],calculation=payload.get('calculation',{}))]
        documents=[(Structure.from_dict(entry['structure']),entry.get('calculation',{})) for entry in entries]
        for structure,configuration in documents:
            validate_structure(structure)
            if not isinstance(configuration,dict):raise ValueError('Invalid calculation settings in project.')
        self._loading_project=True;self._switching_document=True
        try:
            self.current_document=None;self.documents=[];self.document_pages={}
            for sub in self.tabs.subWindowList():self.tabs.removeSubWindow(sub.widget());sub.hide();sub.deleteLater()
            self.project_name=payload.get('name','Molecular Study');self.messages=payload.get('messages',[])
            for structure,configuration in documents:self.create_document(structure,configuration)
            active=max(0,min(len(self.documents)-1,payload.get('active',0)))
            for page,doc in self.document_pages.items():
                if doc is self.documents[active]:self.tabs.setCurrentWidget(page);self.switch_document(page);break
        finally:self._loading_project=False;self._switching_document=False
        self.render_chat();self.update_project_tree();self.persist_session();QTimer.singleShot(0,self.tabs.tileSubWindows)

    def restore_project_session(self):
        path=data_dir()/'project-session-v3.json'
        try:
            payload=json.loads(path.read_text('utf-8'))
            if payload.get('documents'):self.replace_project(payload);return
        except (OSError,ValueError,KeyError):pass
        # A real second structure provides an immediately usable comparison document.
        if len(self.documents)==1:
            original=self.viewer;self.create_document(from_smiles(LIBRARY['咖啡因 · Caffeine'],'Caffeine'))
            self.tabs.setCurrentWidget(original);self.switch_document(original)
        self._loading_project=False;self.persist_session();QTimer.singleShot(0,self.tabs.tileSubWindows)

    def save_project(self):
        path,_=QFileDialog.getSaveFileName(self,'Save Project',self.project_name+'.chatdft.json','ChatDFT project (*.chatdft.json)')
        if path:
            self.project_name=Path(path).name.replace('.chatdft.json','');atomic_json(path,self.project_payload());self.status.setText('Project saved: '+path)
            for d in self.documents:d.modified=False
            self.sync_document_title();self.update_project_tree()

    def open_project(self):
        path,_=QFileDialog.getOpenFileName(self,'Open Project','','ChatDFT project (*.chatdft.json)')
        if not path:return
        try:
            atomic_json(data_dir()/'previous-project.json',self.project_payload());payload=json.loads(Path(path).read_text('utf-8'));payload.setdefault('name',Path(path).stem);self.replace_project(payload)
        except Exception as e:self.error(str(e))

    def update_model_panels(self):
        if not hasattr(self,'structure'):return
        s=self.structure;self.sync_document_title();self.setWindowTitle(self.project_name+' — ChatDFT Studio')
        self.atom_table.blockSignals(True);self.atom_table.setRowCount(len(s.atoms))
        for i,a in enumerate(s.atoms):
            for j,value in enumerate([i+1,a['element'],f"{a['x']:.6f}",f"{a['y']:.6f}",f"{a['z']:.6f}"]):self.atom_table.setItem(i,j,QTableWidgetItem(str(value)))
        self.atom_table.blockSignals(False);self.update_selection([]);self.update_project_tree()
        if s.smiles in LIBRARY.values():self.library.setCurrentText(next(k for k,v in LIBRARY.items() if v==s.smiles))

    def update_project_tree(self):
        if not hasattr(self,'project_tree') or not hasattr(self,'structure'):return
        tree=self.project_tree
        tree_signature=(self.project_name,tuple(d.structure.name for d in self.documents),
                        tuple((j['id'],j['status']) for j in getattr(self,'job_values',[])),bool(self.result))
        if tree_signature==getattr(self,'_tree_signature',None):return
        self._tree_signature=tree_signature
        tree.clear();root=QTreeWidgetItem(tree,[self.project_name]);root.setIcon(0,icon('open'));root.setExpanded(True)
        structures=QTreeWidgetItem(root,['Structures']);structures.setIcon(0,icon('open'));structures.setExpanded(True)
        for index,doc in enumerate(self.documents):
            item=QTreeWidgetItem(structures,[doc.structure.name+'.xyz']);item.setIcon(0,icon('ballstick'));item.setData(0,Qt.UserRole,dict(kind='model',index=index));item.setExpanded(True)
            geom=QTreeWidgetItem(item,['Atomic coordinates']);geom.setData(0,Qt.UserRole,dict(kind='coordinates',index=index))
            setup=QTreeWidgetItem(item,['PySCF calculation']);setup.setIcon(0,icon('settings'));setup.setData(0,Qt.UserRole,dict(kind='settings',index=index))
        calc=QTreeWidgetItem(root,['Calculations']);calc.setIcon(0,icon('open'));calc.setExpanded(True)
        for job in getattr(self,'job_values',[]):
            item=QTreeWidgetItem(calc,[job['name']+' / '+job['id'][:6]]);item.setIcon(0,icon('plot'));item.setData(0,Qt.UserRole,job['id']);item.setToolTip(0,job['status'])
        if self.result:
            result=QTreeWidgetItem(root,['Results']);result.setIcon(0,icon('plot'));result.setData(0,Qt.UserRole,'result')
        self.filter_project(self.project_search.text())

    def filter_project(self,text):
        if not hasattr(self,'project_tree'):return
        text=text.casefold()
        def visit(item):
            children=[visit(item.child(i)) for i in range(item.childCount())];visible=not text or text in item.text(0).casefold() or any(children);item.setHidden(not visible);return visible
        for i in range(self.project_tree.topLevelItemCount()):visit(self.project_tree.topLevelItem(i))

    def project_context_menu(self,pos):
        item=self.project_tree.itemAt(pos);menu=QMenu(self)
        if item:menu.addAction('Open',lambda:self.activate_project_item(item,0))
        menu.addAction('New Molecule…',self.new_structure);menu.addAction('Duplicate Active Structure',self.duplicate_document);menu.addSeparator();menu.addAction('Save Project…',self.save_project);menu.exec(self.project_tree.viewport().mapToGlobal(pos))

    def update_selection(self,indices):
        if not hasattr(self,'properties') or not hasattr(self,'structure'):return
        s=self.structure;pt=Chem.GetPeriodicTable();masses=[pt.GetAtomicWeight(a['element']) for a in s.atoms];mass=sum(masses)
        center=[sum(a[k] for a in s.atoms)/len(s.atoms) for k in ['x','y','z']];com=[sum(a[k]*m for a,m in zip(s.atoms,masses))/mass for k in ['x','y','z']]
        rows=[('Name',s.name),('Chemical Formula',s.formula()),('Net Charge',str(self.charge.value())),('Spin Multiplicity',str(self.multiplicity.value())),('Net Mass / u',f'{mass:.5f}'),('Number of Atoms',str(len(s.atoms))),('Number of Bonds',str(len(s.bonds))),('Geometry Center',', '.join(f'{x:.4f}' for x in center)),('Mass Center',', '.join(f'{x:.4f}' for x in com)),('Coordinate Units','Å')]
        for axis,k in enumerate(['x','y','z']):rows.append(('Extent '+k.upper()+' / Å',f"{max(a[k] for a in s.atoms)-min(a[k] for a in s.atoms):.4f}"))
        if self.property_filter.currentText()=='Selection' or indices:
            selection=[('Selected Atoms',', '.join(str(i+1) for i in indices) or 'None')]
            if len(indices)==1:
                i=indices[0];a=s.atoms[i];selection += [('Element',a['element']),('Atomic Number',str(pt.GetAtomicNumber(a['element']))),('Atomic Mass',f'{masses[i]:.5f}'),('X / Å',f"{a['x']:.6f}"),('Y / Å',f"{a['y']:.6f}"),('Z / Å',f"{a['z']:.6f}")]
            if 2<=len(indices)<=4:selection.append(('Measurement',measurement(s.atoms,indices)))
            rows=selection if self.property_filter.currentText()=='Selection' else rows+selection
        if self.property_filter.currentText()=='Display':rows=[('Display Style',self.viewer.style),('Ball Scale',str(self.viewer.ball_scale)),('Stick Scale',str(self.viewer.stick_scale)),('CPK Scale',str(self.viewer.cpk_scale)),('Atom Labels',str(self.viewer.labels)),('Show Hydrogens',str(self.viewer.show_hydrogens)),('Projection','Perspective' if self.viewer.perspective else 'Orthographic')]
        rows.append(('Source',s.provenance));self.properties.setRowCount(len(rows))
        for i,(k,v) in enumerate(rows):self.properties.setItem(i,0,QTableWidgetItem(k));self.properties.setItem(i,1,QTableWidgetItem(v));self.properties.item(i,1).setToolTip(v)
        if not self._atom_sync:
            self._atom_sync=True;self.atom_table.clearSelection()
            for i in indices:
                for j in range(5):self.atom_table.item(i,j).setSelected(True)
            self._atom_sync=False

    def atom_table_selected(self):
        if self._atom_sync:return
        self._atom_sync=True;self.viewer.select_indices(sorted({item.row() for item in self.atom_table.selectedItems()}));self._atom_sync=False
    def activate_project_item(self,item,column):
        target=item.data(0,Qt.UserRole)
        if isinstance(target,dict):
            doc=self.documents[target['index']]
            for page,value in self.document_pages.items():
                if value is doc:self.tabs.setCurrentWidget(page);self.switch_document(page);break
            if target['kind']=='coordinates':self.output_dock.show();self.output_tabs.setCurrentIndex(2)
            elif target['kind']=='settings':self.show_calculation()
        elif target=='result':self.show_results()
        elif target:
            for i,j in enumerate(getattr(self,'job_values',[])):
                if j['id']==target:self.jobs_table.selectRow(i);self.show_jobs();break

    def sync_view_controls(self):
        if not hasattr(self,'style_actions'):return
        self.style_combo.blockSignals(True);self.style_combo.setCurrentIndex(self.style_combo.findData(self.viewer.style));self.style_combo.blockSignals(False);self.style_actions[self.viewer.style].setChecked(True)
        for action,state in [(self.label_action,self.viewer.labels),(self.axes_action,self.viewer.axes),
                             (self.perspective_action,self.viewer.perspective),(self.background_action,self.viewer.light_background)]:
            action.setChecked(state)
    def set_view_option(self,key,value):setattr(self.viewer,key,value);self.viewer.update()
    def measure_mode(self,count):self.viewer.clear_selection();self.viewer.measure_count=count;self.viewer.set_mode('measure');self.status.setText(f'依次选择 {count} 个原子进行测量。')
    def show_jobs(self):self.output_dock.show();self.output_tabs.setCurrentIndex(0);self.output_dock.raise_()
    def show_results(self):
        if self.tabs.indexOf(self.result_page)<0:self.tabs.addTab(self.result_page,icon('plot'),'计算结果')
        self.tabs.setCurrentWidget(self.result_page);self.update_project_tree()
    def show_assistant(self):self.assistant_dock.show();self.assistant_dock.raise_();self.resizeDocks([self.assistant_dock],[320],Qt.Horizontal);self.prompt.setFocus()
    def show_calculation(self):
        self.calc_dialog.setWindowModality(Qt.WindowModal)
        self.calc_dialog.show();self.calc_dialog.raise_();self.calc_dialog.activateWindow()
    def display_style(self):DisplayEditor(self.viewer,self).exec()
    def edit_property(self,row,column):
        if self.property_filter.currentText()=='Display':self.display_style()
        elif self.properties.item(row,0).text() in ['Net Charge','Spin Multiplicity']:self.show_calculation()
        else:self.edit_coordinates()
    def edit_coordinates(self):
        structure=copy.deepcopy(self.structure);structure.charge=self.charge.value();structure.multiplicity=self.multiplicity.value();dialog=AtomEditor(structure,self)
        for i in self.viewer.picks:dialog.table.selectRow(i)
        if dialog.exec()==QDialog.Accepted:self.set_structure(dialog.result_structure)
    def edit_bond(self):
        if len(self.viewer.picks)!=2:self.status.setText('Select two atoms using Ctrl+click, then choose Bond Order.');return
        i,j=self.viewer.picks;items=['None','Single','Aromatic','Double','Triple'];text,ok=QInputDialog.getItem(self,'Bond Order',f'{i+1} — {j+1}',items,1,False)
        if not ok:return
        structure=copy.deepcopy(self.structure);structure.bonds=[b for b in structure.bonds if set(b[:2])!={i,j}]
        order={'None':0,'Single':1,'Aromatic':1.5,'Double':2,'Triple':3}[text]
        if order:structure.bonds.append([i,j,order])
        structure.smiles='';structure.provenance='User-edited bonding / Cartesian geometry';self.set_structure(structure)
    def recalculate_bonds(self):
        from .chemistry import from_xyz
        try:
            s=from_xyz(self.structure.xyz(),self.structure.name,self.charge.value(),self.multiplicity.value());s.provenance='Distance-based connectivity / bond orders require review';self.set_structure(s)
        except Exception as e:self.error(str(e))
    def transform_atoms(self,rotate):
        d=QDialog(self);d.setWindowTitle('Rotate Atoms' if rotate else 'Translate Atoms');v=QVBoxLayout(d);f=QFormLayout();v.addLayout(f);fields=[]
        from PySide6.QtWidgets import QDoubleSpinBox
        for axis in ['X','Y','Z']:
            w=QDoubleSpinBox();w.setRange(-360 if rotate else -100,360 if rotate else 100);w.setDecimals(4);w.setSuffix(' °' if rotate else ' Å');f.addRow(axis,w);fields.append(w)
        selected=QCheckBox('Selected atoms only');selected.setChecked(bool(self.viewer.picks));selected.setEnabled(bool(self.viewer.picks));v.addWidget(selected);bb=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);v.addWidget(bb);bb.accepted.connect(d.accept);bb.rejected.connect(d.reject)
        if d.exec()!=QDialog.Accepted:return
        ids=self.viewer.picks if selected.isChecked() else list(range(len(self.structure.atoms)));s=copy.deepcopy(self.structure);values=[w.value() for w in fields]
        center=[sum(s.atoms[i][k] for i in ids)/len(ids) for k in ['x','y','z']]
        for i in ids:
            a=s.atoms[i]
            if rotate:
                x,y,z=[a[k]-center[n] for n,k in enumerate(['x','y','z'])];rx,ry,rz=map(math.radians,values);y,z=y*math.cos(rx)-z*math.sin(rx),y*math.sin(rx)+z*math.cos(rx);x,z=x*math.cos(ry)+z*math.sin(ry),-x*math.sin(ry)+z*math.cos(ry);x,y=x*math.cos(rz)-y*math.sin(rz),x*math.sin(rz)+y*math.cos(rz)
                for n,(k,value) in enumerate(zip(['x','y','z'],[x,y,z])):a[k]=value+center[n]
            else:
                for k,value in zip(['x','y','z'],values):a[k]+=value
        s.smiles='';self.set_structure(s)
    def set_distance(self):
        if len(self.viewer.picks)!=2:self.status.setText('Select two atoms using Ctrl+click to set their distance.');return
        i,j=self.viewer.picks;s=copy.deepcopy(self.structure);a,b=s.atoms[i],s.atoms[j];diff=[b[k]-a[k] for k in ['x','y','z']];length=math.sqrt(sum(x*x for x in diff))
        if length<1e-10:return
        value,ok=QInputDialog.getDouble(self,'Set Distance',f'Move atom {j+1}; distance / Å:',length,.4,100,4)
        if ok:
            for k,delta in zip(['x','y','z'],diff):b[k]=a[k]+delta*value/length
            s.smiles='';self.set_structure(s)
    def element_statistics(self):
        from collections import Counter
        counts=Counter(a['element'] for a in self.structure.atoms);table=QTableWidget(len(counts),3);table.setHorizontalHeaderLabels(['Element','Number of Atoms','Atomic Fraction']);table.setEditTriggers(QTableWidget.NoEditTriggers);table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row,(element,count) in enumerate(sorted(counts.items())):
            for col,value in enumerate([element,str(count),f'{count/len(self.structure.atoms):.6f}']):table.setItem(row,col,QTableWidgetItem(value))
        self.tabs.addTab(table,icon('plot'),self.structure.name+' — Element Counts')
    def rebuild_window_menu(self):
        m=self.window_menu;m.clear();m.addAction('Tile',self.tabs.tileSubWindows);m.addAction('Cascade',self.tabs.cascadeSubWindows);m.addAction('Duplicate Structure',self.duplicate_document);m.addSeparator()
        for sub in self.tabs.subWindowList():m.addAction(sub.windowTitle(),lambda checked=False,w=sub:(w.show(),self.tabs.setActiveSubWindow(w)))
    def run_from_dialog(self):self.calc_dialog.accept();self.submit_btn.trigger()
    def insert_library(self,name):self.library.setCurrentText(name);self.load_library()
    def new_structure(self):self.smiles_dialog()
    def smiles_dialog(self):
        d=QDialog(self);d.setWindowTitle('由 SMILES 建立三维结构');d.resize(490,150);v=QVBoxLayout(d);v.addWidget(QLabel('SMILES:'));edit=QLineEdit(self.smiles.text());v.addWidget(edit);bb=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);v.addWidget(bb);bb.accepted.connect(d.accept);bb.rejected.connect(d.reject)
        if d.exec()==QDialog.Accepted and edit.text().strip():
            smiles=edit.text().strip()
            self.background(lambda:from_smiles(smiles,'Molecule'),self.create_document,buttons=(self.build_btn,))
    def undo(self):
        if not self.undo_stack:return
        self.redo_stack.append(copy.deepcopy(self.structure));s=self.undo_stack.pop();self.history_replay=True
        try:self.set_structure(s)
        finally:self.history_replay=False
        self.redo_action.setEnabled(True);self.undo_action.setEnabled(bool(self.undo_stack))
    def redo(self):
        if not self.redo_stack:return
        self.undo_stack.append(copy.deepcopy(self.structure));s=self.redo_stack.pop();self.history_replay=True
        try:self.set_structure(s)
        finally:self.history_replay=False
        self.undo_action.setEnabled(True);self.redo_action.setEnabled(bool(self.redo_stack))
    def export_view(self):
        path,_=QFileDialog.getSaveFileName(self,'导出模型图片','structure.png','PNG (*.png)')
        if path:self.viewer.grab().save(path);self.status.setText('模型图片已保存。')
    def reset_layout(self):
        self.addDockWidget(Qt.LeftDockWidgetArea,self.project_dock);self.addDockWidget(Qt.LeftDockWidgetArea,self.properties_dock);self.splitDockWidget(self.project_dock,self.properties_dock,Qt.Vertical);self.addDockWidget(Qt.BottomDockWidgetArea,self.output_dock);self.addDockWidget(Qt.RightDockWidgetArea,self.assistant_dock)
        for d in [self.project_dock,self.properties_dock,self.output_dock]:d.setFloating(False);d.show()
        self.assistant_dock.hide();self.resizeDocks([self.project_dock],[265],Qt.Horizontal);self.resizeDocks([self.output_dock],[142],Qt.Vertical)
    def restore_layout(self):
        try:
            saved=json.loads((data_dir()/'workbench-layout-v3.json').read_text('utf-8'))
            self.restoreState(QByteArray.fromBase64(saved['state'].encode('ascii')),1)
        except (OSError,ValueError,KeyError):pass
    def closeEvent(self,event):
        if not self.threads:atomic_json(data_dir()/'workbench-layout-v3.json',dict(state=bytes(self.saveState(1).toBase64()).decode('ascii')))
        super().closeEvent(event)
    def show_help(self):QMessageBox.information(self,'操作说明','1 / 2 / 3 / 4：选择、旋转、平移、缩放\nF：适合窗口　Esc：清除选择\nCtrl+单击：选择多个原子\n中键或 Shift+拖动：平移\nF5：运行计算　F6：计算设置　F9：AI 建模助手\n\n面板可以拖动、停靠或关闭。通过 View → Explorers 重新打开。')
