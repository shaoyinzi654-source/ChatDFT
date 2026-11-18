from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QTableWidget,QTableWidgetItem,
    QPushButton,QDialogButtonBox,QSpinBox,QDoubleSpinBox,QComboBox,QLabel,QCheckBox,QColorDialog,QMessageBox,QHeaderView,QTabWidget,QWidget,QGroupBox,QRadioButton,QButtonGroup)
from .chemistry import from_xyz


class AtomEditor(QDialog):
    def __init__(self,structure,parent):
        super().__init__(parent);self.setWindowTitle('Atom Properties — '+structure.name);self.resize(700,540);self.structure=structure;self.result_structure=None
        v=QVBoxLayout(self);self.table=QTableWidget(len(structure.atoms),4);self.table.setHorizontalHeaderLabels(['Element','X / Å','Y / Å','Z / Å']);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.table.setSelectionBehavior(QTableWidget.SelectRows)
        for i,a in enumerate(structure.atoms):
            for j,val in enumerate([a['element'],a['x'],a['y'],a['z']]):self.table.setItem(i,j,QTableWidgetItem(str(val)))
        v.addWidget(self.table);row=QHBoxLayout();add=QPushButton('Add Atom');add.clicked.connect(self.add_atom);remove=QPushButton('Delete Selected');remove.clicked.connect(self.remove_atoms);row.addWidget(add);row.addWidget(remove);row.addStretch();v.addLayout(row)
        f=QFormLayout();self.charge=QSpinBox();self.charge.setRange(-20,20);self.charge.setValue(structure.charge);self.spin=QSpinBox();self.spin.setRange(1,20);self.spin.setValue(structure.multiplicity);f.addRow('Total charge:',self.charge);f.addRow('Spin multiplicity:',self.spin);v.addLayout(f)
        bb=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);bb.accepted.connect(self.accept);bb.rejected.connect(self.reject);v.addWidget(bb)
    def add_atom(self):
        i=self.table.rowCount();self.table.insertRow(i)
        for j,val in enumerate(['H',0,0,float(i+1)*1.5]):self.table.setItem(i,j,QTableWidgetItem(str(val)))
        self.table.selectRow(i)
    def remove_atoms(self):
        for i in sorted({x.row() for x in self.table.selectedItems()},reverse=True):self.table.removeRow(i)
    def accept(self):
        try:
            lines=[]
            for i in range(self.table.rowCount()):lines.append(' '.join(self.table.item(i,j).text().strip() if self.table.item(i,j) else '' for j in range(4)))
            xyz=str(len(lines))+'\nEdited geometry\n'+'\n'.join(lines)
            self.result_structure=from_xyz(xyz,self.structure.name,self.charge.value(),self.spin.value());self.result_structure.provenance='Edited atom coordinates / Å'
        except Exception as e:QMessageBox.warning(self,'Invalid structure',str(e));return
        super().accept()


class DisplayEditor(QDialog):
    def __init__(self,view,parent):
        super().__init__(parent);self.setWindowTitle('Display Style');self.resize(440,385);self.view=view;self.original={k:getattr(view,k) for k in ['style','labels','ball_scale','stick_scale','cpk_scale','line_width','show_bond_orders','show_hydrogens','light_background']}
        layout=QVBoxLayout(self)
        tabs=QTabWidget();layout.addWidget(tabs,1)
        page=QWidget();row=QHBoxLayout(page);tabs.addTab(page,'Atom')
        styles=QGroupBox('Display style');form=QFormLayout(styles);row.addWidget(styles,1)
        self.style=QButtonGroup(self);self.style_buttons={}
        self.ball=self.number(view.ball_scale,.2,3,.1)
        self.stick=self.number(view.stick_scale,.2,3,.1)
        self.cpk=self.number(view.cpk_scale,.2,2,.1)
        self.width=self.number(view.line_width,.5,5,.25)
        for title,key in [('Line','line'),('Stick','stick'),('Ball and stick','ballstick'),('CPK','cpk')]:
            radio=QRadioButton(title);self.style.addButton(radio);self.style_buttons[key]=radio;form.addRow(radio)
            if key=='line':form.addRow('Line width:',self.width)
            elif key=='stick':form.addRow('Stick scale:',self.stick)
            elif key=='ballstick':form.addRow('Ball scale:',self.ball)
            elif key=='cpk':form.addRow('CPK scale:',self.cpk)
        self.style_buttons[view.style].setChecked(True)
        options=QGroupBox('Visibility');right=QVBoxLayout(options);row.addWidget(options,1)
        self.order=QCheckBox('Bond order');self.order.setChecked(view.show_bond_orders);right.addWidget(self.order)
        self.labels=QCheckBox('Atom labels');self.labels.setChecked(view.labels);right.addWidget(self.labels)
        self.hydrogen=QCheckBox('Hydrogen atoms');self.hydrogen.setChecked(view.show_hydrogens);right.addWidget(self.hydrogen);right.addStretch()
        graphics=QWidget();form=QFormLayout(graphics);tabs.addTab(graphics,'Graphics')
        self.background=QComboBox();self.background.addItems(['Black — Optima','White']);self.background.setCurrentIndex(1 if view.light_background else 0);form.addRow('Background:',self.background)
        self.style.buttonClicked.connect(self.preview);self.background.currentIndexChanged.connect(self.preview)
        for number in [self.ball,self.stick,self.cpk,self.width]:number.valueChanged.connect(self.preview)
        for option in [self.order,self.labels,self.hydrogen]:option.toggled.connect(self.preview)
        footer=QHBoxLayout();footer.addWidget(QLabel(f'Editing: {len(view.structure.atoms)} atoms'));footer.addStretch();buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);footer.addWidget(buttons);layout.addLayout(footer)
    def number(self,value,low,high,step):
        w=QDoubleSpinBox();w.setRange(low,high);w.setSingleStep(step);w.setValue(value);return w
    def preview(self):
        v=self.view;v.style=next(key for key,radio in self.style_buttons.items() if radio.isChecked());v.ball_scale=self.ball.value();v.stick_scale=self.stick.value();v.cpk_scale=self.cpk.value();v.line_width=self.width.value();v.show_bond_orders=self.order.isChecked();v.labels=self.labels.isChecked();v.show_hydrogens=self.hydrogen.isChecked();v.light_background=self.background.currentIndex()==1;v.update();v.viewChanged.emit()
    def reject(self):
        for k,value in self.original.items():setattr(self.view,k,value)
        self.view.update();self.view.viewChanged.emit();super().reject()
