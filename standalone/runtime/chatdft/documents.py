from __future__ import annotations
from dataclasses import dataclass,field
from PySide6.QtCore import Qt,Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMdiArea,QMdiSubWindow


@dataclass(eq=False)
class ModelDocument:
    structure: object
    calculation: dict=field(default_factory=dict)
    undo: list=field(default_factory=list)
    redo: list=field(default_factory=list)
    modified: bool=False


class DocumentArea(QMdiArea):
    """Native child windows, with small document-access helpers shared by actions."""
    def __init__(self,parent=None):
        super().__init__(parent);self.setObjectName('documentArea')
        self.setBackground(__import__('PySide6.QtGui',fromlist=['QBrush']).QBrush(__import__('PySide6.QtGui',fromlist=['QColor']).QColor('#a8afb8')))
        self.setActivationOrder(QMdiArea.ActivationHistoryOrder)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded);self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    def addTab(self,widget,icon,title):
        sub=self.addSubWindow(widget);sub.setWindowTitle(title);sub.setWindowIcon(icon);sub.setAttribute(Qt.WA_DeleteOnClose,False);sub.resize(620,510);sub.show();self.setActiveSubWindow(sub)
        return self.indexOf(widget)
    def indexOf(self,widget):
        return next((i for i,w in enumerate(self.subWindowList()) if w.widget() is widget),-1)
    def setTabText(self,index,text):
        windows=self.subWindowList()
        if 0<=index<len(windows):windows[index].setWindowTitle(text)
    def setCurrentWidget(self,widget):
        for w in self.subWindowList():
            if w.widget() is widget:w.show();self.setActiveSubWindow(w);return
    def currentWidget(self):
        w=self.activeSubWindow();return w.widget() if w else None
    def setCurrentIndex(self,index):
        windows=self.subWindowList()
        if 0<=index<len(windows):windows[index].show();self.setActiveSubWindow(windows[index])
    def currentIndex(self):return self.indexOf(self.currentWidget())
    def count(self):return len(self.subWindowList())
