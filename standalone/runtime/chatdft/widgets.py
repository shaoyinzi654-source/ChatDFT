from __future__ import annotations
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor,QPainter,QPen,QFont,QPainterPath
from PySide6.QtWidgets import QWidget
from .viewport import MoleculeView


class PlotView(QWidget):
    def __init__(self):
        super().__init__();self.series=[];self.caption='等待真实日志数据';self.setMinimumHeight(180)

    def set_data(self,values,caption):
        self.series=values;self.caption=caption;self.update()

    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(),QColor('#ffffff'))
        p.setPen(QColor('#647887'));p.drawText(16,25,self.caption)
        if not self.series:
            return
        values=self.series;lo=min(values);hi=max(values)
        span=max(hi-lo,1e-6);w=self.width()-95;h=self.height()-70
        p.setPen(QPen(QColor('#e4eaee'),1))
        for fraction in [0,.5,1]:
            y=45+h*fraction
            p.drawLine(70,int(y),self.width()-20,int(y))
            p.setPen(QColor('#8495a1'));p.drawText(5,int(y+4),f'{hi-span*fraction:.4g}');p.setPen(QColor('#e4eaee'))
        path=QPainterPath()
        for i,value in enumerate(values):
            point=QPointF(70+w*i/max(1,len(values)-1),45+h*(hi-value)/span)
            if i==0:path.moveTo(point)
            else:path.lineTo(point)
        p.setPen(QPen(QColor('#bd8249'),2.2));p.drawPath(path)
        if len(values)<=80:
            p.setBrush(QColor('#bd8249'))
            for i,value in enumerate(values):
                p.drawEllipse(QPointF(70+w*i/max(1,len(values)-1),45+h*(hi-value)/span),2.5,2.5)
        p.setPen(QColor('#8495a1'));p.drawText(self.width()-100,self.height()-8,f'{len(values)} 个数据点')
