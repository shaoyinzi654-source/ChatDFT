from __future__ import annotations
import math
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from PySide6.QtGui import QColor,QPainter,QPen,QBrush,QRadialGradient,QLinearGradient,QFont
from PySide6.QtWidgets import QWidget,QMenu,QApplication

COLORS={'H':'#eeeeee','C':'#8b8b8b','N':'#365cf1','O':'#e82a2a','F':'#83d96a','Cl':'#36c132','Br':'#a64d2d','S':'#ebd32f','P':'#eaa13c','Fe':'#c97640','Cu':'#bf804c'}
VDW={'H':1.2,'C':1.7,'N':1.55,'O':1.52,'F':1.47,'Cl':1.75,'Br':1.85,'S':1.8,'P':1.8}


def measurement(atoms,indices):
    if len(indices)<2:return ''
    xyz=[[atoms[i][k] for k in ('x','y','z')] for i in indices]
    def sub(a,b):return [x-y for x,y in zip(a,b)]
    def dot(a,b):return sum(x*y for x,y in zip(a,b))
    def norm(a):return math.sqrt(dot(a,a))
    def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    if len(indices)==2:return f'{norm(sub(xyz[0],xyz[1])):.4f} Å'
    if len(indices)==3:
        u,v=sub(xyz[0],xyz[1]),sub(xyz[2],xyz[1]);den=norm(u)*norm(v)
        return f'{math.degrees(math.acos(max(-1,min(1,dot(u,v)/den)))):.2f}°' if den>1e-10 else '角度未定义'
    b0,b1,b2=sub(xyz[0],xyz[1]),sub(xyz[2],xyz[1]),sub(xyz[3],xyz[2]);n=norm(b1)
    if n<1e-10:return '二面角未定义'
    b1=[x/n for x in b1];v=[x-dot(b0,b1)*y for x,y in zip(b0,b1)];w=[x-dot(b2,b1)*y for x,y in zip(b2,b1)]
    if norm(v)*norm(w)<1e-10:return '二面角未定义'
    return f'{math.degrees(math.atan2(dot(cross(b1,v),w),dot(v,w))):.2f}°'


class MoleculeView(QWidget):
    selected=Signal(str)
    selectionChanged=Signal(list)
    viewChanged=Signal()
    def __init__(self):
        super().__init__();self.structure=None;self.yaw=.35;self.pitch=-.28;self.zoom=1.
        self.last=None;self.start=None;self.projected=[];self.radii=[];self.picks=[]
        self.labels=False;self.style='ballstick';self.mode='rotate';self.measure_count=2
        self.pan=QPointF();self.light_background=False;self.axes=True;self.perspective=False
        self.ball_scale=1.;self.stick_scale=1.;self.cpk_scale=1.;self.line_width=1.5;self.show_bond_orders=True;self.show_hydrogens=True
        self.grid=False
        self.background_colors=None
        self.fit_padding=(120,110)
        self.setMinimumSize(300,260);self.setFocusPolicy(Qt.StrongFocus);self.setMouseTracking(True)
        self.setToolTip('拖动旋转 · 中键 / Shift+拖动平移 · 滚轮缩放 · 点击选择原子 · F 适合窗口')
    def set_structure(self,s):self.structure=s;self.fit();self.clear_selection()
    def fit(self):self.zoom=1.;self.pan=QPointF();self.update();self.viewChanged.emit()
    def reset_view(self):self.yaw=.35;self.pitch=-.28;self.fit()
    def orient(self,view):self.yaw,self.pitch={'front':(0,0),'right':(math.pi/2,0),'top':(0,math.pi/2),'iso':(.6,-.5)}[view];self.fit()
    def set_style(self,style):self.style=style;self.update();self.viewChanged.emit()
    def set_mode(self,mode):
        self.mode=mode;self.setCursor(Qt.OpenHandCursor if mode in ['rotate','pan'] else Qt.CrossCursor if mode in ['select','measure'] else Qt.SizeVerCursor)
    def clear_selection(self):self.picks=[];self.selectionChanged.emit([]);self.update()
    def select_indices(self,indices):
        self.picks=list(dict.fromkeys(i for i in indices if self.structure and 0<=i<len(self.structure.atoms)));self.selectionChanged.emit(self.picks);self.update()
    def rotated(self,x,y,z):
        x,z=x*math.cos(self.yaw)+z*math.sin(self.yaw),-x*math.sin(self.yaw)+z*math.cos(self.yaw)
        y,z=y*math.cos(self.pitch)-z*math.sin(self.pitch),y*math.sin(self.pitch)+z*math.cos(self.pitch)
        return x,y,z
    def scene_center(self, atoms):
        return [sum(atom[key] for atom in atoms)/len(atoms) for key in ('x','y','z')]

    def framing_points(self, points, center):
        return points

    def paint_environment(self, painter, center, scale, extent):
        pass

    def atom_color(self, element):
        return QColor(COLORS.get(element, '#b49ed4'))

    def paint_sphere(self, painter, x, y, radius, color):
        sphere = QRadialGradient(QPointF(x-radius*.35, y-radius*.4), radius*1.6)
        sphere.setColorAt(0, color.lighter(175))
        sphere.setColorAt(.28, color.lighter(110))
        sphere.setColorAt(.7, color.darker(170))
        sphere.setColorAt(1, color.darker(230))
        painter.setBrush(sphere)
        painter.setPen(QPen(color.darker(170), .6))
        painter.drawEllipse(QPointF(x, y), radius, radius)

    def paint_bond_segment(self, painter, start, end, element, width, normal, pen_style):
        color = self.atom_color(element)
        light = QLinearGradient(start+normal, start-normal)
        light.setColorAt(0, color.darker(170))
        light.setColorAt(.35, color.lighter(135))
        light.setColorAt(1, color.darker(180))
        painter.setPen(QPen(QBrush(light), width, pen_style, Qt.RoundCap))
        painter.drawLine(start, end)

    def scale_bar(self, scale):
        length = 2 if scale < 100 else 1
        return scale*length, f'{length} Å'
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        grad=QLinearGradient(0,0,0,self.height())
        top,bottom=self.background_colors or (('#f5f8fc','#e6eef8') if self.light_background else ('#172d47','#2a4868'))
        grad.setColorAt(0,QColor(top));grad.setColorAt(1,QColor(bottom))
        p.fillRect(self.rect(),grad);text=QColor('#4b535e' if self.light_background else '#b2b8c0')
        if not self.structure:return
        atoms=self.structure.atoms;center=self.scene_center(atoms)
        points=[self.rotated(*[a[k]-center[i] for i,k in enumerate(('x','y','z'))]) for a in atoms]
        margin=3.5 if self.style=='cpk' else 1.8
        frame=self.framing_points(points, center)
        extent_x=max(x for x,y,z in frame)-min(x for x,y,z in frame)+margin;extent_y=max(y for x,y,z in frame)-min(y for x,y,z in frame)+margin
        scale=min(max(1,self.width()-self.fit_padding[0])/extent_x,max(1,self.height()-self.fit_padding[1])/extent_y)*self.zoom;self.scale=scale
        extent=max(3.,max(math.sqrt(x*x+y*y+z*z) for x,y,z in points));factors=[1/(1-z/(extent*5)) if self.perspective else 1 for x,y,z in points]
        if self.grid:
            self.draw_grid(p, center, scale, extent)
        self.paint_environment(p, center, scale, extent)
        self.projected=[(self.width()/2+x*scale*f+self.pan.x(),self.height()/2-y*scale*f+self.pan.y(),z) for (x,y,z),f in zip(points,factors)]
        def radius(i):
            e=atoms[i]['element']
            if self.style=='cpk':return VDW.get(e,1.7)*scale*factors[i]*self.cpk_scale
            if self.style=='line':return 3.
            return (.20 if e=='H' else .29)*scale*factors[i]*self.ball_scale if self.style=='ballstick' else .115*scale*factors[i]*self.stick_scale
        self.radii=[max(3.,radius(i)) for i in range(len(atoms))];primitives=[]
        if self.style!='cpk':
            for i,j,order in self.structure.bonds:
                i,j=int(i),int(j)
                if not self.show_hydrogens and (atoms[i]['element']=='H' or atoms[j]['element']=='H'):continue
                primitives.append(((points[i][2]+points[j][2])/2,'bond',(i,j,order)))
        primitives.extend((xyz[2],'atom',i) for i,xyz in enumerate(points) if self.show_hydrogens or atoms[i]['element']!='H')
        for z,kind,data in sorted(primitives,key=lambda v:v[0]):
            if kind=='bond':
                i,j,order=data;x1,y1,_=self.projected[i];x2,y2,_=self.projected[j];dx,dy=x2-x1,y2-y1;length=max(1,math.hypot(dx,dy));ux,uy=dx/length,dy/length
                trim1=self.radii[i]*.72 if self.style=='ballstick' else 0;trim2=self.radii[j]*.72 if self.style=='ballstick' else 0
                a=QPointF(x1+ux*trim1,y1+uy*trim1);b=QPointF(x2-ux*trim2,y2-uy*trim2);width=self.line_width if self.style=='line' else max(1.4,scale*(.10 if self.style=='ballstick' else .19)*self.stick_scale)
                count=(3 if order>=3 else 2 if order>=1.5 else 1) if self.show_bond_orders else 1;offsets=[0] if count==1 else [-width*.65,width*.65] if count==2 else [-width*1.2,0,width*1.2]
                for offset_number,offset in enumerate(offsets):
                    off=QPointF(-uy*offset,ux*offset);mid=(a+b)/2
                    for start,end,e in [(a+off,mid+off,atoms[i]['element']),(mid+off,b+off,atoms[j]['element'])]:
                        pen_style = Qt.DashLine if order==1.5 and offset_number==1 else Qt.SolidLine
                        self.paint_bond_segment(p,start,end,e,width,QPointF(-uy*width/2,ux*width/2),pen_style)
            else:
                i=data;x,y,_=self.projected[i];r=self.radii[i];e=atoms[i]['element'];color=self.atom_color(e)
                if self.style=='line':p.setBrush(color);p.setPen(Qt.NoPen);p.drawEllipse(QPointF(x,y),r,r)
                else:
                    self.paint_sphere(p, x, y, r, color)
                if i in self.picks:p.setBrush(Qt.NoBrush);p.setPen(QPen(QColor('#ffdb46'),1.7));p.drawEllipse(QPointF(x,y),r+4,r+4)
                if self.labels:p.setFont(QFont('Segoe UI',9));p.setPen(QColor('#202020' if self.light_background else '#ededed'));p.drawText(QPointF(x+r+4,y-4),f'{e}{i+1}')
        if 2<=len(self.picks)<=4:
            p.setPen(QPen(QColor('#ffdf55'),1,Qt.DashLine))
            for i,j in zip(self.picks,self.picks[1:]):p.drawLine(QPointF(*self.projected[i][:2]),QPointF(*self.projected[j][:2]))
            p.setFont(QFont('Segoe UI',10));p.setPen(QColor('#ffdf55') if not self.light_background else QColor('#855c00'));p.drawText(16,25,'—'.join(str(i+1) for i in self.picks)+'    '+measurement(atoms,self.picks))
        p.setFont(QFont('Segoe UI',8));p.setPen(text)
        caption=('透视视图' if self.perspective else '正交视图')+(' · 网格 1 Å' if self.grid else '')
        p.drawText(QRectF(12,5,self.width()-24,23),Qt.AlignRight|Qt.AlignVCenter,caption)
        if self.axes:
            origin=QPointF(52,self.height()-48)
            for vector,c,name in [((1,0,0),'#ed5757','X'),((0,1,0),'#63c35d','Y'),((0,0,1),'#5c9eed','Z')]:
                x,y,z=self.rotated(*vector);tip=origin+QPointF(x*30,-y*30);p.setPen(QPen(QColor(c),1.8));p.drawLine(origin,tip);p.drawText(tip+QPointF(3,-2),name)
            p.setBrush(QColor('#afb6bf'));p.setPen(Qt.NoPen);p.drawEllipse(origin,2.5,2.5)
        bar,bar_label=self.scale_bar(scale)
        if bar<self.width()/3:
            x,y=self.width()-30-bar,self.height()-32;p.setPen(QPen(text,1));p.drawLine(QPointF(x,y),QPointF(x+bar,y));p.drawLine(QPointF(x,y-3),QPointF(x,y+3));p.drawLine(QPointF(x+bar,y-3),QPointF(x+bar,y+3));p.drawText(QRectF(x,y-21,bar,18),Qt.AlignCenter,bar_label)
    def draw_grid(self, painter, center, scale, extent):
        span=min(10,max(4,math.ceil(extent)))
        height=min(atom['z']-center[2] for atom in self.structure.atoms)-.6
        painter.save()
        for coordinate in range(-span,span+1):
            color=QColor('#8298b4' if self.light_background else '#84a9d3')
            color.setAlpha(70 if coordinate==0 else 30)
            painter.setPen(QPen(color,1))
            for endpoints in (
                    ((coordinate,-span,height),(coordinate,span,height)),
                    ((-span,coordinate,height),(span,coordinate,height))):
                projected=[]
                for xyz in endpoints:
                    x,y,z=self.rotated(*xyz)
                    factor=1/(1-z/(extent*5)) if self.perspective else 1
                    projected.append(QPointF(self.width()/2+x*scale*factor+self.pan.x(),
                                             self.height()/2-y*scale*factor+self.pan.y()))
                painter.drawLine(projected[0],projected[1])
        painter.restore()

    def mousePressEvent(self,e):self.setFocus();self.last=e.position();self.start=e.position()
    def mouseMoveEvent(self,e):
        if self.last is not None and e.buttons():
            d=e.position()-self.last
            if e.buttons() & Qt.MiddleButton or e.modifiers() & Qt.ShiftModifier or self.mode=='pan':self.pan+=d
            elif self.mode=='zoom':self.zoom=max(.1,min(8,self.zoom*math.exp(-d.y()*.01)))
            elif self.mode=='rotate':self.yaw+=d.x()*.01;self.pitch+=d.y()*.01
            self.last=e.position();self.update();self.viewChanged.emit()
    def mouseReleaseEvent(self,e):
        if self.structure and self.start is not None and (e.position()-self.start).manhattanLength()<5 and e.button()==Qt.LeftButton:
            hits=[i for i,(x,y,z) in enumerate(self.projected) if (self.show_hydrogens or self.structure.atoms[i]['element']!='H') and math.hypot(x-e.position().x(),y-e.position().y())<=max(9,self.radii[i]+3)]
            if hits:
                i=max(hits,key=lambda k:self.projected[k][2])
                if e.modifiers() & Qt.ControlModifier:self.picks=[x for x in self.picks if x!=i] if i in self.picks else self.picks+[i]
                elif self.mode=='measure':self.picks=([i] if len(self.picks)>=self.measure_count else self.picks+[i])
                else:self.picks=[i]
                self.selected.emit(f'原子 {i+1} · {self.structure.atoms[i]["element"]}');self.selectionChanged.emit(self.picks)
            elif not e.modifiers() & Qt.ControlModifier:self.clear_selection()
        self.last=None;self.update()
    def wheelEvent(self,e):self.zoom=max(.1,min(8,self.zoom*math.exp(e.angleDelta().y()/120*.12)));self.update();self.viewChanged.emit()
    def keyPressEvent(self,e):
        if e.key()==Qt.Key_F:self.fit()
        elif e.key()==Qt.Key_Escape:self.clear_selection()
        else:super().keyPressEvent(e)
    def contextMenuEvent(self,e):
        menu=QMenu(self);menu.addAction('适合窗口',self.fit);menu.addAction('重置视图',self.reset_view);menu.addSeparator()
        for name,style in [('球棍模型','ballstick'),('棍状模型','stick'),('线框模型','line'),('空间填充（CPK）','cpk')]:menu.addAction(name,lambda checked=False,s=style:self.set_style(s))
        menu.addSeparator();menu.addAction('复制视图',lambda:QApplication.clipboard().setPixmap(self.grab()));menu.exec(e.globalPos())
