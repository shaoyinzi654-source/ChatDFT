"""Interactive preview of actual GROMACS coordinates, kept separate from PySCF models."""
from dataclasses import dataclass
from itertools import product
import json
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QPointF, QLineF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF

from chatdft.md_inputs import IONS, rows, system_state
from chatdft.viewport import MoleculeView


@dataclass
class BoxModel:
    xyz: np.ndarray
    bonds: np.ndarray
    names: np.ndarray
    ions: np.ndarray
    water: np.ndarray
    cell: np.ndarray | None
    summary: str
    ion_symbol: str | None


def load_box(folder):
    manifest = json.loads((folder/'manifest.json').read_text('utf-8'))
    config = manifest['config']
    periodic = (folder/'system.gro').is_file()
    path = folder/('system.gro' if periodic else 'solute.gro')
    records = path.read_text('utf-8').splitlines()
    atom_count = int(records[1])
    if atom_count < 1 or len(records) != atom_count + 3:
        raise ValueError('GRO 原子记录数量不符。')
    atoms = records[2:-1]
    xyz = np.array([[float(line[start:start+8]) for start in (20, 28, 36)] for line in atoms])
    if not np.isfinite(xyz).all():
        raise ValueError('GRO 包含无效坐标。')
    atoms_per_pdc = manifest['solute_atoms']
    molecules = config['molecules'] if periodic else 1
    pdc_end = molecules*atoms_per_pdc
    ion_count = manifest['ion_count'] if periodic else 0
    ion_end = pdc_end + ion_count
    if ion_end > atom_count or (atom_count-ion_end) % 3:
        raise ValueError('GRO 的 PDC、离子或水数量与作业清单不符。')
    if any(line[5:10].strip() != manifest['molecule_name'] for line in atoms[:pdc_end]):
        raise ValueError('GRO 的 PDC 原子顺序与作业清单不符。')
    _, ion = system_state(config['system'])
    if ion_count and any(line[5:10].strip() != ion.upper() for line in atoms[pdc_end:ion_end]):
        raise ValueError('GRO 中的离子类型与作业体系不符。')
    if any(line[5:10].strip() != 'SOL' for line in atoms[ion_end:]):
        raise ValueError('GRO 的水分子记录顺序不符。')

    unit_bonds = np.array([[int(bond[0])-1, int(bond[1])-1] for bond in
                           rows((folder/'pdc.itp').read_text('utf-8'), 'bonds')], dtype=int).reshape(-1, 2)
    bonds = (unit_bonds + (np.arange(molecules)*atoms_per_pdc)[:,None,None]).reshape(-1, 2)
    points = xyz[:pdc_end].reshape(molecules, atoms_per_pdc, 3)
    ions, water = xyz[pdc_end:ion_end], xyz[ion_end::3]
    names = np.array([line[10:15].strip() for line in atoms[:pdc_end]])
    cell = None
    if periodic:
        box = [float(value) for value in records[-1].split()]
        if len(box) == 3:
            cell = np.diag(box)
        elif len(box) == 9:
            cell = np.array([[box[0],box[3],box[4]], [box[5],box[1],box[6]], [box[7],box[8],box[2]]])
        if cell is None or not np.isfinite(cell).all() or np.linalg.det(cell) <= 0:
            raise ValueError('GRO 缺少有效的周期盒。')
        inverse = np.linalg.inv(cell)
        points -= (np.floor(points.mean(axis=1) @ inverse) @ cell)[:,None,:]
        ions -= np.floor(ions @ inverse) @ cell
        water -= np.floor(water @ inverse) @ cell
        xyz -= cell.sum(axis=0)/2
        lengths = ' × '.join(f'{length:g}' for length in np.linalg.norm(cell, axis=1))
        counterions = f' · {ion_count} {ion}{"²⁺" if IONS[ion]==2 else "⁺"}' if ion else ''
        text = f'初始周期盒 {lengths} nm · {molecules} PDC · {len(water):,} 水{counterions} · {atom_count:,} 原子'
    else:
        points -= points.mean(axis=1)[:,None,:]
        text = '已生成 PDC 起始分子。完整周期盒会在服务器任务的建盒阶段生成。'
    return BoxModel(points.reshape(-1, 3), bonds, names, ions, water, cell, text, ion)


class PeriodicBoxView(MoleculeView):
    """The workbench renderer with a periodic cell and an optional water layer."""
    CORNERS = np.array(list(product([-.5, .5], repeat=3)))

    def __init__(self, background_colors):
        super().__init__()
        self.setMinimumSize(360, 330)
        self.background_colors = background_colors
        self.fit_padding = (72, 62)
        self.ball_scale = 1.12
        self.show_hydrogens = False
        self.show_bond_orders = False
        self.model = None
        self.cell_frame = None
        self.show_water = False
        self.show_cell = True
        self._sphere_images = {}
        self._bond_pens = {}
        self._bond_width = None
        self.set_mode('rotate')

    def set_model(self, model):
        self.model = model
        self.cell_frame = None
        if model is None:
            self.set_structure(None)
            return
        xyz = np.concatenate((model.xyz, model.ions))*10  # shared viewport uses Å
        elements = [name[0] for name in model.names] + [model.ion_symbol]*len(model.ions)
        atoms = [dict(element=e, x=float(x), y=float(y), z=float(z))
                 for e, (x, y, z) in zip(elements, xyz)]
        bonds = [(int(i), int(j), 1) for i, j in model.bonds]
        if model.cell is not None:
            self.cell_frame = self.CORNERS @ (model.cell*10)
        self.set_structure(SimpleNamespace(atoms=atoms, bonds=bonds))

    def toggle_water(self, visible):
        self.show_water = visible
        self.update()

    def toggle_cell(self, visible):
        self.show_cell = visible
        self.update()

    def toggle_hydrogens(self, visible):
        self.show_hydrogens = visible
        self.clear_selection()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.model is None:
            painter = QPainter(self)
            painter.setPen(QColor('#e1eaf4'))
            painter.drawText(self.rect(), Qt.AlignCenter, '选择已准备的论文体系来查看结构')

    def scene_center(self, atoms):
        return (0., 0., 0.) if self.cell_frame is not None else super().scene_center(atoms)

    def framing_points(self, points, center):
        if self.cell_frame is None:
            return points
        return [self.rotated(*point) for point in self.cell_frame]

    def atom_color(self, element):
        if element == self.model.ion_symbol:
            return QColor('#efbd65')
        return super().atom_color(element)

    def paint_sphere(self, painter, x, y, radius, color):
        key = color.name()
        sprite = self._sphere_images.get(key)
        if sprite is None:
            sprite = QPixmap(34, 34)
            sprite.fill(Qt.transparent)
            brush_painter = QPainter(sprite)
            brush_painter.setRenderHint(QPainter.Antialiasing)
            super().paint_sphere(brush_painter, 17, 17, 16, color)
            brush_painter.end()
            self._sphere_images[key] = sprite
        r = radius * 17/16
        painter.drawPixmap(QRectF(x-r, y-r, 2*r, 2*r), sprite, QRectF(sprite.rect()))

    def paint_bond_segment(self, painter, start, end, element, width, normal, pen_style):
        if width != self._bond_width:
            self._bond_pens.clear()
            self._bond_width = width
        pen = self._bond_pens.get(element)
        if pen is None:
            pen = QPen(self.atom_color(element).lighter(125), width, Qt.SolidLine, Qt.RoundCap)
            self._bond_pens[element] = pen
        painter.setPen(pen)
        painter.drawLine(start, end)

    def scale_bar(self, scale):
        if self.cell_frame is None or scale >= 12:
            return super().scale_bar(scale)
        for nm in (0.5, 1, 2, 5):
            pixels = nm * 10 * scale
            if pixels >= 34:
                return pixels, f'{nm:g} nm'
        return scale*50, '5 nm'

    def paint_environment(self, painter, center, scale, extent):
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if self.cell_frame is None:
            return
        matrix = np.array([self.rotated(*axis) for axis in np.eye(3)])
        origin = [self.width()/2 + self.pan.x(), self.height()/2 + self.pan.y()]

        def project(xyz):
            rotated = (xyz - center) @ matrix
            factor = 1/(1-rotated[:,2]/(extent*5)) if self.perspective else 1.
            return rotated[:,:2]*(np.asarray(factor)[...,None]*[scale,-scale])+origin

        if self.show_water and len(self.model.water):
            points = project(self.model.water*10)
            painter.setPen(QPen(QColor(139, 197, 222, 102), 1.4, Qt.SolidLine, Qt.RoundCap))
            painter.drawPoints(QPolygonF([QPointF(x, y) for x, y in points]))
        if self.show_cell:
            cube = project(self.cell_frame)
            painter.setPen(QPen(QColor(215, 228, 239, 135), 1.15))
            painter.drawLines([QLineF(*cube[i], *cube[i | bit]) for i in range(8)
                               for bit in (1, 2, 4) if not i & bit])
