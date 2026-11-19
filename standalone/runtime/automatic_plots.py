"""Plot molecular geometries and numerical data returned by PySCF."""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from chatdft.assembly import coordinates
from automatic_analysis import align_coordinates
from chart_templates import make_figure, result_templates


ELEMENT_COLORS = {'C': '#394552', 'H': '#e7edf1', 'O': '#c5363d',
                  'N': '#3558ad', 'Mg': '#916aba', 'S': '#bd941e'}


def save_figure(folder, name, figure):
    FigureCanvasAgg(figure)
    for suffix in ('png', 'svg'):
        figure.savefig(folder / f'{name}.{suffix}', dpi=220, facecolor='white')
    return name


def structure_figure(folder, initial, final=None):
    models = [('Initial geometry', initial)]
    values = coordinates(initial)
    values -= values.mean(axis=0)
    geometries = [values]
    if final:
        models.append(('Optimized geometry', final))
        geometries = align_coordinates(initial, final)
    radius = max(max(np.max(np.abs(xyz)) for xyz in geometries), .8) * 1.09
    figure = Figure(figsize=(5.8 * len(models), 5.1), layout='constrained')
    for index, ((title, structure), values) in enumerate(zip(models, geometries), 1):
        ax = figure.add_subplot(1, len(models), index, projection='3d')
        for a, b, _ in structure.bonds:
            ax.plot(*values[[a, b]].T, color='#8f9aa5', lw=1.4, alpha=.9)
        symbols = np.array([atom['element'] for atom in structure.atoms])
        for symbol in sorted(set(symbols)):
            ax.scatter(*values[symbols == symbol].T, s=19 if symbol == 'H' else 52,
                       color=ELEMENT_COLORS.get(symbol, '#26836e'), edgecolors='#546474',
                       linewidths=.35, label=symbol, depthshade=False)
        ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius),
               xlabel='x (Å)', ylabel='y (Å)', zlabel='z (Å)', title=title)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-53)
        ax.legend(loc='upper right', frameon=False, fontsize=8)
    return save_figure(folder, '结构图', figure)


def result_figure(folder, result):
    overview = result_templates(result).get('overview')
    if overview:
        return save_figure(folder, '计算结果图', make_figure(overview))
