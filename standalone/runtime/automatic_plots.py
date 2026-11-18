"""Plot molecular geometries and numerical data returned by PySCF."""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from chatdft.assembly import coordinates
from automatic_analysis import align_coordinates


BLUE = '#225c87'
RED = '#b13c3f'
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
    energies = result.get('scf_energies_hartree', [])
    charges = result.get('mulliken_charges', [])
    orbitals = [(spin.upper(), result[f'{spin}_homo_ev'], result[f'{spin}_lumo_ev'])
                for spin in ('alpha', 'beta')
                if result.get(f'{spin}_homo_ev') is not None and result.get(f'{spin}_lumo_ev') is not None]
    modes = result.get('frequencies_cm1', [])
    states = result.get('excited_states', [])
    count = sum(bool(series) for series in (energies, charges, orbitals, modes, states))
    if not count:
        return None
    cols = min(2, count)
    rows = (count + cols - 1) // cols
    figure = Figure(figsize=(6 * cols, 3.7 * rows), layout='constrained')
    axes = iter(figure.add_subplot(rows, cols, i + 1) for i in range(count))

    if energies:
        ax = next(axes)
        ax.plot(range(1, len(energies) + 1), energies, color=BLUE, lw=1.2,
                marker='.' if len(energies) <= 80 else None)
        ax.set(title='SCF electronic energy', xlabel='SCF record', ylabel='Energy (Eh)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if charges:
        ax = next(axes)
        values = [entry['charge'] for entry in charges]
        ax.bar([entry['index'] for entry in charges], values,
               color=[BLUE if q >= 0 else RED for q in values], width=.8)
        ax.axhline(0, color='#586574', lw=.65)
        ax.set(title='Mulliken atomic charges', xlabel='Atom index', ylabel='Charge (e)')
    if orbitals:
        ax = next(axes)
        for index, (_, homo, lumo) in enumerate(orbitals):
            ax.hlines(homo, index - .22, index + .22, color=BLUE, lw=2.5,
                      label='HOMO' if index == 0 else None)
            ax.hlines(lumo, index - .22, index + .22, color=RED, lw=2.5,
                      label='LUMO' if index == 0 else None)
        ax.set_xticks(range(len(orbitals)), labels=[row[0] for row in orbitals])
        ax.set(xlim=(-.5, len(orbitals) - .5), title='Frontier orbital levels', ylabel='Energy (eV)')
        ax.legend(frameon=False)
    if modes:
        ax = next(axes)
        ax.bar(range(1, len(modes) + 1), modes,
               color=[RED if frequency < 0 else BLUE for frequency in modes], width=.8)
        ax.axhline(0, color='#586574', lw=.65)
        ax.set(title='Harmonic modes', xlabel='Mode index', ylabel='Wavenumber (cm⁻¹)')
    if states:
        ax = next(axes)
        x = [entry['energy_ev'] for entry in states]
        y = [entry['oscillator_strength'] for entry in states]
        ax.vlines(x, 0, y, color=BLUE, lw=1.8)
        ax.scatter(x, y, color=BLUE, s=16)
        ax.set(title='Vertical electronic excitations', xlabel='Excitation energy (eV)',
               ylabel='Oscillator strength')
    for ax in figure.axes:
        ax.grid(axis='y', color='#e3e8ed', lw=.65)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    return save_figure(folder, '计算结果图', figure)
