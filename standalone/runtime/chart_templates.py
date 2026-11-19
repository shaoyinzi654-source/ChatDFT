"""Plot recipes for saved PySCF results and existing numerical CSV curves."""
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator


@dataclass(frozen=True)
class Theme:
    title: str
    paper: str
    ink: str
    grid: str
    colors: tuple[str, ...]
    negative: str


THEMES = {
    'journal': Theme('期刊 · 白底蓝橙', '#ffffff', '#172737', '#dce4eb',
                     ('#17688a', '#c35a29', '#38603e', '#725694'), '#a9364d'),
    'lab': Theme('实验室 · 暖灰与孔雀绿', '#f8f6f0', '#213633', '#e4e2d8',
                 ('#006d63', '#a44a38', '#635597', '#8b6727'), '#b8453a'),
    'dark': Theme('深色汇报 · 青蓝', '#101827', '#f0f5ff', '#314155',
                  ('#55d9ba', '#ffb66b', '#a8aaff', '#f7819e'), '#fc8284'),
    'colorblind': Theme('高辨识 · 蓝橙对照', '#ffffff', '#20252b', '#d6dbe0',
                        ('#0072b2', '#d55e00', '#009e73', '#cc79a7'), '#a53016'),
    'mono': Theme('黑白印刷 · 灰阶', '#ffffff', '#202020', '#dedede',
                  ('#252525', '#555555', '#767676', '#9b9b9b'), '#797979'),
}


@dataclass(frozen=True)
class Chart:
    title: str
    xlabel: str
    ylabel: str
    kind: str
    # Each series supplies its own x values, y values, and label.
    series: tuple
    ticks: tuple = ()
    log_scale: bool = False
    integer_x: bool = False


@dataclass(frozen=True)
class Template:
    name: str
    category: str
    charts: tuple[Chart, ...]


def finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _entries(values):
    return values if isinstance(values, (list, tuple)) else ()


def indexed(values):
    return [(i, float(value)) for i, value in enumerate(_entries(values), 1) if finite(value)]


def result_templates(result):
    """Return only charts whose real input fields exist, including for older jobs."""
    templates = {}

    def add(key, name, category, title, xlabel, ylabel, kind, pairs, **options):
        if pairs:
            x, y = zip(*pairs)
            templates[key] = Template(name, category, (Chart(title, xlabel, ylabel, kind,
                                                               ((x, y, ''),), **options),))

    energies = indexed(result.get('scf_energies_hartree'))
    add('scf', 'SCF 收敛', '收敛', 'SCF electronic energy', 'SCF record', 'Energy (Eh)',
        'line', energies, integer_x=True)
    if len(energies) > 1:
        final = energies[-1][1]
        delta = [(i, abs(value - final)) for i, value in energies if value != final]
        add('scf-error', 'SCF 残差 · 对数', '收敛', 'SCF error vs. final record', 'SCF record',
            '|E − Efinal| (Eh)', 'line', delta, log_scale=True, integer_x=True)
        steps = [(right[0], right[1] - left[1]) for left, right in zip(energies, energies[1:])
                 if right[0] == left[0] + 1]
        add('scf-step', '逐步能量变化', '收敛', 'Successive SCF energy changes', 'SCF record',
            'ΔE (Eh)', 'stem', steps, integer_x=True)

    charges = sorted((float(row['index']), float(row['charge']))
                     for row in _entries(result.get('mulliken_charges'))
                     if isinstance(row, dict) and finite(row.get('index')) and finite(row.get('charge')))
    add('charges', '原子电荷 · 正负柱', '电荷', 'Mulliken atomic charges', 'Atom index', 'Charge (e)',
        'bar', charges, integer_x=True)
    rank = sorted(charges, key=lambda row: abs(row[1]), reverse=True)[:16]
    add('charge-rank', '高电荷原子排序', '电荷', 'Top 16 atomic charges by magnitude', 'Charge (e)',
        'Atom index', 'rank', rank)
    if charges:
        add('charge-hist', '电荷分布直方图', '电荷', 'Distribution of Mulliken charges', 'Charge (e)',
            'Atom count', 'hist', [(q, 0) for _, q in charges])

    levels = [(spin.upper(), float(result[f'{spin}_homo_ev']), float(result[f'{spin}_lumo_ev']))
              for spin in ('alpha', 'beta')
              if finite(result.get(f'{spin}_homo_ev')) and finite(result.get(f'{spin}_lumo_ev'))]
    if levels:
        indices = tuple(range(1, len(levels) + 1))
        ticks = tuple(row[0] for row in levels)
        templates['orbitals'] = Template('HOMO / LUMO 能级', '轨道', (Chart(
            'Frontier orbital levels', 'Spin channel', 'Energy (eV)', 'levels',
            ((indices, tuple(row[1] for row in levels), 'HOMO'),
             (indices, tuple(row[2] for row in levels), 'LUMO')), ticks),))
        add('gap', '自旋通道能隙', '轨道', 'HOMO–LUMO gap', 'Spin channel', 'Gap (eV)',
            'bar', [(i, lumo - homo) for i, (_, homo, lumo) in enumerate(levels, 1)], ticks=ticks)

    modes = indexed(result.get('frequencies_cm1'))
    add('frequencies', '振动频率与虚频', '光谱', 'Harmonic vibrational modes', 'Mode index',
        'Wavenumber (cm⁻¹)', 'bar', modes, integer_x=True)
    strengths = dict(indexed(result.get('ir_intensities_km_mol')))
    add('ir', '红外谱 · 棒图', '光谱', 'Harmonic infrared spectrum', 'Wavenumber (cm⁻¹)',
        'Intensity (km mol⁻¹)', 'stem', sorted((freq, strengths[i]) for i, freq in modes
                                             if i in strengths and freq > 0 and strengths[i] >= 0))
    states = [row for row in _entries(result.get('excited_states')) if isinstance(row, dict)]
    for key, field, name, label in (('uv-ev', 'energy_ev', '电子跃迁 · 能量轴', 'Energy (eV)'),
                                     ('uv-nm', 'wavelength_nm', '电子跃迁 · 波长轴', 'Wavelength (nm)')):
        peaks = sorted((float(row[field]), float(row['oscillator_strength'])) for row in states
                       if finite(row.get(field)) and float(row[field]) > 0 and finite(row.get('oscillator_strength')))
        add(key, name, '光谱', 'Vertical electronic excitations', label, 'Oscillator strength', 'stem', peaks)
    panels = tuple(templates[key].charts[0] for key in ('scf', 'charges', 'orbitals', 'frequencies', 'uv-ev')
                   if key in templates)
    if panels:
        return {'overview': Template('计算结果总览', '组合', panels), **templates}
    return templates


def curve_templates(headers, rows):
    if not rows or len(headers) < 2:
        return {}
    values = tuple(zip(*rows))
    series = tuple((values[0], column, header) for column, header in zip(values[1:], headers[1:]))
    ylabel = headers[1] if len(headers) == 2 else 'Value'
    plots = {}
    for key, name, title, kind in (
        ('line', '多系列折线', 'Series comparison', 'line'),
        ('scatter', '多系列散点', 'Measured points', 'scatter'),
        ('area', '曲线与基线填充', 'Series with baseline', 'area'),
        ('step', '阶梯曲线', 'Step profile', 'step'),
        ('curve-stem', '离散棒图', 'Discrete values', 'stem'),
        ('curve-bar', '并列柱图', 'Grouped values', 'bar'),
        ('curve-hist', '样本分布直方图', 'Value distribution', 'hist-y'),
    ):
        x_label, y_label = (ylabel, 'Count') if kind == 'hist-y' else (headers[0], ylabel)
        if kind == 'bar' and len(rows) > 500:
            continue
        plots[key] = Template(name, 'CSV', (Chart(title, x_label, y_label, kind, series),))
    if len(series) > 1:
        plots['facets'] = Template('独立子图对照', 'CSV', tuple(
            Chart(label, headers[0], label, 'line', ((x, y, ''),)) for x, y, label in series))
    return plots


def _draw(ax, chart, theme):
    styles = ('-', '--', '-.', ':')
    markers = ('o', 's', '^', 'D')
    total = len(chart.series)
    for i, (raw_x, raw_y, label) in enumerate(chart.series):
        x, y = np.array(raw_x), np.array(raw_y)
        color = theme.colors[i % len(theme.colors)]
        opts = dict(color=color, label=label or None)
        if chart.kind in ('bar', 'stem', 'rank'):
            signed = [color if value >= 0 else theme.negative for value in y]
        if chart.kind == 'line':
            ax.plot(x, y, linewidth=1.8, linestyle=styles[i % 4],
                    marker=markers[i % 4] if len(x) <= 40 else None, markersize=3.4, **opts)
        elif chart.kind == 'scatter':
            ax.scatter(x, y, s=20, alpha=.85, marker=markers[i % 4], **opts)
        elif chart.kind == 'area':
            ax.fill_between(x, y, 0, color=color, alpha=.16)
            ax.plot(x, y, linewidth=1.6, linestyle=styles[i % 4], **opts)
        elif chart.kind == 'step':
            ax.step(x, y, where='mid', linewidth=1.7, linestyle=styles[i % 4], **opts)
        elif chart.kind == 'bar':
            steps = np.diff(np.unique(x))
            width = (.74 * float(steps.min()) if steps.size else .74) / total
            ax.bar(x + (i - (total - 1) / 2) * width, y, width=width, color=signed,
                   label=label or None)
        elif chart.kind == 'stem':
            ax.vlines(x, 0, y, color=signed, linewidth=1.5, label=label or None)
            if len(x) < 100:
                ax.scatter(x, y, color=signed, s=13, zorder=3, marker=markers[i % 4])
        elif chart.kind == 'levels':
            ax.hlines(y, x - .23, x + .23, linewidth=3, **opts)
        elif chart.kind == 'rank':
            positions = np.arange(len(x))[::-1]
            ax.hlines(positions, 0, y, color=signed, alpha=.72, lw=1.8)
            ax.scatter(y, positions, color=signed, s=35, zorder=3)
            ax.set_yticks(positions, labels=[f'{value:g}' for value in x])
        elif chart.kind in ('hist', 'hist-y'):
            ax.hist(x if chart.kind == 'hist' else y, bins=min(48, max(5, int(np.sqrt(len(x))))), alpha=.58,
                    edgecolor=theme.paper, linewidth=.65, **opts)
    if chart.kind in ('bar', 'stem'):
        ax.axhline(0, color=theme.ink, lw=.6, alpha=.55)
    elif chart.kind == 'rank':
        ax.axvline(0, color=theme.ink, lw=.6, alpha=.55)
    ax.set(xlabel=chart.xlabel, ylabel=chart.ylabel)
    ax.set_title(chart.title, loc='left', color=theme.ink, weight='bold', pad=12, fontsize=11)
    if chart.log_scale:
        ax.set_yscale('log')
    if chart.ticks:
        ax.set_xticks(range(1, len(chart.ticks) + 1), labels=chart.ticks)
        ax.set_xlim(.5, len(chart.ticks) + .5)
    elif chart.integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    handles, labels = ax.get_legend_handles_labels()
    if handles and total > 1:
        ax.legend(handles, labels, frameon=False, labelcolor=theme.ink, fontsize=8, loc='best')


def apply_chart_style(figure, theme):
    figure.set_facecolor(theme.paper)
    for ax in figure.axes:
        ax.set_facecolor(theme.paper)
        ax.grid(axis='y', color=theme.grid, lw=.65)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        for side in ('bottom', 'left'):
            ax.spines[side].set_color(theme.ink)
            ax.spines[side].set_linewidth(.75)
        ax.tick_params(colors=theme.ink, labelsize=8, length=3, width=.6)
        ax.xaxis.label.set_color(theme.ink)
        ax.yaxis.label.set_color(theme.ink)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)


_CJK_FONTS = ('Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei')


def _apply_cjk(figure):
    for text in figure.findobj(Text):
        if not any('\u3400' <= character <= '\u9fff' for character in text.get_text()):
            continue
        for name in _CJK_FONTS:
            properties = FontProperties(family=name, size=text.get_fontsize(), weight=text.get_fontweight())
            try:
                findfont(properties, fallback_to_default=False)
            except ValueError:
                continue
            text.set_fontproperties(properties)
            break


def make_figure(template, theme_key='journal', title='', size=None, figure=None):
    theme = THEMES[theme_key]
    count = len(template.charts)
    columns = min(2, count)
    if count == 3:
        columns = 3
    rows = (count + columns - 1) // columns
    if figure is None:
        figure = Figure(figsize=size or (max(6.5, 4.3*columns), max(4.3, 3.1*rows)), layout='constrained')
    else:
        figure.clear()
    grid = figure.add_gridspec(rows, columns)
    for i, chart in enumerate(template.charts):
        row, column = divmod(i, columns)
        cell = grid[row, :] if i == count - 1 and column == 0 else grid[row, column]
        _draw(figure.add_subplot(cell), chart, theme)
    if title.strip():
        figure.suptitle(title.strip(), fontsize=16, fontweight='bold', color=theme.ink)
    apply_chart_style(figure, theme)
    _apply_cjk(figure)
    return figure
