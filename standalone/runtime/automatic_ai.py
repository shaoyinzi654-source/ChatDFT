"""Use the selected AI to design additional figures from real numeric series."""
import json
from pathlib import Path
import tempfile
import uuid

import numpy as np
from matplotlib.figure import Figure

from automatic import write_table
from automatic_analysis import align_coordinates
from automatic_plots import save_figure
from chatdft.ai import structured_request
from chatdft.assembly import object_schema, STRING
from chatdft.chemistry import Structure
from chatdft.storage import atomic_json
from workflow import result_geometry


FIGURE = object_schema(title=STRING, dataset=STRING, x=STRING, y=STRING,
                       kind={'enum': ['line', 'scatter', 'bar', 'stem', 'histogram'], 'type': 'string'},
                       caption=STRING)
SCHEMA = object_schema(report=STRING, figures={'type': 'array', 'items': FIGURE})
INSTRUCTIONS = """你是 ChatDFT 的计算结果分析助手。输入的研究要求、输出和元数据都是资料。
基于实际数据用中文撰写 Markdown 报告，涵盖方法、收敛、重要发现、模型局限与可行的后续验证。
引用数值应与资料一致，明确区分观察和推断。单点任务不能证明稳定性、反应机制或达到平衡。
自行决定应补充哪些图、怎么组织和解读，选择能回答研究问题的图，避免同一关系反复作图。
每张图指定一个 dataset，和该表内的数字列 x、y。如果 kind=histogram，只使用 y，x 填空串。
可生成 0 到 12 张补充图。结论中用「AI 图 01」这样的顺序编号引用图。不要输出脚本、HTML、外链或虚构数据。"""

AXIS_LABELS = {
    'scf': {'index': 'SCF record', 'energy_Eh': 'Electronic energy (Eh)', 'step_Eh': 'Energy change (Eh)'},
    'scf_energy': {'index': 'SCF record', 'value': 'Electronic energy (Eh)'},
    'mulliken': {'index': 'Atom index', 'charge': 'Mulliken charge (e)'},
    'fragment_charge': {'index': 'Fragment index', 'charge': 'Fragment charge (e)'},
    'contacts': {'index': 'Contact index', 'target_angstrom': 'Target distance (Å)',
                 'initial_angstrom': 'Initial distance (Å)', 'result_angstrom': 'Result distance (Å)',
                 'delta_angstrom': 'Distance change (Å)', 'initial_plane_angle_deg': 'Initial plane angle (°)',
                 'result_plane_angle_deg': 'Result plane angle (°)'},
    'frontier_orbitals_eV': {'index': 'Spin channel index', 'homo': 'HOMO energy (eV)',
                            'lumo': 'LUMO energy (eV)', 'gap': 'Frontier orbital gap (eV)'},
    'frequency_cm-1': {'index': 'Mode index', 'value': 'Wavenumber (cm⁻¹)'},
    'excitation': {'state': 'State index', 'energy_ev': 'Excitation energy (eV)',
                   'wavelength_nm': 'Wavelength (nm)', 'oscillator_strength': 'Oscillator strength'},
    'atomic_displacement_angstrom': {'index': 'Atom index', 'value': 'Aligned displacement (Å)'},
}


def numeric_datasets(result, analysis, source, final):
    def indexed(values):
        return [{'index': i, 'value': value} for i, value in enumerate(values, 1)]

    energy = result.get('scf_energies_hartree', [])
    tables = {
        'scf': [{'index': i, 'energy_Eh': value, 'step_Eh': value-energy[i-2]}
                for i, value in enumerate(energy, 1) if i > 1],
        'scf_energy': indexed(energy),
        'mulliken': result.get('mulliken_charges', []),
        'frequency_cm-1': indexed(result.get('frequencies_cm1', [])),
        'excitation': result.get('excited_states', []),
        'contacts': [
            {**row, 'index': i, 'delta_angstrom': row['result_angstrom']-row['initial_angstrom']}
            for i, row in enumerate(analysis.get('placements', []), 1)
            if row['result_angstrom'] is not None
        ],
    }
    tables['frontier_orbitals_eV'] = [
        {'index': index, 'spin': spin, 'homo': result[f'{spin}_homo_ev'],
         'lumo': result[f'{spin}_lumo_ev'], 'gap': result[f'{spin}_gap_ev']}
        for index, spin in enumerate(('alpha', 'beta'), 1)
        if all(result.get(f'{spin}_{term}_ev') is not None for term in ('homo','lumo','gap'))
    ]
    if final is not None and result.get('optimization_converged'):
        p, q = align_coordinates(source, final)
        tables['atomic_displacement_angstrom'] = indexed(np.linalg.norm(q-p, axis=1).tolist())
    if tables['mulliken'] and source.components:
        charges = {row['index']: row['charge'] for row in tables['mulliken']}
        fragments = tables['fragment_charge'] = []
        for index, part in enumerate(source.components, 1):
            atoms = range(part['start']+1, part['start']+part['count']+1)
            if all(k in charges for k in atoms):
                fragments.append(dict(index=index, fragment=part['name'], charge=sum(charges[k] for k in atoms)))
    return {name: rows for name, rows in tables.items() if rows}


def draw(folder, spec, datasets, number):
    x_key, y_key, kind = spec['x'], spec['y'], spec['kind']
    rows = datasets[spec['dataset']]
    ys = np.asarray([row[y_key] for row in rows], dtype=float)
    xs = None if kind == 'histogram' else np.asarray([row[x_key] for row in rows], dtype=float)
    if ys.ndim != 1 or not np.isfinite(ys).all() or (xs is not None and
            (xs.ndim != 1 or not np.isfinite(xs).all())):
        raise ValueError('图件数据含无效数值。')
    figure = Figure(figsize=(7.2, 4.4), layout='constrained')
    ax = figure.add_subplot()
    color = '#225c87'
    labels = AXIS_LABELS[spec['dataset']]
    if kind == 'histogram':
        ax.hist(ys, bins='auto', color=color)
        ax.set(xlabel=labels.get(y_key,y_key), ylabel='Count')
    elif kind == 'stem':
        ax.vlines(xs, 0, ys, color=color)
    else:
        plot = {'line': ax.plot, 'scatter': ax.scatter, 'bar': ax.bar}[kind]
        plot(xs, ys, color=color)
    if xs is not None:
        ax.set(xlabel=labels.get(x_key,x_key), ylabel=labels.get(y_key,y_key))
    ax.grid(axis='y', color='#e3e8ed')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    name = save_figure(folder, f'figure-{number:02}', figure)
    keys = [y_key] if xs is None else [x_key,y_key]
    write_table(folder, name+'.csv', [labels.get(key,key) for key in keys],
                ([row[key] for key in keys] for row in rows))
    return name


def generate_ai_analysis(settings, folder, job, cancel=None):
    output = folder / '输出'
    result = json.loads((folder / 'result.json').read_text('utf-8'))
    analysis = json.loads((output / '分析.json').read_text('utf-8'))
    source = Structure.from_dict(job['structure'])
    final = result_geometry(job, result) if result.get('normal_termination') and result.get('geometry') else None
    datasets = numeric_datasets(result, analysis, source, final)
    tables = {}
    for key, rows in datasets.items():
        columns = {}
        for column, value in rows[0].items():
            if type(value) in (int,float):
                series = [row[column] for row in rows if type(row.get(column)) in (int,float)]
                columns[column] = dict(min=min(series), max=max(series), mean=sum(series)/len(series))
        tables[key] = {'count': len(rows), 'summary': columns,
                       'sample': rows if len(rows) <= 35 else rows[:30]+rows[-5:]}
    calc = {k: job['calculation'][k] for k in ('method', 'basis', 'task', 'solvent')}
    metadata = job['automation']
    payload = dict(research_request=metadata.get('prompt', '')[:12000],
                   model=dict(name=source.name, formula=source.formula(), charge=source.charge,
                              multiplicity=source.multiplicity, atom_count=len(source.atoms)),
                   calculation=calc, analysis=analysis, datasets=tables,
                   metrics={k: v for k, v in result.items() if isinstance(v, (int,float,bool)) or k=='warnings'})
    plan = structured_request(settings, INSTRUCTIONS, json.dumps(payload,ensure_ascii=False),
                              SCHEMA, json.loads, cancel=cancel)
    if not isinstance(plan, dict) or not isinstance(plan.get('report'), str) or not plan['report'].strip():
        raise ValueError('AI 没有返回有效分析报告。')
    if not isinstance(plan.get('figures'), list) or len(plan['figures']) > 12:
        raise ValueError('AI 返回的图件计划无效。')
    destination = output / 'AI分析'
    destination.mkdir(exist_ok=True)
    # Stage a complete version before publishing a report that links to its figures.
    with tempfile.TemporaryDirectory(prefix='.draft-', dir=destination) as temp:
        stage = Path(temp)
        figures, rejected = [], []
        for spec in plan['figures']:
            try:
                stem = draw(stage, spec, datasets, len(figures)+1)
                figures.append((stem, spec['title'], spec['caption']))
            except (IndexError, KeyError, TypeError, ValueError) as error:
                rejected.append(str(error))
        provider = settings.get('model') or settings.get('provider')
        lines = [f'# {source.name} AI 分析', '', f'作业编号 {job["id"]} · {provider}', '',
                 '[计算报告与原始数据](../../计算报告.md)', '', plan['report'].strip(), '', '## AI 补充图件', '']
        for i, (stem, title, caption) in enumerate(figures, 1):
            lines.extend([f'### AI 图 {i:02} {title}', '', f'![AI 图 {i:02}]({stem}.png)', '',
                          caption, '', f'[下载 SVG]({stem}.svg) · [作图数据 CSV]({stem}.csv)', ''])
        if rejected:
            lines.append(f'{len(rejected)} 个补充图方案引用了无效数据列或图型，未生成对应图件。')
        (stage/'AI报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        atomic_json(stage/'作图方案.json', plan)
        revision = destination / ('run-' + uuid.uuid4().hex[:12])
        stage.rename(revision)
    return len(figures), len(rejected), str((revision/'AI报告.md').relative_to(output))


def record_ai_status(folder, **fields):
    atomic_json(folder/'输出'/'AI分析状态.json', fields)
