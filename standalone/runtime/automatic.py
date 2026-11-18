"""Persist the inputs and export the actual output of one automatic PySCF job."""
import csv
import json

from chatdft.chemistry import Structure
from chatdft.gaussian import Calculation, build_input, report
from chatdft.pyscf_engine import METHODS
from chatdft.storage import atomic_json
from automatic_analysis import summarize_result
from workflow import result_geometry


TERMINAL = ('completed', 'succeeded', 'failed', 'cancelled')
EXPORT_VERSION = 2


def automatic_calculation(model, requested, resources):
    default = dict(method='B3LYP', basis='6-31G(d)', task='sp', solvent='Gas')
    calculation = Calculation(**(requested or default), **resources)
    build_input(model, calculation)
    if calculation.method not in METHODS or calculation.dispersion:
        raise ValueError('当前 PySCF 服务器不支持所选方法或附加色散校正。')
    if calculation.method == 'MP2' and (calculation.task, calculation.solvent) != ('sp', 'Gas'):
        raise ValueError('MP2 仅支持气相单点。')
    if calculation.task not in ('sp', 'opt', 'optfreq', 'freq', 'td'):
        raise ValueError('当前 PySCF 服务器不支持所选任务类型。')
    return calculation


def save_structure(folder, stem, structure, calculation):
    (folder / f'{stem}.xyz').write_text(structure.xyz(), encoding='utf-8')
    atomic_json(folder / f'{stem}.chatdft.json',
                dict(format='ChatDFT', version=1, structure=structure.to_dict(), calculation=calculation))


def write_table(folder, name, header, rows):
    with (folder / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def report_current(folder, state):
    receipt = folder / '导出状态.json'
    if receipt.is_file():
        try:
            info = json.loads(receipt.read_text('utf-8'))
            return (isinstance(info, dict) and info.get('status') == state and info.get('version') == EXPORT_VERSION
                    and (folder / '计算报告.md').is_file())
        except (ValueError, OSError):
            pass
    return False


def needs_export(folder, job):
    output = folder / '输出'
    if not (output / '初始结构.chatdft.json').is_file():
        return True
    state = job['status']
    return state in TERMINAL and not report_current(output, state) and (
        state not in ('completed', 'succeeded') or (folder / 'result.json').is_file())


def export_job(folder, job):
    """Return the report path only when a terminal report has just been written."""
    folder = folder / '输出'
    folder.mkdir(exist_ok=True)
    structure = Structure.from_dict(job['structure'])
    calculation = job['calculation']
    if not (folder / '初始结构.chatdft.json').exists():
        save_structure(folder, '初始结构', structure, calculation)

    state = job['status']
    if state not in TERMINAL or report_current(folder, state):
        return None

    details = [
        f'# {structure.name}', '',
        f'- 作业编号 {job["id"]}',
        f'- 原子数 {len(structure.atoms)}，电荷 {structure.charge}，多重度 {structure.multiplicity}',
        f'- PySCF {calculation["method"]}/{calculation["basis"]} · {calculation["task"]} · {calculation["solvent"]}',
        '- [初始结构 XYZ](初始结构.xyz) · [PySCF 输入 JSON](../input.json) · [输入模板](../input.gjf)',
        '', '## 建模说明', '', job['automation'].get('description', ''),
    ]
    if structure.warnings:
        details.extend(['', *[f'- {warning}' for warning in structure.warnings]])

    from automatic_plots import result_figure, structure_figure
    final = None
    optimized = False
    result_plot = None
    if state in ('completed', 'succeeded'):
        source = folder.parent / 'result.json'
        if not source.is_file():
            return None
        result = json.loads(source.read_text('utf-8'))
        if result.get('normal_termination') and result.get('geometry'):
            final = result_geometry(job, result)
            optimized = bool(result.get('optimization_converged'))
            stem = '优化结构' if optimized else '计算结构'
            save_structure(folder, stem, final, calculation)
            details.extend(['', f'[{stem} XYZ]({stem}.xyz) · [{stem}项目]({stem}.chatdft.json)'])
        analysis, notes = summarize_result(result, calculation, structure, final)
        atomic_json(folder / '分析.json', analysis)
        details.extend(['', '## 自动分析', '', *notes, '',
                        '[分析 JSON](分析.json) · [原始结果 JSON](../result.json)',
                        '', report(result, 'PySCF 实际结果')])
        placement_header = ('类型', '移动片段', '移动锚点', '参考片段', '参考锚点', '设定距离 / Å',
                            '初始距离 / Å', '结果距离 / Å', '初始面夹角 / °', '结果面夹角 / °')
        metrics = (
            ('电子能', 'final_energy_hartree', 'Eh'), ('偶极矩', 'dipole_debye', 'D'),
            ('零点能', 'zero_point_hartree', 'Eh'), ('焓', 'enthalpy_hartree', 'Eh'),
            ('吉布斯自由能', 'gibbs_hartree', 'Eh'),
        ) + tuple((f'{spin} {orbital}', f'{spin}_{orbital.lower()}_ev', 'eV')
                  for spin in ('alpha', 'beta') for orbital in ('HOMO', 'LUMO', 'Gap'))
        tables = (
            ('几何测量.csv', placement_header, [list(row.values()) for row in analysis['placements']]),
            ('汇总.csv', ('指标', '数值', '单位'),
             [(name, result[key], unit) for name, key, unit in metrics if result.get(key) is not None]),
            ('SCF能量.csv', ('记录序号', '电子能 / Eh'),
             list(enumerate(result.get('scf_energies_hartree', []), 1))),
            ('原子电荷.csv', ('原子序号', '元素', 'Mulliken 电荷 / e'),
             [(row['index'], row['element'], row['charge']) for row in result.get('mulliken_charges', [])]),
            ('振动频率.csv', ('模式序号', '波数 / cm⁻¹'),
             list(enumerate(result.get('frequencies_cm1', []), 1))),
            ('激发态.csv', ('态序号', '能量 / eV', '波长 / nm', '振子强度'),
             [(row['state'], row['energy_ev'], row['wavelength_nm'], row['oscillator_strength'])
              for row in result.get('excited_states', [])]),
        )
        for name, header, rows in tables:
            if rows:
                write_table(folder, name, header, rows)
                details.append(f'- [{name}]({name})')
        result_plot = result_figure(folder, result)
    else:
        status = '作业失败' if state == 'failed' else '作业已取消'
        details.extend(['', f'## {status}', '', job.get('message') or status])
    for filename, label in [('output.log', '计算日志'), ('worker.log', '运行日志'),
                            ('calculation.chk', 'PySCF 检查点')]:
        if (folder.parent / filename).is_file():
            details.extend(['', f'[{label}](../{filename})'])
    geometry_plot = structure_figure(folder, structure, final if optimized else None)
    details.extend(['', '## 图件', '', f'![结构图]({geometry_plot}.png)', '',
                    f'[结构图 SVG]({geometry_plot}.svg)'])
    if result_plot:
        details.extend(['', f'![计算结果图]({result_plot}.png)', '',
                        f'[计算结果图 SVG]({result_plot}.svg)'])

    report_path = folder / '计算报告.md'
    report_path.write_text('\n'.join(details) + '\n', encoding='utf-8')
    atomic_json(folder / '导出状态.json', dict(status=state, version=EXPORT_VERSION, report=report_path.name))
    return report_path
