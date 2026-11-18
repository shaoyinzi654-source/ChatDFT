from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from .chemistry import Structure, validate_structure

METHODS = ['B3LYP', 'PBE1PBE', 'wB97XD', 'M062X', 'HF', 'MP2', 'CAM-B3LYP']
BASES = ['6-31G(d)', '6-31+G(d,p)', '6-311+G(d,p)', 'Def2SVP', 'Def2TZVP', 'cc-pVDZ', 'STO-3G']
TASKS = {'optfreq': 'Opt Freq', 'opt': 'Opt', 'sp': 'SP', 'freq': 'Freq',
         'td': 'TD(NStates=10)', 'nmr': 'NMR=GIAO', 'ts': 'Opt=(TS,CalcFC,NoEigenTest) Freq'}
SOLVENTS = ['Gas', 'Water', 'Ethanol', 'Methanol', 'Acetonitrile', 'DMSO', 'Toluene', 'Chloroform', 'THF']


@dataclass
class Calculation:
    method: str = 'B3LYP'
    basis: str = '6-31G(d)'
    task: str = 'optfreq'
    solvent: str = 'Gas'
    cores: int = 8
    memory_gb: int = 8
    dispersion: bool = False

    def to_dict(self):
        return asdict(self)


def build_input(s: Structure, calc: Calculation):
    validate_structure(s)
    if calc.method not in METHODS or calc.basis not in BASES or calc.task not in TASKS or calc.solvent not in SOLVENTS:
        raise ValueError('计算配置包含尚未支持的关键词，请从配置列表选择。')
    if not 1 <= calc.cores <= 16 or not 1 <= calc.memory_gb <= 14:
        raise ValueError('当前服务器配置限 1–16 核、1–14 GB Gaussian 内存。')
    heavy = [a['element'] for a in s.atoms if __import__('rdkit').Chem.GetPeriodicTable().GetAtomicNumber(a['element']) > 36]
    if heavy:
        raise ValueError('当前内置模板暂限 Kr 及以前元素；重元素需单独指定并核对 ECP / 基组。')
    if calc.dispersion and calc.method not in ['B3LYP', 'PBE1PBE', 'CAM-B3LYP']:
        raise ValueError('该方法的附加 D3BJ 组合未验证；wB97XD 自带色散，请关闭额外 D3BJ。')
    route = f'#p {calc.method}/{calc.basis} {TASKS[calc.task]} SCF=(Tight,XQC) Integral=UltraFine NoSymm'
    if calc.method in ['HF', 'MP2']:
        route = route.replace(' Integral=UltraFine', '')
    if calc.solvent != 'Gas':
        route += f' SCRF=(SMD,Solvent={calc.solvent})'
    if calc.dispersion:
        route += ' EmpiricalDispersion=GD3BJ'
    # Fixed checkpoint basename and generated routes prevent file path / route injection.
    title = re.sub(r'[^\x20-\x7e]', '_', s.name).replace('%', '_')[:100] or 'ChatDFT calculation'
    coords = '\n'.join(f"{a['element']:2} {a['x']: .8f} {a['y']: .8f} {a['z']: .8f}" for a in s.atoms)
    return f'%chk=calculation.chk\n%nprocshared={calc.cores}\n%mem={calc.memory_gb}GB\n{route}\n\n{title}\n\n{s.charge} {s.multiplicity}\n{coords}\n\n'


NUMBER = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[DEde][-+]?\d+)?'


def number(value):
    return float(value.replace('D', 'E').replace('d', 'e'))


def parse_log(text: str, task: str = ''):
    energies = [number(x) for x in re.findall(r'SCF Done:.*?=\s*('+NUMBER+r')', text)]
    mp2 = re.findall(r'EUMP2\s*=\s*('+NUMBER+r')', text)
    freq, ir = [], []
    # Only the final frequency section, when Gaussian prints repeated frequency jobs.
    frequency_sections = re.split(r'Harmonic frequencies\s*\(', text)
    ftext = frequency_sections[-1] if len(frequency_sections) > 1 else text
    for line in ftext.splitlines():
        if 'Frequencies --' in line:
            freq.extend(number(x) for x in re.findall(NUMBER, line.split('--',1)[1]))
        elif 'IR Inten' in line and '--' in line:
            ir.extend(number(x) for x in re.findall(NUMBER, line.split('--',1)[1]))
    normal_pos = text.rfind('Normal termination of Gaussian')
    error_pos = text.rfind('Error termination')
    normal = bool(re.search(r'Normal termination of Gaussian[^\r\n]*\s*\Z', text)) and normal_pos > error_pos
    final_energy_records = [(m.start(),number(m[1])) for m in re.finditer(r'SCF Done:.*?=\s*('+NUMBER+r')',text)]
    final_energy_records += [(m.start(),number(m[1])) for m in re.finditer(r'EUMP2\s*=\s*('+NUMBER+r')',text)]
    result = dict(normal_termination=normal, scf_energies_hartree=energies,
                  final_energy_hartree=max(final_energy_records)[1] if final_energy_records else None,
                  frequencies_cm1=freq, ir_intensities_km_mol=ir,
                  imaginary_frequencies=[f for f in freq if f < -20],
                  optimization_converged='Optimization completed.' in text or 'Stationary point found.' in text,
                  warnings=[], excited_states=[], mulliken_charges=[], geometry=None)
    qm = re.findall(r'Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)', text)
    result['charge'] = int(qm[-1][0]) if qm else None
    result['multiplicity'] = int(qm[-1][1]) if qm else None
    for label, key in [('Zero-point correction=', 'zero_point_hartree'),
                       ('Sum of electronic and thermal Free Energies=', 'gibbs_hartree'),
                       ('Sum of electronic and thermal Enthalpies=', 'enthalpy_hartree')]:
        matches = re.findall(re.escape(label)+r'\s*('+NUMBER+r')', text)
        result[key] = number(matches[-1]) if matches else None
    dipoles = re.findall(r'X=\s*('+NUMBER+r')\s+Y=\s*('+NUMBER+r')\s+Z=\s*('+NUMBER+r')\s+Tot=\s*('+NUMBER+r')', text)
    result['dipole_debye'] = number(dipoles[-1][-1]) if dipoles else None
    for spin in ['Alpha', 'Beta']:
        # Occupied values can span multiple consecutive lines; reset on a new block.
        occupied, virtual, previous = [], [], ''
        for line in text.splitlines():
            m = re.search(spin+r'\s+(occ\.|virt\.) eigenvalues --(.*)', line)
            if m:
                kind = m[1]
                if kind == 'occ.' and previous != 'occ.':
                    occupied, virtual = [], []
                (occupied if kind == 'occ.' else virtual).extend(number(x) for x in re.findall(NUMBER, m[2]))
                previous = kind
            elif not re.search(r'(Alpha|Beta)\s+(occ\.|virt\.) eigenvalues', line):
                previous = ''
        result[spin.lower()+'_homo_ev'] = max(occupied)*27.211386245988 if occupied else None
        result[spin.lower()+'_lumo_ev'] = min(virtual)*27.211386245988 if virtual else None
        result[spin.lower()+'_gap_ev'] = (min(virtual)-max(occupied))*27.211386245988 if occupied and virtual else None
    for m in re.finditer(r'Excited State\s+(\d+):\s+(.*?)\s+('+NUMBER+r') eV\s+('+NUMBER+r') nm\s+f=('+NUMBER+r')', text):
        result['excited_states'].append(dict(state=int(m[1]), label=m[2], energy_ev=number(m[3]), wavelength_nm=number(m[4]), oscillator_strength=number(m[5])))
    # Last standard orientation (Angstroms), falling back to input orientation.
    blocks = re.findall(r'(?:Standard|Input) orientation:\s*\n\s*-+\n.*?\n.*?\n\s*-+\n(.*?)\n\s*-+', text, re.S)
    if blocks:
        from rdkit import Chem
        atoms = []
        for line in blocks[-1].splitlines():
            p = line.split()
            if len(p) >= 6 and p[1].isdigit() and int(p[1]) > 0:
                atoms.append(dict(element=Chem.GetPeriodicTable().GetElementSymbol(int(p[1])), x=number(p[-3]), y=number(p[-2]), z=number(p[-1])))
        if atoms:
            result['geometry'] = atoms
    charges = re.findall(r'Mulliken charges(?: and spin densities)?:\s*\n.*?\n(.*?)\n\s*Sum of Mulliken', text, re.S)
    if charges:
        for line in charges[-1].splitlines():
            p = line.split()
            if len(p) >= 3 and p[0].isdigit():
                result['mulliken_charges'].append(dict(index=int(p[0]), element=p[1], charge=number(p[2])))
    if not normal:
        result['warnings'].append('日志没有最终正常终止标记；数值不能视为已完成计算。')
    if task in ['opt', 'optfreq', 'ts'] and not result['optimization_converged']:
        result['warnings'].append('未找到结构优化收敛标记。')
    if task in ['optfreq', 'freq', 'ts'] and not freq:
        result['warnings'].append('没有频率数据，无法判断驻点性质。')
    if result['imaginary_frequencies']:
        result['warnings'].append(f"检测到 {len(result['imaginary_frequencies'])} 个低于 -20 cm⁻¹ 的虚频；需检查振动模式。")
    if any(-20 <= f < 0 for f in freq):
        result['warnings'].append('存在 -20 到 0 cm⁻¹ 的小负频率，需检查数值精度及软模式。')
    if task == 'ts' and len(result['imaginary_frequencies']) != 1:
        result['warnings'].append('过渡态应结合一个显著虚频的模式和 IRC 验证；本次未满足一个显著虚频。')
    if task == 'nmr':
        result['warnings'].append('NMR 输出为绝对屏蔽；化学位移需使用同方法、基组、溶剂的参照物。')
    shieldings = re.findall(r'\s+(\d+)\s+([A-Z][a-z]?)\s+Isotropic\s*=\s*('+NUMBER+r')', text)
    result['nmr_shieldings_ppm'] = [dict(index=int(i), element=e, shielding=number(v)) for i,e,v in shieldings]
    return result


def report(result: dict, title='Gaussian 结果'):
    engine=result.get('engine','Gaussian')
    lines = [f'# {title}', '', f'**数据来源：{engine} 计算输出；未填充缺失数值。**', '',
             f"- 最终正常终止：{'是' if result['normal_termination'] else '否'}",
             f"- 优化收敛标记：{'有' if result['optimization_converged'] else '未检出'}"]
    if result.get('calculation'):
        c=result['calculation'];lines += [f"- 计算级别：{c['method']}/{c['basis']}",f"- 溶剂：{c['solvent']} / {result.get('solvent_model','')}"]
    if result.get('thermochemistry'):lines.append('- 热化学约定：'+result['thermochemistry'])
    if result.get('spin_square') is not None:lines.append(f"- <S²>：{result['spin_square']:.6f}")
    for key, label, unit in [('final_energy_hartree','最终电子能','Hartree'),('gibbs_hartree','热修正 Gibbs 自由能','Hartree'),('enthalpy_hartree','热修正焓','Hartree'),('dipole_debye','偶极矩','D'),('alpha_gap_ev','Alpha HOMO–LUMO 能隙','eV'),('beta_gap_ev','Beta HOMO–LUMO 能隙','eV')]:
        if result.get(key) is not None:
            lines.append(f'- {label}：{result[key]:.8f} {unit}')
    lines += [f"- 频率数量：{len(result['frequencies_cm1'])}", f"- 显著虚频数量（< -20 cm⁻¹）：{len(result['imaginary_frequencies'])}"]
    lines.extend('\n注意：'+w for w in result['warnings'])
    if result['frequencies_cm1']:
        lines += ['', '## 谐振频率 / cm⁻¹', ', '.join(f'{f:.2f}' for f in result['frequencies_cm1'])]
    if result['excited_states']:
        lines += ['', '## 垂直激发态', '|态|能量 / eV|波长 / nm|振子强度|', '|---|---|---|---|']
        lines += [f"|{e['state']}|{e['energy_ev']}|{e['wavelength_nm']}|{e['oscillator_strength']}|" for e in result['excited_states']]
    if result['nmr_shieldings_ppm']:
        lines += ['', '## NMR 绝对屏蔽 / ppm'] + [f"- {x['index']} {x['element']}：{x['shielding']}" for x in result['nmr_shieldings_ppm']]
    lines += ['', '轨道能隙不是光学激发能。不同组成体系的绝对电子能不能直接用来判断稳定性。', '谐振频率与热化学数据受方法、基组、温度、标准态影响；本报告不自动做频率缩放或标准态修正。']
    return '\n'.join(lines)
