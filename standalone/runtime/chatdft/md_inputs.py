"""Reconstructed Li 2024 PDC inputs with provenance and explicit force-field choices."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from .storage import atomic_json


ASSETS = Path(__file__).resolve().parents[1] / 'assets' / 'pdc2024'
COMMIT = 'fb868ea2646ecf98aa4faed389a5ef64528069a5'
PDC_PAPER_DOI = '10.1021/acsomega.4c05431'
PDC_PAPER_PROMPT = '复刻 Li 2024 PDC 论文，按论文参数准备全部 21 个体系的 GROMACS 输入。'
IONS = {'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
        'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2}


def catalog():
    return sorted(str(p.parent.relative_to(ASSETS)).replace('\\', '/')
                  for p in ASSETS.rglob('poly_solv.top'))


def identifies_pdc_paper(text):
    text = ' '.join(text.casefold().split())
    return PDC_PAPER_DOI in text or all(term in text for term in
                                       ('acid dissociation', 'ionic solutions', 'pyrone', 'dicarboxylic'))


def rows(text, section):
    current = ''
    for line in text.splitlines():
        line = line.split(';', 1)[0].strip()
        match = re.fullmatch(r'\[\s*(\w+)\s*\]', line)
        if match:
            current = match[1].lower()
        elif line and not line.startswith('#') and current == section:
            yield line.split()


def clean_sections(text):
    """Keep molecule definitions, excluding embedded global atomtype declarations."""
    output, keep = [], False
    for line in text.splitlines():
        match = re.match(r'\[\s*(\w+)\s*\]', line.strip())
        if match:
            keep = match[1].lower() not in ('atomtypes', 'defaults', 'nonbond_params', 'system', 'molecules')
        if keep and not line.strip().startswith('#'):
            output.append(line)
    return '\n'.join(output) + '\n'


@dataclass
class MDConfig:
    system: str = 'PDC_neutral'
    molecules: int = 100
    box_nm: float = 6.0
    temperature_k: float = 293.15
    pressure_bar: float = 1.0
    timestep_fs: float = 2.0
    equilibration_ns: float = 2.0
    production_ns: float = 100.0
    output_ps: float = 1000.0
    cores: int = 8
    seed: int = 20261118
    stable_after_ns: float = 20.0
    distance_nm: float = 0.31
    angle_deg: float = 20.0

    def validate(self):
        if self.system not in catalog():
            raise ValueError('请选择已固定版本的论文体系。')
        if not 1 <= self.molecules <= 1000 or not 2.6 <= self.box_nm <= 30:
            raise ValueError('分子数量或盒子尺寸超出支持范围。')
        if not 1 <= self.cores <= 16 or not 0 < self.timestep_fs <= 2:
            raise ValueError('核数需为 1–16，时间步需大于 0 且不超过 2 fs。')
        values = asdict(self)
        if any(not math.isfinite(v) for v in values.values() if isinstance(v, (float, int))):
            raise ValueError('模拟参数必须为有限数值。')
        if not (0 < self.equilibration_ns <= 100 and 0 < self.production_ns <= 1000
                and 0 < self.output_ps <= self.production_ns * 1000
                and 0 < self.temperature_k <= 1000 and 0 < self.pressure_bar <= 1000
                and 0 <= self.stable_after_ns < self.production_ns
                and 0 < self.distance_nm < self.box_nm / 2 and 0 < self.angle_deg <= 90):
            raise ValueError('请检查温压、模拟时长、采样间隔和分析阈值。稳定区间起点需早于生产结束时间。')
        state, ion = system_state(self.system)
        if ion and state * self.molecules % IONS[ion]:
            raise ValueError('分子数与二价离子数不能形成整数电中和体系。')


def system_state(system):
    if system == 'PDC_neutral':
        return 0, None
    return int(system[3]), system.split('/')[-1].split('_')[1]


def paper_config(system):
    _, ion = system_state(system)
    return MDConfig(system=system, stable_after_ns=30 if IONS.get(ion)==2 else 20)


def template(system):
    base = ASSETS / system
    state, ion = system_state(system)
    filename = 'product_PDC.itp' if state == 0 else 'PDC_ion.itp'
    text = (base / filename).read_text('utf-8-sig')
    atoms = list(rows(text, 'atoms'))
    bonds = list(rows(text, 'bonds'))
    atomtypes = {}
    # Embedded definitions take precedence, as in the author's PDC2 monovalent inputs.
    inputs = sorted(base.glob('*.prm')) + [base / filename]
    for path in inputs:
        for row in rows(path.read_text('utf-8-sig'), 'atomtypes'):
            atomtypes[row[0]] = row
    missing = {row[1] for row in atoms} - set(atomtypes)
    if missing:
        raise ValueError('作者参数缺少原子类型：' + ', '.join(sorted(missing)))
    charge = sum(float(row[6]) for row in atoms)
    if abs(charge + state) > .002:
        raise ValueError('作者 PDC 拓扑电荷与所选质子化状态不符。')
    if ion and ion.upper() not in atomtypes:
        alias = next((row for key, row in atomtypes.items() if key.casefold() == ion.casefold()), None)
        if alias:
            atomtypes[ion.upper()] = [ion.upper(), *alias[1:]]
    return base, text, atoms, bonds, atomtypes, state, ion


def solute_geometry(atoms, bonds, charge, seed):
    molecule = Chem.RWMol()
    for row in atoms:
        mass = float(row[7])
        element = 'H' if mass < 2 else 'C' if mass < 13 else 'O'
        molecule.AddAtom(Chem.Atom(element))
    for row in bonds:
        molecule.AddBond(int(row[0]) - 1, int(row[1]) - 1, Chem.BondType.SINGLE)
    molecule = molecule.GetMol()
    molecule.AddConformer(Chem.Conformer(len(atoms)))
    rdDetermineBonds.DetermineBondOrders(molecule, charge=charge, allowChargedFragments=True, embedChiral=False)
    Chem.SanitizeMol(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    if AllChem.EmbedMolecule(molecule, parameters) < 0:
        raise ValueError('无法从作者拓扑生成 PDC 初始构象。')
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=1000)
    else:
        AllChem.UFFOptimizeMolecule(molecule, maxIters=1000)
    xyz = np.asarray(molecule.GetConformer().GetPositions()) / 10
    xyz -= xyz.mean(axis=0)
    return xyz


def write_gro(path, name, atoms, xyz, box=6.0):
    lines = ['ChatDFT ' + name, str(len(atoms))]
    for index, (row, position) in enumerate(zip(atoms, xyz), 1):
        atom_name = row[4]
        lines.append(f'{1:5d}{name[:5]:<5}{atom_name[:5]:>5}{index:5d}'
                     + ''.join(f'{value:8.3f}' for value in position))
    lines.append(f'{box:.6f} {box:.6f} {box:.6f}')
    path.write_text('\n'.join(lines) + '\n', encoding='ascii')


def make_mdp(config, stage, tabulated=False):
    c = config
    dt = c.timestep_fs / 1000
    data = dict(integrator='steep' if stage == 'minimize' else 'md',
                nsteps=2000 if stage == 'minimize' else round(
                    (c.equilibration_ns if stage == 'equilibrate' else c.production_ns) * 1000 / dt),
                dt=dt, emtol=500, emstep=.005, pbc='xyz',
                **{'cutoff-scheme': 'group' if tabulated else 'Verlet',
                   'nstlist': 1 if tabulated else 20, 'rlist': 1.2,
                   'coulombtype': 'PME', 'rcoulomb': 1.2,
                   'fourierspacing': .14, 'pme-order': 4,
                   'vdwtype': 'User' if tabulated else 'Cut-off', 'rvdw': 1.2,
                   'vdw-modifier': 'None' if tabulated else 'Potential-shift',
                   'DispCorr': 'no', 'constraints': 'h-bonds', 'constraint-algorithm': 'lincs',
                   'lincs-order': 4, 'lincs-iter': 1})
    if stage != 'minimize':
        data.update({'tcoupl': 'V-rescale' if stage == 'equilibrate' else 'Nose-Hoover',
                     'tc-grps': 'System', 'tau-t': 2, 'ref-t': c.temperature_k,
                     'pcoupl': 'Berendsen' if stage == 'equilibrate' else 'Parrinello-Rahman',
                     'pcoupltype': 'isotropic', 'tau-p': 2, 'ref-p': c.pressure_bar,
                     'compressibility': '4.5e-5', 'gen-vel': 'yes' if stage == 'equilibrate' else 'no',
                     'gen-temp': c.temperature_k, 'gen-seed': c.seed,
                     'continuation': 'no' if stage == 'equilibrate' else 'yes',
                     'nstxout-compressed': max(1, round(c.output_ps / dt)),
                     'nstxout': 0, 'nstvout': 0, 'nstenergy': 500,
                     'nstlog': 500, 'nstcalcenergy': 100})
    if tabulated:
        data.update({'energygrps': ' '.join(c4_groups(c.system)),
                     'energygrp-table': ' '.join(f'{a} {b}' for a, b in c4_pairs(c.system))})
    return '\n'.join(f'{key} = {value}' for key, value in data.items()) + '\n'


def c4_groups(system):
    _, _, atoms, _, _, _, ion = template(system)
    return ['ION'] + ['T_' + value for value in sorted({a[1] for a in atoms})] + ['T_OW', 'T_HW'] if ion else []


def c4_pairs(system):
    return [('ION', name) for name in c4_groups(system)[1:]]


def literature_c4(system):
    """Equation 4/Table 2, Li & Merz; uniform-kappa reconstruction, not author tables."""
    ion = system_state(system)[1]
    values = {'Be': 9.800, 'Mg': .987, 'Ca': .230, 'Sr': .137, 'Ba': .072}
    if ion not in values:
        return None
    return dict(source='Li & Merz, JCTC 2014, Eq. 4 / Table 2 (SPC/E), DOI:10.1021/ct400751u',
                units='nm^-2', kappa={group: values[ion]*100 for _, group in c4_pairs(system)},
                method='Uniform ion kappa for all ion–PDC/water pairs; C4=kappa*C6.',
                warning='Reconstructed from cited literature. Original PDC author table files are unavailable; transfer to PDC atom types requires author confirmation.')


def write_table(path, kappa=0.0, cutoff=1.2):
    # GROMACS evaluates C6*g(r) + C12*h(r). kappa is C4/C6 in nm^-2.
    # Shift the full VdW potential to zero at cutoff, preserving its derivative.
    with Path(path).open('w', encoding='ascii') as stream:
        stream.write('# r f -df/dr g -dg/dr h -dh/dr; r in nm; kappa=C4/C6 nm^-2\n')
        # The legacy engine requires cutoff + table-extension (default 1 nm).
        for i in range(3001):
            r = i * .001
            if r < .01:
                values = (r, 0, 0, 0, 0, 0, 0)
            else:
                inside = cutoff is None or r < cutoff
                shift_g = cutoff**-6+kappa*cutoff**-4 if cutoff else 0
                shift_h = cutoff**-12 if cutoff else 0
                g = -r**-6 - kappa * r**-4 + shift_g if inside else 0
                dg = -6 * r**-7 - 4 * kappa * r**-5 if inside else 0
                h = r**-12 - shift_h if inside else 0
                dh = 12 * r**-13 if inside else 0
                values = (r, 1/r, 1/r**2, g, dg, h, dh)
            stream.write(' '.join(f'{value:.12e}' for value in values) + '\n')


def prepare_inputs(folder, config, c4=None):
    config.validate()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base, text, atoms, bonds, types, state, ion = template(config.system)
    divalent = ion is not None and IONS[ion] == 2
    if divalent:
        required = {name for _, name in c4_pairs(config.system)}
        if not isinstance(c4, dict) or not c4.get('source') or c4.get('units') != 'nm^-2':
            raise ValueError('二价离子需要带出处的 kappa=C4/C6 参数 JSON（单位 nm^-2）。作者仓库未附原始表格势。')
        if set(c4.get('kappa', {})) != required or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in c4['kappa'].values()):
            raise ValueError('kappa 表必须明确包含所有离子–原子类型对，零值也需显式列出。')
    molecule_name = next(rows(text, 'moleculetype'))[0]
    (folder / 'pdc.itp').write_text(clean_sections(text), encoding='ascii')
    # Use the explicitly reported SPC/E water rather than the inconsistent water
    # partial charges and shared hydroxyl types found in some author directories.
    water = '''[ moleculetype ]
SOL 2
[ atoms ]
1 OW 1 SOL OW 1 -0.8476 15.9994
2 HW 1 SOL HW1 1 0.4238 1.008
3 HW 1 SOL HW2 1 0.4238 1.008
[ settles ]
1 1 0.1 0.16330
[ exclusions ]
1 2 3
2 1 3
3 1 2
'''
    if divalent:
        # Legacy group tables require each charge group to belong to one energy group.
        water = water.replace('2 HW 1 SOL HW1 1 ', '2 HW 1 SOL HW1 2 ').replace(
            '3 HW 1 SOL HW2 1 ', '3 HW 1 SOL HW2 3 ')
    (folder / 'water.itp').write_text(water, encoding='ascii')
    top = ['; Reconstructed from pinned author files; see manifest.json', '[ defaults ]',
           f'1 {1 if divalent else 2} yes 0.5 0.8333', '[ atomtypes ]']
    used = {a[1] for a in atoms}
    if ion:
        used.add(ion.upper())
    for key in sorted(used):
        if key not in types:
            raise ValueError('缺少离子原子类型 ' + key)
        top.append(' '.join(types[key]))
    if divalent:
        sigma, epsilon = .316557, .650194
        top.extend([f'OW 8 15.9994 0 A {4*epsilon*sigma**6:.12g} {4*epsilon*sigma**12:.12g}',
                    'HW 1 1.008 0 A 0 0'])
        nonbond = [row for row in rows((base / 'poly_solv.top').read_text(), 'nonbond_params')
                   if row[0] in used and row[1] in used]
        top.append('[ nonbond_params ]')
        top.extend(' '.join(row) for row in nonbond)
        # comb-rule 1 mixes C6/C12 geometrically. Explicit pairs preserve the
        # Lorentz-Berthelot mixing used by the cited Li/Merz ion/water model.
        for key in sorted(used):
            c6, c12 = map(float, types[key][-2:])
            if c6 > 0 and c12 > 0:
                other_sigma, other_epsilon = (c12/c6)**(1/6), c6*c6/(4*c12)
                mixed_sigma = (sigma+other_sigma)/2
                mixed_epsilon = math.sqrt(epsilon*other_epsilon)
                a, b = 4*mixed_epsilon*mixed_sigma**6, 4*mixed_epsilon*mixed_sigma**12
            else:
                a, b = 0, 0
            top.append(f'{key} OW 1 {a:.12g} {b:.12g}')
    else:
        top.extend(['OW 8 15.9994 0 A 0.316557 0.650194', 'HW 1 1.008 0 A 0 0'])
    top.extend(['#include "pdc.itp"', '#include "water.itp"'])
    if ion:
        top.extend(['[ moleculetype ]', f'{ion.upper()} 1', '[ atoms ]',
                    f'1 {ion.upper()} 1 {ion.upper()} {ion.upper()} 1 {IONS[ion]}',
                    '[ system ]', 'PDC in SPC/E water', '[ molecules ]',
                    f'{molecule_name} {config.molecules}', f'{ion.upper()} {state*config.molecules//IONS[ion]}'])
        write_gro(folder / 'ion.gro', ion.upper(), [[None, None, None, None, ion.upper()]], [[0, 0, 0]])
    else:
        top.extend(['[ system ]', 'PDC in SPC/E water', '[ molecules ]', f'{molecule_name} {config.molecules}'])
    (folder / 'topol.top').write_text('\n'.join(top) + '\n', encoding='ascii')
    write_gro(folder / 'solute.gro', molecule_name, atoms,
              solute_geometry(atoms, bonds, -state, config.seed), config.box_nm)
    for stage in ('minimize', 'equilibrate', 'production'):
        (folder / f'{stage}.mdp').write_text(make_mdp(config, stage, divalent), encoding='ascii')
    warnings = ['初始坐标由作者键连接重新生成，随机种子和原作者初始构型可能不同。',
                '采用正文 SPC/E：独立 OW/HW 类型、SPC/E 电荷和刚性几何。作者部分水文件与正文不一致。',
                '水数量由重新溶剂化得到，以实际生成的体系和浓度为准。',
                '保留正文 tau-t=tau-p=2 ps。GROMACS 的温压耦合共振警告将逐项记录，仅此已识别警告允许继续；需检查生产阶段稳定性。']
    if divalent:
        warnings.append('表格势使用 GROMACS 2018 group 截断；正文 Verlet 与用户表格势不兼容。')
        warnings.append('水氧交叉 LJ 显式采用 Lorentz-Berthelot 混合；为逐类型表格势，将水的 charge group 拆为逐原子。')
        atomic_json(folder / 'c4.json', c4)
        write_table(folder / 'table.xvg')
        if c4.get('warning'):
            warnings.append(c4['warning'])
        write_table(folder / 'tablep.xvg', cutoff=None)
        for _, group in c4_pairs(config.system):
            write_table(folder / f'table_ION_{group}.xvg', c4['kappa'][group])
    manifest = dict(format='ChatDFT MD', version=1, source_commit=COMMIT,
                    paper_doi='10.1021/acsomega.4c05431', config=asdict(config),
                    molecule_name=molecule_name, solute_atoms=len(atoms), ion=ion,
                    solute_atom_names=[a[4] for a in atoms],
                    ion_count=state*config.molecules//IONS[ion] if ion else 0,
                    ring_atoms=[i for i, a in enumerate(atoms) if a[4] in ('C8', 'C9', 'C10')]
                    if state else [7, 8, 9], warnings=warnings,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in base.iterdir() if p.is_file()},
                    tabulated=divalent, c4_source=c4.get('source') if divalent else None)
    atomic_json(folder / 'manifest.json', manifest)
    return manifest
