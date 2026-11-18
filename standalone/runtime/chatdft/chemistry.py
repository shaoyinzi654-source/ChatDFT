from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


@dataclass
class Structure:
    name: str
    atoms: list[dict]
    bonds: list[list] = field(default_factory=list)
    smiles: str = ""
    charge: int = 0
    multiplicity: int = 1
    provenance: str = "用户导入"
    warnings: list[str] = field(default_factory=list)
    components: list[dict] = field(default_factory=list)
    modeling_recipe: dict = field(default_factory=dict)

    def component_count(self):
        parents = list(range(len(self.atoms)))

        def root(index):
            while parents[index] != index:
                index = parents[index]
            return index

        for first, second, _ in self.bonds:
            parents[root(int(first))] = root(int(second))
        return len({root(index) for index in range(len(parents))})

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})

    def xyz(self):
        return f"{len(self.atoms)}\n{self.name}\n" + "\n".join(
            f"{a['element']:2} {a['x']: .8f} {a['y']: .8f} {a['z']: .8f}" for a in self.atoms) + "\n"

    def formula(self):
        from collections import Counter
        counts = Counter(a['element'] for a in self.atoms)
        order = sorted(counts, key=lambda x: (0 if x == 'C' else 1 if x == 'H' else 2, x))
        return ''.join(e + (str(counts[e]) if counts[e] != 1 else '') for e in order)


LIBRARY = {
    "水 · Water": "O", "苯 · Benzene": "c1ccccc1", "乙醇 · Ethanol": "CCO",
    "苯酚 · Phenol": "Oc1ccccc1", "吡啶 · Pyridine": "c1ccncc1",
    "乙酸 · Acetic acid": "CC(=O)O", "甲醛 · Formaldehyde": "C=O",
    "咖啡因 · Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "水二聚体 · Water dimer": "O.O", "苯–水复合物": "c1ccccc1.O",
}


def validate_structure(structure: Structure):
    if not 1 <= len(structure.atoms) <= 500:
        raise ValueError("结构必须包含 1–500 个原子。")
    pt = Chem.GetPeriodicTable()
    electrons = -int(structure.charge)
    for a in structure.atoms:
        try:
            number = pt.GetAtomicNumber(a['element'])
        except Exception:
            raise ValueError(f"未知元素：{a.get('element')}") from None
        if number < 1:
            raise ValueError("不接受虚原子。")
        electrons += number
        if not all(math.isfinite(float(a[k])) and abs(float(a[k])) <= 10000 for k in ('x', 'y', 'z')):
            raise ValueError("坐标必须是有限数值，单位为 Å。")
    m = int(structure.multiplicity)
    if electrons < 1 or m < 1 or m > electrons + 1 or (electrons - (m - 1)) % 2:
        raise ValueError(f"电子数 {electrons} 与自旋多重度 {m} 不匹配，请检查电荷和自旋。")
    for i, a in enumerate(structure.atoms):
        for b in structure.atoms[i+1:]:
            if sum((float(a[k]) - float(b[k]))**2 for k in ('x', 'y', 'z')) < .16:
                raise ValueError("存在距离小于 0.4 Å 的原子；请先修正重叠。")
    if any(type(i) is not int or type(j) is not int or not 0 <= i < len(structure.atoms)
           or not 0 <= j < len(structure.atoms) or i == j or order not in (1, 1.5, 2, 3)
           for i, j, order in structure.bonds):
        raise ValueError('结构中有无效的键或原子编号。')
    return electrons


def from_smiles(smiles: str, name="新分子", conformers=8) -> Structure:
    if len(smiles) > 3000:
        raise ValueError("SMILES 过长。")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("SMILES 无法解析，请检查价态、环闭合与立体化学。")
    if mol.GetNumAtoms() > 150:
        raise ValueError("自动建模暂限 150 个重原子，更大模型请导入 XYZ。")
    canonical = Chem.MolToSmiles(mol)
    charge = Chem.GetFormalCharge(mol)
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    fragments = Chem.GetMolFrags(mol, asMols=True)
    atoms, bonds, offset, right_edge = [], [], 0, None
    warnings = []
    if len(fragments) > 1:
        warnings.append("多片段模型按几何间距摆放；相对取向是初始猜测，需要人工检查和几何优化。")
    for frag in fragments:
        frag = Chem.AddHs(frag)
        if offset + frag.GetNumAtoms() > 500:
            raise ValueError('加氢后的原子数超过 500。')
        if frag.GetNumAtoms() == 1:
            x = 0. if right_edge is None else right_edge + 2.5
            atoms.append(dict(element=frag.GetAtomWithIdx(0).GetSymbol(), x=x, y=0., z=0.))
            right_edge = x
            offset += 1
            continue
        params = AllChem.ETKDGv3()
        params.randomSeed = 20261117
        params.numThreads = 1
        params.pruneRmsThresh = .3
        ids = list(AllChem.EmbedMultipleConfs(frag, numConfs=max(1, min(20, conformers)), params=params))
        if not ids:
            raise ValueError("无法生成三维构象；请导入经过检查的 XYZ。")
        if AllChem.MMFFHasAllMoleculeParams(frag):
            energies = AllChem.MMFFOptimizeMoleculeConfs(frag, numThreads=1, maxIters=500)
            force_field = "MMFF94"
        elif AllChem.UFFHasAllMoleculeParams(frag):
            energies = AllChem.UFFOptimizeMoleculeConfs(frag, numThreads=1, maxIters=500)
            force_field = "UFF"
        else:
            energies = [(1, 0)] * len(ids)
            force_field = "未预优化"
            warnings.append("存在缺少力场参数的元素；仅生成初始坐标。")
        converged = [(i, e) for i, (status, e) in enumerate(energies) if status == 0]
        idx = min(converged or list(enumerate(e for _, e in energies)), key=lambda t: t[1])[0]
        if not converged:
            warnings.append("力场预优化未收敛，结构仅作初始猜测。")
        conf = frag.GetConformer(ids[idx])
        positions = [[conf.GetAtomPosition(i)[j] for j in range(3)] for i in range(frag.GetNumAtoms())]
        center = [sum(p[j] for p in positions) / len(positions) for j in range(3)]
        positions = [[p[j] - center[j] for j in range(3)] for p in positions]
        shift = 0 if right_edge is None else right_edge + 2.5 - min(p[0] for p in positions)
        for a, p in zip(frag.GetAtoms(), positions):
            atoms.append(dict(element=a.GetSymbol(), x=p[0]+shift, y=p[1], z=p[2]))
        right_edge = max(p[0]+shift for p in positions)
        bonds.extend([[b.GetBeginAtomIdx()+offset, b.GetEndAtomIdx()+offset, b.GetBondTypeAsDouble()] for b in frag.GetBonds()])
        offset += len(positions)
    s = Structure(name, atoms, bonds, canonical, charge, radicals+1,
                  "RDKit ETKDGv3 + 构象筛选 / 力场预优化（非量子计算）", warnings)
    if radicals > 1:
        s.warnings.append("多自由基自旋耦合需人工选择；默认采用高自旋初始值。")
    validate_structure(s)
    return s


def from_xyz(text: str, name="导入结构", charge=0, multiplicity=1):
    lines = text.strip().splitlines()
    try:
        count = int(lines[0].strip())
    except (ValueError, IndexError):
        raise ValueError("XYZ 第一行必须是原子数，第二行为注释。") from None
    if len(lines) != count + 2:
        raise ValueError("XYZ 原子数与坐标行数不一致；一次只支持一个结构。")
    atoms = []
    for line in lines[2:]:
        p = line.split()
        if len(p) != 4:
            raise ValueError("XYZ 每行必须是：元素 X Y Z。")
        atoms.append(dict(zip(['element', 'x', 'y', 'z'], [p[0], *map(float, p[1:])])))
    s = Structure(name or lines[1], atoms, charge=charge, multiplicity=multiplicity)
    # Connectivity is a visual distance heuristic only; never changes charge or coordinates.
    pt = Chem.GetPeriodicTable()
    for i, a in enumerate(atoms):
        for j in range(i):
            b = atoms[j]
            try:
                cutoff = 1.2 * (pt.GetRcovalent(a['element']) + pt.GetRcovalent(b['element']))
                d = math.sqrt(sum((a[k]-b[k])**2 for k in ('x','y','z')))
                if .4 < d < cutoff:
                    s.bonds.append([j, i, 1])
            except Exception:
                pass
    validate_structure(s)
    return s


def combine(a: Structure, b: Structure, gap=2.5):
    import copy
    atoms = copy.deepcopy(a.atoms)
    shift = max(p['x'] for p in a.atoms) + gap - min(p['x'] for p in b.atoms)
    atoms.extend(dict(p, x=p['x']+shift) for p in b.atoms)
    s = Structure(a.name+' + '+b.name, atoms, a.bonds+[[i+len(a.atoms),j+len(a.atoms),o] for i,j,o in b.bonds],
                  charge=a.charge+b.charge, multiplicity=a.multiplicity+b.multiplicity-1,
                  provenance="片段组合模型 / 用户指定间距", warnings=["相对取向未经量子优化；请检查接触距离、自旋耦合和基组叠加误差。"])
    validate_structure(s)
    return s
