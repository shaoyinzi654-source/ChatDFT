from pathlib import Path
import xml.etree.ElementTree as ET

from rdkit import Chem

from .chemistry import Structure, from_xyz, validate_structure


def read_structure(path: Path, charge: int, multiplicity: int) -> Structure:
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError('结构文件超过 10 MB。')
    if path.suffix.lower() != '.xsd':
        return from_xyz(path.read_text('utf-8-sig'), path.stem, charge, multiplicity)

    # MS writes an empty DOCTYPE; reject entity definitions and external DTDs.
    raw = path.read_bytes()
    xml_signatures = raw.replace(b'\x00', b'').upper()
    if b'<!ENTITY' in xml_signatures:
        raise ValueError('XSD 文件包含不支持的 XML 实体。')
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError('无法读取 Materials Studio XSD 文件。') from exc
    if root.tag != 'XSD' or root.find('AtomisticTreeRoot') is None:
        raise ValueError('请选择 Materials Studio 的 Atomistic XSD 结构文件。')
    if any(node.tag in {'SymmetrySystem', 'SpaceGroup', 'Lattice3D'} for node in root.iter()):
        raise ValueError('该 XSD 包含周期结构。请先导出非周期分子或簇模型，再交给当前分子计算模块。')

    atoms, bonds, indices = [], [], {}
    periodic_table = Chem.GetPeriodicTable()
    try:
        for node in root.iter('Atom3d'):
            element = node.attrib['Components']
            if not element.isalpha() or not periodic_table.GetAtomicNumber(element):
                raise ValueError(f'不支持的原子组分 {element!r}。')
            xyz = node.get('XYZ', '0,0,0').split(',')
            x, y, z = (float(value) for value in xyz)
            atom_id = node.attrib['ID']
            if atom_id in indices:
                raise ValueError('XSD 中存在重复的原子编号。')
            indices[atom_id] = len(atoms)
            atoms.append(dict(element=element, x=x, y=y, z=z))
        bond_types = {'Single': 1, 'Double': 2, 'Triple': 3, 'Aromatic': 1.5, 'PartialDouble': 1.5}
        seen = set()
        for node in root.iter('Bond'):
            if 'Connects' not in node.attrib:
                continue
            a, b = (indices[key.strip()] for key in node.attrib['Connects'].split(','))
            pair = tuple(sorted((a, b)))
            if a == b or pair in seen:
                continue
            kind = node.get('Type', 'Single')
            if kind not in bond_types:
                raise ValueError(f'暂不支持 XSD 键类型 {kind!r}。')
            seen.add(pair)
            bonds.append([a, b, bond_types[kind]])
    except (KeyError, RuntimeError) as exc:
        raise ValueError('XSD 原子或键记录不完整。') from exc
    # XSD atomic charges are commonly partial charges, not the requested molecular charge.
    structure = Structure(path.stem, atoms, bonds, charge=charge, multiplicity=multiplicity,
                          provenance='Materials Studio XSD / Cartesian Å')
    validate_structure(structure)
    return structure
