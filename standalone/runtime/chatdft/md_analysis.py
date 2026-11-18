"""PBC-aware Li 2024 aggregate metrics, spatial density, and observable comparison."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re

import mdtraj as md
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def minimum_image(displacement, box):
    inverse = np.linalg.inv(box)
    fractional = displacement @ inverse
    fractional -= np.rint(fractional)
    wrapped = fractional @ box
    if np.allclose(box, np.diag(np.diag(box))):
        return wrapped
    # Reduced triclinic GROMACS cells may need a neighboring image after wrapping.
    shifts = np.array([(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)]) @ box
    candidates = wrapped[...,None,:] + shifts
    selected = np.argmin(np.sum(candidates*candidates,axis=-1),axis=-1)
    return np.take_along_axis(candidates,selected[...,None,None],axis=-2)[...,0,:]


def frame_metrics(xyz, box, molecules, ring_atoms, distance=.31, angle=20):
    """Distances use all atoms; ring normals follow original molecule atom order."""
    count = len(molecules)
    if not count:
        raise ValueError('没有 PDC 分子。请核对拓扑与分子数量。')
    normals = []
    for atoms in molecules:
        a, b, c = xyz[np.asarray(atoms)[ring_atoms]]
        first = minimum_image(a - b, box)
        second = minimum_image(c - b, box)
        normal = np.cross(first, second)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            raise ValueError('环平面三个原子共线。请核对原子映射。')
        normals.append(normal / length)
    normals = np.asarray(normals)
    aggregation = np.eye(count, dtype=bool)
    stacking = np.eye(count, dtype=bool)
    for i in range(count):
        for j in range(i + 1, count):
            vector = xyz[molecules[i]][:, None] - xyz[molecules[j]][None, :]
            close = np.min(np.linalg.norm(minimum_image(vector, box), axis=-1)) <= distance
            aggregation[i, j] = aggregation[j, i] = close
            aligned = abs(np.dot(normals[i], normals[j])) >= math.cos(math.radians(angle))
            stacking[i, j] = stacking[j, i] = close and aligned
    _, aggregates = connected_components(csr_matrix(aggregation), directed=False)
    _, clusters = connected_components(csr_matrix(stacking), directed=False)
    aggregate_sizes = Counter(aggregates)
    cluster_sizes = Counter(clusters)
    frequencies = Counter(size for size in cluster_sizes.values() if size >= 2)
    records = []
    covered_aggregates = set()
    for label, size in cluster_sizes.items():
        if size < 2:
            continue
        member = int(np.flatnonzero(clusters == label)[0])
        total = aggregate_sizes[aggregates[member]]
        covered_aggregates.add(aggregates[member])
        records.append(dict(Nc=int(size), Lp=int(total),
                            Rc=float(size / total), Mc=int(size * frequencies[size])))
    for label, size in aggregate_sizes.items():
        if label not in covered_aggregates:
            records.append(dict(Nc=0,Lp=int(size),Rc=0.0,Mc=0))
    parallel_members = np.array([cluster_sizes[label] >= 2 for label in clusters], dtype=bool)
    return dict(aggregates=int(len(aggregate_sizes)),
                parallel_clusters=int(sum(size >= 2 for size in cluster_sizes.values())),
                isolated=int(sum(size == 1 for size in aggregate_sizes.values())),
                records=records, parallel_members=parallel_members)


def read_manifest(folder):
    return json.loads((Path(folder) / 'manifest.json').read_text('utf-8'))


def validate_topology(trajectory, manifest):
    """Reject a same-sized trajectory with a different molecule/atom ordering."""
    config = manifest['config']
    nsolute = manifest['solute_atoms'] * config['molecules']
    water_start = nsolute + manifest['ion_count']
    if trajectory.n_atoms <= water_start or (trajectory.n_atoms-water_start)%3:
        raise ValueError('轨迹原子数与 PDC / 离子 / 三原子水顺序不符。')
    atoms = list(trajectory.topology.atoms)
    expected = manifest.get('solute_atom_names')
    for index in range(nsolute):
        if atoms[index].residue.name != manifest['molecule_name'] or (
                expected and atoms[index].name != expected[index%len(expected)]):
            raise ValueError('轨迹 PDC 残基或原子顺序与作者拓扑不符。')
    if manifest['ion'] and any(a.residue.name.upper()!=manifest['ion'].upper()
                               for a in atoms[nsolute:water_start]):
        raise ValueError('轨迹离子顺序不符。')
    if any(a.residue.name not in ('SOL','HOH') for a in atoms[water_start:]):
        raise ValueError('轨迹水分子顺序不符。')
    if trajectory.unitcell_vectors is None or np.any(np.linalg.det(trajectory.unitcell_vectors)<=0):
        raise ValueError('轨迹缺少有效的周期盒信息。')


def write_csv(path, header, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def histogram(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    norm = counts.sum()
    return [[float((a+b)/2), float(c/norm) if norm else 0]
            for a, b, c in zip(edges[:-1], edges[1:], counts)]


def thermodynamics(folder):
    folder = Path(folder)
    path = folder/'thermodynamics.xvg'
    lines = path.read_text('utf-8').splitlines()
    names = [re.search(r'legend\s+"([^"]+)"',line)[1]
             for line in lines if re.search(r'@\s+s\d+\s+legend\s+"',line)]
    values = np.asarray([[float(v) for v in line.split()] for line in lines
                         if line.strip() and not line.lstrip().startswith(('#','@'))])
    if values.ndim!=2 or values.shape[1]!=len(names)+1 or not np.isfinite(values).all():
        raise ValueError('GROMACS 热力学输出格式不完整。')
    write_csv(folder/'thermodynamics.csv',['Time (ps)',*names],values)
    start = read_manifest(folder)['config']['stable_after_ns']*1000
    sampled = values[values[:,0]+1e-6>=start]
    report = dict(samples=len(sampled), interval_ps=[float(sampled[0,0]),float(sampled[-1,0])] if len(sampled) else None,
                  columns={name:dict(mean=float(np.mean(sampled[:,i+1])),
                            standard_deviation=float(np.std(sampled[:,i+1])))
                           for i,name in enumerate(names)} if len(sampled) else {},
                  note='Energy-file sample cadence differs from XTC. Means/std alone do not establish equilibration; inspect traces and duration.')
    atomic_json(folder/'thermodynamics.json',report)
    return report


def analyze(folder, start_ns=None, distance=None, angle=None, cancelled=lambda: False):
    folder = Path(folder)
    manifest = read_manifest(folder)
    config = manifest['config']
    start_ns = config['stable_after_ns'] if start_ns is None else start_ns
    distance = config['distance_nm'] if distance is None else distance
    angle = config['angle_deg'] if angle is None else angle
    if not math.isfinite(start_ns) or start_ns<0 or not 0<distance<config['box_nm']/2 or not 0<angle<=90:
        raise ValueError('分析时间、距离或角度无效。')
    trajectory = folder / 'production.xtc'
    topology = folder / 'system.gro'
    if not trajectory.exists():
        raise ValueError('未找到 production.xtc。请先运行生产模拟或导入轨迹。')
    count, atoms = config['molecules'], manifest['solute_atoms']
    molecules = [np.arange(i * atoms, (i+1) * atoms) for i in range(count)]
    frames, metrics, all_records = [], [], []
    hits = np.zeros(count, dtype=int)
    last_time = -math.inf
    for chunk in md.iterload(str(trajectory), top=str(topology), chunk=10):
        validate_topology(chunk, manifest)
        for index, xyz in enumerate(chunk.xyz):
            if cancelled():
                raise InterruptedError('分析已取消。')
            if chunk.unitcell_vectors is None:
                raise ValueError('轨迹缺少周期盒信息。')
            time_ns = float(chunk.time[index]) / 1000
            if time_ns <= last_time or not np.isfinite(xyz).all():
                raise ValueError('轨迹时间需严格递增，坐标需为有限数值。')
            last_time = time_ns
            info = frame_metrics(xyz, chunk.unitcell_vectors[index], molecules,
                                 manifest['ring_atoms'], distance, angle)
            frames.append([time_ns, info['aggregates'], info['parallel_clusters'], info['isolated']])
            if time_ns + 1e-9 >= start_ns:
                all_records.extend(info['records'])
                hits += info['parallel_members']
                metrics.append(time_ns)
    if not metrics:
        raise ValueError('稳定区间没有采样帧。请缩短起始时间或增加模拟长度。')
    write_csv(folder / 'cluster_timeseries.csv', ['Time (ns)', 'Aggregates', 'Parallel clusters', 'Isolated'], frames)
    write_csv(folder / 'cluster_records.csv', ['Nc', 'Lp', 'Rc', 'Mc'],
              [[r[k] for k in ('Nc', 'Lp', 'Rc', 'Mc')] for r in all_records])
    rc = histogram([r['Rc'] for r in all_records], np.linspace(0, 1.000001, 21))
    mc = histogram([r['Mc'] for r in all_records], np.arange(-.5, count+1.5))
    write_csv(folder / 'Rc_distribution.csv', ['Rc', 'Probability'], rc)
    write_csv(folder / 'Mc_distribution.csv', ['Mc', 'Probability'], mc)
    summary = dict(engine='GROMACS', system=config['system'], sampled_frames=len(metrics),
                   analyzed_interval_ns=[min(metrics), max(metrics)], total_frames=len(frames),
                   distance_nm=distance, angle_deg=angle, reference_molecule=int(np.argmax(hits)) + 1,
                   average_aggregates=float(np.mean([r[1] for r in frames if r[0]+1e-9 >= start_ns])),
                   reference_parallel_frames=int(max(hits)),
                   distributions=dict(Rc=rc, Mc=mc),
                   conventions=['All PDC atoms are used for the minimum inter-molecular distance.',
                                'Periodic minimum-image distances and ring normals are used.',
                                'Clusters are undirected connected components.',
                                'Aggregates without a multi-molecule parallel component contribute one Rc=Mc=0 record.',
                                'Each cluster contributes once per sampled frame; histograms are unsmoothed.',
                                'Rc=0 convention and histogram binning require comparison with author analysis code.'])
    atomic_json(folder / 'analysis.json', summary)
    return summary


def cube(path, grid, origin_nm, spacing_nm, reference_xyz, elements):
    numbers = {'H': 1, 'C': 6, 'O': 8, 'Li': 3, 'Na': 11, 'K': 19, 'Rb': 37,
               'Cs': 55, 'Be': 4, 'Mg': 12, 'Ca': 20, 'Sr': 38, 'Ba': 56}
    bohr_per_nm = 18.8972612462577
    with Path(path).open('w', encoding='ascii') as stream:
        stream.write('ChatDFT number density (particles/nm^3), aligned reference\n'
                     'Sampled MD spatial density; coordinates in bohr\n')
        stream.write(f'{len(elements):5d}' + ''.join(f'{x*bohr_per_nm:13.6f}' for x in origin_nm) + '\n')
        for axis, size in enumerate(grid.shape):
            vector = np.zeros(3)
            vector[axis] = spacing_nm * bohr_per_nm
            stream.write(f'{size:5d}' + ''.join(f'{x:13.6f}' for x in vector) + '\n')
        for element, position in zip(elements, reference_xyz):
            stream.write(f'{numbers.get(element, 0):5d}{0.0:13.6f}'
                         + ''.join(f'{x*bohr_per_nm:13.6f}' for x in position) + '\n')
        flat = grid.ravel(order='C')
        buffer=[]
        for start in range(0, len(flat), 6):
            buffer.append(' '.join(f'{x:.6e}' for x in flat[start:start+6]))
            if len(buffer)==1000:
                stream.write('\n'.join(buffer)+'\n')
                buffer.clear()
        if buffer:
            stream.write('\n'.join(buffer)+'\n')


def spatial_density(folder, reference=None, extent_nm=1.5, bins=60, start_ns=None,
                    cancelled=lambda: False):
    folder = Path(folder)
    manifest = read_manifest(folder)
    config = manifest['config']
    analysis = json.loads((folder / 'analysis.json').read_text('utf-8'))
    reference = analysis['reference_molecule'] if reference is None else reference
    if not 1 <= reference <= config['molecules'] or not 8 <= bins <= 120 or not .1 <= extent_nm <= 5:
        raise ValueError('参考分子、网格或范围超限。')
    start_ns = config['stable_after_ns'] if start_ns is None else start_ns
    atoms, count = manifest['solute_atoms'], config['molecules']
    ref = np.arange((reference-1)*atoms, reference*atoms)
    ring = ref[manifest['ring_atoms']]
    pdc_atoms = np.setdiff1d(np.arange(count*atoms), ref)
    first_water = atoms*count + manifest['ion_count']
    total = md.load_frame(str(folder/'production.xtc'), 0, top=str(folder/'system.gro')).n_atoms
    groups = dict(PDC=pdc_atoms, Water=np.arange(first_water, total, 3),
                  Ions=np.arange(atoms*count, first_water))
    grids = {name: np.zeros((bins, bins, bins)) for name in groups}
    sampled, reference_xyz, elements = 0, None, []
    extent = [(-extent_nm, extent_nm)]*3
    for chunk in md.iterload(str(folder/'production.xtc'), top=str(folder/'system.gro'), chunk=10):
        validate_topology(chunk, manifest)
        for frame, xyz in enumerate(chunk.xyz):
            if cancelled():
                raise InterruptedError('空间分布已取消。')
            if chunk.time[frame]/1000 + 1e-9 < start_ns:
                continue
            box = chunk.unitcell_vectors[frame]
            center = xyz[ring[1]]
            x = minimum_image(xyz[ring[0]] - center, box)
            second = minimum_image(xyz[ring[2]] - center, box)
            x /= np.linalg.norm(x)
            z = np.cross(x, second)
            z /= np.linalg.norm(z)
            y = np.cross(z, x)
            rotation = np.stack([x, y, z], axis=1)
            for name, indexes in groups.items():
                positions = minimum_image(xyz[indexes] - center, box) @ rotation
                grids[name] += np.histogramdd(positions, bins=bins, range=extent)[0]
            if reference_xyz is None:
                reference_xyz = minimum_image(xyz[ref] - center, box) @ rotation
                elements = [a.element.symbol for a in chunk.topology.atoms if a.index in set(ref)]
            sampled += 1
    if not sampled:
        raise ValueError('选定区间没有可用轨迹帧。')
    spacing = 2*extent_nm/bins
    for name, grid in grids.items():
        grid /= sampled * spacing**3
        cube(folder/f'SDF_{name}.cube', grid, np.full(3,-extent_nm+spacing/2), spacing,
             reference_xyz, elements)
        # Three orthographic maximum-intensity projections support immediate UI preview.
        np.save(folder/f'SDF_{name}.npy', grid)
    report = dict(reference_molecule=reference, frames=sampled, bins=bins, extent_nm=extent_nm,
                  units='particles/nm^3', representation='All other PDC atoms, water oxygens, ion positions',
                  comparison_note='Paper uses gmx spatial and selected isovalues. Normalization and selection must be matched before comparing surfaces.')
    atomic_json(folder/'spatial.json', report)
    return report


def compare_distribution(folder, reference_csv, observable, source):
    if observable not in ('Rc', 'Mc') or not source.strip():
        raise ValueError('请选择 Rc 或 Mc，并填写论文图号和数据出处。')
    folder = Path(folder)
    current = np.loadtxt(folder/f'{observable}_distribution.csv', delimiter=',', skiprows=1)
    reference = np.loadtxt(reference_csv, delimiter=',', skiprows=1, ndmin=2)
    if reference.shape[1] != 2 or len(reference) < 2 or not np.isfinite(reference).all():
        raise ValueError('参考 CSV 需要两列有限数值及一行表头，至少两个数据点。')
    reference = reference[np.argsort(reference[:, 0])]
    if np.any(np.diff(reference[:, 0]) <= 0) or np.any(reference[:, 1] < 0):
        raise ValueError('参考横坐标必须唯一递增，概率值不能为负。')
    overlap = (current[:, 0] >= reference[0, 0]) & (current[:, 0] <= reference[-1, 0])
    if np.sum(overlap) < 2:
        raise ValueError('参考曲线与模拟结果没有足够重合区间。')
    x, y = current[overlap].T
    target = np.interp(x, reference[:, 0], reference[:, 1])
    report = dict(observable=observable, source=source, points=len(x),
                  rmse=float(np.sqrt(np.mean((y-target)**2))),
                  maximum_absolute_error=float(np.max(abs(y-target))),
                  note='Raw values compared without rescaling or smoothing. Match probability mass/density and bins first.')
    write_csv(folder/f'{observable}_comparison.csv', [observable, 'Simulation', 'Reference'], np.column_stack([x,y,target]))
    atomic_json(folder/f'{observable}_comparison.json', report)
    return report
