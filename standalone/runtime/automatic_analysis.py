"""Geometry checks based on the saved recipe and the actual result coordinates."""
import math

import numpy as np

from chatdft.assembly import atom_selector, coordinates


def align_coordinates(reference, target):
    """Return centered coordinates in the reference orientation and in atom order."""
    first = coordinates(reference)
    second = coordinates(target)
    first -= first.mean(axis=0)
    second -= second.mean(axis=0)
    left, _, right = np.linalg.svd(second.T @ first)
    rotation = np.eye(3)
    rotation[-1, -1] = np.linalg.det(left @ right)
    return first, second @ left @ rotation @ right


def geometry_comparison(reference, target):
    initial, final = align_coordinates(reference, target)
    displacements = np.linalg.norm(final - initial, axis=1)
    return {
        'aligned_rmsd_angstrom': float(np.sqrt(np.mean(displacements**2))),
        'max_aligned_displacement_angstrom': float(displacements.max()),
        'max_displacement_atom': int(displacements.argmax()) + 1,
    }


def measure_placement(structure, placement):
    fragments = []
    for key in ('moving', 'reference'):
        fragment = structure.components[placement[key] - 1]
        atoms = structure.atoms[fragment['start']:fragment['start'] + fragment['count']]
        values = np.array([[atom[axis] for axis in 'xyz'] for atom in atoms])
        symbols = [atom['element'] for atom in atoms]
        if placement['mode'] == 'stack':
            heavy = values[np.array(symbols) != 'H']
            center = heavy.mean(axis=0)
            _, spread, axes = np.linalg.svd(heavy - center, full_matrices=False)
            if len(spread) != 3 or spread[1] < .1:
                raise ValueError('无法定义重原子平面')
            fragments.append((center, axes[-1]))
        else:
            index = atom_selector(symbols, placement[f'{key}_atom'])
            fragments.append(values.mean(axis=0) if index is None else values[index])
    first, second = fragments
    if placement['mode'] == 'stack':
        (moving_center, moving_normal), (reference_center, reference_normal) = first, second
        distance = abs(float((moving_center - reference_center) @ reference_normal))
        cosine = min(1., abs(float(moving_normal @ reference_normal)))
        return distance, math.degrees(math.acos(cosine))
    return float(np.linalg.norm(first - second)), None


def analyze_placements(initial, final):
    recipe = initial.modeling_recipe
    if recipe.get('keep_current') or not initial.components:
        return []
    rows = []
    for row in recipe.get('placements', []):
        try:
            initial_distance, initial_angle = measure_placement(initial, row)
            final_distance, final_angle = measure_placement(final, row) if final else (None, None)
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        rows.append({
            'mode': row['mode'], 'moving': row['moving'], 'moving_atom': row['moving_atom'],
            'reference': row['reference'], 'reference_atom': row['reference_atom'],
            'target_angstrom': row['distance'], 'initial_angstrom': initial_distance,
            'result_angstrom': final_distance, 'initial_plane_angle_deg': initial_angle,
            'result_plane_angle_deg': final_angle,
        })
    return rows


def summarize_result(result, calculation, initial, final):
    summary = {'task': calculation['task'], 'scf_converged': result.get('scf_converged'),
               'optimization_converged': result.get('optimization_converged'),
               'scf_records': len(result.get('scf_energies_hartree', [])),
               'imaginary_frequencies_cm1': result.get('imaginary_frequencies', []),
               'placements': analyze_placements(initial, final)}
    metrics = geometry_comparison(initial, final) if final else None
    summary['geometry_comparison'] = metrics
    status = {True: '收敛', False: '未收敛', None: '结果未提供'}
    notes = [f'- SCF {status[summary["scf_converged"]]}，共记录 {summary["scf_records"]} 条能量。']
    if calculation['task'] in ('opt', 'optfreq'):
        notes.append(f'- 几何优化{status[summary["optimization_converged"]]}。')
    if calculation['task'] not in ('opt', 'optfreq'):
        notes.append('- 本任务使用输入几何。返回坐标可用于核对计算实际采用的结构。')
    if metrics:
        rmsd = metrics['aligned_rmsd_angstrom']
        atom_id = metrics['max_displacement_atom']
        notes.append(f'- 按原子次序刚体对齐后的结构 RMSD 为 {rmsd:.4f} Å，'
                     f'最大位移出现在 {atom_id} 号原子，为 {metrics["max_aligned_displacement_angstrom"]:.4f} Å。')
    if summary['placements']:
        notes.append(f'- 已复测 {len(summary["placements"])} 项建模约束。'
                     '见 [几何测量表](几何测量.csv)，其中堆叠距离为重原子拟合平面投影。')
    if summary['imaginary_frequencies_cm1']:
        notes.append(f'- 检出显著虚频 {len(summary["imaginary_frequencies_cm1"])} 个。请结合振动模式判断驻点性质。')
    if result.get('mulliken_charges'):
        total = sum(row['charge'] for row in result['mulliken_charges'])
        summary['mulliken_charge_sum_e'] = total
        notes.append(f'- Mulliken 电荷合计 {total:+.6f} e。各原子的分配值受基组与布居分析方法影响。')
    return summary, notes
