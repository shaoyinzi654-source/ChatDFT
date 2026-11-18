from copy import deepcopy

from chatdft.chemistry import Structure, validate_structure
from chatdft.gaussian import build_input


def submit(remote, store, structure, calculation, retry_of=None, automation=None):
    if calculation.method == 'MP2' and (calculation.task, calculation.solvent) != ('sp', 'Gas'):
        raise ValueError('MP2 仅支持气相单点计算。')
    gjf = build_input(structure, calculation)
    job = store.create(structure, calculation, gjf, remote.settings)
    if retry_of is not None:
        job['retry_of'] = retry_of
    if automation is not None:
        job['automation'] = automation
        store.save(job)
    return store.launch(job, remote)


def result_geometry(job, result):
    if not result.get('normal_termination') or not result.get('geometry'):
        raise ValueError('任务没有返回正常结束的几何。')
    source = Structure.from_dict(job['structure'])
    geometry = result['geometry']
    if [a['element'] for a in geometry] != [a['element'] for a in source.atoms]:
        raise ValueError('服务器返回的原子顺序与原结构不一致。')
    source.atoms = deepcopy(geometry)
    source.provenance = 'PySCF server result'
    validate_structure(source)
    return source
