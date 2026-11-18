"""Read-only inventory run by the configured server's system Python."""
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time


JOB_ID = re.compile(r'[0-9a-f]{16}\Z')
STATES = {'queued', 'running', 'completed', 'failed', 'cancelled'}


def document(folder, name):
    path = folder/name
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 16*1024*1024:
        return {}
    value = json.loads(path.read_text('utf-8'))
    return value if isinstance(value, dict) else {}


def tail(path, limit=24000):
    if not path.is_file() or path.is_symlink():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-limit))
        return stream.read().decode('utf-8', errors='replace')


def read_job(folder, engine, now):
    if engine == 'pyscf' and document(folder, 'spec.json').get('engine', 'pyscf') != 'pyscf':
        return None
    state = document(folder, 'state.json')
    source = document(folder, 'manifest.json' if engine == 'gromacs' else 'input.json')
    if not source:
        return None
    config = source.get('config', {})
    calculation = source.get('calculation', {})
    status = state.get('status')
    if status not in STATES:
        status = 'unknown'
    stage = state.get('stage', '')
    heartbeat = state.get('heartbeat')
    stale = status in ('queued', 'running') and bool(heartbeat) and now-heartbeat > 120
    span = config.get('equilibration_ns' if stage == 'equilibrate' else 'production_ns')
    position = state.get('simulation_ps') if stage == 'production' else None
    if engine == 'gromacs' and status == 'running' and stage in ('equilibrate', 'production'):
        records = re.findall(r'\bStep\s+Time\s*\n\s*\d+\s+([\d.eE+-]+)', tail(folder/(stage+'.log')))
        if records:
            position = float(records[-1])
    iteration = None
    if engine == 'pyscf' and status == 'running' and calculation.get('task') in ('opt', 'optfreq'):
        steps = re.findall(r'^\s*Step\s+(\d+)\s*:', tail(folder/'output.log'), re.M)
        if steps:
            iteration = int(steps[-1])
    return dict(id=folder.name, engine=engine, name=(config.get('system') or
                source.get('structure', {}).get('name') or folder.name),
                status=status, stage=stage, stale=stale, heartbeat=heartbeat,
                created=state.get('submitted') or state.get('started') or folder.stat().st_mtime,
                message=str(state.get('message', ''))[:1200],
                cancel_requested=status in ('queued', 'running') and (folder/'cancel').is_file(),
                position_ps=position, target_ps=span*1000 if isinstance(span, (int, float)) else None,
                iteration=iteration, task=calculation.get('task', ''), method=calculation.get('method', ''),
                basis=calculation.get('basis', ''), cores=state.get('effective_cores') or config.get('cores'),
                has_trajectory=(folder/'production.xtc').is_file(),
                has_checkpoint=(folder/'production.cpt').is_file() or (folder/'equilibrate.cpt').is_file())


def cpu_ticks():
    values = [int(value) for value in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(values), values[3]+values[4]


def inventory(root):
    first = cpu_ticks()
    started = time.monotonic()
    now = time.time()
    jobs = []
    for engine, parent in (('pyscf', root/'jobs'), ('gromacs', root/'md'/'jobs')):
        if not parent.is_dir() or parent.resolve() != parent:
            continue
        for folder in parent.iterdir():
            if not JOB_ID.fullmatch(folder.name) or not folder.is_dir() or folder.is_symlink():
                continue
            try:
                entry = read_job(folder, engine, now)
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
            if entry is not None:
                jobs.append(entry)
    time.sleep(max(0, .18-(time.monotonic()-started)))
    second = cpu_ticks()
    total = max(1, second[0]-first[0])
    busy = max(0., min(100., 100*(1-(second[1]-first[1])/total)))
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            memory[key] = int(value.split()[0])*1024
    disk = shutil.disk_usage(root if root.exists() else root.parent)
    return dict(host=os.uname().nodename, cores=os.cpu_count(), cpu_percent=round(busy, 1),
                mem_total=memory['MemTotal'], mem_used=memory['MemTotal']-memory['MemAvailable'],
                disk_total=disk.total, disk_used=disk.used, time=now, jobs=jobs)


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    if not root.is_absolute() or len(root.parts) < 3:
        raise ValueError('Invalid ChatDFT job root')
    print(json.dumps(inventory(root), ensure_ascii=False, allow_nan=False))
