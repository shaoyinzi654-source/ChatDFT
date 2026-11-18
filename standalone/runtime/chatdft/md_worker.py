"""Linux MD worker. Commands are fixed and inputs are generated or reviewed files."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from md_analysis import analyze, atomic_json, spatial_density, thermodynamics


class Cancelled(Exception):
    pass


def command(folder, args, state, input_text=None):
    record = dict(command=args, started=time.time())
    with (folder/'commands.jsonl').open('a', encoding='utf-8') as log:
        log.write(json.dumps(record)+'\n')
    with (folder/'output.log').open('ab') as log:
        log.write(('\n$ ' + ' '.join(args) + '\n').encode())
        process = subprocess.Popen(args, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                                   start_new_session=True)
        if input_text is not None:
            process.stdin.write(input_text.encode())
            process.stdin.close()
        while process.poll() is None:
            state['heartbeat'] = time.time()
            state['child_pid'] = process.pid
            production_log = folder/'production.log'
            if production_log.exists():
                with production_log.open('rb') as stream:
                    stream.seek(max(0, production_log.stat().st_size-20000))
                    tail = stream.read().decode(errors='replace')
                steps = re.findall(r'Step\s+Time\s*\n\s*(\d+)\s+([\d.eE+-]+)', tail)
                if steps:
                    state['simulation_ps'] = float(steps[-1][1])
            atomic_json(folder/'state.json', state)
            if (folder/'cancel').exists():
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise Cancelled('用户取消，已保留可用检查点。')
            time.sleep(.5)
        if process.returncode:
            raise RuntimeError(f'{args[1] if len(args)>1 else args[0]} 返回 {process.returncode}，请查看日志。')


def preprocess(folder, args, state, stage):
    """Commit only validated TPRs; allow one precisely identified paper warning."""
    certified = folder/(stage+'.validated.json')
    if certified.exists() and (folder/(stage+'.tpr')).exists():
        return
    temporary = folder/(stage+'.pending.tpr')
    args = list(args)
    args[args.index('-o')+1] = temporary.name
    log = folder/'output.log'
    offset = log.stat().st_size if log.exists() else 0
    accepted = None
    try:
        command(folder, args, state)
    except RuntimeError:
        with log.open('rb') as stream:
            stream.seek(offset)
            output = stream.read().decode(errors='replace')
        warnings = re.findall(r'WARNING\s+\d+\s*\[.*?\]:(.*?)(?=\n\s*\n)', output, re.S)
        message = ' '.join(warnings[0].split()) if len(warnings)==1 else ''
        expected = ('With Nose-Hoover T-coupling and Parrinello-Rahman p-coupling, '
                    'tau-p (2) should be at least twice as large as tau-t (2) to avoid resonances')
        if stage!='production' or message!=expected or 'Too many warnings (1)' not in output:
            temporary.unlink(missing_ok=True)
            raise
        accepted = dict(message=message, reason='Retain paper-reported tau-p=tau-t=2 ps.',
                        risk='Possible thermostat/barostat resonance; inspect production stability.')
        command(folder, args+['-maxwarn','1'], state)
    if not temporary.exists():
        raise ValueError('预处理没有生成有效 TPR。')
    os.replace(temporary, folder/(stage+'.tpr'))
    atomic_json(certified, dict(validated=time.time(), accepted_warning=accepted))


def write_index(folder, manifest):
    config = manifest['config']
    count, atoms = config['molecules'], manifest['solute_atoms']
    lines = []
    def group(name, indexes):
        lines.append(f'[ {name} ]')
        indexes = list(indexes)
        for offset in range(0,len(indexes),15):
            lines.append(' '.join(map(str,indexes[offset:offset+15])))
    coordinates = (folder/'system.gro').read_text().splitlines()
    total = int(coordinates[1])
    group('System', range(1,total+1))
    if manifest['tabulated']:
        group('ION',range(atoms*count+1,atoms*count+manifest['ion_count']+1))
        groups = {}
        atom_rows = []
        active = False
        for line in (folder/'pdc.itp').read_text().splitlines():
            if line.strip().startswith('['):
                active = bool(re.match(r'\[\s*atoms\s*\]',line.strip(),re.I))
            elif active and line.split(';')[0].strip():
                atom_rows.append(line.split(';')[0].split())
        for row in atom_rows:
            groups.setdefault('T_'+row[1], []).extend(molecule*atoms+int(row[0]) for molecule in range(count))
        start = atoms*count+manifest['ion_count']
        groups['T_OW'] = list(range(start+1, total+1, 3))
        groups['T_HW'] = [i for j in range(start+1,total+1,3) for i in (j+1,j+2)]
        for name, indexes in groups.items():
            group(name,indexes)
    (folder/'index.ndx').write_text('\n'.join(lines)+'\n')


def probe(executable, python):
    output = subprocess.run([executable,'--version'],capture_output=True,text=True,timeout=15)
    if output.returncode:
        raise ValueError('GROMACS 版本检查失败。')
    match = re.search(r'GROMACS version:\s*(\S+)',output.stdout)
    return dict(engine='GROMACS', version=match[1] if match else output.stdout[:100],
                executable=executable, python=python, cores=os.cpu_count(),
                free_disk_gb=round(shutil.disk_usage('/opt').free/1024**3,2))


def build_system(folder, gmx, manifest, state):
    config = manifest['config']
    if not (folder/'system.gro').exists():
        state['stage']='build'
        command(folder,[gmx,'insert-molecules','-ci','solute.gro','-nmol',str(config['molecules']),
                        '-box',*[str(config['box_nm'])]*3,'-seed',str(config['seed']),'-try','500',
                        '-o','packed.gro'],state)
        coordinate='packed.gro'
        if manifest['ion_count']:
            command(folder,[gmx,'insert-molecules','-f',coordinate,'-ci','ion.gro',
                            '-nmol',str(manifest['ion_count']),'-seed',str(config['seed']+1),
                            '-try','500','-o','ions.gro'],state)
            coordinate='ions.gro'
        # gmx solvate updates a temporary topology, committed with its resulting coordinates.
        shutil.copyfile(folder/'topol.top',folder/'build.top')
        command(folder,[gmx,'solvate','-cp',coordinate,'-cs','spc216.gro',
                        '-p','build.top','-o','built.gro'],state)
        coordinates=(folder/'built.gro').read_text().splitlines()
        total=int(coordinates[1])
        first_water=config['molecules']*manifest['solute_atoms']+manifest['ion_count']
        if total <= first_water or (total-first_water)%3:
            raise ValueError('溶剂化后原子数量与拓扑不一致。')
        expected_residue=manifest['molecule_name']
        solutes=coordinates[2:2+config['molecules']*manifest['solute_atoms']]
        if any(line[5:10].strip()!=expected_residue for line in solutes):
            raise ValueError('PDC 插入数量不足。请增大盒子或减少分子数量。')
        ion_rows=coordinates[2+len(solutes):2+first_water]
        if manifest['ion'] and any(line[5:10].strip()!=manifest['ion'].upper() for line in ion_rows):
            raise ValueError('离子插入数量不足。请核对盒子和体系电荷。')
        if any(line[5:10].strip()!='SOL' for line in coordinates[2+first_water:2+total]):
            raise ValueError('水分子顺序与生成的拓扑不同。')
        water_count=(total-first_water)//3
        if water_count < 1:
            raise ValueError('盒子没有水。')
        os.replace(folder/'build.top',folder/'topol.top')
        os.replace(folder/'built.gro',folder/'system.gro')
        atomic_json(folder/'system.json',dict(atoms=total,water=water_count,
                                             pdc=config['molecules'],ions=manifest['ion_count'],
                                             concentration_molar=config['molecules']/(.602214076*config['box_nm']**3)))
    write_index(folder,manifest)


def run(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'manifest.json').read_text('utf-8'))
    spec = json.loads((folder/'md-spec.json').read_text('utf-8'))
    config = manifest['config']
    from compute_queue import try_locks

    lock = None
    state = dict(status='queued', stage='waiting', heartbeat=time.time(), pid=os.getpid(),
                 started=time.time(), message='', child_pid=None)
    atomic_json(folder/'state.json',state)
    try:
        light_box = config['cores'] <= 8 and config['box_nm'] <= 8 and config['molecules'] <= 100
        locks = ('compute.lock',) if light_box else ('quantum.lock', 'compute.lock')
        while lock is None:
            if (folder/'cancel').exists():
                raise Cancelled('排队任务已取消。')
            lock = try_locks(folder.parents[2], locks)
            if lock is None:
                state['heartbeat']=time.time()
                atomic_json(folder/'state.json',state)
                time.sleep(2)
        if (folder/'cancel').exists():
            raise Cancelled('任务已取消。')
        gmx = spec['gromacs']
        health = probe(gmx, sys.executable)
        if manifest['tabulated'] and not health['version'].startswith(('2018','2019','5.')):
            raise ValueError('用户表格势任务需要兼容 group 截断的 GROMACS 版本。')
        atomic_json(folder/'engine.json',health)
        state['status']='running'
        build_system(folder, gmx, manifest, state)
        previous='system.gro'
        for stage in ('minimize','equilibrate','production'):
            state['stage']=stage
            if (folder/f'{stage}.done').exists():
                previous=stage+'.gro'
                continue
            args=[gmx,'grompp','-f',stage+'.mdp','-c',previous,'-p','topol.top',
                  '-n','index.ndx','-o',stage+'.tpr','-po',stage+'-resolved.mdp']
            if stage=='production':
                args.extend(['-t','equilibrate.cpt'])
            preprocess(folder,args,state,stage)
            args=[gmx,'mdrun','-deffnm',stage,'-nt',str(config['cores']),'-pin','off']
            if manifest['tabulated']:
                args.extend(['-ntomp','1','-table','table.xvg','-tablep','tablep.xvg'])
            if (folder/(stage+'.cpt')).exists():
                args.extend(['-cpi',stage+'.cpt','-append'])
            command(folder,args,state)
            if not (folder/(stage+'.gro')).exists():
                raise ValueError('模拟结束但没有完成的结构文件。')
            (folder/f'{stage}.done').write_text(str(time.time()))
            previous=stage+'.gro'
        state['stage']='analysis'
        atomic_json(folder/'state.json',state)
        command(folder,[gmx,'energy','-f','production.edr','-o','thermodynamics.xvg'],state,
                'Potential\nKinetic-En.\nTotal-Energy\nTemperature\nPressure\nDensity\n0\n')
        thermodynamics(folder)
        stop=lambda:(folder/'cancel').exists()
        result=analyze(folder,cancelled=stop)
        spatial_density(folder,cancelled=stop)
        state.update(status='completed',stage='completed',message='MD 与聚集分析完成。')
        state['simulation_ps'] = config['production_ns']*1000
        state['frames']=result['sampled_frames']
    except (Cancelled,InterruptedError) as error:
        state.update(status='cancelled',message=str(error))
    except Exception as error:
        state.update(status='failed',message=str(error))
    finally:
        state.update(finished=time.time(),heartbeat=time.time())
        atomic_json(folder/'state.json',state)
        if lock is not None:
            lock.close()


if __name__=='__main__':
    if sys.argv[1]=='probe':
        print(json.dumps(probe(sys.argv[2],sys.executable)))
    elif sys.argv[1]=='run':
        run(sys.argv[2])
