"""MD jobs share the existing pinned SSH connection and use a separate artifact store."""
import json
import os
from dataclasses import asdict
from pathlib import Path
import re
import shlex
import shutil
from tempfile import TemporaryDirectory
import time
import uuid
import zipfile

from .md_inputs import ASSETS, catalog, paper_config, prepare_inputs
from .jobs import download_file
from .storage import atomic_json, data_dir


class MDJobs:
    def __init__(self):
        self.root = data_dir() / 'md-jobs'
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, job):
        if not re.fullmatch(r'[0-9a-f]{16}', job['id']):
            raise ValueError('任务编号无效。')
        return self.root / job['id']

    def list(self):
        jobs = []
        for path in self.root.glob('*/job.json'):
            try:
                job = json.loads(path.read_text('utf-8'))
                state = path.parent / 'state.json'
                if state.exists():
                    job.update(json.loads(state.read_text('utf-8')))
                if 'production_ns' not in job:
                    manifest = json.loads((path.parent/'manifest.json').read_text('utf-8'))
                    job['production_ns'] = manifest['config']['production_ns']
                jobs.append(job)
            except (OSError, ValueError):
                continue
        return sorted(jobs, key=lambda j:j['created'], reverse=True)

    def create(self, config, settings, c4=None):
        job = self.new_job(config, settings)
        folder = self.folder(job)
        # A directory appears in the monitor only once all generated inputs validate.
        prepare_inputs(folder, config, c4)
        atomic_json(folder/'job.json',job)
        return job

    @staticmethod
    def new_job(config, settings):
        return dict(id=uuid.uuid4().hex[:16], created=time.time(), status='prepared',
                    system=config.system, production_ns=config.production_ns, target='ssh', engine='gromacs',
                    connection={k: settings[k] for k in ('ssh_host', 'ssh_port', 'ssh_user', 'remote_root', 'host_fingerprint')})

    def prepare_paper_boxes(self, settings):
        """Reuse the verified starting boxes, including when running offline."""
        candidates = self.list()
        archive_path = ASSETS.parent / 'pdc-paper-models.zip'
        required = {'system.gro', 'system.json', 'manifest.json', 'preflight.json', 'topol.top',
                    'index.ndx', 'pdc.itp', 'water.itp', 'solute.gro', 'minimize.mdp',
                    'equilibrate.mdp', 'production.mdp'}
        with zipfile.ZipFile(archive_path) as archive:
            jobs = []
            for system in catalog():
                config = paper_config(system)
                job = self.new_job(config, settings)
                prefix = f'Li-2024-PDC/{system}/'
                members = {name[len(prefix):]: name for name in archive.namelist() if name.startswith(prefix)}
                if not required.issubset(members):
                    raise ValueError(f'{system} 的论文模型包不完整。')
                reference = json.loads(archive.read(members['manifest.json']))
                if reference['config'] != asdict(config):
                    raise ValueError(f'{system} 的论文模型参数版本不一致。')
                previous = next((entry for entry in candidates if self.matches_paper_box(entry, job, reference)), None)
                if previous is not None:
                    jobs.append(previous)
                    continue
                with TemporaryDirectory(dir=self.root, prefix='.pdc-') as scratch:
                    pending = Path(scratch) / job['id']
                    pending.mkdir()
                    for name, member in members.items():
                        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.(gro|json|top|ndx|itp|mdp|xvg)', name):
                            raise ValueError('论文模型包包含无法识别的输入文件。')
                        with archive.open(member) as source, (pending / name).open('wb') as target:
                            shutil.copyfileobj(source, target)
                    job['stage'] = '周期盒已校验'
                    atomic_json(pending / 'job.json', job)
                    pending.replace(self.folder(job))
                jobs.append(job)
        return jobs

    def matches_paper_box(self, entry, new_job, reference):
        if entry['system'] != new_job['system'] or entry['connection'] != new_job['connection']:
            return False
        folder = self.folder(entry)
        if not (folder/'system.gro').exists() or (folder/'trajectory-import.json').exists():
            return False
        try:
            receipt = json.loads((folder/'preflight.json').read_text('utf-8'))
            manifest = json.loads((folder/'manifest.json').read_text('utf-8'))
            return receipt['status'] == 'passed' and manifest == reference
        except (OSError, ValueError, KeyError):
            return False


class MDRemote:
    def __init__(self, remote):
        self.remote = remote
        self.root = remote.settings['remote_root'].rstrip('/') + '/md'
        self.python = remote.settings.get('remote_python','/opt/chatdft/venv/bin/python')
        self.gromacs = remote.settings.get('gromacs_exe','/usr/bin/gmx')
        self.legacy = remote.settings.get('gromacs_legacy_exe',self.root+'/gromacs-2018/bin/gmx')

    def deploy(self):
        self.remote.execute(['mkdir','-p',self.root,self.root+'/jobs'])
        source = Path(__file__).resolve().parent
        with self.remote.sftp() as sftp:
            for name in ('md_worker.py','md_analysis.py','compute_queue.py'):
                target=self.root+'/'+name
                sftp.put(str(source/name),target+'.upload')
                sftp.posix_rename(target+'.upload',target)

    def health(self):
        self.deploy()
        result = json.loads(self.remote.execute([self.python,self.root+'/md_worker.py','probe',self.gromacs]))
        try:
            result['legacy'] = json.loads(self.remote.execute(
                [self.python,self.root+'/md_worker.py','probe',self.legacy]))
        except (ValueError, RuntimeError):
            result['legacy'] = None
        return result

    def directory(self, job):
        self.remote.require_connection(job)
        if not re.fullmatch(r'[0-9a-f]{16}',job['id']):
            raise ValueError('任务编号无效。')
        return self.root+'/jobs/'+job['id']

    def submit(self, job, folder):
        self.deploy()
        target=self.directory(job)
        self.remote.execute(['mkdir','-p',target])
        manifest=json.loads((folder/'manifest.json').read_text('utf-8'))
        atomic_json(folder/'md-spec.json',dict(gromacs=self.legacy if manifest['tabulated'] else self.gromacs))
        with self.remote.sftp() as sftp:
            for source in folder.iterdir():
                if source.is_file() and source.name not in ('job.json','state.json'):
                    sftp.put(str(source),target+'/'+source.name)
        self.launch(job)
        job.update(status='queued',stage='waiting',remote_dir=target)
        atomic_json(folder/'job.json',job)
        return job

    def launch(self, job):
        target=self.directory(job)
        # flock protects launch itself against duplicate resume clicks; the worker
        # holds a separate lifetime lock and the shared server compute lock.
        args=['flock','-n',target+'/launch.lock',self.python,self.root+'/md_worker.py','run',target]
        command='nohup '+shlex.join(args)+' >> '+shlex.quote(target+'/worker.log')+' 2>&1 < /dev/null &'
        _,out,error=self.remote.connect().exec_command(command,timeout=15)
        if out.channel.recv_exit_status():
            raise RuntimeError(error.read().decode('utf-8',errors='replace')[:300])

    def sync(self, job, folder):
        if (folder/'trajectory-import.json').exists():
            raise ValueError('此任务已导入本地轨迹，不能用原服务器结果覆盖。原文件保存在 before-import 目录。')
        target=self.directory(job)
        small=('state.json','system.json','engine.json','analysis.json','spatial.json',
               'cluster_timeseries.csv','cluster_records.csv','Rc_distribution.csv','Mc_distribution.csv',
               'minimize.validated.json','equilibrate.validated.json','production.validated.json',
               'minimize-resolved.mdp','equilibrate-resolved.mdp','production-resolved.mdp','commands.jsonl')
        small += ('thermodynamics.csv','thermodynamics.json')
        with self.remote.sftp() as sftp:
            for name in small:
                try:
                    if sftp.stat(target+'/'+name).st_size>20*1024**2:
                        continue
                    download_file(sftp, target+'/'+name, folder/name)
                except FileNotFoundError:
                    pass
            for name in ('output.log','worker.log'):
                try:
                    with sftp.open(target+'/'+name,'rb') as stream:
                        stream.seek(max(0,stream.stat().st_size-100000))
                        (folder/name).write_bytes(stream.read())
                except FileNotFoundError:
                    pass
        return json.loads((folder/'state.json').read_text('utf-8')) if (folder/'state.json').exists() else dict(status='queued')

    def cancel(self, job):
        with self.remote.sftp() as sftp:
            with sftp.open(self.directory(job)+'/cancel','w') as stream:
                stream.write('cancel')

    def resume(self, job, folder):
        if (folder/'trajectory-import.json').exists():
            raise ValueError('此任务已导入本地轨迹，请保留本地导入的分析。')
        self.resume_server(job)
        atomic_json(folder/'state.json',dict(status='queued',stage='resume',heartbeat=time.time()))

    def resume_server(self, job):
        target = self.directory(job)
        self.deploy()
        with self.remote.sftp() as sftp, sftp.open(target+'/state.json', 'rb') as stream:
            state = json.loads(stream.read(65536))
        if state.get('status') not in ('failed','cancelled'):
            raise ValueError('请同步状态，并选择已停止的任务继续。')
        with self.remote.sftp() as sftp:
            try:
                sftp.remove(target+'/cancel')
            except FileNotFoundError:
                pass
        self.launch(job)

    def download(self, job, folder, trajectory=False):
        if (folder/'trajectory-import.json').exists():
            raise ValueError('本地轨迹已导入，请使用备份查看原服务器结果。')
        target=self.directory(job)
        names=['system.gro','production.gro','production.log','production.cpt',
               'topol.top','index.ndx','production.tpr','production.edr',
               'SDF_PDC.cube','SDF_Water.cube','SDF_Ions.cube',
               'SDF_PDC.npy','SDF_Water.npy','SDF_Ions.npy']
        if trajectory:
            names.append('production.xtc')
        with self.remote.sftp() as sftp:
            for name in names:
                try:
                    download_file(sftp, target+'/'+name, folder/name)
                except FileNotFoundError:
                    pass
        return self.sync(job,folder)
