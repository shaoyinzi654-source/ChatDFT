"""Manage ChatDFT jobs on the configured SSH server through their worker interfaces."""
from dataclasses import asdict
import json
from pathlib import Path, PurePosixPath
import stat

from .chemistry import Structure
from .gaussian import Calculation
from .jobs import JobStore, download_file
from .md_inputs import MDConfig
from .md_jobs import MDJobs, MDRemote
from .server_inventory import JOB_ID
from .storage import atomic_json


class ServerJobs:
    def __init__(self, remote):
        self.remote = remote
        root = PurePosixPath(remote.settings['remote_root'])
        if not root.is_absolute() or '..' in root.parts or len(root.parts) < 3:
            raise ValueError('服务器工作目录须为独立的绝对目录，例如 /opt/chatdft。')
        self.root = root
        self.quantum = JobStore()
        self.md = MDJobs()
        self.md_remote = MDRemote(remote)

    def snapshot(self):
        script = Path(__file__).with_name('server_inventory.py').read_text('utf-8')
        return json.loads(self.remote.execute(['python3', '-c', script, str(self.root)]))

    def record(self, entry):
        job_id, engine = entry['id'], entry['engine']
        if not JOB_ID.fullmatch(job_id) or engine not in ('pyscf', 'gromacs'):
            raise ValueError('无法识别该服务器任务。')
        parent = self.root/'md'/'jobs' if engine == 'gromacs' else self.root/'jobs'
        keys = ('ssh_host', 'ssh_port', 'ssh_user', 'remote_root', 'host_fingerprint')
        return {**entry, 'target': 'ssh', 'remote_dir': str(parent/job_id),
                'connection': {key: self.remote.settings[key] for key in keys}}

    def job_folder(self, entry):
        job = self.record(entry)
        return (self.md if job['engine'] == 'gromacs' else self.quantum).folder(job)

    def read_json(self, job, name):
        path = job['remote_dir']+'/'+name
        with self.remote.sftp() as sftp:
            info = sftp.lstat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4*1024**2:
                raise ValueError('服务器任务文件无效或体积过大。')
            with sftp.open(path, 'rb') as stream:
                return json.loads(stream.read(4*1024**2+1))

    def log(self, entry):
        job = self.record(entry)
        tail = []
        with self.remote.sftp() as sftp:
            for name in ('worker.log', 'output.log'):
                path = job['remote_dir']+'/'+name
                try:
                    info = sftp.lstat(path)
                    if not stat.S_ISREG(info.st_mode) or not info.st_size:
                        continue
                    with sftp.open(path, 'rb') as stream:
                        stream.seek(max(0, info.st_size-48000))
                        value = stream.read(60000).decode('utf-8', errors='replace')
                    tail.append(f'[{name}]\n{value}')
                except FileNotFoundError:
                    continue
        return '\n\n'.join(tail).strip()

    def fresh(self, entry, states):
        job = self.record(entry)
        state = self.read_json(job, 'state.json')
        if state.get('status') not in states:
            raise ValueError('作业状态已变化，请刷新后再操作。')
        return {**job, 'status': state['status']}

    def cancel(self, entry):
        job = self.fresh(entry, ('queued', 'running'))
        engine = self.md_remote if job['engine'] == 'gromacs' else self.remote
        engine.cancel(job)

    def continue_job(self, entry):
        job = self.fresh(entry, ('failed', 'cancelled'))
        if job['engine'] == 'gromacs':
            self.md_remote.resume_server(job)
            return job['id']
        structure, calculation = self.quantum_input(job)
        from workflow import submit

        return submit(self.remote, self.quantum, structure, calculation, retry_of=job['id'])['id']

    def quantum_input(self, job):
        value = self.read_json(job, 'input.json')
        return Structure.from_dict(value['structure']), Calculation(**value['calculation'])

    def download(self, entry, trajectory=False):
        job = self.fresh(entry, ('completed', 'failed', 'cancelled'))
        folder = self.job_folder(job)
        metadata_path = folder/'job.json'
        previous = json.loads(metadata_path.read_text('utf-8')) if metadata_path.exists() else {}
        if previous and previous.get('connection') != job['connection']:
            raise ValueError('本地已有来自其他服务器的同编号任务。')
        if (folder/'trajectory-import.json').exists():
            raise ValueError('该位置已有手动导入的轨迹，原轨迹保持保留。')
        folder.mkdir(parents=True, exist_ok=True)
        meta = {**previous, 'id': job['id'], 'target': 'ssh', 'engine': job['engine'],
                'name': job['name'], 'created': previous.get('created', job['created']),
                'status': job['status'], 'remote_dir': job['remote_dir'], 'connection': job['connection']}
        if job['engine'] == 'gromacs':
            manifest = self.read_json(job, 'manifest.json')
            config = MDConfig(**manifest['config'])
            config.validate()
            meta.update(system=config.system, production_ns=config.production_ns)
            with self.remote.sftp() as sftp:
                for name in ('pdc.itp', 'water.itp', 'solute.gro'):
                    download_file(sftp, job['remote_dir']+'/'+name, folder/name)
            atomic_json(folder/'manifest.json', manifest)
            self.md_remote.download(job, folder, trajectory)
        else:
            structure, calculation = self.quantum_input(job)
            meta.update(structure=structure.to_dict(), calculation=asdict(calculation))
            self.remote.download_artifacts(job, folder)
        atomic_json(metadata_path, meta)
        return folder
