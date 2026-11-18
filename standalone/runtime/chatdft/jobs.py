from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time
import uuid
import paramiko
from .storage import data_dir, atomic_json
from . import worker


SSH_AUTH_ERROR = 'SSH 认证失败。请重新连接并核对服务器密码、密钥和用户名。'


def fingerprint(key):
    return 'SHA256:'+base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')


def download_file(sftp, remote, destination):
    if not stat.S_ISREG(sftp.lstat(remote).st_mode):
        raise ValueError('服务器作业文件不是普通文件。')
    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.download')
    try:
        sftp.get(remote, str(temporary), max_concurrent_prefetch_requests=32)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class PinnedPolicy(paramiko.MissingHostKeyPolicy):
    def __init__(self, expected):
        self.expected = expected

    def missing_host_key(self, client, hostname, key):
        actual = fingerprint(key)
        if actual != self.expected:
            raise ValueError('SSH 主机指纹不匹配。实际指纹：'+actual+'。请通过可信渠道核实后在设置中更新。')


class Remote:
    def __init__(self, settings, password=''):
        self.settings = dict(settings)
        self.password = password
        self.client = None

    def connect(self):
        if self.client and self.client.get_transport() and self.client.get_transport().is_active():
            return self.client
        self.close()
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(PinnedPolicy(self.settings['host_fingerprint']))
        try:
            client.connect(self.settings['ssh_host'],port=int(self.settings['ssh_port']),username=self.settings['ssh_user'],
                           password=self.password or None,key_filename=self.settings.get('ssh_key') or None,
                           look_for_keys=not bool(self.password),allow_agent=not bool(self.password),timeout=15,auth_timeout=15,banner_timeout=15)
        except paramiko.AuthenticationException as exc:
            client.close()
            raise paramiko.AuthenticationException(SSH_AUTH_ERROR) from exc
        except Exception:
            client.close()
            raise
        client.get_transport().set_keepalive(20)
        self.client = client
        return client

    def close(self):
        if self.client:
            self.client.close()
        self.client = None

    def sftp(self):
        client = self.connect().open_sftp()
        client.get_channel().settimeout(20)
        return client

    def require_connection(self, job):
        expected=job.get('connection',{})
        if any(self.settings.get(k)!=expected.get(k) for k in ['ssh_host','ssh_port','ssh_user','remote_root','host_fingerprint']):
            raise ValueError('当前连接与该任务的服务器配置不同；请恢复任务原来的连接设置。')

    def execute(self, args):
        _, out, err = self.connect().exec_command(shlex.join(args),timeout=30)
        value = out.read().decode('utf-8',errors='replace')
        error = err.read().decode('utf-8',errors='replace')
        if out.channel.recv_exit_status():
            raise RuntimeError(error[:500] or value[:500] or '远程命令失败')
        return value

    def deploy_worker(self):
        root = self.settings['remote_root']
        if not root.startswith('/') or root in ['/','/root','/home','/opt']:
            raise ValueError('远程工作目录必须是独立的绝对目录，例如 /opt/chatdft。')
        self.execute(['mkdir','-p',root+'/jobs'])
        with self.sftp() as sftp:
            for filename in ('compute_queue.py', 'pyscf_engine.py', 'worker.py'):
                target=root+'/'+filename;tmp=target+'.'+uuid.uuid4().hex+'.upload'
                sftp.put(str(Path(worker.__file__).parent/filename),tmp)
                sftp.chmod(tmp,0o700);sftp.posix_rename(tmp,target)
            sftp.chmod(root,0o700)
            sftp.chmod(root+'/jobs',0o700)

    def install_worker(self):
        self.deploy_worker()
        return self.health()

    def health(self):
        engine=self.settings.get('engine','pyscf')
        exe=self.settings['remote_python'] if engine=='pyscf' else self.settings['gaussian_exe']
        profile='' if engine=='pyscf' else self.settings.get('gaussian_profile','')
        return json.loads(self.execute(['python3',self.settings['remote_root']+'/worker.py','probe',exe,profile,engine]))

    def launch(self, job, folder):
        self.deploy_worker()
        root = self.settings['remote_root']
        remote = root+'/jobs/'+job['id']
        self.execute(['mkdir','-p',remote])
        with self.sftp() as sftp:
            for name in ['spec.json','input.gjf','input.json']:
                sftp.put(str(folder/name),remote+'/'+name)
                sftp.chmod(remote+'/'+name,0o600)
        command = 'nohup '+shlex.join(['python3',root+'/worker.py','run',remote])+' > '+shlex.quote(remote+'/worker.log')+' 2>&1 < /dev/null &'
        _,out,err=self.connect().exec_command(command,timeout=15)
        if out.channel.recv_exit_status():
            raise RuntimeError(err.read().decode('utf-8',errors='replace')[:500])
        return remote

    def sync(self, job, folder):
        self.require_connection(job)
        with self.sftp() as sftp:
            for name in ['state.json','output.log','result.json']:
                try:
                    download_file(sftp, job['remote_dir']+'/'+name, folder/name)
                except FileNotFoundError:
                    pass
            if job.get('automation') and (folder/'state.json').is_file():
                state = json.loads((folder/'state.json').read_text('utf-8')).get('status')
                if state in ('completed', 'failed', 'cancelled'):
                    for name in ('calculation.chk', 'worker.log'):
                        try:
                            download_file(sftp, job['remote_dir']+'/'+name, folder/name)
                        except FileNotFoundError:
                            pass
                    atomic_json(folder/'terminal-sync.json', {'status': state})

    def cancel(self, job):
        self.require_connection(job)
        with self.sftp() as sftp:
            with sftp.open(job['remote_dir']+'/cancel','w') as f:
                f.write('cancel')

    def download_artifacts(self, job, folder):
        self.require_connection(job)
        with self.sftp() as sftp:
            for name in ['input.gjf','input.json','output.log','result.json','calculation.chk','state.json']:
                try:
                    download_file(sftp, job['remote_dir']+'/'+name, folder/name)
                except FileNotFoundError:
                    pass


class JobStore:
    def __init__(self):
        self.root = data_dir()/'jobs'
        self.root.mkdir(exist_ok=True)

    def list(self):
        jobs = []
        for p in self.root.glob('*/job.json'):
            try:
                job = json.loads(p.read_text('utf-8'))
                state = p.parent/'state.json'
                if state.exists():
                    job.update(json.loads(state.read_text('utf-8')))
                if job['status'] in ('queued','running') and time.time()-job.get('heartbeat',job['created'])>60:
                    job['connection_warning']='状态超过 60 秒未更新；请检查连接，任务可能仍在运行。'
                jobs.append(job)
            except (ValueError,OSError):
                continue
        return sorted(jobs,key=lambda j:j['created'],reverse=True)

    def folder(self, job):
        return self.root/job['id']

    def create(self, structure, calc, gjf, settings):
        job = dict(id=uuid.uuid4().hex[:16],name=structure.name,created=time.time(),status='prepared',
                   target=settings['target'],engine=settings.get('engine','pyscf'),structure=structure.to_dict(),calculation=calc.to_dict(),
                   connection={k:settings[k] for k in ['ssh_host','ssh_port','ssh_user','remote_root','host_fingerprint']})
        folder=self.folder(job)
        folder.mkdir()
        engine=settings.get('engine','pyscf')
        spec=dict(executable=(settings['remote_python'] if engine=='pyscf' else settings['gaussian_exe']) if settings['target']=='ssh' else settings['local_gaussian'],
                  profile=settings.get('gaussian_profile','') if settings['target']=='ssh' and engine=='gaussian' else '',max_seconds=86400,engine=engine,cores=calc.cores)
        atomic_json(folder/'spec.json',spec)
        atomic_json(folder/'input.json',dict(structure=structure.to_dict(),calculation=calc.to_dict()))
        (folder/'input.gjf').write_text(gjf,encoding='ascii',newline='\n')
        self.save(job)
        return job

    def save(self, job):
        atomic_json(self.folder(job)/'job.json',job)

    def launch(self, job, remote=None):
        folder=self.folder(job)
        try:
            if job['target']=='ssh':
                if remote is None:
                    raise ValueError('请先连接计算服务器。')
                job['remote_dir']=remote.launch(job,folder)
            else:
                if getattr(sys,'frozen',False):
                    cmd=[sys.executable,'--worker',str(folder)]
                else:
                    cmd=[sys.executable,str(Path(worker.__file__)),'run',str(folder)]
                with (folder/'worker.log').open('wb') as out:
                    subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                     creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS) if os.name=='nt' else 0,
                                     start_new_session=os.name!='nt')
            job['status']='queued' if job['target']=='ssh' else 'running'
        except Exception as exc:
            job.update(status='failed',message=str(exc))
            self.save(job)
            raise
        self.save(job)
        return job
