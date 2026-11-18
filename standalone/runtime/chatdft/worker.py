"""Standard-library-only Gaussian runner, shared by desktop and remote server.

No model output is executed. Job JSON contains only a configured executable/profile.
Gaussian software and its licence are supplied separately by the owner.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    os.replace(temporary,path)


def probe(exe, profile='', engine='gaussian'):
    resolved = shutil.which(exe) if not os.path.isabs(exe) else exe
    ready = bool(resolved and Path(resolved).is_file() and os.access(resolved,os.X_OK))
    version = ''
    if engine=='pyscf' and ready:
        checked=subprocess.run([resolved,'-c','import pyscf, geometric; print(pyscf.__version__)'],capture_output=True,text=True,timeout=30)
        ready=checked.returncode==0;version=checked.stdout.strip()
    return dict(host=__import__('socket').gethostname(),cores=os.cpu_count(),engine=engine,engine_version=version,
                executable=resolved or exe, gaussian_installed=ready,engine_ready=ready,
                profile_exists=bool(profile and Path(profile).is_file()),
                free_disk_gb=round(shutil.disk_usage(Path.home()).free/1024**3,1),
                note='PySCF 环境已检测，实际任务以收敛与结果文件为准。' if engine=='pyscf' else '检测可执行文件存在；授权与运行状态需以真实测试计算验证。')


def run(folder):
    folder = Path(folder).resolve()
    spec = json.loads((folder/'spec.json').read_text('utf-8'))
    status = dict(status='queued',pid=os.getpid(),submitted=time.time(),heartbeat=time.time(),
                  returncode=None,message='服务器已接收，等待可用计算资源。')
    if (folder/'cancel').exists():
        status.update(status='cancelled',finished=time.time())
        write(folder/'state.json',status)
        return
    write(folder/'state.json',status)
    proc = None
    resource_lock = None
    try:
        exe, profile = spec['executable'], spec.get('profile','')
        engine=spec.get('engine','gaussian')
        health = probe(exe,profile,engine)
        if not health['gaussian_installed']:
            raise RuntimeError('未找到 '+engine+' 计算环境；请检查服务器安装。')
        if profile and not Path(profile).is_file():
            raise RuntimeError('Gaussian 环境脚本不存在，请检查安装路径。')
        requested = int(spec.get('cores', 8))
        effective = requested
        if os.name != 'nt':
            if __package__:
                from .compute_queue import can_share_with_md, try_locks
            else:
                from compute_queue import can_share_with_md, try_locks
            while resource_lock is None:
                if (folder/'cancel').exists():
                    status.update(status='cancelled', message='排队任务已取消。')
                    return
                share = engine == 'pyscf' and can_share_with_md(folder)
                locks = ('quantum.lock',) if share else ('quantum.lock', 'compute.lock')
                resource_lock = try_locks(folder.parent.parent, locks)
                if resource_lock is None:
                    status['heartbeat'] = time.time()
                    write(folder/'state.json', status)
                    time.sleep(2)
            if share:
                effective = min(4, requested)
        status.update(status='running', started=time.time(), effective_cores=effective,
                      message=f'正在计算，使用 {effective} 个 CPU 核。')
        write(folder/'state.json', status)
        scratch = folder/'scratch'
        scratch.mkdir(exist_ok=True)
        env = dict(os.environ,GAUSS_SCRDIR=str(scratch))
        cmd = [health['executable']]
        if engine=='pyscf':
            cmd += [str(Path(__file__).resolve().parent/'pyscf_engine.py'),str(folder/'input.json'),str(folder/'result.json')]
            env.update(OMP_NUM_THREADS=str(effective), OPENBLAS_NUM_THREADS=str(effective),
                       MKL_NUM_THREADS=str(effective), CHATDFT_EFFECTIVE_CORES=str(effective))
        if profile and os.name != 'nt':
            cmd = ['/bin/bash','-c','source "$1" && exec "$2"','chatdft',profile,health['executable']]
        with (folder/'input.gjf').open('rb') as inp, (folder/'output.log').open('wb') as out:
            proc = subprocess.Popen(cmd,cwd=folder,stdin=inp,stdout=out,stderr=subprocess.STDOUT,env=env,
                                    start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            while proc.poll() is None:
                cancelled = (folder/'cancel').exists()
                timed_out = time.time()-status['started'] > int(spec.get('max_seconds',86400))
                if cancelled or timed_out:
                    if os.name == 'nt':
                        subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait(timeout=15)
                    status.update(status='cancelled' if cancelled else 'failed',message='用户取消' if cancelled else '超过 24 小时运行上限')
                    break
                status['heartbeat'] = time.time()
                write(folder/'state.json',status)
                time.sleep(1)
        status['returncode'] = proc.returncode
        if status['status']=='running':
            with (folder/'output.log').open('rb') as log:
                log.seek(max(0,log.seek(0,2)-65536))
                tail = log.read().decode('utf-8',errors='replace')
            normal = bool(re.search(r'Normal termination of Gaussian[^\r\n]*\s*\Z', tail)) and tail.rfind('Normal termination of Gaussian') > tail.rfind('Error termination')
            if engine=='pyscf':
                normal=(folder/'result.json').exists() and json.loads((folder/'result.json').read_text('utf-8')).get('normal_termination') is True
            status.update(status='completed' if proc.returncode==0 and normal else 'failed',
                          message=engine+' 计算完成' if proc.returncode==0 and normal else engine+' 未正常结束；请检查日志')
    except Exception as exc:
        status.update(status='failed',message=str(exc))
    finally:
        if resource_lock:
            resource_lock.close()
        status.update(finished=time.time(),heartbeat=time.time())
        write(folder/'state.json',status)


if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[1]=='run':
        run(sys.argv[2])
    elif len(sys.argv)>2 and sys.argv[1]=='probe':
        print(json.dumps(probe(sys.argv[2],sys.argv[3] if len(sys.argv)>3 else '',sys.argv[4] if len(sys.argv)>4 else 'gaussian'),ensure_ascii=False))
    else:
        raise SystemExit('Usage: worker.py run JOB_DIR | probe EXECUTABLE [PROFILE]')
