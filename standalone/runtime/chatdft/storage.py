from __future__ import annotations
import base64
import ctypes
import json
import os
from pathlib import Path


def data_dir():
    path = Path(os.environ.get('CHATDFT_DATA_DIR') or Path(os.environ.get('LOCALAPPDATA', Path.home()/'.local/share')) / 'ChatDFT-Studio')
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


DEFAULTS = dict(provider='codex', model='', api_base='https://api.openai.com/v1', api_protocol='responses',
                api_secret='', codex_path='', target='ssh', ssh_host='', ssh_port=22,
                ssh_user='', ssh_key='', host_fingerprint='',
                remote_root='/opt/chatdft', gaussian_exe='/opt/gaussian/g16/g16', gaussian_profile='/opt/gaussian/g16/bsd/g16.profile',
                local_gaussian='', timeout=300,engine='pyscf',remote_python='/opt/chatdft/venv/bin/python',
                gromacs_exe='/usr/bin/gmx',gromacs_legacy_exe='/opt/chatdft/md/gromacs-2018/bin/gmx')


def load_settings():
    try:
        return {**DEFAULTS, **json.loads((data_dir()/'settings.json').read_text('utf-8'))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_settings(settings):
    atomic_json(data_dir()/'settings.json', settings)


def protect_secret(text, decrypt=False):
    if not text:
        return ''
    if os.name != 'nt':
        if decrypt:
            return os.environ.get('CHATDFT_API_KEY', '')
        raise ValueError('非 Windows 系统请通过 CHATDFT_API_KEY 环境变量设置密钥。')
    from ctypes import wintypes
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_char))]
    raw = base64.b64decode(text) if decrypt else text.encode('utf-8')
    buf = ctypes.create_string_buffer(raw)
    src, dst = Blob(len(raw), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), Blob()
    fn = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    fn.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    fn.restype = wintypes.BOOL
    if not fn(ctypes.byref(src), None, None, None, None, 1, ctypes.byref(dst)):
        raise ValueError('Windows 凭据加解密失败。')
    try:
        value = ctypes.string_at(dst.data, dst.size)
        return value.decode('utf-8') if decrypt else base64.b64encode(value).decode('ascii')
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(dst.data, ctypes.c_void_p))
