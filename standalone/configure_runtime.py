"""Configure the native Windows launcher for the selected Python environment."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import sys
import sysconfig
import tempfile


def main():
    if sysconfig.get_platform() != 'win-amd64' or sys.version_info[:2] != (3, 13):
        raise ValueError('ChatDFT requires 64-bit Python 3.13 on Windows.')
    import PySide6.QtWidgets
    import PySide6QtAds
    import matplotlib
    import mdtraj
    import paramiko
    import qtawesome
    import rdkit
    import requests
    import scipy

    locate = ctypes.windll.kernel32.GetModuleFileNameW
    locate.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    locate.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    length = locate(sys.dllhandle, buffer, len(buffer))
    if not 0 < length < len(buffer):
        raise OSError('Could not find the loaded Python DLL.')
    root = Path(__file__).resolve().parent
    exe = Path(sys.executable).resolve(strict=True)
    dll = Path(buffer.value).resolve(strict=True)
    with tempfile.NamedTemporaryFile(dir=root, delete=False, prefix='.runtime-') as stream:
        temp = Path(stream.name)
        stream.write(f'{exe}\n{dll}\n'.encode('utf-8'))
    try:
        temp.replace(root / 'ChatDFT.runtime')
    finally:
        temp.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
