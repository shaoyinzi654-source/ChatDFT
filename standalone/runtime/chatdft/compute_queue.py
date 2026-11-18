"""Acquire the required Linux worker locks together, without holding a partial set."""
from contextlib import ExitStack
import fcntl
from pathlib import Path
import os


def try_locks(root, names):
    resources = ExitStack()
    try:
        for name in names:
            lock = resources.enter_context((root/name).open('a'))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        resources.close()
        return None
    except BaseException:
        resources.close()
        raise
    return resources


def can_share_with_md(folder):
    """Reserve enough headroom for one standard MD box and one small quantum job."""
    import json

    if (os.cpu_count() or 1) < 16:
        return False
    try:
        job = json.loads((folder/'input.json').read_text('utf-8'))
        memory = int(job['calculation']['memory_gb'])
        meminfo = {line.split(':', 1)[0]: int(line.split()[1])
                   for line in Path('/proc/meminfo').read_text().splitlines() if line.split()}
        return (len(job['structure']['atoms']) <= 100 and memory <= 8 and
                meminfo['MemTotal'] >= 14*1024**2 and meminfo['MemAvailable'] >= (memory+2)*1024**2)
    except (OSError, KeyError, IndexError, ValueError, TypeError):
        return False
