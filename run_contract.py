"""Run just the contract gate against a live server (mutation-test harness)."""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.gates import Server

srv = Server(8000)
if not srv.ensure():
    print("server failed to start")
    sys.exit(2)
try:
    p = subprocess.run([sys.executable, "-u", "-m", "backend.contract"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, cwd=os.getcwd())
    print(p.stdout)
    print("EXIT", p.returncode)
finally:
    srv.stop()
