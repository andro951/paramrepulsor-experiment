"""Create/reuse this project's isolated CPU environment; never change defaults."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not (3,10) <= sys.version_info[:2] < (3,12):
        raise SystemExit('Use Python 3.10 (tested), or 3.11; upstream excludes 3.12+.')
    root=Path(__file__).resolve().parent
    target=root/'.venv'
    if not target.exists():
        venv.EnvBuilder(with_pip=True).create(target)
    python=target/'Scripts'/'python.exe'
    if not python.is_file():
        raise SystemExit('Existing .venv is not a Windows Python environment; preserve it and choose a fresh checkout.')
    subprocess.run([str(python),'-m','pip','install','--index-url','https://download.pytorch.org/whl/cpu','torch==2.5.1'],check=True)
    subprocess.run([str(python),'-m','pip','install','--index-url','https://pypi.org/simple','-r',str(root/'requirements-windows-lock.txt')],check=True)
    subprocess.run([str(python),'-m','pip','check'],check=True)
    print('Ready: double-click Start Simulator.bat, then open http://127.0.0.1:8765 manually.')

if __name__=='__main__':
    main()
