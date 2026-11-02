#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PYTHON:=python3.11}"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
.venv/bin/python -m pip install -r experiment/requirements-runtime.txt
printf '\nReady. Run: .venv/bin/python experiment/run_all.py\nPlots: .venv/bin/python experiment/analyze_plot.py\nSimulator: .venv/bin/python experiment/simulator/server.py\n'
