# ParamRepulsor experiment and actual-training demo

Pinned official ParamRepulsor, a separate sampler-only modification, exact
36/50/100-point datasets, all 60 matched experiment outcomes, and a local viewer
of genuine pair selections and training states.

The measured result is a **trade-off**: the modification produced tighter,
better-separated groups, but generally preserved fewer exact original neighbors.
See [the original experiment report](experiment/REPORT.md) for metrics, plots,
methodology and limitations. The official baseline is unchanged at
`be8df72b1ac9041be3aae3d99f16f0d392b492dc`.

## Windows setup and launch

Use Python **3.10** (the tested desktop environment). Upstream requires Python
below 3.12; the Windows dependency snapshot also needs Python 3.10 or newer.
Annoy 1.17.3 may require the Microsoft Visual C++ build tools. CPU PyTorch is
used; no paid compute or GPU hosting is required.

1. Clone this repository and open its folder.
2. Double-click **Setup Windows.bat** to create an isolated `.venv` and install
   pinned dependencies from official PyTorch/PyPI indexes. Existing desktop
   installations can keep their already-prepared environment.
3. Double-click **Start Simulator.bat**.
4. Manually open **http://127.0.0.1:8765** on the same computer.

To select a specific interpreter, run `path\to\python.exe setup_windows.py`.
Python defaults and system packages are not changed. Stop the server with Ctrl+C
in its command window. After HTML-only changes, Ctrl+F5 refreshes the viewer;
Python server changes require a server restart.

The viewer loads the input immediately and keeps its geometry/group colors
fixed. Each purple anchor shows all actual NN, MN and FP selections together,
with overlap rings and retained duplicate counts. It then replays all 451
genuine model states in order. The estimated whole-demo duration defaults to
90 seconds and is editable during playback. Graphs are vertically stacked and
contained by both window width and height. Empty original rings and full-length
faint correspondence lines are retained, without arrowheads.

Selection is paced replay of real sampler results, not fabricated internal
search. Raw embeddings and input scores are different coordinate spaces; their
connecting lines show identity correspondence, not physical movement or a
meaningful cross-space distance. See [viewer details](experiment/simulator/README.md).

## Verification and reproduction

These checks do not rerun the 60 experiments:

```powershell
python verify_snapshot.py
& .\.venv\Scripts\python.exe -m pytest experiment\tests -q
& .\.venv\Scripts\python.exe -m unittest discover -s experiment\simulator -p 'test_*.py' -v
node experiment\simulator\test_ui.cjs
```

Node.js is required only for the optional UI logic test. `verify_windows.py`
explicitly runs fresh representative training checks; stop the viewer first
if you choose to run it. It does not rerun the complete matched experiment.

`experiment/setup.sh` and `experiment/requirements-runtime.txt` preserve the
original Python 3.11/Linux reproduction setup. The report documents the original
60-trial commands. `requirements-windows-lock.txt` records the tested Windows
dependency versions, including CPU PyTorch 2.5.1, NumPy 2.0.2, Numba 0.60.0
and Annoy 1.17.3. Desktop and Linux environments are documented separately.

## Contents and provenance

- `upstream/`: unchanged official source and Apache-2.0 license.
- `modified/`: separate source copy; only its sampler differs from the baseline.
- `inputs/`: exact original datasets and their small source archive.
- `experiment/results/`: all 60 JSON/NPZ outcomes and CSV summaries.
- `experiment/plots/`: original PNG/SVG research figures.
- `experiment/simulator/`: current local demo, observer and focused tests.
- `experiment/upstream.bundle`: small pinned upstream Git bundle.
- `SHA256SUMS.json`: historical checksums of the original delivered archive.
- `PROJECT-CHECKSUMS.json`: exact bytes committed in this project snapshot.

The report, provenance and original checksum file are historical records of the
completed experiment; the viewer subsequently received presentation updates.
[Attribution](ATTRIBUTION.md) identifies the official source and paper.
Local virtual environments, credentials, caches, raw logs, desktop backups and
repeated QA epoch recordings are excluded from Git and remain on the desktop.
No upstream GitHub repository is modified by this private project.
