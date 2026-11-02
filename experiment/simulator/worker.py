"""Train the real selected implementation; observe, never modify, optimizer steps."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import math
import struct
import sys
import time
import traceback


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    # Windows readers/antivirus can briefly deny replacement of an open file.
    # Retry only this status-file operation; model computation stays unchanged.
    for attempt in range(101):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if os.name != 'nt' or attempt == 100:
                raise
            time.sleep(0.01)


def run(root: Path, variant: str, n: int, seed: int, state_path: Path) -> None:
    started = time.perf_counter()
    source = root / ('upstream' if variant == 'baseline' else 'modified') / 'src'
    if not source.is_dir():
        raise FileNotFoundError(f'Missing selected implementation: {source}')
    # Publish real inputs before importing Torch/Numba or compiling any sampler.
    with (root / 'inputs' / 'datasets' / f'paramrep_2d_{n}.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    f32 = lambda value: struct.unpack('f', struct.pack('f', float(value)))[0]
    values = [[f32(row['math_score']), f32(row['english_score'])] for row in rows]
    if len(rows) != n or not all(math.isfinite(v) for row in values for v in row):
        raise ValueError('Dataset size or numeric coordinates are invalid.')
    state = dict(status='preparing', stage='loading_runtime', replay_version=1,compute_started_at=time.time(),
                 variant=variant, n=n, seed=seed, epoch=0, total_epochs=450,
                 source=str(source), points=[dict(point_id=r['point_id'], group=r['group'], x=v[0], y=v[1]) for r,v in zip(rows,values)],
                 coordinates=None, initial_coordinates=None, elapsed_seconds=0, pairs_ready=False)
    atomic_json(state_path, state)
    os.environ['TORCH_DEVICE'] = 'cpu'
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(source))
    from parampacmap import ParamPaCMAP
    from parampacmap.utils import data
    from pair_trace import PairRecorder
    x = np.array(values, dtype=np.float32)
    state.update(stage='selecting_pairs', elapsed_seconds=time.perf_counter()-started)
    atomic_json(state_path, state)
    frames = []
    journal_path = state_path.with_name('frames.jsonl')
    journal_path.write_text('', encoding='utf-8')

    def publish_pairs(document):
        atomic_json(state_path.with_name('pairs.json'), document)
        state.update(stage='training', pairs_ready=True, elapsed_seconds=time.perf_counter()-started)
        atomic_json(state_path, state)

    class ObservedParamPaCMAP(ParamPaCMAP):
        def _inference(self, test_loader):
            # Adapter for cloud runtimes that forbid multiprocessing worker IPC.
            # These <=100 rows fit the original single inference batch; the model
            # and row ordering are identical, but no worker transport is created.
            with torch.inference_mode():
                tensor = torch.as_tensor(test_loader.dataset.data, dtype=self._dtype, device=self.device)
                return self.model(tensor).detach().float().cpu().numpy()

        def snapshot(self, epoch):
            # The default backbone has no stochastic or state-mutating forward layers.
            # Direct forward avoids creating a DataLoader iterator (which consumes RNG).
            random_state = torch.get_rng_state().clone()
            with torch.inference_mode():
                tensor = torch.as_tensor(x, dtype=self._dtype, device=self.device)
                coordinates = self.model(tensor).detach().float().cpu().numpy()
            if not torch.equal(random_state, torch.get_rng_state()):
                raise RuntimeError('Observation consumed training RNG; aborting rather than changing the experiment.')
            if not np.isfinite(coordinates).all():
                raise ValueError(f'Non-finite model embedding after epoch {epoch}.')
            values = coordinates.tolist()
            frames.append(dict(epoch=epoch, coordinates=values))
            # An append-only journal exposes every genuine intermediate frame.
            with journal_path.open('a', encoding='utf-8') as journal:
                journal.write(json.dumps(frames[-1], allow_nan=False)+'\n')
            if epoch == 0:
                state['initial_coordinates'] = values
            state.update(status='running', stage='training', epoch=epoch, coordinates=values,
                         elapsed_seconds=time.perf_counter() - started)
            atomic_json(state_path, state)

        def _train_epoch(self, train_loader, epoch, optimizer, has_labels=False):
            if epoch == 0:
                self.snapshot(0)
            super()._train_epoch(train_loader, epoch, optimizer, has_labels=has_labels)
            self.snapshot(epoch + 1)

    reducer = ObservedParamPaCMAP(seed=seed)
    with PairRecorder(data, variant, seed, publish_pairs):
        reducer.fit(x)
    final = reducer._embedding
    np.testing.assert_allclose(final, np.asarray(frames[-1]['coordinates']), rtol=1e-6, atol=1e-6)
    archive = state_path.with_name('epochs.json')
    atomic_json(archive, dict(variant=variant, n=n, seed=seed, frames=frames,
                            note='Actual initial and post-epoch raw model embeddings; no interpolation.'))
    state.update(status='completed', stage='completed', elapsed_seconds=time.perf_counter() - started,
                 frame_count=len(frames), saved_epochs=archive.name)
    atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'modified'], required=True)
    parser.add_argument('--n', type=int, choices=[36, 50, 100], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.root.resolve(), args.variant, args.n, args.seed, args.state)
    except Exception as exc:
        atomic_json(args.state, dict(status='error', error=f'{type(exc).__name__}: {exc}'))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
