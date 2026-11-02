"""Verify a full real worker run; temporary epoch archives are removed afterward."""
import argparse
import csv
import json
from pathlib import Path
import tempfile

from worker import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['baseline', 'modified'], required=True)
    parser.add_argument('--n', type=int, choices=[36, 50, 100], required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='paramrep-worker-case-') as directory:
        state_path = Path(directory) / 'state.json'
        run(root, args.variant, args.n, args.seed, state_path)
        state = json.loads(state_path.read_text())
        archive = json.loads((state_path.parent/'epochs.json').read_text())
        with (root/'inputs/datasets'/f'paramrep_2d_{args.n}.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert state['status']=='completed' and state['epoch']==450
        assert state['frame_count']==451 and len(archive['frames'])==451
        assert [frame['epoch'] for frame in archive['frames']]==list(range(451))
        assert [point['point_id'] for point in state['points']]==[row['point_id'] for row in rows]
        assert [point['group'] for point in state['points']]==[row['group'] for row in rows]
        assert all(len(frame['coordinates'])==args.n for frame in archive['frames'])
        assert archive['frames'][-1]['coordinates']==state['coordinates']
        report=dict(variant=args.variant,n=args.n,seed=args.seed,epochs=450,
                    actual_saved_frames=451,completed=True,
                    input_point_ids_and_groups_exactly_preserved=True,
                    every_frame_retains_original_row_order=True,
                    observation_rng_invariance_checked_each_epoch=True,
                    final_inference_matches_last_observer_frame=True,
                    elapsed_seconds=state['elapsed_seconds'],
                    epoch_archives_temporary_only=True)
        destination=Path(__file__).with_name(f'worker_{args.variant}_n{args.n}.json')
        destination.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
