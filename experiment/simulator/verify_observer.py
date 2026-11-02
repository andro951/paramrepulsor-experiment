"""Compare observed versus unobserved real training for one isolated variant."""
import argparse
import csv
import json
from pathlib import Path
import tempfile

from worker import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['baseline', 'modified'], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='paramrep-observer-check-') as temporary:
        state_path = Path(temporary) / 'state.json'
        run(root, args.variant, 36, 42, state_path)
        state = json.loads(state_path.read_text())
        archive = json.loads((state_path.parent/'epochs.json').read_text())
        assert len(archive['frames']) == 451
        assert [frame['epoch'] for frame in archive['frames']] == list(range(451))
        assert all(len(frame['coordinates']) == 36 for frame in archive['frames'])

        import numpy as np
        import torch
        from parampacmap import ParamPaCMAP
        class InProcessInferenceOnly(ParamPaCMAP):
            def _inference(self, loader):
                with torch.inference_mode():
                    value = torch.as_tensor(loader.dataset.data, dtype=self._dtype, device=self.device)
                    return self.model(value).detach().float().cpu().numpy()
        with (root/'inputs/datasets/paramrep_2d_36.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        x = np.array([[float(row['math_score']), float(row['english_score'])] for row in rows],dtype=np.float32)
        reference = InProcessInferenceOnly(seed=42)
        reference.fit(x)
        np.testing.assert_array_equal(reference._embedding, np.asarray(state['coordinates'],dtype=np.float32))
        report = dict(variant=args.variant,n=36,seed=42,epochs=450,
                      actual_saved_frames=451,observer_final_exactly_matches_unobserved_training=True,
                      each_frame_preserves_row_identity=True,
                      observation_rng_invariance_checked_each_epoch=True)
        destination = Path(__file__).with_name(f'observer_{args.variant}.json')
        destination.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
