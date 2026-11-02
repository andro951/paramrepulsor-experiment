"""Read-only instrumentation of real pair generation; no replacement sampler."""
from __future__ import annotations

import time


class PairRecorder:
    """Tap actual Annoy queries/distances and the unchanged sampler's returns.

    The ANN library's internal tree traversal and Numba MN/FP candidate/retry
    loops are not exposed. They must never be presented as a fabricated trace.
    """

    def __init__(self, module, variant, seed, publish):
        self.module, self.variant, self.seed, self.publish = module, variant, seed, publish
        self.calls = []
        self.scaled = None

    def __enter__(self):
        recorder = self
        module = self.module
        self.original = {name: getattr(module, name) for name in ('AnnoyIndex', 'scale_dist', 'generate_pair')}

        class IndexTap:
            def __init__(self, *args, **kwargs):
                self.inner = recorder.original['AnnoyIndex'](*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def get_nns_by_item(self, source, *args, **kwargs):
                result = self.inner.get_nns_by_item(source, *args, **kwargs)
                recorder.calls.append(dict(sequence=len(recorder.calls), kind='annoy_query', source=int(source), returned=[int(v) for v in result]))
                return result

            def get_distance(self, source, target):
                result = self.inner.get_distance(source, target)
                recorder.calls.append(dict(sequence=len(recorder.calls), kind='distance_check', source=int(source), target=int(target), distance=float(result)))
                return result

        def scaled_tap(*args, **kwargs):
            result = recorder.original['scale_dist'](*args, **kwargs)
            recorder.scaled = dict(values=result.tolist(), candidates=args[2].tolist(), sigma=args[1].tolist())
            return result

        def generate_tap(*args, **kwargs):
            start = time.perf_counter()
            result = recorder.original['generate_pair'](*args, **kwargs)
            nn, mn, fp, _ = result
            document = dict(schema_version=1, variant=recorder.variant, seed=recorder.seed,
                            n=int(args[0].shape[0]), sampler_input=args[0].tolist(),
                            calls=recorder.calls, scaled_distances=recorder.scaled,
                            pairs=dict(NN=nn.tolist(), MN=mn.tolist(), FP=fp.tolist()),
                            capture_seconds=time.perf_counter()-start,
                            provenance='Actual Annoy get_nns_by_item/get_distance calls and unchanged generate_pair return arrays.',
                            presentation='Paced per-source replay. Final NN choices occur after global distance scaling; display groups checks and final results by source, not wall-clock execution order.',
                            not_traced=['Annoy internal tree traversal', 'Numba MN/FP candidate draws and rejection loops', 'parallel internal ordering of pair-array writes'])
            recorder.publish(document)
            return result

        module.AnnoyIndex, module.scale_dist, module.generate_pair = IndexTap, scaled_tap, generate_tap
        return self

    def __exit__(self, *_):
        for name, value in self.original.items():
            setattr(self.module, name, value)
