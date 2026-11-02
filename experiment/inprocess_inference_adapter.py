"""Pytest-only local inference transport adapter for the sandbox.

Run with ``pytest -p experiment.inprocess_inference_adapter upstream/tests``.
The official test files are unchanged. This is not a pristine-runtime test run:
only the inference DataLoader transport is replaced (workers=0, same data/order,
batch size and model math). No training or sampling code is patched.
"""
import torch
from parampacmap import ParamPaCMAP


def pytest_configure(config):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original = ParamPaCMAP._inference

    def inprocess_inference(self, test_loader):
        loader = torch.utils.data.DataLoader(
            test_loader.dataset,
            batch_size=test_loader.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=0,
            persistent_workers=False,
        )
        return original(self, loader)

    ParamPaCMAP._inference = inprocess_inference
