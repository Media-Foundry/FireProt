"""Diagnostic controls for locating the N1536 confidence-only parity failure."""
from contextlib import contextmanager

from fireprot.backends.amd.softmax_fp32 import SoftmaxFP32

from .owned_scores import OwnedScoreSoftmax


class ScopedScoreSoftmax(OwnedScoreSoftmax):
    def __init__(self, mode):
        if mode not in ('none', 'trunk', 'all'):
            raise ValueError('Unknown owned-score diagnostic scope')
        super().__init__()
        self.mode = mode
        self.in_confidence = 0
        self.out_of_place_calls = {}

    def __call__(self, scores, biases):
        if self.mode == 'none' or (self.mode == 'trunk' and self.in_confidence):
            result = SoftmaxFP32.__call__(self, scores, biases)
            key = f"{'confidence' if self.in_confidence else 'other'}_B{scores.shape[0]}_N{scores.shape[-1]}"
            self.out_of_place_calls[key] = self.out_of_place_calls.get(key, 0) + 1
            return result
        return super().__call__(scores, biases)

    def experiment_manifest(self):
        return dict(**super().experiment_manifest(), diagnostic_scope=self.mode,
                    out_of_place_calls=self.out_of_place_calls)


@contextmanager
def confidence_scope(model, kernel):
    """Observe the whole native confidence forward without changing its inputs."""
    module = model.aux_heads
    original = module.forward
    had_local = 'forward' in module.__dict__
    local = module.__dict__.get('forward')

    def call(*args, **kwargs):
        kernel.in_confidence += 1
        try:
            return original(*args, **kwargs)
        finally:
            kernel.in_confidence -= 1

    module.forward = call
    try:
        yield
    finally:
        if had_local:
            module.forward = local
        else:
            del module.forward
