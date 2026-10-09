"""CPU fallback contract; accepted GPU arithmetic is covered by the GPU screen."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("global_softmax_experiment", Path(__file__).with_name("global_softmax.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unsupported_input_preserves_fallback_arguments():
    kernel = SimpleNamespace(device=0)
    captured = []
    native = object()
    expected = object()
    def fallback(*args, **kwargs):
        captured.append((args, kwargs))
        return expected
    dispatch = module.GlobalSoftmaxDispatcher(fallback, kernel)
    q, k, v, biases = object(), object(), object(), [object()]
    with torch.inference_mode():
        actual = dispatch(native, q, k, v, biases, use_high_precision=True)
    assert actual is expected
    assert captured == [((native, q, k, v, biases), dict(use_high_precision=True))]
    assert dispatch.calls == dispatch.groups_launched == 0


def test_cpu_probabilities_rejected_before_kernel_access():
    dispatch = module.GlobalSoftmaxDispatcher(None, SimpleNamespace(device=0))
    with torch.inference_mode(), pytest.raises(ValueError, match="Unsupported"):
        dispatch.probabilities(torch.zeros(1), [torch.zeros(1), torch.zeros(1)])
