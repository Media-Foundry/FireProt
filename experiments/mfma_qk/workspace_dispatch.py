"""Opt-in Rust-registry workspace dispatch around the accepted model adapter."""
import hashlib
import json
from pathlib import Path

import torch

from fireprot.ops.registry import OperationRegistry, signature_from_tensor
from experiments.mfma_qk.model_workspace import ModelWorkspaceFusion


def load_workspace(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Workspace control file hash mismatch')
    control = json.loads(data)
    options = ({'allocation_policy': control['allocation_policy']}
               if 'allocation_policy' in control else {})
    return ModelWorkspaceFusion(control['fusion'], microbatch=control.get('microbatch', 256),
                                **options)


class WorkspaceDispatcher:
    """Fallback is the frozen accepted dispatcher, including its D16 coverage."""
    def __init__(self, accepted, workspace):
        self.accepted = accepted
        self.kernel = accepted.kernel
        self.workspace = workspace
        self.fallback_reasons = {}
        self.registry = OperationRegistry()
        self.registry.register(operation='attention_workspace', name='hip_qk_softmax_workspace_fp32',
            version='combined-v12-rust-lease-v1', priority=20, function=self._run,
            supports=lambda signature: (dict(signature.attributes).get('eligible') == 'True',
                                       'Outside the measured workspace signature'))

    def __call__(self, native, query, key, value, biases, use_high_precision=False):
        if self.workspace is None:
            return self.accepted(native, query, key, value, biases, use_high_precision=use_high_precision)
        eligible = False
        reason = 'accepted_adapter_ineligible'
        if self.accepted._eligible(query, key, value, biases):
            eligible, reason = self.workspace.eligibility(query, key, value, *biases)
        if not eligible:
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
        if type(query) is not torch.Tensor:
            return self.accepted(native, query, key, value, biases, use_high_precision=use_high_precision)
        signature = signature_from_tensor('attention_workspace', query, architecture='gfx90a',
            mode='training' if torch.is_grad_enabled() else 'inference', mask='pair_and_token',
            inplace=False, attributes={'eligible': eligible, 'high_precision': use_high_precision})
        def fallback(*args, **kwargs):
            return self.accepted(native, *args, **kwargs)
        return self.registry.dispatch(signature, fallback, query, key, value, biases,
                                      use_high_precision=use_high_precision)

    def _run(self, query, key, value, biases, use_high_precision=False):
        # FP32 is the accepted boundary for both upstream precision modes.
        with torch.amp.autocast('cuda', dtype=torch.float32):
            return self.workspace(query, key, value, *biases)

    def manifest(self):
        return dict(**self.accepted.manifest(), experimental_workspace=dict(
            enabled=self.workspace is not None, registry=self.registry.manifest(),
            fallback_reasons=dict(self.fallback_reasons),
            workspace=None if self.workspace is None else self.workspace.manifest()))
