"""Synthetic GPU check against the pinned upstream conditioning module.

Uses randomized parameters, not published model weights. No timing claim.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning

from verified_reuse import verified_reuse


class Diffusion(torch.nn.Module):
    def __init__(self, conditioning):
        super().__init__()
        self.diffusion_conditioning = conditioning

    def forward(self, inputs):
        return self.diffusion_conditioning(**inputs)


class Sample(torch.nn.Module):
    def __init__(self, conditioning):
        super().__init__()
        self.diffusion_module = Diffusion(conditioning)

    def _sample_rollout(self, inputs):
        for noise in (4608.0, 4473.96728515625, 105.991943359375, 0.0076895058155):
            values = dict(inputs, t=torch.tensor(noise, device="cuda"))
            self.diffusion_module(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    source = Path(inspect.getfile(DiffusionConditioning))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != "5bf5170d836c3b56cf665a9dc68bb935018ef21f779f516bc97ee0be5b1c6305":
        raise RuntimeError("Upstream conditioning source changed")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(101)
    module = DiffusionConditioning(449, 384, 128, 256, 32, 2, 16.0,
                                  tune_chunk_size=True).cuda().eval()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            # Avoid all-zero initialized residual projections hiding differences.
            if "layer_norm" not in name:
                parameter.normal_(0, 0.03)
    sample = Sample(module).eval()
    checks = []
    with torch.inference_mode():
        for length in (1000, 1536):
            indices = torch.arange(length, device="cuda")[None, None, :]
            zeros = torch.zeros_like(indices)
            inputs = dict(si_input=torch.randn(1, 1, length, 449, device="cuda"),
                          si_trunk=torch.randn(1, 1, length, 384, device="cuda"),
                          zij_trunk=torch.randn(1, 1, length, length, 128, device="cuda"),
                          chunk_size=1024)
            for mask_kind in ("ones", "partial", "zero"):
                mask = torch.ones_like(indices, dtype=torch.float32)
                if mask_kind == "partial":
                    mask[..., ::3] = 0
                elif mask_kind == "zero":
                    mask.zero_()
                batch = dict(token_mask=mask, residue_index=indices, token_index=indices,
                             asym_id=zeros, entity_id=zeros, sym_id=zeros)
                for enabled in (True, False):
                    destination = args.output / f"n{length}-{mask_kind}-{enabled}"
                    destination.mkdir()
                    with verified_reuse(sample, destination) as report:
                        sample._sample_rollout(dict(inputs, batch=batch,
                                                    use_conditioning=enabled))
                    checks.append(dict(length=length, mask=mask_kind,
                                       use_conditioning=enabled, report=report))
            del inputs
    torch.cuda.synchronize()
    result = dict(schema_version=1, source_sha256=source_sha, seed=101,
                  torch=torch.__version__, hip=torch.version.hip,
                  device=str(torch.cuda.get_device_properties(0)),
                  weights="synthetic randomized parameters", performance_evidence=False,
                  checks=checks)
    (args.output / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=len(checks), output=str(args.output / "screen.json"))))


if __name__ == "__main__":
    main()
