"""GPU guard checks for the experimental duplicate-template scope.

Run with HIP_VISIBLE_DEVICES=7 and the pinned OpenFold3 Torch environment.
This is a guard/aliasing screen, not a model or performance benchmark.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from template_reuse import template_reuse_scope


class Block:
    training = False

    def __init__(self):
        self.slots = []

    def forward(self, t, mask, **kwargs):
        self.slots.append(t.shape[1])
        # Independent slots, matching the native template block's dependency.
        return t.add_(1)


class Stack:
    def __init__(self, block):
        self.blocks = [block]


@torch.inference_mode()
def main(destination):
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    out = Path(destination)
    out.mkdir(exist_ok=False)
    block = Block()
    model = SimpleNamespace(template_embedder=SimpleNamespace(template_pair_stack=Stack(block)))
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    t = torch.zeros(1, 4, 1536, 1536, 64, device='cuda', dtype=torch.float32)
    mask = torch.ones(1, 4, 1536, 1536, device='cuda', dtype=torch.float32)
    original = block.forward
    checks = []
    with template_reuse_scope(model, out, source_sha):
        value = block.forward(t, mask, inplace_safe=True)
        assert value.data_ptr() == t.data_ptr() and block.slots[-1] == 1
        assert torch.all(t == 1).item()
        checks.append('identical slots computed once; all outputs and alias preserved')

        t.zero_()
        t[0, 3, 0, 0, 0] = -0.0
        block.forward(t, mask, inplace_safe=True)
        assert block.slots[-1] == 4 and torch.all(t == 1).item()
        checks.append('signed-zero difference selects native fallback')

        t.fill_(1)
        mask[0, 3, 0, 0] = 0
        block.forward(t, mask, inplace_safe=True)
        assert block.slots[-1] == 4 and torch.all(t == 2).item()
        checks.append('different mask selects native fallback')
        mask.fill_(1)

        block.training = True
        block.forward(t, mask, inplace_safe=True)
        assert block.slots[-1] == 4
        checks.append('training selects native fallback')
        block.training = False

        block.forward(t, mask, inplace_safe=False)
        assert block.slots[-1] == 4
        checks.append('non-inplace call selects native fallback')
    assert block.forward == original and 'forward' not in block.__dict__
    checks.append('original bound method restored')
    counts = json.loads((out / 'template-reuse.json').read_text())['counts']
    assert counts == dict(eligible=1, distinct=2, unsupported=2, reused_slots=3)
    (out / 'screen.json').write_text(json.dumps(dict(passed=True, checks=checks,
        counts=counts, torch=str(torch.__version__), hip=torch.version.hip), indent=2) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
