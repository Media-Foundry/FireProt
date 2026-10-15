"""HIP7 synthetic gate for invariant right packing; no full-model speedup claim."""
from pathlib import Path
import hashlib
import json
import os
import statistics
import sys

import torch
from esm.models.esmfold2.layers import TriangleMultiplicativeBlock
from fireprot.adapters.esmfold2.triangle import packed_contract, LAYERS_SHA256


def run(destination):
    destination = Path(destination)
    assert os.environ.get('HIP_VISIBLE_DEVICES') == '7'
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx90a'
    import esm.models.esmfold2.layers as layers
    assert hashlib.sha256(Path(layers.__file__).read_bytes()).hexdigest() == LAYERS_SHA256
    torch.manual_seed(101)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    with torch.no_grad():
        for n in (600, 1000):
            backing = torch.randn(1, n, n, 512, device='cuda', dtype=torch.float32)
            left, right = backing.chunk(2, dim=-1)
            for flow in ('outgoing', 'incoming'):
                engine = TriangleMultiplicativeBlock(256, 256, flow).eval()
                for autocast in (False, True):
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                        ref = engine._triangular_contract_chunked(left, right, 64)
                        got = packed_contract(left, right, 64, flow)
                        torch.cuda.synchronize()
                        exact = torch.equal(ref, got)
                        record = dict(n=n, flow=flow, autocast=autocast,
                            output_dtype=str(ref.dtype), bitwise_equal=exact,
                            max_abs_delta=float((ref-got).abs().max().item()))
                        if exact:
                            timings = {'native': [], 'reuse_right_pack': []}
                            for i in range(8):
                                for arm in (('native','reuse_right_pack') if i % 2 == 0 else ('reuse_right_pack','native')):
                                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                    begin.record()
                                    output = engine._triangular_contract_chunked(left,right,64) if arm=='native' else packed_contract(left,right,64,flow)
                                    end.record();end.synchronize()
                                    if i >= 2:timings[arm].append(begin.elapsed_time(end))
                                    del output
                            record['samples_ms']=timings
                            record['medians_ms']={k:statistics.median(v) for k,v in timings.items()}
                        records.append(record)
                        print(json.dumps(record),flush=True)
                        del ref,got
                del engine
            del left,right,backing
    result=dict(schema_version=1,records=records,all_exact=all(r['bitwise_equal'] for r in records),
        scope='Synthetic operator screen only; seed101, strict deterministic, chunk64, FP32/BF16-autocast, HIP7. Native scope includes repeated operand layout conversion.',
        torch=str(torch.__version__),hip=torch.version.hip)
    with destination.open('x') as f:json.dump(result,f,indent=2)


if __name__ == '__main__':
    run(sys.argv[1])
