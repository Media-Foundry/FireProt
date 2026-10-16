"""Qualify packaged N1000 batch operators against the same native tensor batch."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from fireprot_workers.plm import load_model
from fireprot_workers.plm_batch import prepare_forward
from fireprot_workers.atlasfold import read_sequence
from fireprot_workers.transport import save_arrays
from fireprot.adapters.plm.runtime import optimized_scope


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    library = Path(sys.argv[2]).resolve()
    output = Path(sys.argv[3]); output.mkdir(exist_ok=False)
    name = request['workload']['model_name']
    torch.set_num_threads(8); torch.manual_seed(request['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, sequence = read_sequence(request['input']['path'])
    assert len(sequence) == 1000
    report = dict(model=name, hip_device=6, seed=request['seed'], precision='native',
                  boundary='warm packaged forward including context entry/exit; not complete invocation',
                  reference='native identical batch grouping; maintained ESM scope', cases=[])
    with torch.inference_mode():
        model, _, _ = load_model(request, sequence)
        request['workload']['batch'] = dict(items=4,max_batch_size=4,max_padded_tokens=4008,grouping='exact_length')
        request['resources']['hip_esm'] = dict(path=str(library),sha256=hashlib.sha256(library.read_bytes()).hexdigest())
        for size in (1,2,4):
            sequences = [(sequence[i*137:] + sequence[:i*137]) for i in range(size)]
            forward = prepare_forward(model, name, sequences)
            expected = {k:v.clone() for k,v in forward().items()}
            assert all(torch.isfinite(v).all() for v in expected.values())
            originals = [(m,m.forward) for m in model.modules()]
            runtime = None

            def run(candidate):
                nonlocal runtime
                if not candidate: return forward()
                with optimized_scope(model,request,1000,batch_size=size) as state:
                    actual = forward()
                runtime = state
                return actual

            pairs = []
            for order in [(False,True),(True,False),(False,True)]:
                pair = {}
                for candidate in order:
                    for _ in range(2): run(candidate)
                    values = []
                    for _ in range(3):
                        torch.cuda.synchronize(); start = time.perf_counter()
                        actual = run(candidate); torch.cuda.synchronize()
                        values.append(time.perf_counter()-start)
                        assert all(torch.equal(v.view(torch.uint8),actual[k].view(torch.uint8)) for k,v in expected.items()), (name,size)
                        assert all(m.forward == original for m,original in originals)
                    pair['candidate' if candidate else 'native'] = values
                pairs.append(pair)
            medians = {arm:statistics.median([x for p in pairs for x in p[arm]]) for arm in ('native','candidate')}
            save_arrays(output/f'b{size}-native.npz',expected)
            save_arrays(output/f'b{size}-candidate.npz',actual)
            result = dict(batch_size=size,pairs=pairs,medians=medians,
                          speedup=medians['native']/medians['candidate'],all_arrays_bitwise_equal=True,runtime=runtime)
            report['cases'].append(result)
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','runtime')}),flush=True)


if __name__ == '__main__':
    main()
