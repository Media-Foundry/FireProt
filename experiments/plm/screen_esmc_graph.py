"""Measure HIP graph replay after exact ESMC capture, including changed-token checks."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from fireprot.adapters.plm.runtime import optimized_scope
from fireprot.ops._bridge import bridge
from fireprot_workers.plm import load_model
from fireprot_workers.plm_batch import prepare_forward
from screen_esmc_ck import difference


def main():
    assert os.environ['HIP_VISIBLE_DEVICES']=='6'
    request=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    torch.set_num_threads(8);torch.manual_seed(request['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert not torch.cuda.tunable.is_enabled()
    source=Path(request['input']['path']);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha==request['input']['sha256']
    plan=json.loads(bridge.plan_plm_batch(source.read_text(),json.dumps(request['workload']['batch'])))
    sequences=[x['sequence'] for x in plan['sequences']]
    report=dict(model=request['workload']['model_name'],hip_device=6,seed=request['seed'],length=1000,
                precision='native BF16',input_sha256=sha,torch=torch.__version__,hip=torch.version.hip,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                boundary='warm graph replay; capture and side-stream warmup reported separately; not E2E',
                cache_state='resident model, warmed operators and graph, caches not flushed',cases=[])
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with torch.inference_mode():
        model,_,_=load_model(request,sequences[0])
        for size in (1,4):
            forward=prepare_forward(model,request['workload']['model_name'],sequences[:size])
            expected={k:v.clone() for k,v in forward().items()}
            assert not model._use_flash_attn
            assert bool((expected['tokens'] != model.config.pad_token_id).all())
            graph_tokens=expected['tokens'].clone()
            def eager():
                with optimized_scope(model,request,1000,batch_size=size):return forward()
            def graph_forward():
                # Native model.py selects exactly this branch for unpadded inputs,
                # without requested intermediate states, attentions or SAE output.
                # Validate its data-dependent mask decision outside capture.
                with optimized_scope(model,request,1000,batch_size=size):
                    hidden,*_=model.transformer(model.embed(graph_tokens),sequence_id=None,
                                               layers_to_collect=[],output_attentions=False)
                return dict(tokens=graph_tokens,embeddings=hidden)
            case=dict(batch=size);report['cases'].append(case);save()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            start=time.perf_counter()
            with torch.cuda.stream(stream):
                for _ in range(2):graph_forward()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            case['warmup_seconds']=time.perf_counter()-start
            before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            graph=torch.cuda.CUDAGraph();start=time.perf_counter()
            try:
                with torch.cuda.graph(graph,stream=stream):static=graph_forward()
            except RuntimeError as error:
                case['capture_error']=str(error);save();raise
            torch.cuda.synchronize();case['capture_seconds']=time.perf_counter()-start
            case['allocated_before_capture']=before;case['allocated_after_capture']=torch.cuda.memory_allocated()
            case['capture_peak_allocated']=torch.cuda.max_memory_allocated()
            graph.replay();torch.cuda.synchronize()
            case['initial_difference']={k:difference(v,static[k]) for k,v in expected.items()}
            assert all(v['bitwise_equal'] for v in case['initial_difference'].values())
            alternate=sequences[1:2] if size==1 else list(reversed(sequences[:size]))
            alternate_expected=prepare_forward(model,request['workload']['model_name'],alternate)()
            assert not torch.equal(alternate_expected['tokens'],expected['tokens'])
            assert bool((alternate_expected['tokens'] != model.config.pad_token_id).all())
            static['tokens'].copy_(alternate_expected['tokens']);graph.replay();torch.cuda.synchronize()
            case['changed_tokens_difference']={k:difference(v,static[k]) for k,v in alternate_expected.items()}
            assert all(v['bitwise_equal'] for v in case['changed_tokens_difference'].values())
            static['tokens'].copy_(expected['tokens']);graph.replay();torch.cuda.synchronize()
            assert all(torch.equal(v.view(torch.uint8),static[k].view(torch.uint8)) for k,v in expected.items())
            pairs=[]
            for order in ((False,True),(True,False),(False,True)):
                pair={}
                for candidate in order:
                    def run():
                        if candidate:graph.replay();return static
                        return eager()
                    for _ in range(2):run()
                    times=[]
                    for _ in range(3):
                        torch.cuda.synchronize();start=time.perf_counter();actual=run();torch.cuda.synchronize()
                        times.append(time.perf_counter()-start)
                        assert all(torch.equal(v.view(torch.uint8),actual[k].view(torch.uint8)) for k,v in expected.items())
                    pair['graph' if candidate else 'eager']=times
                pairs.append(pair)
            case['pairs']=pairs;case['medians']={a:statistics.median(t for p in pairs for t in p[a]) for a in ('graph','eager')}
            case['speedup']=case['medians']['eager']/case['medians']['graph'];save()
            print(json.dumps({k:v for k,v in case.items() if k!='pairs'}),flush=True)
            del graph,static,actual,alternate_expected


if __name__=='__main__':main()
