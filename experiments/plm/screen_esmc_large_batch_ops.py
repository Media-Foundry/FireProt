"""Qualify large ESMC HIP batches against frozen official same-group outputs."""
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import sys
import time

from screen_esmc_large_batch import pin


def main():
    assert os.environ['HIP_VISIBLE_DEVICES']=='6'
    request_path,work,reference=map(Path,sys.argv[1:4]);output=work/'screen';output.mkdir(exist_ok=False)
    request=json.loads(request_path.read_text());library=json.loads((work/'library.json').read_text());assert pin(library['path'])==library
    prior=json.loads((reference/'report.json').read_text());assert prior['passed'] and prior['assets_rehashed_after']
    def verify(value):assert pin(value['path'])==value
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verify,prior['assets']))
    import numpy as np
    import torch
    from fireprot_workers.plm import load_model
    from fireprot_workers.plm_batch import prepare_forward
    from fireprot_workers.transport import save_arrays
    from fireprot.adapters.plm.runtime import optimized_scope
    from fireprot.adapters.plm.esm import esm_scope
    from esmc_pointwise_scope import Pointwise
    torch.set_num_threads(8);torch.manual_seed(101)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert not torch.cuda.tunable.is_enabled()
    assert str(torch.__version__)=='2.12.0a0+git78258b9' and str(torch.version.hip)=='7.14.60850'
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx90a'
    items=prior['inputs'];assert len(items)==16 and all(len(i['sequence'])==1000 for i in items)
    expected={}
    for n in (4,8,16):
        samples=[]
        for i in range(16):
            with np.load(reference/f'native-b{n}-item-{i}.npz') as z:
                samples.append({k:torch.from_numpy(z[k].copy()).view(torch.bfloat16) if k=='embeddings' else torch.from_numpy(z[k].copy()) for k in z.files})
        expected[n]={k:torch.cat([v[k] for v in samples]) for k in samples[0]}
    report=dict(schema_version=1,boundary='Warm resident forward of 16 N1000 inputs; not E2E; no fixed-GEMM substitution at B8/B16.',
        request=pin(request_path),library=library,reference=pin(reference/'report.json'),script=pin(__file__),cases={},coverage={})
    def save(): (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def cpu(arrays):return {k:torch.cat([a[k].detach().cpu() for a in arrays]).contiguous() for k in arrays[0]}
    def equal(actual,ref):return all(v.shape==ref[k].shape and v.dtype==ref[k].dtype and torch.equal(v.view(torch.uint8),ref[k].view(torch.uint8)) and torch.isfinite(v).all() for k,v in actual.items())
    with torch.inference_mode():
        model,_,_=load_model(request,items[0]['sequence'])
        forwards={n:[prepare_forward(model,'esmc-fold2-6b',[v['sequence'] for v in items[i:i+n]]) for i in range(0,16,n)] for n in (4,8,16)}
        selected=dict(request,resources=dict(request['resources'],hip_esm=library))
        pointwise=Pointwise(library['path'],4)
        def run(n,audit=False):
            outputs=[];coverage=[]
            for fn in forwards[n]:
                if n==4:
                    with optimized_scope(model,request,1000,batch_size=4) as state:outputs.append(fn())
                    coverage.append(state)
                else:
                    with pointwise.scope(model,'residual',audit=audit) as residual,esm_scope(model,selected,n) as state:outputs.append(fn())
                    coverage.append(dict(hip=state,residual=residual))
            return outputs,coverage
        for n in (4,8,16):
            report['cases'][str(n)]=dict(seconds=[],peak_allocated_bytes=[]);save()
            arrays,coverage=run(n,audit=True);actual=cpu(arrays);exact=bool(equal(actual,expected[n]))
            report['cases'][str(n)]['all_arrays_exact']=exact;report['coverage'][str(n)]=coverage;save()
            if not exact:raise RuntimeError(f'B{n} final output differs from official same-group output')
            for i in range(16):save_arrays(output/f'hip-b{n}-item-{i}.npz',{k:v[i:i+1] for k,v in actual.items()})
            del arrays,actual
        for n in (4,8,16):
            for _ in range(2):
                arrays,_=run(n);assert equal(cpu(arrays),expected[n]);del arrays
        for order in ((4,8,16),(16,8,4),(4,8,16)):
            for n in order:
                row=report['cases'][str(n)]
                for _ in range(3):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    arrays,_=run(n);torch.cuda.synchronize();seconds=time.perf_counter()-start
                    peak=torch.cuda.max_memory_allocated();assert equal(cpu(arrays),expected[n]);del arrays
                    row['seconds'].append(seconds);row['peak_allocated_bytes'].append(peak)
                save();print('B',n,'median',statistics.median(row['seconds']),flush=True)
        for row in report['cases'].values():row['median_seconds']=statistics.median(row['seconds'])
        report['passed']=True;save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verify,prior['assets']))
    assert pin(library['path'])==library;report['assets_rehashed_after']=True;save()
    print(json.dumps({n:row['median_seconds'] for n,row in report['cases'].items()},indent=2),flush=True)


if __name__=='__main__':main()
