"""Screen equal-length native B8/B16 against native and optimized resident B4.

Grouping changes may change rounding. Preserve each group's official native
outputs; this is warm forward throughput evidence, not accepted E2E inference.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


def pin(path):
    path=Path(path);before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    return dict(path=str(path),sha256=h.hexdigest())


def main():
    assert os.environ['HIP_VISIBLE_DEVICES']=='6'
    request_path=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(exist_ok=False)
    request=json.loads(request_path.read_text());assert request['hip_device']==6 and request['seed']==101
    assets=list(request['checkpoints'].values())+list(request['resources'].values())+[request['input']]
    def verify(value):assert pin(value['path'])==value
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verify,assets))
    import torch
    from fireprot_workers.plm import load_model
    from fireprot_workers.plm_batch import prepare_forward
    from fireprot_workers.transport import save_arrays
    from fireprot.adapters.plm.runtime import optimized_scope
    from contextlib import nullcontext
    manifest=json.loads(Path(request['input']['path']).read_text());items=manifest['sequences'][:16]
    assert len(items)==16 and all(len(v['sequence'])==1000 for v in items)
    torch.set_num_threads(8);torch.manual_seed(101)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert not torch.cuda.tunable.is_enabled()
    report=dict(schema_version=1,model='ESMC bundled 6B',length=1000,items=16,seed=101,dtype='native BF16',hip_device=6,
        torch=str(torch.__version__),hip=str(torch.version.hip),affinity=sorted(os.sched_getaffinity(0)),
        boundary='Warm model-resident forward for the same 16 inputs; no loading/export/preflight; not E2E.',
        cache_state='Unflushed; two warmups per route; alternating route order; no output cache.',
        script=pin(__file__),request=pin(request_path),assets=assets,inputs=items,routes={},rounds=[])
    def save(): (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    def cpu(arrays):return {k:torch.cat([a[k].detach().cpu() for a in arrays]).contiguous() for k in arrays[0]}
    def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.view(torch.uint8),b.view(torch.uint8))
    def validate(arrays,reference):
        actual=cpu(arrays)
        assert all(torch.isfinite(v).all() and exact(v,reference[k]) for k,v in actual.items())
    with torch.inference_mode():
        start=time.perf_counter();model,_,_=load_model(request,items[0]['sequence']);torch.cuda.synchronize()
        report['load_seconds']=time.perf_counter()-start
        prepared={n:[prepare_forward(model,'esmc-fold2-6b',[v['sequence'] for v in items[i:i+n]]) for i in range(0,16,n)] for n in (4,8,16)}
        def run(n,optimized=False):
            arrays=[]
            for fn in prepared[n]:
                with optimized_scope(model,request,1000,batch_size=n) if optimized else nullcontext():arrays.append(fn())
            return arrays
        references={}
        for n in (4,8,16):
            values=run(n);torch.cuda.synchronize();references[n]=cpu(values);del values
            for i,item in enumerate(items):
                save_arrays(output/f'native-b{n}-item-{i}.npz',{k:v[i:i+1] for k,v in references[n].items()})
            for _ in range(1):validate(run(n),references[n])
        drifts={}
        for n in (8,16):
            per_item=[]
            for i,item in enumerate(items):
                fields={}
                for key,v in references[4].items():
                    a=v[i];b=references[n][key][i];delta=(a.float()-b.float()).abs()
                    fields[key]=dict(bitwise_equal=exact(a,b),max_abs=float(delta.max()),mean_abs=float(delta.mean()),different_values=int((a!=b).sum()))
                assert fields['tokens']['bitwise_equal'];per_item.append(dict(id=item['id'],fields=fields))
            drifts[str(n)]=per_item
        report['native_grouping_drift_vs_b4']=drifts
        validate(run(4,True),references[4]);report['optimized_b4_exact']=True;save()
        routes={'native_b4':(4,False),'native_b8':(8,False),'native_b16':(16,False),'optimized_b4':(4,True)}
        for name,(n,enabled) in routes.items():
            for _ in range(2):validate(run(n,enabled),references[n])
            report['routes'][name]=dict(batch=n,optimized=enabled,seconds=[],peak_allocated_bytes=[])
        orders=[list(routes),list(reversed(routes)),list(routes)]
        for order in orders:
            report['rounds'].append(order)
            for name in order:
                n,enabled=routes[name];row=report['routes'][name]
                for _ in range(3):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    values=run(n,enabled);torch.cuda.synchronize();seconds=time.perf_counter()-start
                    peak=torch.cuda.max_memory_allocated();validate(values,references[n]);del values
                    row['seconds'].append(seconds);row['peak_allocated_bytes'].append(peak)
                save();print(name,'median',statistics.median(row['seconds']),flush=True)
        for name,row in report['routes'].items():
            row['median_seconds']=statistics.median(row['seconds']);row['sequences_per_second']=16/row['median_seconds']
        # Instrument just one B4 group, outside the measured throughput samples.
        for optimized in (False,True):
            name='optimized' if optimized else 'native'
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with optimized_scope(model,request,1000,batch_size=4) if optimized else nullcontext():values=prepared[4][0]()
                torch.cuda.synchronize()
            actual={k:v.detach().cpu().contiguous() for k,v in values.items()}
            assert all(exact(v,references[4][k][:4]) for k,v in actual.items());del values
            prof.export_chrome_trace(str(output/f'{name}-b4-trace.json'))
            report[name+'_b4_profile']=[dict(name=e.key,count=e.count,self_device_us=e.self_device_time_total,cpu_us=e.self_cpu_time_total)
                for e in sorted(prof.key_averages(),key=lambda e:e.self_device_time_total,reverse=True)[:25]]
        report['passed']=True;save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verify,assets))
    report['assets_rehashed_after']=True;save()
    print(json.dumps({name:dict(seconds=row['median_seconds'],seq_per_second=row['sequences_per_second']) for name,row in report['routes'].items()},indent=2),flush=True)


if __name__=='__main__':main()
