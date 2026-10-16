"""Check every parameter and output, then time ESMC loaders in fresh processes."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch
import fireprot_workers.plm_loading as loading
from fireprot.ops._bridge import bridge
from fireprot_workers.plm import load_model
from fireprot_workers.plm_batch import prepare_forward
from fireprot_workers.transport import save_arrays
from esmc_cpu_loading import cpu_streamed_esmc_state, reused_pinned_esmc_state


def main():
    assert os.environ['HIP_VISIBLE_DEVICES']=='6'
    req=json.loads(Path(sys.argv[1]).read_text());mode=sys.argv[2];output=Path(sys.argv[3]);output.mkdir(exist_ok=False)
    assert mode in ('qualify','qualify_direct','gpu_bf16','cpu_bf16','pinned_bf16','pinned_direct')
    torch.set_num_threads(8);torch.manual_seed(req['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    src=Path(req['input']['path']);sha=hashlib.sha256(src.read_bytes()).hexdigest();assert sha==req['input']['sha256']
    plan=json.loads(bridge.plan_plm_batch(src.read_text(),json.dumps(req['workload']['batch'])))
    sequences=[s['sequence'] for s in plan['sequences']];assert len(sequences)==4
    report=dict(model=req['workload']['model_name'],mode=mode,hip_device=6,seed=req['seed'],batch=4,length=1000,
                torch=torch.__version__,hip=torch.version.hip,precision='native BF16',input_sha256=sha,
                boundary='loader diagnostic, excludes process import/CUDA initialization; not E2E',
                cache_state='fresh process; file cache not flushed; no converted checkpoint cache',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                loader_sha256=hashlib.sha256(Path(__file__).with_name('esmc_cpu_loading.py').read_bytes()).hexdigest())
    native=loading.streamed_esmc_state
    def set_mode(choice):
        if choice=='pinned_direct':
            loading.streamed_esmc_state=reused_pinned_esmc_state
            return
        loading.streamed_esmc_state=(native if choice=='gpu_bf16' else
            lambda root,mapping,convert:cpu_streamed_esmc_state(root,mapping,convert,pinned=choice=='pinned_bf16'))
    def save(): (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        with torch.inference_mode():
            torch.cuda.init()
            if mode.startswith('qualify'):
                reference,_,_=load_model(req,sequences[0])
                expected=reference.state_dict()
                expected_arrays={k:v.clone() for k,v in prepare_forward(reference,req['workload']['model_name'],sequences)().items()}
                save_arrays(output/'reference.npz',expected_arrays)
                report['cases']=[];save()
                for choice in (('pinned_direct',) if mode=='qualify_direct' else ('cpu_bf16','pinned_bf16')):
                    set_mode(choice);torch.manual_seed(req['seed'])
                    candidate,_,_=load_model(req,sequences[0]);actual=candidate.state_dict()
                    assert actual.keys()==expected.keys()
                    differences=[]
                    for key,value in expected.items():
                        if value.shape!=actual[key].shape or value.dtype!=actual[key].dtype or not torch.equal(value.view(torch.uint8),actual[key].view(torch.uint8)):
                            differences.append(key)
                    arrays=prepare_forward(candidate,req['workload']['model_name'],sequences)()
                    exact={k:torch.equal(v.view(torch.uint8),arrays[k].view(torch.uint8)) for k,v in expected_arrays.items()}
                    row=dict(mode=choice,parameter_tensors=len(expected),parameter_elements=sum(v.numel() for v in expected.values()),
                             differing_parameters=differences,output_exact=exact,finite=bool(torch.isfinite(arrays['embeddings']).all()))
                    report['cases'].append(row);save();print(json.dumps(row),flush=True)
                    assert not differences and all(exact.values()) and row['finite']
                    del candidate,actual,arrays
                    torch.cuda.empty_cache()
            else:
                set_mode(mode);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                model,_,_=load_model(req,sequences[0]);torch.cuda.synchronize()
                report['loader_seconds']=time.perf_counter()-start
                report['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
                report['peak_host_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                arrays=prepare_forward(model,req['workload']['model_name'],sequences)();torch.cuda.synchronize()
                save_arrays(output/'sample.npz',arrays)
                import zipfile
                with zipfile.ZipFile(sys.argv[4]) as ref,zipfile.ZipFile(output/'sample.npz') as actual:
                    report['output_exact']=ref.namelist()==actual.namelist() and all(ref.read(k)==actual.read(k) for k in ref.namelist())
                save();print(json.dumps(report),flush=True);assert report['output_exact']
    finally:loading.streamed_esmc_state=native


if __name__=='__main__':main()
