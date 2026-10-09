"""Three-arm complete trained attention replay; retained workspace in all arms."""
from contextlib import ExitStack
from pathlib import Path
import ctypes,hashlib,importlib.util,json,os,statistics,sys
import torch
import torch.utils.deterministic
from fireprot_workers.norm_experiment.screen import verified_captures,Kernel
from fireprot_workers.norm_experiment.model_scope import normalization_scope
from fireprot_workers.triangle_memory.attention_model_scope import key_scope
from fireprot_workers.owned_scores import OwnedScoreSoftmax
from fireprot.adapters.openfold3.attention import AttentionDispatcher,attention_scope
from experiments.mfma_qk.workspace_dispatch import load_workspace,WorkspaceDispatcher
from experiments.mfma_qk.workspace_attention import WorkspaceFusion
from experiments.mfma_qk.probe_fusion import validate_inputs
from openfold3.core.model.layers.triangular_attention import TriangleAttention

def pin(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def same(a,b):return torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@torch.inference_mode()
def main(root,source):
 root,source=Path(root),Path(source);out=root/'benchmark';out.mkdir(exist_ok=False)
 assert os.environ['HIP_VISIBLE_DEVICES']=='7'
 captures,capture_pins=verified_captures('/media/IntelSSD/fireprot/outputs/triatt-profile-n1536-v1')
 torch.use_deterministic_algorithms(True,warn_only=False);torch.utils.deterministic.fill_uninitialized_memory=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.preferred_blas_library('cublas');torch.manual_seed(101)
 canonical=root.parent/'openfold3-canonical-source-20261007/screen.py';Kernel.__call__=load('canonical_norm',canonical).Kernel.__call__
 cp=pin(os.environ['FIREPROT_QK_WORKSPACE_CONTROL']);assert cp['sha256']==os.environ['FIREPROT_QK_WORKSPACE_CONTROL_SHA256']
 entries={};libraries=[];pins=[cp,pin(canonical),pin(__file__)]
 for name,directory in [('fused','openfold3-qkv-fused-20261009'),('index32','openfold3-qkv-index32-20261010')]:
  directory=root.parent/directory;status=json.loads((directory/'driver-status.json').read_text());assert status['exit_code']==0 and status['correctness_passed']
  receipt=json.loads((directory/'build/build.json').read_text());assert pin(receipt['library'])['sha256']==receipt['library_sha256']
  lib=ctypes.CDLL(receipt['library']);libraries.append(lib);entry=lib.fireprot_qkv_softmax_f32
  entry.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]+[ctypes.c_int64]*5+[ctypes.c_void_p];entry.restype=ctypes.c_int;entries[name]=entry
  pins.extend([pin(directory/'screen.json'),pin(directory/'build/build.json'),pin(receipt['library'])])
 labels=['baseline','fused','index32'];orders=[labels[i:]+labels[:i] for i in range(3)]+[list(reversed(labels[i:]+labels[:i])) for i in range(3)];orders*=2
 records=[];counts={n:0 for n in entries}
 for capture in captures:
  dest=out/capture.stem;dest.mkdir();snap=torch.load(capture,map_location='cpu',weights_only=True,mmap=True)
  module=TriangleAttention(128,32,4,starting=snap['starting']).eval().cuda();module.load_state_dict(snap['state_dict'],strict=True)
  x=snap['z'].cuda();mask=snap['mask'].cuda();reference_flags=dict(snap['flags'],chunk_size=256)
  expected=module(x.clone(),mask=mask,**reference_flags);torch.cuda.synchronize()
  with ExitStack() as stack:
   dispatchers={}
   for name in labels:
    workspace=stack.enter_context(load_workspace(cp['path'],cp['sha256']))
    dispatchers[name]=WorkspaceDispatcher(AttentionDispatcher(OwnedScoreSoftmax()),workspace)
   active=[None];selected=['baseline']
   def dispatch(*args,**kwargs):return active[0](*args,**kwargs)
   stack.enter_context(attention_scope(dispatch))
   stack.enter_context(key_scope(dest,'/media/IntelSSD/fireprot/runtime/recovered-openfold3-20261007/fireprot-triangle-e2e-v1/python/fireprot_workers/triangle_memory/key-copy.so'))
   original=WorkspaceFusion.__call__;packer=original.__globals__['_key_packer']
   def fused_call(self,query,key,value,mask,bias):
    name=selected[0]
    if name=='baseline':return original(self,query,key,value,mask,bias)
    if not self._lock.acquire(blocking=False):raise RuntimeError('Concurrent workspace use')
    submitted=False
    try:
     self._check_inputs(query,key,value,mask,bias);stream=torch.cuda.current_stream(query.device).cuda_stream
     assert self._scores is not None and self._device==query.device and self._stream==stream
     assert not torch.cuda.is_current_stream_capturing()
     q=query.reshape(self.batch*4,1536,32);k=packer(key);v=value.reshape(self.batch*4,1536,32)
     if self.contiguous_bias:bias=bias.contiguous()
     assert validate_inputs(q,k,mask,bias)==self.batch
     assert v.is_contiguous()
     output=torch.empty((self.batch,4,1536,32),device=query.device,dtype=query.dtype)
     submitted=True
     rc=entries[name](q.data_ptr(),k.data_ptr(),v.data_ptr(),mask.data_ptr(),bias.data_ptr(),output.data_ptr(),None,self.batch,mask.stride(0) if mask.shape[0]!=1 else 0,mask.stride(-1),bias.stride(1),bias.stride(2),bias.stride(3),stream)
     if rc:raise RuntimeError(f'Fusion launch failed: {rc}')
     self.calls+=1;counts[name]+=1;return output
    except BaseException:
     if submitted:self._closed=True
     raise
    finally:self._lock.release()
   WorkspaceFusion.__call__=fused_call
   stack.callback(setattr,WorkspaceFusion,'__call__',original)
   stack.enter_context(normalization_scope(module,dest,str(source/'norm.so'),'a06828d77438cd417298f19221dca4cff371ed81ba4df041283eb2ab8ebbf03a',str(source/'strided.so'),'88d43a5129b3e0abb2ecfcc321a9e8bf29e00f409a611ec0f1adca542511e65e'))
   checks={n:[] for n in labels};samples={n:[] for n in labels}
   def invoke(name,timed):
    active[0]=dispatchers[name];selected[0]=name;inp=x.clone();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();value=module(inp,mask=mask,**snap['flags']);b.record();b.synchronize()
    if timed:samples[name].append(a.elapsed_time(b))
    check=dict(bitwise_equal=same(value,expected),max_abs_delta=float((value-expected).abs().max().item()));checks[name].append(check)
    assert check['bitwise_equal'],(name,check)
   for name in labels:
    for _ in range(2):invoke(name,False)
   for order in orders:
    for name in order:invoke(name,True)
   result=dict(capture=pin(capture),flags=snap['flags'],reference_flags=reference_flags,exact=all(c['bitwise_equal'] for arm in checks.values() for c in arm),checks=checks,samples_ms=samples,medians_ms={n:statistics.median(v) for n,v in samples.items()},orders=orders,dispatchers={n:d.manifest() for n,d in dispatchers.items()})
   records.append(result);print(json.dumps(dict(capture=capture.name,medians_ms=result['medians_ms'],exact=result['exact'])),flush=True)
  del module,x,mask,expected,snap
 assert all(v>0 for v in counts.values())
 report=dict(correctness_passed=all(r['exact'] for r in records),records=records,fusion_calls=counts,pins=pins,capture_pins=capture_pins,scope='Complete trained TriangleAttention; two warmups and twelve balanced samples per arm/capture; canonical HIP normalization, HIP K packing, strict allocations, same retained score workspace; probabilities unused by fused arms; no model/E2E claim; exact checks between timings affect cache state',strict=True,seed=101,hip_device=7)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main(*sys.argv[1:])
