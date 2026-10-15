"""Bounded exact-output pLM operator trials on pretrained N1000 inputs."""
import ctypes,json,os,sys,time,statistics,types
from pathlib import Path
import torch
from fireprot_workers.plm import load_model
from fireprot_workers.atlasfold import read_sequence
from fireprot_workers.transport import save_arrays
r=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);output.mkdir(exist_ok=False)
torch.set_num_threads(8);torch.manual_seed(r['seed']);_,seq=read_sequence(r['input']['path'])
model,tokens,native=load_model(r,seq);originals=[];calls={'verified':0};verify=True
if r['workload']['model_name']=='esm2-3b':
 for layer in model.layers:
  module=layer.self_attn;original=module.forward
  def call(*args,_original=original,**kwargs):
   if kwargs.get('need_head_weights',False):raise ValueError('Head weights are outside this standalone workload')
   kwargs['need_weights']=False
   return _original(*args,**kwargs)
  originals.append((module,original,call))
else:
 lib=ctypes.CDLL(str(Path(sys.argv[3]).resolve()));entry=lib.fireprot_atlaslm_rope
 entry.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int,ctypes.c_void_p];entry.restype=ctypes.c_int
 for block in model.transformer.blocks:
  module=block.attn.rotary;original=module.forward
  # Same native table cast, performed once for the fixed sequence/dtype.
  c=module._cos_cached[:len(seq)+2].to(torch.bfloat16).contiguous();s=module._sin_cached[:len(seq)+2].to(torch.bfloat16).contiguous()
  def call(q,k,pos_id=None,_original=original,_c=c,_s=s):
   if pos_id is not None or q.shape!=k.shape or tuple(q.shape)!=(1,len(seq)+2,36,64) or q.dtype!=torch.bfloat16 or not q.is_contiguous() or not k.is_contiguous():raise ValueError('Outside AtlasLM exact signature')
   qo=torch.empty_like(q);ko=torch.empty_like(k)
   rc=entry(q.data_ptr(),k.data_ptr(),_c.data_ptr(),_s.data_ptr(),qo.data_ptr(),ko.data_ptr(),q.shape[1],torch.cuda.current_stream().cuda_stream)
   if rc:raise RuntimeError(rc)
   if verify:
    a,b=_original(q,k)
    if not torch.equal(a,qo) or not torch.equal(b,ko):raise RuntimeError('AtlasLM trained rotary mismatch')
    calls['verified']+=1
   return qo,ko
  originals.append((module,original,call))
def install(enabled):
 for module,original,candidate in originals:module.forward=candidate if enabled else original
def timed(enabled):
 install(enabled)
 for _ in range(2):native()
 times=[]
 for _ in range(5):
  torch.cuda.synchronize();start=time.perf_counter();out=native();torch.cuda.synchronize();times.append(time.perf_counter()-start)
 return times,out
report={'model':r['workload']['model_name'],'length':len(seq),'hip_visible_devices':os.environ['HIP_VISIBLE_DEVICES'],'torch':str(torch.__version__),'hip':str(torch.version.hip)}
with torch.inference_mode():
 baseline=native();install(True);candidate=native();torch.cuda.synchronize();report['all_outputs_exact']=all(torch.equal(v,candidate[k]) for k,v in baseline.items());report['trained_rotary_calls_verified']=calls['verified'];verify=False
 if not report['all_outputs_exact']:raise RuntimeError('Complete output mismatch')
 save_arrays(output/'native.npz',baseline);save_arrays(output/'candidate.npz',candidate)
 report['pairs']=[]
 for order in [(False,True),(True,False),(False,True)]:
  pair={}
  for enabled in order:
   values,out=timed(enabled)
   if not all(torch.equal(v,out[k]) for k,v in baseline.items()):raise RuntimeError('Complete output drift')
   pair['candidate' if enabled else 'native']=values
  report['pairs'].append(pair)
 report['native_median']=statistics.median(sum([p['native'] for p in report['pairs']],[]));report['candidate_median']=statistics.median(sum([p['candidate'] for p in report['pairs']],[]));report['speedup']=report['native_median']/report['candidate_median']
(output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
