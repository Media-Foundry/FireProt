"""Exact pretrained ESMC RoPE HIP screening; diagnostic warm-forward benchmark."""
import ctypes,json,os,sys,time,statistics,types
from pathlib import Path
import torch
from fireprot_workers.plm import load_model
from fireprot_workers.atlasfold import read_sequence
from fireprot_workers.transport import save_arrays
request=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);output.mkdir(exist_ok=False)
torch.set_num_threads(8);torch.manual_seed(request['seed'])
_,seq=read_sequence(request['input']['path']);model,tokens,native=load_model(request,seq)
lib=ctypes.CDLL(sys.argv[3]);call=lib.fireprot_esmc_rope
call.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int,ctypes.c_void_p];call.restype=ctypes.c_int
originals=[];counters={'calls':0,'verified':0};verify=True
for block in model.transformer.blocks:
 rotary=block.attn.rotary;original=rotary.forward;originals.append((rotary,original))
 def wrap(self,q,k,seqlen_offset=0,_original=original):
  global verify
  if q.shape!=k.shape or q.shape[0]!=1 or q.shape[2:]!=(40,64) or q.dtype!=torch.bfloat16 or not q.is_contiguous() or not k.is_contiguous() or seqlen_offset!=0:
   raise ValueError('Outside exact ESMC RoPE signature')
  self._update_cos_sin_cache(q.shape[1],device=q.device,dtype=q.dtype)
  qo=torch.empty_like(q);ko=torch.empty_like(k)
  rc=call(q.data_ptr(),k.data_ptr(),self._cos_cached.data_ptr(),self._sin_cached.data_ptr(),qo.data_ptr(),ko.data_ptr(),q.shape[1],torch.cuda.current_stream().cuda_stream)
  if rc:raise RuntimeError(f'HIP launch failed {rc}')
  counters['calls']+=1
  if verify:
   expected=_original(q,k)
   if not torch.equal(qo,expected[0]) or not torch.equal(ko,expected[1]):raise RuntimeError('Trained RoPE mismatch')
   counters['verified']+=1
  return qo,ko
 rotary._fireprot_candidate=types.MethodType(wrap,rotary)
def install(enabled):
 for rotary,original in originals:rotary.forward=rotary._fireprot_candidate if enabled else original
def timed(enabled):
 install(enabled);values=[]
 for _ in range(3):native()
 for _ in range(10):
  torch.cuda.synchronize();start=time.perf_counter();r=native();torch.cuda.synchronize();values.append(time.perf_counter()-start)
 return values,r
report={'length':len(seq),'precision':'bfloat16','hip_visible_devices':os.environ['HIP_VISIBLE_DEVICES'],'torch':str(torch.__version__),'hip':str(torch.version.hip)}
with torch.inference_mode():
 baseline=native();install(True);candidate=native();torch.cuda.synchronize()
 report['all_outputs_exact']=all(torch.equal(v,candidate[k]) for k,v in baseline.items())
 report['trained_rotary_calls_verified']=counters['verified'];verify=False
 if not report['all_outputs_exact']:raise RuntimeError('Full model mismatch')
 save_arrays(output/'native.npz',baseline);save_arrays(output/'hip.npz',candidate)
 report['pairs']=[]
 for order in [(False,True),(True,False),(False,True)]:
  row={}
  for enabled in order:
   values,r=timed(enabled)
   if not all(torch.equal(v,r[k]) for k,v in baseline.items()):raise RuntimeError('Output drift')
   row['hip' if enabled else 'native']=values
  report['pairs'].append(row)
 report['native_median']=statistics.median(sum([p['native'] for p in report['pairs']],[]));report['hip_median']=statistics.median(sum([p['hip'] for p in report['pairs']],[]));report['speedup']=report['native_median']/report['hip_median']
(output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
