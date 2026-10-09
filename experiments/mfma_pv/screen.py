"""PV K-order hypotheses, then full outputs; no timing or inference claim."""
from pathlib import Path
import ctypes,hashlib,json,os,sys
import torch
import torch.utils.deterministic
from trace_probe import trace_operations

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
 return h.hexdigest()
@torch.inference_mode()
def main(root):
 root=Path(root);assert os.environ['HIP_VISIBLE_DEVICES']=='7'
 torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.preferred_blas_library('cublas');torch.manual_seed(101)
 build=json.loads((root/'build/build.json').read_text());assert sha(build['library'])==build['library_sha256']
 lib=ctypes.CDLL(build['library']);entry=lib.fireprot_pv_order;entry.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p];entry.restype=ctypes.c_int
 pin=json.loads((root/'native-pin.json').read_text());assert sha(pin['code_object']['path'])==pin['code_object']['sha256']
 report=dict(completed=False,scope='Bounded synthetic PV only; no fused-attention/model/timing result',strict=True,seed=101,hip_device=7,records=[],library_sha256=build['library_sha256'],native_pin=pin)
 def operands(batch,case):
  p=torch.randn(batch,1536,1536,device='cuda');v=torch.randn(batch,1536,32,device='cuda')
  if case=='probability':p=p.softmax(-1)
  elif case=='cancellation':
   p.fill_(1.);v.zero_();v[:,0]=1.e20;v[:,1]=1.;v[:,2]=-1.e20;v[:,3]=1.
  elif case=='subnormal':p.mul_(1.e-40)
  elif case=='signed_zero':p.zero_();v.zero_();v[:,::2]=-0.
  elif case=='nonfinite':p[:,0,0]=float('nan');v[:,0,0]=float('inf');v[:,1,1]=-float('inf')
  return p,v
 def check(p,v,expected,order,tiles):
  actual=torch.full_like(expected,7.);status=entry(p.data_ptr(),v.data_ptr(),actual.data_ptr(),p.shape[0],tiles,order,torch.cuda.current_stream().cuda_stream);assert status==0
  rows=tiles*16;a=actual[:,:rows].contiguous();e=expected[:,:rows].contiguous()
  mismatch=int((a.view(torch.int32)!=e.view(torch.int32)).sum().item())
  if rows<1536:assert bool((actual[:,rows:]==7.).all())
  return dict(order=order,query_rows=rows,bitwise_equal=mismatch==0,mismatch_count=mismatch,max_abs_delta=float((a-e).abs().nan_to_num().max().item()))
 try:
  candidates={0,1,2}
  for case in ('random','probability','cancellation','subnormal','signed_zero','nonfinite'):
   p,v=operands(128,case)
   expected=torch.bmm(p,v)
   if case=='random':
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as profile:
     with torch.profiler.record_function('fireprot.attention.PV'):traced=torch.bmm(p,v)
    trace=root/'native-pv.chrome.json';profile.export_chrome_trace(str(trace));data=dict(einsums=[dict(label='PV')]);trace_operations(data,trace)
    bmm=[x for x in data['einsums'][0]['operations'] if x['operation']=='aten::bmm'];assert len(bmm)==1
    kernels=bmm[0]['kernels'];assert len(kernels)==1 and kernels[0]['name']==pin['kernel'],kernels
    assert torch.equal(traced.view(torch.int32),expected.view(torch.int32));report['trace']=dict(path=str(trace),sha256=sha(trace),operations=data)
    del traced,profile
   checks=[check(p,v,expected,o,1) for o in range(3)]
   candidates&={x['order'] for x in checks if x['bitwise_equal']}
   record=dict(case=case,batch=128,checks=checks);report['records'].append(record);print(json.dumps(record),flush=True)
   del p,v,expected
  report['bounded_passing_orders']=sorted(candidates);full=[]
  for order in sorted(candidates):
   for batch in (128,512):
    for case in ('random','probability'):
     p,v=operands(batch,case);expected=torch.bmm(p,v);record=dict(batch=batch,case=case,**check(p,v,expected,order,96));full.append(record);print(json.dumps(record),flush=True);del p,v,expected
  report['full_checks']=full;report['correctness_passed']=bool(full) and all(x['bitwise_equal'] for x in full);report['completed']=True
 finally:(root/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main(sys.argv[1])
