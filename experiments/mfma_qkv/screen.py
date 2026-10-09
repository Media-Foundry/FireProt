"""Packed synthetic QK/softmax/PV equality; no inference or timing claim."""
from pathlib import Path
import ctypes,hashlib,json,os,sys
import torch
import torch.utils.deterministic
@torch.inference_mode()
def main(root):
 root=Path(root);assert os.environ['HIP_VISIBLE_DEVICES']=='7'
 prerequisite=json.loads((root.parent/'openfold3-pv-order-20261009/screen.json').read_text())
 assert prerequisite['completed'] and prerequisite['correctness_passed'] and 1 in prerequisite['bounded_passing_orders']
 assert all(c['bitwise_equal'] for c in prerequisite['full_checks'] if c['order']==1)
 torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.preferred_blas_library('cublas');torch.manual_seed(101)
 build=json.loads((root/'build/build.json').read_text());assert hashlib.sha256(Path(build['library']).read_bytes()).hexdigest()==build['library_sha256']
 lib=ctypes.CDLL(build['library']);entry=lib.fireprot_qkv_softmax_f32
 entry.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]+[ctypes.c_int64]*5+[ctypes.c_void_p];entry.restype=ctypes.c_int
 report=dict(completed=False,correctness_passed=False,scope='Packed synthetic probabilities and PV outputs only; no performance/model qualification',seed=101,strict=True,hip_device=7,library_sha256=build['library_sha256'],records=[])
 def mismatch(a,b):
  assert a.shape==b.shape and a.dtype==b.dtype==torch.float32
  return sum(int((a[i:i+8].contiguous().view(torch.int32)!=b[i:i+8].contiguous().view(torch.int32)).sum().item()) for i in range(0,a.shape[0],8))
 def guarded(shape):
  n=1
  for dimension in shape:n*=dimension
  storage=torch.full((n+512,),17.,device='cuda');return storage,storage[256:-256].view(shape)
 def guard(storage):return bool((storage[:256]==17.).all()) and bool((storage[-256:]==17.).all())
 cases=[('random',32),('random',128),('transposed_bias',128),('masked',128),('cancellation',32),('subnormal',32),('signed_zero',32),('nonfinite',32)]
 try:
  for case,batch in cases:
   q=torch.randn(batch*4,1536,32,device='cuda')*.125;k=torch.randn(batch*4,32,1536,device='cuda')*.125;v=torch.randn(batch*4,1536,32,device='cuda')
   mask=torch.zeros((batch if case=='masked' else 1,1,1,1536),device='cuda');bias=torch.randn(1,4,1536,1536,device='cuda')*.05
   if case=='transposed_bias':bias=bias.transpose(-1,-2)
   elif case=='masked':mask[...,100:130]=float('-inf');mask[0].fill_(float('-inf'))
   elif case=='cancellation':q[:]=torch.tensor([16777216.,1.,-16777216.,1.]+[0.]*28,device='cuda');k.fill_(1.)
   elif case=='subnormal':q.fill_(torch.finfo(torch.float32).tiny/8);k.fill_(1.);v.mul_(1.e-38)
   elif case=='signed_zero':q.fill_(-0.);k.fill_(1.);v.fill_(-0.)
   elif case=='nonfinite':q[:,0,0]=float('nan');q[:,1,0]=float('inf');k[:,0,0]=0.;v[:,1,1]=float('inf')
   scores=torch.bmm(q,k).view(batch,4,1536,1536);scores.add_(mask);scores.add_(bias)
   probabilities=scores.softmax(-1);del scores
   expected=torch.bmm(probabilities.view(batch*4,1536,1536),v)
   checks=[]
   for capture in (False,True):
    out_storage,out=guarded(expected.shape)
    if capture:prob_storage,prob=guarded(probabilities.shape)
    else:prob_storage=prob=None
    rc=entry(q.data_ptr(),k.data_ptr(),v.data_ptr(),mask.data_ptr(),bias.data_ptr(),out.data_ptr(),prob.data_ptr() if capture else None,batch,mask.stride(0) if mask.shape[0]!=1 else 0,mask.stride(-1),bias.stride(1),bias.stride(2),bias.stride(3),torch.cuda.current_stream().cuda_stream);assert rc==0
    bad=mismatch(out,expected);prob_bad=mismatch(prob,probabilities) if capture else None
    guards=guard(out_storage) and (not capture or guard(prob_storage));assert guards
    checks.append(dict(capture_probabilities=capture,output_mismatches=bad,probability_mismatches=prob_bad,guards_unchanged=guards,bitwise_equal=bad==0 and (not capture or prob_bad==0)))
    del out_storage,out,prob_storage,prob
   record=dict(case=case,batch=batch,checks=checks);report['records'].append(record);print(json.dumps(record),flush=True)
   del q,k,v,mask,bias,probabilities,expected
  report['correctness_passed']=all(c['bitwise_equal'] for r in report['records'] for c in r['checks']);report['completed']=True
 finally:(root/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main(sys.argv[1])
