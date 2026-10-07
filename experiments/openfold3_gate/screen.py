import hashlib,json,os,sys
from pathlib import Path
import torch
import torch.utils.deterministic
from runtime import Gate
@torch.inference_mode()
def main(root):
 root=Path(root);assert os.environ['HIP_VISIBLE_DEVICES']=='7'
 torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True;torch.manual_seed(101)
 receipt=json.loads((root/'build/build.json').read_text());library=Path(receipt['library']);assert hashlib.sha256(library.read_bytes()).hexdigest()==receipt['library_sha256'];gate=Gate(str(library));rows=[]
 for shape in [(1,17,31,128),(1,256,1536,128),(1,1536,256,128)]:
  for variant in ['random','extreme','strided_mask']:
   g=torch.randn(shape,device='cuda');p=torch.randn_like(g)
   m=torch.randint(0,2,shape[:-1]+(1,),device='cuda').float()
   if variant=='extreme':
    v=torch.tensor([0.,-0.,1e-40,-1e-40,88.,-88.,100.,-100.,float('inf'),-float('inf'),float('nan')],device='cuda')
    g.flatten()[:len(v)]=v;p.flatten()[:len(v)]=v
   if variant=='strided_mask':m=m.transpose(1,2).contiguous().transpose(1,2)
   reference=g.sigmoid();reference.mul_(p);reference.mul_(m);value=gate(g,p,m)
   bits=torch.equal(value.view(torch.int32),reference.view(torch.int32))
   rows.append(dict(shape=list(shape),variant=variant,exact=bits,max_abs_finite_delta=float(torch.nan_to_num((value-reference).abs(),nan=0.,posinf=0.).max().item())))
   print(json.dumps(rows[-1]),flush=True)
 report=dict(correctness_passed=all(r['exact'] for r in rows),checks=rows,seed=101,strict=True,hip_device=7,library_sha256=receipt['library_sha256'])
 (root/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main(sys.argv[1])
