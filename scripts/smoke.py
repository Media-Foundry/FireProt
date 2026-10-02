"""Bounded asset smoke checks; these are not speed or scientific-quality benchmarks."""
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
models=['protenix_mini_esm_v0.5.0','protenix_mini_default_v0.5.0','protenix_tiny_default_v0.5.0']
start=time.time();processes=[]
for model,gcd in zip(models,[0,2,4]):
 out=root/'outputs'/('smoke_'+model);out.mkdir(parents=True,exist_ok=False)
 log=(out/'run.log').open('w')
 args=[str(root/'scripts/infer.sh'),'--input',str(root/'cases/1izr_L124/input.json'),'--out_dir',str(out),'--model_name',model]
 p=subprocess.Popen(args,env=dict(os.environ,HIP_VISIBLE_DEVICES=str(gcd),CUDA_VISIBLE_DEVICES=str(gcd)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 processes.append((model,gcd,out,p,args))
(root/'provenance/smoke_handles.json').write_text(json.dumps([dict(model=m,gcd=g,pid=p.pid,args=a) for m,g,o,p,a in processes],indent=2))
while any(p.poll() is None for _,_,_,p,_ in processes):
 if time.time()-start>600:
  import signal
  for _,_,_,p,_ in processes:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  break
 time.sleep(2)
rows=[]
for model,gcd,out,p,args in processes:
 code=p.wait();files=list(out.rglob('*.cif'));assert code==0 and len(files)==1,(model,code,files)
 from biotite.structure.io.pdbx import CIFFile,get_structure
 import numpy as np
 atoms=get_structure(CIFFile.read(files[0]),model=1);assert len(atoms)>0 and np.isfinite(atoms.coord).all()
 rows.append(dict(model=model,gcd=gcd,exit_code=code,atoms=len(atoms),finite=True,output=str(files[0].relative_to(root)),sha256=hashlib.sha256(files[0].read_bytes()).hexdigest()))
report=dict(complete=True,rows=rows,wall_seconds=time.time()-start,source_commit=subprocess.check_output(['git','-C',str(root/'Protenix'),'rev-parse','HEAD'],text=True).strip(),cycles=4,steps=1,samples=1,dtype='fp32',msa=False,scope='load and finite-output smoke only; no comparative latency or accuracy claim')
(root/'provenance/smoke_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
