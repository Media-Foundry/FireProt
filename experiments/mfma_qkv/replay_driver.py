from pathlib import Path
import hashlib,json,os,subprocess
b=Path('/media/IntelSSD/fireprot/outputs/experiments/openfold3-qkv-replay-20261010')
source=b.parent/'openfold3-post-norm-hotspots-20261007'
paired=b.parent/'openfold3-isolated-data-pairs-20261009'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for data in iter(lambda:f.read(4194304),b''):h.update(data)
 return h.hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def write(n,d):
 with (b/n).open('w') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
try:
 assert read(source/'collected.json')['correctness_passed']
 prep=read(source/'preparation.json')
 for p in prep['files']+[prep['cli'],prep['reference_npz'],prep['reference_cif'],prep['paired_selection'],prep['probe_source']]:assert sha(p['path'])==p['sha256'],p['path']
 for n in ['openfold3-pv-order-20261009','openfold3-qkv-fused-20261009','openfold3-qkv-index32-20261010']:
  status=read(b.parent/n/'driver-status.json');assert status['exit_code']==0 and status['correctness_passed']
 d=json.loads(subprocess.check_output(['/usr/bin/amd-smi','process','--gpu','5','--json']))
 assert len(d)==1 and d[0]['gpu']==5 and d[0]['process_list'] in ([],[{'process_info':'No running processes detected'}]),d
 current=read(paired/'candidate-profile.json');inventory=current['environment_inventory'];assert sha(inventory['path'])==inventory['sha256'];inv=read(Path(inventory['path']))
 cli=read(paired/'preparation.json')['cli'];assert sha(cli['path'])==cli['sha256']
 (b/'environment-spec.json').write_text(json.dumps(inv['specification'],indent=2)+'\n')
 with (b/'environment.json').open('x') as f:subprocess.run([cli['path'],'capture-environment',str(b/'environment-spec.json')],stdout=f,check=True)
 assert read(b/'environment.json')==inv
 profile=read(source/'profile.json');request=read(source/'request.json');env=dict(os.environ)
 for k in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):env.pop(k,None)
 env.update(profile['environment']);env.update(HIP_VISIBLE_DEVICES='7',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join([str(source/'python'),request['source']['root'],str(source)]))
 identity=json.loads(subprocess.check_output([profile['hip_probe']],env=env));assert identity==dict(pci_address='0000:93:00.0',architecture='gfx90a')
 write('preparation.json',dict(files=[pin(p) for p in [b/'benchmark.py',Path(__file__),source/'preparation.json',source/'profile.json',paired/'candidate-profile.json']],device=identity,environment_inventory=inventory,profile_environment=profile['environment'],source=str(source)))
 with (b/'benchmark.log').open('x') as log:
  child=subprocess.Popen([profile['python'],str(b/'benchmark.py'),str(b),str(source)],env=env,cwd=str(b),stdout=log,stderr=subprocess.STDOUT)
  write('status.json',dict(stage='replay',child_pid=child.pid));rc=child.wait()
 if rc:raise RuntimeError(f'Replay exited {rc}')
 report=read(b/'benchmark/report.json');assert len(report['records'])==2 and all(v>0 for v in report['fusion_calls'].values())
 write('driver-status.json',dict(exit_code=0,correctness_passed=report['correctness_passed']))
except Exception as e:
 write('driver-status.json',dict(exit_code=1,error=repr(e)));raise
