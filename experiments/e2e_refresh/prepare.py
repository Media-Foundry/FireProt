"""Prepare fresh matched runs from qualified model routes; no GPU work."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path('/media/IntelSSD/fireprot')

def pin(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def load(p):return json.loads(Path(p).read_text())

def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
 return pin(p)

def checked(p):
 assert pin(p['path'])==p,p
 return load(p['path'])

def main():
 out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok=False)
 shutil.copy2(ROOT/'target/release/fireprot',out/'fireprot')
 cases=[]
 prior={
  'esmc-b4':'esmc-snapshot-monitor-20261016/native-pairs/spec.json',
  'esm2-b4':'esm-speed-20261016/esm2-3b-pairs/spec.json',
  'esm3-b4':'esm-speed-20261016/esm3-open-pairs/spec.json',
  'atlaslm-b4':'atlaslm-speed-20261016/qualified-assign/atlaslm-3b-pairs.json',
  'esmfold2-n1000':'esmfold2-integration-recovery3-20261015/n1000-pairs/specification.json',
  'openfold3-n1536':'openfold3-direct-native-latest-20261010/pairs.json',
 }
 def prepare(name,spec,origin):
  case=out/name;case.mkdir();old=spec['execution']['pairs'][0]
  arms={}
  for arm in ('baseline','candidate'):
   req=checked(old[arm]['request']);profile=checked(old[arm]['profile'])
   req['hip_device']=6
   arms[arm]=(req,write(case/f'{arm}-profile.json',profile))
  pairs=[]
  for i in range(3):
   pair=dict(first='candidate' if i==1 else 'baseline')
   for arm,(base,profile) in arms.items():
    req=dict(base,run_id=f'e2e-refresh-20261016-{name}-{arm}-{i}',output_dir=str(case/f'{arm}-{i}'))
    pair[arm]=dict(request=write(case/f'{arm}-{i}.json',req),profile=profile)
   pairs.append(pair)
  spec['execution']=dict(schema_version=1,timeout_seconds=14400,invocation_timeout_seconds=3600,pairs=pairs)
  spec['comparison']['cache_state']='Fresh workers; unflushed caches; published/offline operators; complete hashes and all outputs; serial HIP6, PCI0000:8e:00.0; asset transfer paused during measurements.'
  spec['comparison']['treatment_description']+=' Fresh 2026-10-16 E2E refresh on HIP6; scientific settings preserved.'
  changes=spec['comparison']['expected_changes']
  if 'invocation.profile_sha256' in changes:changes['invocation.profile_sha256']={a:arms[a][1]['sha256'] for a in arms}
  write(case/'spec.json',spec)
  cases.append(dict(name=name,spec=pin(case/'spec.json'),historical_source=origin,policy='model-specific Rust benchmark-pairs gates'))
 for name,p in prior.items():
  path=ROOT/'outputs/experiments'/p;prepare(name,load(path),pin(path))
 bridge=pin(ROOT/'target/release/lib_fireprot_bridge.so')
 new=[
 ('mini-n1000','protenix-n1000-native.json','protenix-n1000-fireprot.json','protenix-all-published.json'),
 ('mini-n1536','protenix-n1536-native.json','protenix-n1536-fireprot.json','protenix-all-published.json'),
 ('mini-esm-n1000','protenix-mini-esm-n1000-native.json','protenix-mini-esm-n1000-fireprot.json','protenix-all-published.json'),
 ('atlasfold-n1000','atlasfold-n1000-native.json','atlasfold-n1000-hip-fused.json','atlasfold-all-published.json'),
 ('atlasfold-n1536','atlasfold-n1536-chunked.json','atlasfold-n1536-hip-fused.json','atlasfold-all-published.json'),
 ]
 for name,b,c,p in new:
  b=ROOT/'configs/requests'/b;c=ROOT/'configs/requests'/c;p=ROOT/'configs/diamondhill'/p
  assert load(b)['workload']==load(c)['workload'] and load(b)['seed']==load(c)['seed']
  spec=dict(schema_version=1,execution=dict(pairs=[dict(baseline=dict(request=pin(b),profile=pin(p)),candidate=dict(request=pin(c),profile=pin(p)))]),comparison=dict(scope='end_to_end',expected_changes={
   'backend':dict(baseline=load(b)['backend'],candidate=load(c)['backend']),
   'launch.inputs.rust-bridge':dict(baseline=None,candidate=dict(name='rust-bridge',canonical_path=bridge['path'],sha256=bridge['sha256']))},treatment_description=f'{name}: original reference versus current published operators, including complete preprocessing and language-model work.'))
  prepare(name,spec,dict(baseline=pin(b),candidate=pin(c),profile=pin(p)))
 write(out/'plan.json',dict(schema_version=1,cli=pin(out/'fireprot'),hip_device=6,pci='0000:8e:00.0',cases=cases,base='Separate unchanged user aligned-coordinate policy; prepare independently.',excluded={'OpenFold2':'Not integrated','ESMFold-v1':'Not integrated','Tiny':'Separate L124 development case; not a long-sequence optimization.'}))
 print('Prepared',len(cases),'three-pair studies',flush=True)

if __name__=='__main__':main()
