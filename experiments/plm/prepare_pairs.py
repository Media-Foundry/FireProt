"""Prepare a fresh frozen deployment and delegate all paired gates/timing to Rust.

DiamondHill reproduction example; choose a new root before reusing it.
"""
import json,pathlib,hashlib,shutil,subprocess,os
root=pathlib.Path('/media/IntelSSD/fireprot/outputs/experiments/plm-20261015');freeze=root/'qualified';freeze.mkdir(exist_ok=False)
for name in ['python','fireprot','kernels']:
 shutil.copytree(root/name,freeze/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(freeze/'target/release').mkdir(parents=True)
for src,dst in [('fireprot-v3','fireprot'),('lib_fireprot_bridge.so','lib_fireprot_bridge.so')]:shutil.copy2(root/'target/release'/src,freeze/'target/release'/dst)
exe=freeze/'target/release/fireprot';bridge=freeze/'target/release/lib_fireprot_bridge.so'
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,r):p.write_text(json.dumps(r,indent=2)+'\n');return pin(p)
profiles={}
for model in ['esmc-fold2-6b','esm2-3b','atlaslm-3b']:
 p=root/('plm-'+model+'.json');r=json.loads(p.read_text());r['workers']=str(freeze/'python');r['environment']['FIREPROT_BRIDGE_LIBRARY']=str(bridge)
 profile=write(freeze/(model+'-profile.json'),r);profiles[model]=profile
 pairs=[]
 candidate=json.loads((root/('plm-'+model+'-n1000-fireprot.json')).read_text())
 for i in range(3):
  pair={'first':'candidate' if i==1 else 'baseline'}
  for arm in ['baseline','candidate']:
   r=json.loads(json.dumps(candidate));r['backend']='native' if arm=='baseline' else 'fireprot';r['hip_device']=6
   r['run_id']=f'plm-{model}-{arm}-pair{i+1}';r['output_dir']=str(freeze/r['run_id'])
   pair[arm]={'request':write(freeze/(r['run_id']+'.json'),r),'profile':profile}
  pairs.append(pair)
 spec={'schema_version':1,'execution':{'schema_version':1,'timeout_seconds':1800,'invocation_timeout_seconds':300,'pairs':pairs},'comparison':{'scope':'end_to_end','cache_state':'Fresh processes; filesystem/GPU library caches unflushed; prior N1000 qualification; no output feature cache.','treatment_description':f'Native versus qualified standalone {model} opt-in execution; native precision and exact output gate.','expected_changes':{'backend':{'baseline':'native','candidate':'fireprot'},'launch.inputs.rust-bridge':{'baseline':None,'candidate':{'canonical_path':str(bridge),'name':'rust-bridge','sha256':pin(bridge)['sha256']}}}}}
 write(freeze/(model+'-pairs.json'),spec)
# All timed GPU and CPU model work is serialized on HIP6. HIP7 remains free.
env=dict(os.environ,HIP_VISIBLE_DEVICES='6');env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('ROCR_VISIBLE_DEVICES',None)
for model in profiles:
 with (freeze/(model+'-pairs.log')).open('w') as log:
  result=subprocess.run([str(exe),'benchmark-pairs',str(freeze/(model+'-pairs.json')),str(freeze/(model+'-measurements'))],cwd=freeze,env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise SystemExit(f'{model} pairs failed {result.returncode}')
 print(model,'completed',flush=True)
