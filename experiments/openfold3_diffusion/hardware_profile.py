"""Separate exact full-model trace/counter diagnostics after timing studies."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
b=Path('/media/IntelSSD/fireprot/outputs/experiments/openfold3-hardware-followup-20261010')
source=b.parent/'openfold3-hip-bias-pack-model-20261010'
mode=sys.argv[1]
if mode not in ('trace','counters'):raise ValueError('Unknown diagnostic mode')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4194304),b''):h.update(x)
 return h.hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def write(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
prep=read(source/'preparation.json');qualification=read(source/'collected.json')
assert qualification['correctness_passed'] and qualification['all11_exact'] and qualification['full_cif_exact']
for x in prep['files']+[prep[k] for k in ('cli','paired_qualification','screen','packing','reference_npz','reference_cif','reference_runtime')]:assert sha(x['path'])==x['sha256'],x['path']
root=b/mode;root.mkdir(exist_ok=False)
metrics=['GRBM_COUNT','GRBM_GUI_ACTIVE','SQ_BUSY_CU_CYCLES','MeanOccupancyPerCU','OccupancyPercent','MfmaUtil','VALUBusy','FetchSize','WriteSize']
command=['/opt/rocm/bin/rocprofv3']
if mode=='trace':command+=['--kernel-trace','--stats']
else:command+=['--pmc',*metrics,'--kernel-include-regex','qkv_softmax|Cijk_|norm|softmax|pack_h16|gate|copy','--kernel-iteration-range','10-12']
command+=['--output-format','csv','-d',str(root/'profiler'),'--','python3','-c','import os,json;print(json.dumps({k:v for k,v in os.environ.items() if k.startswith(("ROCPROF","ROCP_")) or k=="LD_PRELOAD"}))']
env=dict(os.environ)
for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):env.pop(key,None)
env['HIP_VISIBLE_DEVICES']='7'
setup=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
(root/'setup.stdout').write_text(setup.stdout);(root/'setup.stderr').write_text(setup.stderr)
lines=[line for line in setup.stdout.splitlines() if line.startswith('{')]
assert len(lines)==1,setup.stdout
profiler_environment=json.loads(lines[0]);assert profiler_environment.get('LD_PRELOAD')
profile=read(source/'profile.json');profile['environment'].update(profiler_environment)
request=read(source/'request.json');request.update(run_id=f'openfold3-hardware-{mode}-20261010',output_dir=f'/media/IntelSSD/fireprot/outputs/openfold3-hardware-{mode}-20261010')
assert request['hip_device']==7 and not Path(request['output_dir']).exists()
write(root/'request.json',request);write(root/'profile.json',profile)
write(root/'spec.json',dict(schema_version=1,timeout_seconds=3600,request=pin(root/'request.json'),profile=pin(root/'profile.json')))
paths=[Path('/opt/rocm/bin/rocprofv3'),Path('/opt/rocm/share/rocprofiler-sdk/config.yaml'),Path('/opt/rocm/share/rocprofiler-sdk/basic_counters.xml'),Path('/opt/rocm/share/rocprofiler-sdk/derived_counters.xml'),*(Path(p) for p in profiler_environment['LD_PRELOAD'].split(':'))]
write(root/'plan.json',dict(mode=mode,metrics=metrics if mode=='counters' else [],command=command,profiler_environment=profiler_environment,profiler_inputs=[pin(p) for p in paths],model_qualification=pin(source/'collected.json'),source_preparation=pin(source/'preparation.json'),specification=pin(root/'spec.json'),scope='Instrumented strict FP32 N1536 R3/D200/S1 seed101 on HIP7; diagnostic times are not E2E speedups; sampled metrics do not represent the entire model.'))
subprocess.run([prep['cli']['path'],'measure-inference',str(root/'spec.json'),str(root/'measurement')],check=True)
out=Path(request['output_dir']);pred=read(out/'predictions.json')
arrays=json.loads(subprocess.check_output([prep['cli']['path'],'compare-npz',qualification['npz']['path'],str(out/'sample-0.npz')]))
assert len(arrays)==11 and all(v['passed'] and v['bitwise_equal'] for v in arrays.values())
cif=out/pred['predictions'][0]['structure'];assert cif.read_bytes()==Path(qualification['cif']['path']).read_bytes()
for x in read(root/'plan.json')['profiler_inputs']:assert sha(x['path'])==x['sha256'],x['path']
for name,key in [('global-softmax','global_softmax'),('normalization','normalization'),('qkv-fusion','qkv'),('gate-fusion','gate')]:assert read(out/(name+'.json'))==qualification[key],name
raw=[pin(p) for p in sorted((root/'profiler').rglob('*.csv'))];assert raw,'Profiler produced no CSV evidence'
write(root/'collected.json',dict(correctness_passed=True,all11_exact=True,full_cif_exact=True,arrays=arrays,npz=pin(out/'sample-0.npz'),cif=pin(cif),plan=pin(root/'plan.json'),raw=raw,limits=['Diagnostic execution only; instrumented timings must not enter E2E comparisons.','Counter file completeness and per-grid sample coverage still require analysis.']))
