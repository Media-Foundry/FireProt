"""Add a matched N1000 Tiny comparison; acceleration remains unqualified until measured."""
import json
from pathlib import Path
import sys
from prepare import ROOT,pin,load,write

root=Path(sys.argv[1]);case=root/'tiny-n1000';case.mkdir(exist_ok=False)
request=load(ROOT/'configs/requests/protenix-tiny-default-l124-smoke-v1.json')
request['input']=load(ROOT/'configs/requests/protenix-n1000-native.json')['input'];request['hip_device']=6
profile=write(case/'profile.json',load(ROOT/'configs/diamondhill/protenix-all-published.json'))
pairs=[]
for i in range(3):
 pair=dict(first='candidate' if i==1 else 'baseline')
 for arm,backend in [('baseline','native'),('candidate','fireprot')]:
  r=dict(request,backend=backend,run_id=f'e2e-refresh-20261016-tiny-{arm}-{i}',output_dir=str(case/f'{arm}-{i}'))
  pair[arm]=dict(request=write(case/f'{arm}-{i}.json',r),profile=profile)
 pairs.append(pair)
bridge=pin(ROOT/'target/release/lib_fireprot_bridge.so')
spec=dict(schema_version=1,execution=dict(schema_version=1,timeout_seconds=7200,invocation_timeout_seconds=1200,pairs=pairs),comparison=dict(scope='end_to_end',cache_state='Fresh workers; unflushed caches; complete hashes/all outputs; serial HIP6; asset transfer paused.',treatment_description='First controlled Tiny-default N1000 FP32 C4/D1/S1 seed101 native versus published FireProt operators; no prior speedup assumed.',expected_changes={'backend':dict(baseline='native',candidate='fireprot'),'launch.inputs.rust-bridge':dict(baseline=None,candidate=dict(name='rust-bridge',canonical_path=bridge['path'],sha256=bridge['sha256']))}))
write(case/'spec.json',spec)
plan=load(root/'plan.json');plan['cases'].append(dict(name='tiny-n1000',spec=pin(case/'spec.json'),policy='Unchanged Rust Protenix coordinate/confidence gates'))
plan['excluded'].pop('Tiny',None)
(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Added Tiny N1000; no performance claim before acceptance')
