"""Prepare an actual AtlasLM B4 refresh with explicit workload assertions."""
from pathlib import Path
import sys
from prepare import ROOT,pin,load,write

root=Path(sys.argv[1]);name='atlaslm-b4-refresh';case=root/name;case.mkdir(exist_ok=False)
profile_data=load(ROOT/'outputs/experiments/atlaslm-speed-20261016/qualified-assign/profile.json')
profile=write(case/'profile.json',profile_data);bridge=pin(Path(profile_data['environment']['FIREPROT_BRIDGE_LIBRARY']))
pairs=[]
for i in range(3):
 pair=dict(first='candidate' if i==1 else 'baseline')
 for arm,backend in [('baseline','native'),('candidate','fireprot')]:
  q=load(ROOT/f'configs/requests/plm-atlaslm-3b-b4-n1000-{backend}.json')
  assert q['workload']['batch']==dict(items=4,max_batch_size=4,max_padded_tokens=4008,grouping='length_sorted') and q['seed']==102
  q.update(run_id=f'e2e-refresh-20261016-atlaslm-b4-{arm}-{i}',output_dir=str(case/f'{arm}-{i}'),hip_device=6)
  pair[arm]=dict(request=write(case/f'{arm}-{i}.json',q),profile=profile)
 pairs.append(pair)
spec=dict(schema_version=1,execution=dict(schema_version=1,timeout_seconds=3600,invocation_timeout_seconds=600,pairs=pairs),comparison=dict(scope='end_to_end',cache_state='Fresh workers; unflushed caches; full hashes/all outputs; serial HIP6; asset transfer paused.',treatment_description='AtlasLM-3B true B4, four N1000 inputs, native BF16, seed102: native versus qualified eager loading and HIP operators. Explicit grouping assertions; not the separately measured seed102 singleton.',expected_changes={'backend':dict(baseline='native',candidate='fireprot'),'launch.inputs.rust-bridge':dict(baseline=None,candidate=dict(name='rust-bridge',canonical_path=bridge['path'],sha256=bridge['sha256']))}))
write(case/'spec.json',spec)
plan=load(root/'plan.json')
for item in plan['cases']:
 if item['name']=='atlaslm-b4':item.update(name='atlaslm-singleton',directory='atlaslm-b4',expected_items=1,expected_seed=102)
 elif item['name'] in ('esmc-b4','esm2-b4','esm3-b4'):item.update(expected_items=4,expected_seed=101)
plan['cases'].append(dict(name=name,spec=pin(case/'spec.json'),expected_items=4,expected_seed=102,policy='Strict official same-batch byte equality'))
import json
(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Prepared true AtlasLM B4 / seed102; prior singleton labelled explicitly')
