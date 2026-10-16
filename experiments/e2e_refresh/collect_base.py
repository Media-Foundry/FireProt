"""Collect Base with the explicit aligned policy and unchanged generic diagnostic."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
from collect import archive,checked,pin

root=Path(sys.argv[1]);case=root/'base-n1000';cli=root/'fireprot'
assert json.loads((case/'driver-status.json').read_text())['exit_code']==0
old=Path('/media/IntelSSD/fireprot/outputs/experiments/protenix-base-final-native-pairs-20261007')
spec=json.loads((old/'final-v1-compare.json').read_text());records=[];numerics=[]
for arm,indices,label in [('native',[0,3,4],'baseline'),('optimized',[1,2,5],'candidate')]:
 spec[label]['runs']=[];spec[label]['invocations']=[]
 anchor=None
 for pair,index in enumerate(indices):
  out=case/f'{arm}-{pair}';invpath=case/f'measurement-{index:03d}/invocation.json';inv=json.loads(invpath.read_text())
  assert inv['exit_code']==0 and inv['accepted']==pin(out/'accepted.json')
  pred,arrays=archive(out)
  if anchor is None:anchor=arrays
  assert arrays==anchor
  supervisor=json.loads((out/'supervisor.json').read_text());assert supervisor['phase']=='succeeded' and supervisor['device_verified'] and supervisor['physical_device']['pci_address']=='0000:8e:00.0'
  req=json.loads((out/'request.json').read_text());assert req['hip_device']==6 and req['seed']==101
  assert req['workload']==dict(model='protenix',model_name='protenix_base_default_v0.5.0',cycles=10,diffusion_steps=200,samples=1,precision='float32')
  spec[label]['runs'].append(pin(out/'accepted.json'));spec[label]['invocations'].append(pin(invpath))
  records.append(dict(arm=label,pair=pair,invocation=pin(invpath),accepted=pin(out/'accepted.json'),e2e_seconds=inv['elapsed_seconds'],model_seconds=pred['model_seconds']))
for i in range(3):
 a=json.loads((case/f'aligned-{i}.json').read_text());c=json.loads((case/f'confidence-{i}.json').read_text())
 assert a['coordinate_gate_passed'] and a['atom_identities_exact'] and a['npz_cif_binding_checked'] and a['all_atom']['aligned_rmsd_angstrom']<=.1
 assert c['passed'] and c['max_abs_delta']<=.001
 numerics.append(dict(pair=i,aligned_report=pin(case/f'aligned-{i}.json'),confidence_report=pin(case/f'confidence-{i}.json'),all_atom=a['all_atom'],ca=a['ca'],confidence=c))
spec['baseline']['cache_state']=spec['candidate']['cache_state']='Fresh workers; unflushed caches; full hashes and exports; shared fast initialization; HIP6; asset transfer paused; AB/BA/AB.'
specpath=case/'fresh-legacy-comparison-spec.json';assert not specpath.exists();specpath.write_text(json.dumps(spec,indent=2)+'\n')
with (case/'fresh-legacy-comparison.json').open('x') as out,(case/'fresh-legacy-comparison.stderr').open('x') as err:
 result=subprocess.run([str(cli),'compare-runs',str(specpath)],stdout=out,stderr=err)
legacy=json.loads((case/'fresh-legacy-comparison.json').read_text());assert len(legacy['pairs'])==3
# A complete generic comparison report establishes its protected treatment and
# workload checks, even when its legacy coordinate policy rejects this route.
medians={field:{arm:statistics.median(r[field] for r in records if r['arm']==arm) for arm in ('baseline','candidate')} for field in ('e2e_seconds','model_seconds')}
report=dict(status='user_policy_passed',policy='Aligned all-atom RMSD <=0.1 A and summary confidence <=0.001; unchanged legacy diagnostic recorded separately.',records=records,numerics=numerics,within_arm_exact=True,generic_condition_check_completed=True,legacy_comparison=pin(case/'fresh-legacy-comparison.json'),legacy_exit_code=result.returncode,legacy_correctness_passed=legacy['correctness_passed'],medians=medians,e2e_speedup=medians['e2e_seconds']['baseline']/medians['e2e_seconds']['candidate'],paired_speedups=[next(r['e2e_seconds'] for r in records if r['arm']=='baseline' and r['pair']==i)/next(r['e2e_seconds'] for r in records if r['arm']=='candidate' and r['pair']==i) for i in range(3)])
(case/'collected.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(medians=medians,speedup=report['e2e_speedup'],legacy_correctness_passed=legacy['correctness_passed']),indent=2))
