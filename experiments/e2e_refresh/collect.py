"""Independently rehash refreshed Rust evidence and retain explicit timing boundaries."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile


def pin(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return dict(path=str(p),sha256=h.hexdigest())

def checked(v):
 assert pin(v['path'])==v
 return json.loads(Path(v['path']).read_text())

def archive(out):
 receipt=json.loads((out/'accepted.json').read_text())
 for item in receipt['metadata']+receipt['outputs']:
  p=out/item['path'];assert p.resolve().is_relative_to(out.resolve()) and pin(p)['sha256']==item['sha256']
 pred=json.loads((out/'predictions.json').read_text());arrays=[]
 for item in pred['predictions']:
  with zipfile.ZipFile(out/item['archive']) as z:arrays.append({n:z.read(n) for n in z.namelist()})
 return pred,arrays

def main():
 root=Path(sys.argv[1]);plan=json.loads((root/'plan.json').read_text());cases={}
 for case in plan['cases']:
  directory=root/case.get('directory',case['name']);path=directory/'measurements/benchmark.json'
  if not path.exists():cases[case['name']]=dict(status='pending_or_failed_before_report');continue
  b=json.loads(path.read_text())
  if b['phase']!='succeeded':cases[case['name']]=dict(status='failed',benchmark=pin(path),report=b);continue
  for key in ('execution_plan','comparison_settings'):checked(b[key])
  assert pin(b['order']['path'])==b['order'];comparison=checked(b['comparison']);assert comparison['correctness_passed']
  for item in b['checks']:assert checked(item)['correctness_passed']
  records=[];anchor={};within_exact={};cif_anchor=None;all_exact=True;cif_exact=True
  for job in b['measurements']:
   inv=checked(job['invocation']);checked(job['accepted']);assert inv['exit_code']==0 and inv['accepted']==job['accepted']
   out=Path(job['accepted']['path']).parent;pred,arrays=archive(out);arm=job['slot']['arm']
   req=json.loads((out/'request.json').read_text());sup=json.loads((out/'supervisor.json').read_text())
   if 'expected_items' in case:
    assert req['workload'].get('batch',{}).get('items',1)==case['expected_items'] and len(arrays)==case['expected_items']
   if 'expected_seed' in case:assert req['seed']==case['expected_seed']
   assert req['hip_device']==6 and sup['phase']=='succeeded' and sup['device_verified'] and sup['physical_device']['pci_address']=='0000:8e:00.0'
   if arm not in anchor:anchor[arm]=arrays;within_exact[arm]=True
   within_exact[arm]=within_exact[arm] and arrays==anchor[arm]
   # Protenix Mini uses the existing Rust coordinate/confidence tolerance policy.
   # Byte equality is an additional reported diagnostic, not that model's gate.
   if req['workload']['model']!='protenix':assert within_exact[arm],f'Within-arm output drift: {case["name"]}/{arm}'
   if 'baseline' in anchor:all_exact=all_exact and arrays==anchor['baseline']
   cifs=[p.read_bytes() for p in sorted((out/'native').rglob('*.cif'))]
   if cif_anchor is None:cif_anchor=cifs
   cif_exact=cif_exact and cifs==cif_anchor
   records.append(dict(slot=job['slot'],invocation=job['invocation'],accepted=job['accepted'],e2e_seconds=inv['elapsed_seconds'],model_seconds=pred['model_seconds'],workload=req['workload'],seed=req['seed'],input=req['input'],archives=len(arrays),cif_files=len(cifs)))
  if records[0]['workload']['model'] in ('plm','atlasfold','openfold3','esmfold2'):assert all_exact
  if records[0]['workload']['model'] in ('atlasfold','openfold3','esmfold2'):assert cif_exact and records[0]['cif_files']>0
  cases[case['name']]=dict(status='verified',benchmark=pin(path),comparison=b['comparison'],performance=comparison['performance'],numerical_pairs=comparison['pairs'],all_exported_arrays_exact=all_exact,within_arm_exact=all(within_exact.values()),within_arm_exact_by_arm=within_exact,full_cif_exact=cif_exact if cif_anchor else None,records=records,medians={field:{arm:statistics.median(r[field] for r in records if r['slot']['arm']==arm) for arm in ('baseline','candidate')} for field in ('e2e_seconds','model_seconds')})
 result=dict(schema_version=1,root=str(root),plan=pin(root/'plan.json'),cli=plan['cli'],cases=cases)
 output=root/'collected-latest.json';output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:{a:v[a] for a in ('status','medians') if a in v} for k,v in cases.items()},indent=2))

if __name__=='__main__':main()
