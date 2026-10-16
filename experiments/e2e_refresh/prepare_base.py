"""Prepare the unchanged Base aligned-coordinate policy with fresh run identities."""
import json
from pathlib import Path
import shutil
import sys
from prepare import ROOT,pin,load,write


def main():
 root=Path(sys.argv[1]);case=root/'base-n1000';case.mkdir(exist_ok=False)
 old=ROOT/'outputs/experiments/protenix-base-final-native-pairs-20261007'
 script=(old/'run.sh').read_text()
 script=script.replace(str(old),str(case)).replace(str(ROOT/'target/release/fireprot'),str(root/'fireprot'))
 replacements={}
 for arm in ('native','optimized'):
  shutil.copy2(old/f'{arm}-profile.json',case/f'{arm}-profile.json')
  for i in range(3):
   request=load(old/f'{arm}-r{i}-request.json');prior_id=request['run_id'];prior_out=request['output_dir']
   request.update(run_id=f'e2e-refresh-20261016-base-{arm}-{i}',output_dir=str(case/f'{arm}-{i}'),hip_device=6)
   replacements[prior_out]=request['output_dir']
   request_pin=write(case/f'{arm}-r{i}-request.json',request)
   write(case/f'{arm}-r{i}-spec.json',dict(schema_version=1,timeout_seconds=1800,request=request_pin,profile=pin(case/f'{arm}-profile.json')))
 for before,after in replacements.items():script=script.replace(before,after)
 shutil.copy2(old/'anchors.sha256',case/'anchors.sha256')
 shutil.copy2(Path(__file__).with_name('check_base.py'),case/'check_base.py')
 for i in range(3):
  needle=f"jq -e '.coordinate_gate_passed == true' {case}/aligned-{i}.json > /dev/null"
  assert script.count(needle)==1
  script=script.replace(needle,needle+f'\npython3 {case}/check_base.py {case} {i}')
 (case/'run.sh').write_text(script)
 write(case/'preparation.json',dict(cli=pin(root/'fireprot'),historical_driver=pin(old/'run.sh'),driver=pin(case/'run.sh'),policy='Unchanged aligned all-atom RMSD <= 0.1 A, summary confidence <= 0.001; exact nine arrays within each arm against frozen anchors; legacy coordinate threshold remains unchanged.',scientific=load(case/'native-r0-request.json')['workload']))
 print('Prepared Base native/optimized three-pair study')

if __name__=='__main__':main()
