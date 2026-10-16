"""Measure direct native-to-current ESMC E2E, including the native monitor."""
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_esmc_monitor_pairs import pin, write


def main():
    root=Path('/media/IntelSSD/fireprot');work=Path(sys.argv[1]).resolve()
    assert json.loads((work/'qualification-v3/qualification.json').read_text())['passed']
    assert json.loads((work/'isolation-check/report.json').read_text())['passed']
    case=work/'native-pairs';case.mkdir(exist_ok=False)
    request=json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
    source_profiles={
        'baseline':root/'outputs/experiments/esmc-host-20261016/native-final/baseline-profile.json',
        'candidate':work/'b4-pairs/candidate-profile.json'}
    profiles={};values={}
    for arm,path in source_profiles.items():
        values[arm]=json.loads(path.read_text());profiles[arm]=write(case/f'{arm}-profile.json',values[arm])
    assert values['baseline']['workers']==values['candidate']['workers']
    changes={'backend':dict(baseline='native',candidate='fireprot'),
        'invocation.profile_sha256':dict(baseline=profiles['baseline']['sha256'],candidate=profiles['candidate']['sha256'])}
    for key,value in values['baseline']['environment'].items():
        other=values['candidate']['environment'][key]
        if value!=other:changes['launch.environment.'+key]=dict(baseline=value,candidate=other)
    def identity(path):
        p=Path(path).resolve();return dict(name='amd-smi',canonical_path=str(p),sha256=pin(p)['sha256'])
    changes['launch.inputs.amd-smi']=dict(baseline=identity(values['baseline']['amd_smi']),candidate=identity(values['candidate']['amd_smi']))
    pairs=[]
    for index in range(3):
        pair=dict(first='candidate' if index==1 else 'baseline')
        for arm in ('baseline','candidate'):
            value=dict(request,backend='native' if arm=='baseline' else 'fireprot',
                       run_id=f'esmc-monitor-native-{arm}-{index+1}',output_dir=str(case/f'{arm}-{index+1}'))
            pair[arm]=dict(request=write(case/f'{arm}-{index+1}.json',value),profile=profiles[arm])
        pairs.append(pair)
    spec=dict(schema_version=1,execution=dict(schema_version=1,timeout_seconds=3600,invocation_timeout_seconds=600,pairs=pairs),
        comparison=dict(scope='end_to_end',monitor_treatment=True,
            cache_state='Fresh workers; unflushed caches; full original checkpoint hashes and all exports; operators and monitor built offline.',
            treatment_description='Direct native ESMC B4 reference (hash4/inherited CPU mask/original loader/sequential imports/original SMI) versus current exact HIP/pinned loader/import overlap/hash8/local CPU/native SMI. N1000 x4, native BF16, seed101, HIP6. Same new Rust CLI and frozen workers; no multiplied incremental ratios.',expected_changes=changes))
    write(case/'spec.json',spec)
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for key in ('ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'):env.pop(key,None)
    with (case/'pairs.log').open('x') as log:
        subprocess.run([str(work/'fireprot-bench'),'benchmark-pairs',str(case/'spec.json'),str(case/'measurements')],
                       cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Direct native pairs complete',flush=True)


if __name__=='__main__':main()
