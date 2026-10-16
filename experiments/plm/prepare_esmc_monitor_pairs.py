"""Compare qualified native monitoring against the original AMD SMI CLI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    return pin(path)


def main():
    root=Path('/media/IntelSSD/fireprot');work=Path(sys.argv[1]).resolve()
    qualification=json.loads((work/'qualification-v3/qualification.json').read_text())
    assert qualification['passed'] and len(qualification['holders'])==2
    assert all(h['passed'] for h in qualification['holders'])
    native=qualification['binaries']['native'];assert pin(Path(native['path']))==native
    official=qualification['official'];assert pin(Path(official['path']))==official
    cli=work/'fireprot-bench';assert pin(cli)==json.loads((work/'benchmark-cli.json').read_text())
    library=Path('/opt/rocm/lib/libamd_smi.so.27.0.0');library_pin=pin(library)
    write(work/'runtime-library.json',library_pin)
    def identity(value):return dict(name='amd-smi',canonical_path=str(Path(value['path']).resolve()),sha256=value['sha256'])
    cases=[]
    for count in (4,64):
        case=work/f'b{count}-pairs';case.mkdir(exist_ok=False)
        request=json.loads((root/('configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json' if count==4
            else 'configs/requests/plm-esmc-fold2-6b-resident64-gemm.json')).read_text())
        profile_path=root/('configs/diamondhill/plm-esmc-fold2-6b-host-scheduling.json' if count==4
            else 'configs/diamondhill/plm-esmc-fold2-6b-resident-gemm.json')
        profiles={}
        for arm in ('baseline','candidate'):
            profile=json.loads(profile_path.read_text())
            assert Path(profile['amd_smi']).resolve()==Path(official['path']).resolve()
            if arm=='candidate':profile['amd_smi']=native['path']
            profiles[arm]=write(case/f'{arm}-profile.json',profile)
        pairs=[]
        for index in range(3):
            pair=dict(first='candidate' if index==1 else 'baseline')
            for arm in ('baseline','candidate'):
                value=dict(request,run_id=f'esmc-monitor-{count}-{arm}-{index+1}',output_dir=str(case/f'{arm}-{index+1}'))
                assert value['hip_device']==6
                pair[arm]=dict(request=write(case/f'{arm}-{index+1}.json',value),profile=profiles[arm])
            pairs.append(pair)
        spec=dict(schema_version=1,execution=dict(schema_version=1,timeout_seconds=3600,invocation_timeout_seconds=600,pairs=pairs),
            comparison=dict(scope='end_to_end',monitor_treatment=True,
                cache_state='Fresh workers and query processes; unflushed caches; full checkpoint hashes and all exports; native probe built offline.',
                treatment_description=f'Only AMD SMI query executable changes: original CLI versus qualified one-shot native probe. ESMC N1000 x {count}, B4, native BF16, seed101, HIP6. Same model/worker/CLI/assets and unchanged Rust mapping/process checks and 250ms sampling interval.',
                expected_changes={
                    'launch.inputs.amd-smi':dict(baseline=identity(official),candidate=identity(native)),
                    'invocation.profile_sha256':dict(baseline=profiles['baseline']['sha256'],candidate=profiles['candidate']['sha256'])}))
        write(case/'spec.json',spec);cases.append(case)
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for name in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):env.pop(name,None)
    for case in cases:
        with (case/'pairs.log').open('x') as log:
            subprocess.run([str(cli),'benchmark-pairs',str(case/'spec.json'),str(case/'measurements')],
                           cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert pin(library)==library_pin and pin(Path(native['path']))==native and pin(Path(official['path']))==official
        print(case.name,'complete',flush=True)


if __name__=='__main__':main()
