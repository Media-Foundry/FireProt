"""Compare four optimized cold invocations with one optimized resident B4 job.

Each invocation uses Rust measurement/acceptance. This aggregate intentionally
changes grouping, so it is not submitted to the identical-workload comparator.
Singleton and batch arrays must match their own retained official references.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

root = Path('/media/IntelSSD/fireprot')
experiment = Path(sys.argv[1]).resolve()
deployment = experiment/'qualified-v3'
work = experiment/'atlaslm-batch-reuse'; work.mkdir(exist_ok=False)
cli = deployment/'target/release/fireprot'

def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def write(path,value):
    with path.open('x') as handle: json.dump(value,handle,indent=2);handle.write('\n')
    return pin(path)

profile = json.loads((experiment/'atlaslm-3b-pairs/profile.json').read_text())
profile_pin = write(work/'profile.json',profile)
base = json.loads((root/'configs/requests/plm-atlaslm-3b-n1000-composed.json').read_text())
batch_base = json.loads((experiment/'atlaslm-3b-pairs/candidate-1.json').read_text())
base['resources']['hip_batch'] = batch_base['resources']['hip_batch']
batch_base['resources'] = base['resources']
sequences = json.loads(Path(batch_base['input']['path']).read_text())['sequences']
inputs = []
for index,item in enumerate(sequences):
    inputs.append(write(work/f'input-{index}.json', [dict(name=item['id'],sequences=[dict(proteinChain=dict(sequence=item['sequence'],count=1))])]))
env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'): env.pop(key,None)
result=dict(schema_version=1,model='atlaslm-3b',hip_device=6,pci_address='0000:8e:00.0',seed=102,
    precision='native',items=4,length=1000,backend='fireprot',
    scope='Sum of four complete optimized singleton invocation intervals versus one complete optimized B4 interval. Rust measures and validates each invocation. Caller orchestration and post-measurement verification excluded from both arms. Fresh processes, unflushed caches; AB/BA/AB, no observations discarded.',
    numerical_policy='Each singleton matches official singleton output; B4 matches official identical B4 grouping. Singleton-to-batch drift is separate and is not asserted zero.',
    cli=pin(cli),profile=profile_pin,pairs=[])
for pair_index,order in enumerate([('serial','batch'),('batch','serial'),('serial','batch')]):
    pair=dict(order=list(order))
    for arm in order:
        observations=[]
        for item in (range(4) if arm=='serial' else [None]):
            name=f'pair-{pair_index+1}-{arm}'+(f'-{item}' if item is not None else '')
            request=dict(base if arm=='serial' else batch_base,run_id=name,output_dir=str(work/name),hip_device=6)
            if item is not None: request['input']=inputs[item]
            request_pin=write(work/f'{name}-request.json',request)
            spec_pin=write(work/f'{name}-spec.json',dict(schema_version=1,timeout_seconds=300,request=request_pin,profile=profile_pin))
            measurement=work/f'{name}-measurement'
            with (work/f'{name}.log').open('x') as log:
                subprocess.run([str(cli),'measure-inference',spec_pin['path'],str(measurement)],cwd=root,env=env,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            invocation_path=measurement/'invocation.json'
            invocation=json.loads(invocation_path.read_text())
            assert invocation['exit_code']==0
            output=Path(request['output_dir'])
            accepted=json.loads((output/'accepted.json').read_text())
            assert pin(output/'accepted.json')==invocation['accepted']
            for artifact in accepted['metadata']+accepted['outputs']:
                file=output/artifact['path']
                assert file.resolve().is_relative_to(output.resolve()) and pin(file)['sha256']==artifact['sha256']
            supervisor=json.loads((output/'supervisor.json').read_text())
            assert supervisor['phase']=='succeeded' and supervisor['device_verified']
            assert supervisor['physical_device']['pci_address']=='0000:8e:00.0'
            for output_index,reference_index in ([(0,item)] if item is not None else enumerate(range(4))):
                actual=output/f'sample-{output_index}.npz'
                expected=experiment/'atlaslm/1000-1000-1000-1000'/f'{"single" if arm=="serial" else "batch"}-{reference_index}.npz'
                with zipfile.ZipFile(actual) as a,zipfile.ZipFile(expected) as b:
                    assert set(a.namelist())==set(b.namelist())
                    assert all(a.read(k)==b.read(k) for k in a.namelist())
            observations.append(dict(invocation=pin(invocation_path),accepted=invocation['accepted'],
                                     elapsed_seconds=invocation['elapsed_seconds'],official_reference_exact=True))
        pair[arm]=dict(invocations=observations,total_seconds=sum(x['elapsed_seconds'] for x in observations))
        print(pair_index+1,arm,pair[arm]['total_seconds'],flush=True)
    pair['speedup']=pair['serial']['total_seconds']/pair['batch']['total_seconds']
    result['pairs'].append(pair)
    (work/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
result['medians']={arm:statistics.median(p[arm]['total_seconds'] for p in result['pairs']) for arm in ('serial','batch')}
result['median_speedup']=result['medians']['serial']/result['medians']['batch']
write(work/'report.json',result)
print(json.dumps(result['medians']),result['median_speedup'],flush=True)
