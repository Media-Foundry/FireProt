"""Recheck accepted B4 archives and collect matched complete-invocation results."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

root = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2])

def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def checked(item):
    path = Path(item['path'])
    assert path.is_absolute() and pin(path) == item
    return json.loads(path.read_text())

models = {}
for model, calls, array_count in [('atlaslm-3b',48,3),('esmc-fold2-6b',80,2),('esm2-3b',36,3)]:
    path = root/f'{model}-pairs/measurements/benchmark.json'
    benchmark = json.loads(path.read_text())
    assert benchmark['phase'] == 'succeeded' and len(benchmark['measurements']) == 6
    for key in ('execution_plan','comparison_settings'): checked(benchmark[key])
    assert pin(Path(benchmark['order']['path'])) == benchmark['order']
    comparison = checked(benchmark['comparison'])
    assert comparison['correctness_passed']
    for item in benchmark['checks']: assert checked(item)['correctness_passed']
    anchor = None
    records = []
    for job in benchmark['measurements']:
        invocation = checked(job['invocation']); accepted = checked(job['accepted'])
        assert invocation['exit_code'] == 0 and invocation['accepted'] == job['accepted']
        output = Path(job['accepted']['path']).parent
        for artifact in accepted['metadata'] + accepted['outputs']:
            file = output/artifact['path']
            assert file.resolve().is_relative_to(output.resolve())
            assert pin(file)['sha256'] == artifact['sha256']
        request = json.loads((output/'request.json').read_text())
        prediction = json.loads((output/'predictions.json').read_text())
        supervisor = json.loads((output/'supervisor.json').read_text())
        assert supervisor['phase'] == 'succeeded' and supervisor['device_verified']
        assert supervisor['physical_device']['pci_address'] == '0000:8e:00.0'
        assert request['hip_device'] == 6 and request['seed'] == (102 if model=='atlaslm-3b' else 101)
        assert request['workload'] == dict(model='plm',model_name=model,precision='native',
            batch=dict(items=4,max_batch_size=4,max_padded_tokens=4008,grouping='length_sorted'))
        retained = json.loads((output/'input-snapshot.json').read_text())
        assert all(len(x['sequence']) == 1000 for x in retained['sequences'])
        assert prediction['schema_version'] == 6 and prediction['batch_groups'] == [[0,1,2,3]]
        assert len(prediction['predictions']) == 4
        values = []
        for index, item in enumerate(prediction['predictions']):
            assert item['sample'] == index and item['input_id'] == retained['sequences'][index]['id']
            assert len(item['arrays']) == array_count and item.get('structure') is None
            with zipfile.ZipFile(output/item['archive']) as archive:
                values.append({name:archive.read(name) for name in archive.namelist()})
        if anchor is None: anchor=values
        assert values == anchor
        if request['backend'] == 'fireprot':
            runtimes = json.loads((output/'operator-runtime.json').read_text())
            assert len(runtimes) == 1 and runtimes[0]['calls'] == calls and runtimes[0]['batch_size'] == 4
            if model == 'atlaslm-3b':
                assert runtimes[0]['swiglu_calls'] == 48 and runtimes[0]['rotary_modules'] == 1
                assert runtimes[0]['swiglu_library'] == request['resources']['hip_batch']
        records.append(dict(index=job['index'],slot=job['slot'],run_id=request['run_id'],
            invocation=job['invocation'],accepted=job['accepted'],e2e_seconds=invocation['elapsed_seconds'],
            first_forward_seconds=prediction['model_seconds'],request_seconds=prediction['request_seconds']))
    for pair in comparison['pairs']:
        assert len(pair['samples']) == 4
        for sample in pair['samples']:
            assert sample['passed'] and sample['policy'] == 'all-arrays-exact-v1'
            assert len(sample['exact_arrays']) == array_count and all(sample['exact_arrays'].values())
    models[model] = dict(benchmark=pin(path),comparison=benchmark['comparison'],records=records,
        performance=comparison['performance'],all_arrays_exact=True,
        first_forward_medians={arm:statistics.median(r['first_forward_seconds'] for r in records if r['slot']['arm']==arm)
                               for arm in ('baseline','candidate')})
result=dict(schema_version=1,hip_device=6,batch_size=4,length=1000,models=models,
    scope='Three AB/BA/AB complete Rust invocation pairs per model, same four distinct N1000 inputs/grouping/native precision. Seed102 AtlasLM, seed101 ESMC/ESM2. Unflushed caches, fresh processes, no feature cache. Every retained array bitwise equal. This compares native B4 to FireProt B4, not four independent cold single-sequence processes.')
with destination.open('x') as handle: json.dump(result,handle,indent=2);handle.write('\n')
