"""Verify host scheduling pairs, all asset checks, and official same-batch output."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(item):
    path = Path(item['path'])
    assert path.is_absolute() and pin(path) == item
    return json.loads(path.read_text())


def arrays(output, prediction):
    result = []
    for index, item in enumerate(prediction['predictions']):
        assert item['sample'] == index and set(item['arrays']) == {'tokens', 'embeddings'}
        with zipfile.ZipFile(output/item['archive']) as archive:
            result.append({name: archive.read(name) for name in archive.namelist()})
    return result


def verify_archive(output):
    accepted = json.loads((output/'accepted.json').read_text())
    for artifact in accepted['metadata'] + accepted['outputs']:
        file = output/artifact['path']
        assert file.resolve().is_relative_to(output.resolve())
        assert pin(file)['sha256'] == artifact['sha256']
    return json.loads((output/'predictions.json').read_text())


work = Path(sys.argv[1]).resolve()
reference = Path('/media/IntelSSD/fireprot/outputs/experiments/esmc-load-20261016/validation/native-pairs/baseline-1')
reference_prediction = verify_archive(reference)
anchor = arrays(reference, reference_prediction)
path = work/'measurements/benchmark.json'
benchmark = json.loads(path.read_text())
assert benchmark['phase'] == 'succeeded' and len(benchmark['measurements']) == 6
for key in ('execution_plan', 'comparison_settings'):
    checked(benchmark[key])
assert pin(Path(benchmark['order']['path'])) == benchmark['order']
comparison = checked(benchmark['comparison'])
assert comparison['correctness_passed']
for item in benchmark['checks']:
    assert checked(item)['correctness_passed']
records = []
provenance_anchor = None
for job in benchmark['measurements']:
    invocation = checked(job['invocation'])
    checked(job['accepted'])
    assert invocation['exit_code'] == 0 and invocation['accepted'] == job['accepted']
    output = Path(job['accepted']['path']).parent
    prediction = verify_archive(output)
    provenance = json.loads((output/'provenance.json').read_text())
    if provenance_anchor is None:
        provenance_anchor = provenance
    assert provenance == provenance_anchor
    execution = json.loads((output/'execution.json').read_text())
    profile = json.loads((work/f"{job['slot']['arm']}-profile.json").read_text())
    for name in ('FIREPROT_PREFLIGHT_HASH_THREADS', 'FIREPROT_WORKER_CPU_AFFINITY'):
        assert execution['environment'][name] == profile['environment'][name]
    request = json.loads((output/'request.json').read_text())
    supervisor = json.loads((output/'supervisor.json').read_text())
    assert supervisor['phase'] == 'succeeded' and supervisor['device_verified']
    assert supervisor['physical_device']['pci_address'] == '0000:8e:00.0'
    assert request['hip_device'] == 6 and request['seed'] == 101
    assert request['workload'] == dict(model='plm', model_name='esmc-fold2-6b', precision='native',
        batch=dict(items=4, max_batch_size=4, max_padded_tokens=4008, grouping='length_sorted'))
    retained = json.loads((output/'input-snapshot.json').read_text())
    assert all(len(x['sequence']) == 1000 for x in retained['sequences'])
    assert prediction['schema_version'] == 6 and prediction['batch_groups'] == [[0, 1, 2, 3]]
    assert len(prediction['predictions']) == 4
    assert arrays(output, prediction) == anchor
    for index, item in enumerate(prediction['predictions']):
        assert item['input_id'] == retained['sequences'][index]['id']
    if request['backend'] == 'fireprot':
        runtime, = json.loads((output/'operator-runtime.json').read_text())
        assert runtime['calls'] == 160 and runtime['swiglu_calls'] == 80 and runtime['batch_size'] == 4
        assert runtime['library'] == request['resources']['hip_esm']
    records.append(dict(index=job['index'], slot=job['slot'], invocation=job['invocation'],
        accepted=job['accepted'], e2e_seconds=invocation['elapsed_seconds'],
        host_outside_supervisor_seconds=invocation['elapsed_seconds']-supervisor['elapsed_seconds'],
        hash_threads=execution['environment']['FIREPROT_PREFLIGHT_HASH_THREADS'],
        worker_cpu_affinity=execution['environment']['FIREPROT_WORKER_CPU_AFFINITY'],
        supervisor_seconds=supervisor['elapsed_seconds'],
        model_forward_seconds=prediction['model_seconds'], request_seconds=prediction['request_seconds']))
for pair in comparison['pairs']:
    assert len(pair['samples']) == 4
    for sample in pair['samples']:
        assert sample['passed'] and sample['policy'] == 'all-arrays-exact-v1'
        assert len(sample['exact_arrays']) == 2 and all(sample['exact_arrays'].values())
result = dict(schema_version=1, boundary='Complete invocation; reference and treatment are sealed in comparison settings.',
    benchmark=pin(path), comparison=benchmark['comparison'], reference_accepted=pin(reference/'accepted.json'),
    records=records, performance=comparison['performance'], all_arrays_exact=True,
    medians={field: {arm: statistics.median(r[field] for r in records if r['slot']['arm'] == arm)
                    for arm in ('baseline', 'candidate')}
             for field in ('e2e_seconds', 'supervisor_seconds', 'model_forward_seconds', 'request_seconds', 'host_outside_supervisor_seconds')})
with (work/'collected.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps(dict(performance=result['performance'], medians=result['medians']), indent=2))
