"""Rehash all accepted standalone pLM pair archives and collect exact results."""
from pathlib import Path
import hashlib
import json
import statistics
import sys
import zipfile


def collect(root, destination, *selected):
    root = Path(root).resolve()
    def pin(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def checked(item):
        path = Path(item['path'])
        assert path.is_absolute() and pin(path)['sha256'] == item['sha256']
        return json.loads(path.read_text())
    models = {}
    specifications = [('esmc-fold2-6b',80,2),('esm2-3b',36,3),('atlaslm-3b',48,3)]
    if 'esm3-open' in selected:
        specifications.append(('esm3-open', 0, 3))
    assert set(selected).issubset({item[0] for item in specifications})
    for model, calls, arrays in specifications:
        if selected and model not in selected:
            continue
        path = root / (model+'-measurements/benchmark.json')
        benchmark = json.loads(path.read_text())
        assert benchmark['phase'] == 'succeeded' and len(benchmark['measurements']) == 6
        for key in ('execution_plan','comparison_settings'):
            checked(benchmark[key])
        assert pin(Path(benchmark['order']['path'])) == benchmark['order']
        comparison = checked(benchmark['comparison'])
        assert comparison['correctness_passed']
        for item in benchmark['checks']:
            assert checked(item)['correctness_passed']
        anchor = None
        records = []
        for job in benchmark['measurements']:
            invocation = checked(job['invocation']); accepted = checked(job['accepted'])
            assert invocation['exit_code'] == 0 and invocation['accepted'] == job['accepted']
            output = Path(job['accepted']['path']).parent
            for artifact in accepted['metadata'] + accepted['outputs']:
                file = output / artifact['path']
                assert file.resolve().is_relative_to(output.resolve())
                assert pin(file)['sha256'] == artifact['sha256']
            request = json.loads((output/'request.json').read_text())
            prediction = json.loads((output/'predictions.json').read_text())
            supervisor = json.loads((output/'supervisor.json').read_text())
            assert supervisor['phase'] == 'succeeded' and supervisor['device_verified']
            assert supervisor['physical_device']['pci_address'] == '0000:8e:00.0'
            assert request['hip_device'] == 6
            assert request['seed'] == (102 if model == 'atlaslm-3b' else 101)
            assert request['workload'] == dict(model='plm',model_name=model,precision='native')
            retained = json.loads((output/'input-snapshot.json').read_text())
            assert len(retained[0]['sequences'][0]['proteinChain']['sequence']) == 1000
            assert prediction['schema_version'] == 5 and len(prediction['predictions']) == 1
            assert len(prediction['predictions'][0]['arrays']) == arrays
            assert prediction['predictions'][0].get('structure') is None
            if request['backend'] == 'fireprot':
                runtime = json.loads((output/'operator-runtime.json').read_text())
                assert runtime['calls'] == calls
            with zipfile.ZipFile(output/'sample-0.npz') as archive:
                values = {name: archive.read(name) for name in archive.namelist()}
            if anchor is None: anchor = values
            assert values == anchor
            records.append(dict(index=job['index'], slot=job['slot'], run_id=request['run_id'],
                invocation=job['invocation'],accepted=job['accepted'],e2e_seconds=invocation['elapsed_seconds'],
                first_forward_seconds=prediction['model_seconds'],request_seconds=prediction['request_seconds']))
        for pair in comparison['pairs']:
            assert pair['cif_atom_identities_checked'] is False
            for sample in pair['samples']:
                assert sample['passed'] and sample['policy'] == 'all-arrays-exact-v1'
                assert len(sample['exact_arrays']) == arrays and all(sample['exact_arrays'].values())
        models[model] = dict(benchmark=pin(path),comparison=benchmark['comparison'],order=benchmark['order'],
            records=records,performance=comparison['performance'],all_arrays_exact=True,
            first_forward_medians={arm:statistics.median(r['first_forward_seconds'] for r in records if r['slot']['arm']==arm) for arm in ('baseline','candidate')})
    result = dict(schema_version=1, length=1000, hip_device=6, pci_address='0000:8e:00.0',models=models,
        scope='Three AB/BA/AB fresh-process pairs per model under frozen Rust CLI, sequentially on HIP6. Native precision, seed101 except seed102 AtlasLM, one sequence forward, unflushed filesystem/GPU library caches. All arrays bitwise equal; no observations discarded. Cold first-forward timings include lazy GPU initialization and are distinct from warm-forward screens. No folding speedup is claimed.')
    with Path(destination).open('x') as handle:
        json.dump(result,handle,indent=2);handle.write('\n')


if __name__ == '__main__':
    collect(*sys.argv[1:])
