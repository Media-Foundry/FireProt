"""Collect a successful Rust ESMFold2 benchmark without rerunning inference."""
from pathlib import Path
import hashlib
import json
import statistics
import sys


def collect(root, destination):
    root = Path(root).resolve()
    measurement = root / 'n1000-pairs/measurements'

    def pin(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def checked(item):
        path = Path(item['path'])
        assert path.is_absolute() and pin(path)['sha256'] == item['sha256'], item
        return json.loads(path.read_text())

    bench_path = measurement / 'benchmark.json'
    bench = json.loads(bench_path.read_text())
    assert bench['phase'] == 'succeeded' and len(bench['measurements']) == 6
    for key in ('execution_plan', 'comparison_settings'):
        checked(bench[key])
    assert pin(Path(bench['order']['path'])) == bench['order']
    comparison = checked(bench['comparison'])
    assert comparison['correctness_passed'] and comparison['performance']['baseline']['count'] == 3
    for item in bench['checks']:
        assert checked(item)['correctness_passed']
    records = []
    anchor_cif = None
    for job in bench['measurements']:
        inv = checked(job['invocation'])
        accepted = checked(job['accepted'])
        assert inv['accepted'] == job['accepted'] and inv['exit_code'] == 0
        output = Path(job['accepted']['path']).parent
        for item in accepted['metadata'] + accepted['outputs']:
            path = output / item['path']
            assert path.resolve().is_relative_to(output.resolve())
            assert pin(path)['sha256'] == item['sha256']
        request = json.loads((output / 'request.json').read_text())
        prediction = json.loads((output / 'predictions.json').read_text())
        runtime = json.loads((output / 'operator-runtime.json').read_text())
        supervisor = json.loads((output / 'supervisor.json').read_text())
        assert supervisor['phase'] == 'succeeded' and supervisor['device_verified']
        assert supervisor['physical_device']['pci_address'] == '0000:93:00.0'
        retained = json.loads((output / 'input-snapshot.json').read_text())
        assert len(retained[0]['sequences'][0]['proteinChain']['sequence']) == 1000
        assert request['hip_device'] == 7 and request['seed'] == 101
        assert request['workload'] == dict(model='esmfold2', model_name='biohub/ESMFold2', loops=3,
            sampling_steps=50, samples=1, precision='native', chunk_size=64, deterministic=True)
        assert runtime['counts'] == dict(model=1, language_model=1, trunk=4, lm_encoder=4, diffusion=34)
        assert runtime['profiling'] is False and runtime['bundled_esmc'] is True
        assert len(prediction['predictions'][0]['arrays']) == 12
        if request['backend'] == 'fireprot':
            assert runtime['fireprot']['completed_dispatches'] == 460
        cif = (output / 'native/sample-0.cif').read_bytes()
        if anchor_cif is None:
            anchor_cif = cif
        assert cif == anchor_cif
        records.append(dict(index=job['index'], slot=job['slot'], run_id=request['run_id'],
            invocation=job['invocation'], accepted=job['accepted'],
            e2e_seconds=inv['elapsed_seconds'], model_seconds=prediction['model_seconds'],
            request_seconds=prediction['request_seconds'],
            peak_allocated_bytes=runtime['peak_allocated_bytes'], peak_reserved_bytes=runtime['peak_reserved_bytes']))
    for pair in comparison['pairs']:
        assert pair['cif_atom_identities_checked']
        for sample in pair['samples']:
            assert sample['passed'] and sample['policy'] == 'all-arrays-exact-v1'
            assert len(sample['exact_arrays']) == 12 and all(sample['exact_arrays'].values())
    model_medians = {arm: statistics.median(r['model_seconds'] for r in records if r['slot']['arm'] == arm)
                     for arm in ('baseline', 'candidate')}
    result = dict(schema_version=1, model='biohub/ESMFold2 including bundled ESMC', length=1000,
        workload=request['workload'], seed=101, hip_device=7, actual_denoiser_calls=34,
        benchmark=pin(bench_path), comparison=bench['comparison'], order=bench['order'],
        qualification=pin(root / 'qualification-result.json'), records=records,
        performance=comparison['performance'], model_medians=model_medians,
        model_median_speedup=model_medians['baseline'] / model_medians['candidate'],
        all12_exact=True, full_cif_exact=True,
        scope='Three fresh-process AB/BA/AB pairs under the same Rust launcher. Native BF16/FP32 boundaries, query-only MSA, bundled ESMC included, unflushed caches, profiling disabled. This loops3/schedule50 development workload is not the upstream loops20/schedule200 default. No observations discarded.')
    with Path(destination).open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    collect(*sys.argv[1:])
