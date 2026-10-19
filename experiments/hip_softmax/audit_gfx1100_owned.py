"""Independently verify owned-score trials against the maintained native prediction."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def pin(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(path.read_text())


def checked(item, base):
    path = Path(item['path'])
    if not path.is_absolute():
        path = base/path
    assert pin(path)['sha256'] == item['sha256'], str(path)
    return path


def audit(repo, root, destination):
    repo, root, destination = map(lambda v: Path(v).resolve(), (repo, root, destination))
    preparation = read(root/'preparation.json')
    for item in preparation['files'] + [preparation[k] for k in ('library', 'cli', 'bridge')]:
        checked(item, root)
    reference = repo/'outputs/experiments/gfx11-openfold3-model-20261017-v1/native-prediction'
    for item in read(reference/'accepted.json')['metadata'] + read(reference/'accepted.json')['outputs']:
        checked(item, reference)
    ref_sample = read(reference/'predictions.json')['predictions'][0]
    paths = sorted(root.glob('measurement-*/invocation.json')) + sorted(root.glob('paired/measurement-*/invocation.json'))
    records = []
    for path in paths:
        invocation = read(path)
        assert invocation['exit_code'] == 0
        profile = read(checked(invocation['profile'], path.parent))
        request = read(checked(invocation['request'], path.parent))
        assert request['workload'] == dict(model='openfold3', model_name='openbind-0',
            recycles=3, diffusion_steps=200, samples=1, precision='float32', deterministic=True)
        assert request['hip_device'] == 0 and request['seed'] == 101
        accepted = checked(invocation['accepted'], path.parent)
        directory = accepted.parent
        receipt = read(accepted)
        for item in receipt['metadata'] + receipt['outputs']:
            checked(item, directory)
        assert {'triangle-copy.json', 'reuse-runtime.json', 'operator-runtime.json'} <= {v['path'] for v in receipt['outputs']}
        supervisor = read(directory/'supervisor.json')
        assert supervisor['phase'] == 'succeeded' and supervisor['device_verified']
        assert supervisor['physical_device']['pci_address'] == '0000:53:00.0'
        bundle = read(directory/'predictions.json')
        sample = bundle['predictions'][0]
        command = [str(repo/'target/release/fireprot'), 'compare-npz',
                   str(reference/ref_sample['archive']), str(directory/sample['archive'])]
        replay = subprocess.run(command, capture_output=True, text=True, timeout=60)
        comparison = json.loads(replay.stdout)
        exact = replay.returncode == 0 and len(comparison) == 11 and all(v['bitwise_equal'] and v['passed'] for v in comparison.values())
        cif_exact = pin(directory/sample['structure'])['sha256'] == pin(reference/ref_sample['structure'])['sha256']
        copy = read(directory/'triangle-copy.json')
        reuse = read(directory/'reuse-runtime.json')
        operator = read(directory/'operator-runtime.json')
        enabled = profile['environment']['FIREPROT_GFX1100_OWNED'] == '1'
        owned = operator['kernel']['owned_scores']
        coverage = (copy['completed'] and copy['enabled'] is True
            and reuse['completed'] and reuse['mode'] == 3
            and operator['launches_submitted'] == 1696
            and owned['sha256'] == preparation['library']['sha256']
            and owned['enabled'] == enabled and owned['fallbacks'] == 0
            and copy['copies'] == 3440 and copy['counts']['accepted'] == 430)
        if enabled:
            coverage &= owned['calls'] == 1696
        else:
            coverage &= owned['calls'] == 0
        for cache in reuse['caches'].values():
            scopes = cache['scopes']
            coverage &= len(scopes) == 1 and scopes[0]['completed'] and all(scopes[0][k] == v
                for k,v in [('calls',200),('hits',199),('misses',1),('fallbacks',0)])
        coverage &= set(reuse['caches']) == {'pair','atom'} and reuse['template']['counts']['reused_slots'] == 27
        records.append(dict(invocation=pin(path), accepted=pin(accepted), enabled=enabled,
            e2e_seconds=invocation['elapsed_seconds'], model_seconds=bundle['model_seconds'],
            all_11_arrays_exact=exact, full_cif_exact=cif_exact, coverage_passed=bool(coverage),
            copy=copy, owned_scores=owned, softmax_launches=operator['launches_submitted'],
            peak_allocated_bytes=reuse['peak_allocated_bytes'], peak_reserved_bytes=reuse['peak_reserved_bytes']))
    result = dict(preparation=pin(root/'preparation.json'), auditor=pin(Path(__file__).resolve()),
        reference=pin(reference/'accepted.json'), records=records,
        all_passed=bool(records) and all(r['all_11_arrays_exact'] and r['full_cif_exact'] and r['coverage_passed'] for r in records))
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    audit(*sys.argv[1:])
