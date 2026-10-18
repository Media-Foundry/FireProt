"""Offline evidence replay for the gfx1100 reuse cycles and Rust paired runs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def pin(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(path.read_text())


def checked(item, base):
    path = Path(item['path'])
    if not path.is_absolute():
        path = base / path
    assert pin(path)['sha256'] == item['sha256'], str(path)
    return path


def audit(repo, root, destination):
    repo, root, destination = map(lambda p: Path(p).resolve(), (repo, root, destination))
    preparation = read(root / 'preparation.json')
    for item in preparation['files']:
        checked(item, root)
    base = read(checked(preparation['base'], root))
    for name in ('library', 'bridge', 'cli'):
        checked(base[name], repo)
    reference = repo / 'outputs/experiments/gfx11-openfold3-model-20261017-v1/native-prediction'
    for item in read(reference / 'accepted.json')['metadata'] + read(reference / 'accepted.json')['outputs']:
        checked(item, reference)
    ref_sample = read(reference / 'predictions.json')['predictions'][0]
    paths = sorted(root.glob('measurement-*/invocation.json')) + sorted(root.glob('paired/measurement-*/invocation.json'))
    records = []
    for path in paths:
        invocation = read(path)
        assert invocation['exit_code'] == 0
        profile = read(checked(invocation['profile'], path.parent))
        checked(invocation['request'], path.parent)
        expected_mode = int(profile['environment']['FIREPROT_GFX1100_REUSE_MODE'])
        assert expected_mode in (0, 1, 2, 3)
        accepted = checked(invocation['accepted'], path.parent)
        directory = accepted.parent
        receipt = read(accepted)
        for item in receipt['metadata'] + receipt['outputs']:
            checked(item, directory)
        assert 'reuse-runtime.json' in {item['path'] for item in receipt['outputs']}
        bundle = read(directory / 'predictions.json')
        sample = bundle['predictions'][0]
        replay = subprocess.run([str(repo / 'target/release/fireprot'), 'compare-npz',
            str(reference / ref_sample['archive']), str(directory / sample['archive'])],
            text=True, capture_output=True, timeout=60)
        comparison = json.loads(replay.stdout)
        exact = replay.returncode == 0 and len(comparison) == 11 and all(v['bitwise_equal'] and v['passed'] for v in comparison.values())
        cif = pin(directory / sample['structure'])
        reuse = read(directory / 'reuse-runtime.json')
        operator = read(directory / 'operator-runtime.json')
        expected_caches = set() if expected_mode == 0 else {'pair'} if expected_mode == 1 else {'pair', 'atom'}
        coverage = (reuse['completed'] and operator['launches_submitted'] > 0
                    and reuse['mode'] == expected_mode
                    and set(reuse['caches']) == expected_caches)
        for name, cache in reuse['caches'].items():
            scopes = cache['scopes']
            coverage &= len(scopes) == 1 and scopes[0]['completed'] and all(scopes[0][k] == v for k, v in
                [('calls',200), ('hits',199), ('misses',1), ('fallbacks',0)])
        if reuse['mode'] == 3:
            coverage &= reuse['template']['counts']['reused_slots'] > 0
        records.append(dict(invocation=pin(path), accepted=pin(accepted), run_id=bundle['run_id'],
            mode=reuse['mode'], e2e_seconds=invocation['elapsed_seconds'], model_seconds=bundle['model_seconds'],
            request_seconds=bundle['request_seconds'], all_11_arrays_exact=exact,
            full_cif_exact=cif['sha256'] == pin(reference / ref_sample['structure'])['sha256'],
            npz=pin(directory / sample['archive']), cif=cif, coverage_passed=bool(coverage),
            launches_submitted=operator['launches_submitted'], reuse=reuse,
            reuse_receipt=pin(directory / 'reuse-runtime.json')))
    result = dict(schema_version=1, preparation=pin(root / 'preparation.json'),
        auditor=pin(Path(__file__)), reference=pin(reference / 'accepted.json'), records=records,
        all_passed=bool(records) and all(r['all_11_arrays_exact'] and r['full_cif_exact'] and r['coverage_passed'] for r in records),
        boundary='Single cycle timings are diagnostics; only the complete Rust paired report establishes repeated comparison.')
    for name in ('benchmark', 'comparison'):
        path = root / f'paired/{name}.json'
        if path.exists():
            result[name] = pin(path)
    with destination.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(audit=pin(destination), all_passed=result['all_passed'],
        runs=[{k:r[k] for k in ('mode','e2e_seconds','model_seconds','all_11_arrays_exact','full_cif_exact','coverage_passed')} for r in records]), indent=2))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo', 'root', 'destination'):
        parser.add_argument(name)
    args = parser.parse_args()
    audit(args.repo, args.root, args.destination)
