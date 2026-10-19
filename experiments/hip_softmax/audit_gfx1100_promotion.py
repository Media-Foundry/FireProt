"""Independently replay published-route length gates, including full CIF bytes."""
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
        path = base / path
    assert pin(path)['sha256'] == item['sha256'], path
    return path


repo, root, destination = map(lambda p: Path(p).resolve(), sys.argv[1:4])
standard = '--standard' in sys.argv[4:]
lengths = [int(v) for v in sys.argv[4:] if v != '--standard']
assert lengths and len(set(lengths)) == len(lengths)
records = []
for length in lengths:
    directory = root / f'paired-{"standard-" if standard else ""}n{length}'
    benchmark = read(directory / 'benchmark.json')
    assert benchmark['phase'] == 'succeeded'
    comparison = read(checked(benchmark['comparison'], directory))
    assert comparison['correctness_passed']
    predictions = []
    runs = []
    for index, arm in enumerate(('baseline', 'candidate')):
        invocation_path = directory / f'measurement-{index:03d}/invocation.json'
        invocation = read(invocation_path)
        assert invocation['exit_code'] == 0
        accepted_path = checked(invocation['accepted'], invocation_path.parent)
        output = accepted_path.parent
        accepted = read(accepted_path)
        for item in accepted['metadata'] + accepted['outputs']:
            checked(item, output)
        bundle = read(output / 'predictions.json')
        assert len(bundle['predictions']) == 1
        sample = bundle['predictions'][0]
        predictions.append((output / sample['archive'], output / sample['structure']))
        run = dict(arm=arm, invocation=pin(invocation_path), accepted=pin(accepted_path),
            e2e_seconds=invocation['elapsed_seconds'], model_seconds=bundle['model_seconds'])
        if arm == 'candidate':
            assert 'gfx1100-composition.json' in {item['path'] for item in accepted['outputs']}
            composition = read(output / 'gfx1100-composition.json')
            operator = read(output / 'operator-runtime.json')
            assert composition['completed'] and operator['kernel']['mode'] == 'published'
            if length in (1000, 1536):
                assert composition['token_count'] == length and composition['reuse_enabled']
                assert set(composition['caches']) == {'pair', 'atom'}
                for cache in composition['caches'].values():
                    assert len(cache['scopes']) == 1
                    scope = cache['scopes'][0]
                    assert scope['completed'] and all(scope[k] == v for k, v in
                        [('calls', 200), ('hits', 199), ('misses', 1), ('fallbacks', 0)])
                assert composition['template']['counts']['reused_slots'] > 0
                assert operator['launches_submitted'] > 0
            else:
                assert not composition['reuse_enabled'] and not composition['caches']
                assert operator['launches_submitted'] == 0
            run.update(composition=composition, operator=operator)
        runs.append(run)
    replay = subprocess.run([str(repo / 'target/release/fireprot'), 'compare-npz',
        str(predictions[0][0]), str(predictions[1][0])], capture_output=True, text=True, timeout=60)
    arrays = json.loads(replay.stdout)
    assert replay.returncode == 0 and len(arrays) == 11
    assert all(v['bitwise_equal'] and v['passed'] for v in arrays.values())
    assert pin(predictions[0][1])['sha256'] == pin(predictions[1][1])['sha256']
    records.append(dict(length=length, all_11_arrays_exact=True, full_cif_exact=True,
        benchmark=pin(directory / 'benchmark.json'), comparison=benchmark['comparison'],
        runs=runs, performance=comparison['performance']))
for item in read(root / 'promotion-map.json')['files']:
    assert pin(repo / item['relative_path'])['sha256'] == item['sha256'], item['relative_path']
result = dict(schema_version=1, all_passed=True, lengths=lengths,
    ready_for_default=set(lengths) == {124, 1000, 1536}, records=records,
    profile=pin(root / ('profile-standard.json' if standard else 'profile-qualified.json')),
    environment_rescan='optional_strict_profile' if standard else 'every_invocation',
    published_screen=pin(root / 'published-screen-v2.json'),
    source_map=pin(root / 'promotion-map.json'), auditor=pin(Path(__file__)),
    boundary='One matched qualification pair per length; no repeated-performance claim. N124 is an intentional native fallback.')
with destination.open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(audit=pin(destination), all_passed=True, lengths=lengths,
                     ready_for_default=result['ready_for_default']), indent=2))
