"""Bind composed E2E evidence, historical output anchors and plan startup probes."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_archive(output):
    accepted = json.loads((output/'accepted.json').read_text())
    for item in accepted['metadata'] + accepted['outputs']:
        file = output/item['path']
        assert file.resolve().is_relative_to(output.resolve())
        assert pin(file)['sha256'] == item['sha256']
    prediction = json.loads((output/'predictions.json').read_text())
    values = []
    for item in prediction['predictions']:
        with zipfile.ZipFile(output/item['archive']) as archive:
            values.append({name: archive.read(name) for name in archive.namelist()})
    return values


work = Path(sys.argv[1]).resolve()
previous = work.parent
collected = json.loads((work/'collected.json').read_text())
anchors = {
    4: previous/'esmc-startup-20261016/native/baseline-1',
    64: previous/'esmc-gemm-order-20261016/lt-validation/esmc-fixed-64-pairs/baseline-1',
}
initialization = {}
for count, reference in anchors.items():
    label = f'esmc-fixed-{count}'
    expected = read_archive(reference)
    current = work/f'{label}-pairs/baseline-1'
    assert read_archive(current) == expected
    values = []
    for index in range(1, 4):
        runtime = json.loads((work/f'{label}-pairs/candidate-{index}/operator-runtime.json').read_text())
        for group, item in enumerate(runtime):
            gemm = item['fixed_gemm']
            assert gemm['plan_created'] == (group == 0)
            if group == 0:
                values.append(gemm['initialization_host_seconds'])
            else:
                assert gemm['initialization_host_seconds'] == 0
    initialization[label] = dict(seconds=values, median=statistics.median(values))
probe = work/'startup-probe-v2'
reports = {}
for mode in ('index-1', 'restore-1', 'index-2', 'restore-2'):
    report = json.loads((probe/mode/'report.json').read_text())
    if mode.startswith('index'):
        assert report['status'] == 'accepted'
        assert report['all_80_projections_exact'] and all(report['output_exact'].values())
        assert report['warm_restore']['all_80_projections_exact'] and report['warm_restore']['output_exact']
        assert pin(probe/mode/'algorithm.bin')['sha256'] == report['algorithm_sha256']
    else:
        assert report['status'] == 'rejected' and report['return_code'] == -9
    assert report['hip_device'] == 6 and report['batch'] == 4
    reports[mode] = report
assert len({r['algorithm_sha256'] for r in reports.values()}) == 1
manifest = json.loads((work/'deployment-manifest.json').read_text())
for name, digest in manifest['files'].items():
    assert pin(work/'qualified-fixed'/name)['sha256'] == digest
files = [work/'collected.json', work/'deployment-manifest.json', work/'prepare-measured.py',
         work/'esmc_lt_scope.py', work/'collect_esmc_fixed_pairs.py', Path(__file__),
         probe/'esmc_lt_serialized.hip', probe/'probe_esmc_lt_startup.py',
         probe/'run_esmc_lt_startup_probe.py', probe/'build-spec.json', probe/'build-receipt.json']
files += [probe/mode/'report.json' for mode in reports]
files += [work/'startup-probe/restore-1.log', work/'startup-probe/index-1/report.json']
lazy = work/'lazy'
lazy_result = json.loads((lazy/'collected.json').read_text())
assert read_archive(lazy/'esmc-fixed-4-pairs/baseline-1') == read_archive(anchors[4])
lazy_times = []
for index in range(1, 4):
    runtime, = json.loads((lazy/f'esmc-fixed-4-pairs/candidate-{index}/operator-runtime.json').read_text())
    assert runtime['fixed_gemm']['plan_created']
    lazy_times.append(runtime['fixed_gemm']['initialization_host_seconds'])
initialization['lazy-b4'] = dict(seconds=lazy_times, median=statistics.median(lazy_times))
lazy_manifest = json.loads((lazy/'deployment-manifest.json').read_text())
for name, digest in lazy_manifest['files'].items():
    assert pin(lazy/'qualified-fixed'/name)['sha256'] == digest
files += [lazy/name for name in ('collected.json', 'deployment-manifest.json',
          'prepare-measured.py', 'esmc_lt_lazy_scope.py', 'collect_esmc_lazy_pairs.py')]
result = dict(schema_version=1, date='2026-10-16', baseline_commit='7a70cf5', experiment=str(work),
    status='Experimental composition and initialization diagnostics; no default promotion.',
    e2e=collected, lazy_e2e=lazy_result, initialization=initialization, probes=reports,
    output_anchors={str(n): pin(p/'accepted.json') for n, p in anchors.items()},
    anchor_scope={'4': 'official same-batch native output', '64': 'previous exact FireProt resident64 output'},
    files={str(p.relative_to(work)): pin(p)['sha256'] for p in files},
    limitations=['Incremental E2E only; do not multiply by the published direct-native ratio.',
                 'Startup probes are separate fresh-process diagnostics, not E2E comparisons.',
                 'Native precision and bitwise output acceptance remain unchanged.'])
with (work/'evidence.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps(dict(initialization=initialization,
    performance={k: v['performance'] for k, v in collected['models'].items()}), indent=2))
