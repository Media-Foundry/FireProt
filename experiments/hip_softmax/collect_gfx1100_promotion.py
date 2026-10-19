"""Collect independently audited deployment evidence; never infer a gate from timing."""
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def pin(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest())


repo, root, destination = map(lambda value: Path(value).resolve(), sys.argv[1:4])
waived = sys.argv[4:] == ['--user-waived-long-reruns']
assert not sys.argv[4:] or waived
audit_name = 'audit-strict-n124.json' if waived else 'audit-standard-all.json'
audit = read(root / audit_name)
assert audit['all_passed']
if not waived:
    assert audit['ready_for_default'] and set(audit['lengths']) == {124, 1000, 1536}
strict = read(root / 'audit-strict-n124.json')
assert strict['all_passed'] and strict['lengths'] == [124]
screen = read(root / 'published-screen-v2.json')
assert screen['status'] == 'passed' and len(screen['records']) == 54
assert all(item['exact'] for item in screen['records'])
assert sum(item['launches'] for item in screen['records']) == 42
assert screen['exception_restoration'] and screen['wrong_manifest_rejected']
tests = read(root / 'tests-final.json')
assert tests['returncode'] == 0
assert pin(Path(tests['log']['path']))['sha256'] == tests['log']['sha256']
standard_profile = read(root / 'profile-standard.json')
strict_profile = read(root / 'profile-qualified.json')
assert read(repo / 'configs/precision/openfold3-default.json') == standard_profile
assert read(repo / 'configs/precision/openfold3-strict.json') == strict_profile
without_inventory = dict(strict_profile)
inventory = without_inventory.pop('environment_inventory')
assert without_inventory == standard_profile
assert pin(Path(inventory['path']))['sha256'] == inventory['sha256']
for item in read(root / 'promotion-map.json')['files']:
    assert pin(repo / item['relative_path'])['sha256'] == item['sha256']
for length in (124, 1000, 1536):
    request = read(repo / f'configs/requests/precision-openfold3-n{length}-default.json')
    qualified = read(root / f'request-standard-n{length}-candidate.json')
    for key in ('workload', 'seed', 'hip_device', 'source', 'checkpoints'):
        assert request[key] == qualified[key], (length, key)
    assert request['backend'] == ('native' if waived and length == 1536 else qualified['backend'])
    assert request['input']['sha256'] == qualified['input']['sha256']
    assert pin(Path(request['input']['path']))['sha256'] == request['input']['sha256']

summaries = []
for record in audit['records']:
    baseline, candidate = record['runs']
    composition = candidate['composition']
    summaries.append(dict(length=record['length'],
        all_11_arrays_exact=record['all_11_arrays_exact'], full_cif_exact=record['full_cif_exact'],
        baseline_e2e_seconds=baseline['e2e_seconds'], candidate_e2e_seconds=candidate['e2e_seconds'],
        baseline_forward_seconds=baseline['model_seconds'], candidate_forward_seconds=candidate['model_seconds'],
        observed_single_pair_ratio=baseline['e2e_seconds'] / candidate['e2e_seconds'],
        softmax_launches=candidate['operator']['launches_submitted'],
        reuse_enabled=composition['reuse_enabled'], caches=composition['caches'],
        template=composition.get('template'),
        peak_allocated_bytes=composition['peak_allocated_bytes'],
        peak_reserved_bytes=composition['peak_reserved_bytes'],
        benchmark=record['benchmark'], comparison=record['comparison'],
        accepted=[run['accepted'] for run in record['runs']],
        performance=record['performance']))

result = dict(schema_version=1, status='promoted_precision_default', date='2026-10-19',
    deployment_decision='user_authorized_without_additional_long_model_reruns' if waived else 'all_three_packaged_length_gates_passed',
    all_three_packaged_length_gates_passed=not waived,
    user_override='算了不跑了，直接推广即可' if waived else None,
    n1536_default_backend='native' if waived else 'hip_fused',
    canceled_study=pin(root / 'paired-standard-n1536/benchmark.json') if waived else None,
    previous_n1000_qualification=pin(repo / 'provenance/precision_openfold3_reuse_gfx1100_20261018.json'),
    scope='Published out-of-place gfx1100 wave32 softmax plus rollout-local pair/atom and exact duplicate-template reuse. Copy/private-score follow-ups are NOT included.',
    host='Precision', gpu=dict(target='gfx1100', pci='0000:53:00.0', hip_device=0),
    scientific_contract='OpenBind-0 strict deterministic FP32 with deterministic fill; recycles3/four trunk passes, diffusion200, sample1, seed101, query-only/no templates. No Triton.',
    user_choice='Ordinary verification is default; the extra full Python/ROCm/system-content inventory rescan is a separate strict profile.',
    qualification=summaries,
    strict_n124=dict(audit=pin(root / 'audit-strict-n124.json'),
        e2e_seconds=[run['e2e_seconds'] for run in strict['records'][0]['runs']]),
    publication=read(root / 'publication.json'),
    operator_checks=dict(count=54, hip_launches=42, native_fallbacks=12,
        exception_restoration=True, wrong_manifest_rejected=True),
    tests=tests,
    profiles={name: pin(repo / f'configs/precision/openfold3-{name}.json') for name in ('default', 'strict', 'native')},
    requests=[pin(repo / f'configs/requests/precision-openfold3-n{length}-default.json') for length in (124, 1000, 1536)],
    evidence={name: pin(root / name) for name in (
        audit_name, 'audit-strict-n124.json', 'published-screen-v2.json',
        'publication.json', 'promotion-map.json', 'environment-discovery.json',
        'environment-spec-v3.json', 'environment-exclusions-v3.json', 'environment-v3.json',
        'standard-controls.json', 'qualification-controls.json', 'build/build.json', 'tests-final.json')},
    environment_inventory=inventory,
    limitations=[
        'Under the user override, only strict N124 has a completed packaged full-model pair. Standard N1536 was canceled during baseline; standard N1000/N124 did not run. Promotion is a user-authorized deployment decision, not a three-length validation pass.' if waived else 'Each length has one matched qualification pair, not a repeated-performance estimate; retain Rust insufficient_repeats.',
        'N124 is a real 1izq 124-residue input with intentional native computation; no short-sequence speedup claim.',
        'N1000 has earlier frozen-composition three-pair evidence, not a new maintained-package full-model run. N1536 optimized execution remains unqualified and its default request selects native.' if waived else 'N1000/N1536 and the pinned software/workload are qualified, not arbitrary lengths, workloads, platforms or versions.',
        'Other lengths/workloads retain native feature computations and attention falls back outside the screened shape signatures.',
        'Caches rely on pinned upstream immutable-input contracts and are scoped to one rollout; no arbitrary mutation support.',
        'The strict inventory covers recorded current dependencies, not a hermetic OS. It partitions around an existing dangling Qt link and checks absence of /usr/share/qtchooser; no system files were changed.',
        'Failed initial inventory attempts and the initial wrong-pin test harness exception remain recorded; final v3 inventory and v2 screen passed.',
        'Caches unflushed and clocks unlocked. Forward allocator peaks are not hardware or complete-invocation memory peaks.',
        'No general asset migration gate was released or modified.'
    ], collector=pin(Path(__file__)))
with destination.open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(provenance=pin(destination), status=result['status']), indent=2))
