"""Summarize completed gfx1100 reuse evidence without running model jobs."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def pin(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(path.read_text())


def compact(record):
    fields = ('invocation', 'accepted', 'run_id', 'mode', 'e2e_seconds', 'model_seconds',
              'all_11_arrays_exact', 'full_cif_exact', 'coverage_passed', 'launches_submitted',
              'npz', 'cif', 'reuse_receipt')
    result = {key: record[key] for key in fields}
    reuse = record['reuse']
    result['caches'] = {name: item['scopes'] for name, item in reuse['caches'].items()}
    result['template_counts'] = reuse.get('template', {}).get('counts')
    result['forward_allocator_peak_bytes'] = {name: reuse[name]
        for name in ('peak_allocated_bytes', 'peak_reserved_bytes')}
    return result


def collect(repo, root, destination):
    repo, root, destination = map(lambda p: Path(p).resolve(), (repo, root, destination))
    audit = read(root / 'audit-final.json')
    assert audit['all_passed'] and len(audit['records']) == 9
    assert read(root / 'template-guard/screen.json')['passed']
    assert '50 passed' in (root / 'tests.log').read_text()
    assert audit['auditor'] == pin(repo / 'experiments/hip_softmax/audit_gfx1100_reuse.py')
    for name in ('benchmark', 'comparison'):
        assert audit[name] == pin(root / f'paired/{name}.json')
    benchmark = read(root / 'paired/benchmark.json')
    comparison = read(root / 'paired/comparison.json')
    assert benchmark['phase'] == 'succeeded' and comparison['correctness_passed']
    assert benchmark['comparison'] == pin(root / 'paired/comparison.json')
    cycles = [r for r in audit['records'] if '-cycle-' in r['run_id']]
    paired = [r for r in audit['records'] if '-paired-' in r['run_id']]
    assert len(cycles) == 3 and {r['mode'] for r in cycles} == {1, 2, 3}
    assert len(paired) == 6 and [r['mode'] for r in paired].count(0) == 3
    assert [r['mode'] for r in paired].count(3) == 3
    model = {arm: [r['model_seconds'] for r in paired if r['mode'] == mode]
             for arm, mode in [('baseline', 0), ('candidate', 3)]}
    forward = {arm: dict(seconds=values, median=median(values)) for arm, values in model.items()}
    forward['median_speedup'] = median(model['baseline']) / median(model['candidate'])
    previous_path = repo / 'provenance/precision_openfold3_model_gfx1100_20261017.json'
    previous = read(previous_path)
    result = {key: previous[key] for key in ('host', 'architecture', 'hip_device', 'pci_address',
        'workload', 'seed', 'source', 'checkpoints', 'input', 'runtime', 'precision')}
    result.update(schema_version=1, status='three_cycles_and_three_pairs_passed',
        date_utc='2026-10-18', base_repository_revision='f4288f4cd7cc061f198a983e0ea6a90bfa381136',
        summary_builder=pin(Path(__file__)), previous_softmax_evidence=pin(previous_path),
        boundary='Incremental mode 0 softmax-only versus mode 3 composed reuse, not direct native speedup',
        experiment_root=str(root), preparation=pin(root / 'preparation.json'),
        paired_preparation=pin(root / 'paired-preparation.json'), specification=pin(root / 'pairs.json'),
        benchmark=pin(root / 'paired/benchmark.json'), comparison=pin(root / 'paired/comparison.json'),
        final_audit=pin(root / 'audit-final.json'), native_reference=audit['reference'],
        template_guard=pin(root / 'template-guard/screen.json'),
        unit_tests=dict(passed=50, log=pin(root / 'tests.log'),
            scope='21 conditioning tests, 12 composition tests, 17 maintained worker/scope tests'),
        diagnostic_cycles=[compact(r) for r in cycles],
        paired_runs=[compact(r) for r in paired],
        end_to_end_performance=comparison['performance'], model_forward=forward,
        implementation=dict(
            pair='Rollout-local immutable pair conditioning; native first miss, 199 hits; recompute time-dependent single branch',
            atom='Rollout-local static cl/plm; native first miss, 199 hits; recompute noisy-coordinate projection',
            template='Frozen N1000 guard specialization; compare exact slot/mask bits, retain first native calculation and native reduction',
            maintained_worker_changed=False, precision_or_work_reduced=False),
        telemetry=dict(cycle3=pin(root / 'gpu-cycle3.json'),
            memory_scope='Torch allocator peaks reset immediately before forward; not whole-invocation/hardware peak'),
        limitations=[
            'Only one N1000 query-only/no-template input, FP32/R3/D200/S1/seed101 on this gfx1100 device.',
            'Pair/atom identity keys require source-pinned immutable rollout inputs and cannot detect arbitrary in-place mutation.',
            'Internal padded template slots were identical; this does not establish gains for distinct real templates.',
            'Single cycle timings are diagnostics, not independently repeated ablation results.',
            'Only the composed reuse-versus-softmax comparison has three alternating pairs; do not multiply historical speedups.',
            'Caches unflushed, clocks unlocked; the retained cycle-3 AMD SMI snapshot reports THROTTLED.',
            'Three-sample p95 is the observed maximum; descriptive trends are not statistical significance.',
            'Complete inherited-environment content inventory and Rust publication of the raw HIP library remain open.',
            'Frozen experiment only; no maintained route promotion or N1536 full-model claim.',
            'General asset migration gate and waiting jobs remain unchanged.'])
    with destination.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(provenance=pin(destination), end_to_end=comparison['performance'],
                          forward=forward), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo', 'root', 'destination'):
        parser.add_argument(name)
    args = parser.parse_args()
    collect(args.repo, args.root, args.destination)
