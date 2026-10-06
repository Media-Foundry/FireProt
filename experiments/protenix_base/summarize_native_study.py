"""Summarize completed Rust-measured native pairs under the explicit Base policy.

This does not launch inference or relax compare-runs. Rust authenticates archives,
scientific conditions, numerical deltas and confidence. The historical result is
preserved, while descriptive timings use the user's separate aligned-RMSD rule.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def pinned_read(descriptor):
    if pin(descriptor['path']) != descriptor:
        raise RuntimeError('Control or reference pin changed: ' + descriptor['path'])
    return read(descriptor['path'])


def summarize(study, executable, tag='final-v1'):
    study, executable = Path(study), Path(executable)
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in tag):
        raise ValueError('Unsafe report tag')
    if read(study/'driver-status.json') != {'exit_code': 0}:
        raise RuntimeError('The complete study did not succeed')
    plan = read(study/'study-plan.json')
    if pin(plan['driver']['path']) != plan['driver']:
        raise RuntimeError('Driver changed after preparation')
    expected_order = [(0, 'native'), (0, 'optimized'), (1, 'optimized'),
                      (1, 'native'), (2, 'native'), (2, 'optimized')]
    if [(r['pair'], r['arm']) for r in plan['runs']] != expected_order:
        raise RuntimeError('Unexpected study order')
    journal = [json.loads(line) for line in (study/'order.jsonl').read_text().splitlines()]
    expected_journal = [dict(index=i, pair=p, arm=a, event=e)
                        for i, (p, a) in enumerate(expected_order)
                        for e in ('started', 'verified')]
    if journal != expected_journal:
        raise RuntimeError('Missing or unexpected journal events')
    groups = {arm: [] for arm in ('native', 'optimized')}
    exact_checks = []
    scientific = None
    profiles = {}
    keys = {'coordinates', 'confidence.atom_is_polymer', 'confidence.atom_plddt',
            'confidence.atom_to_token_idx', 'confidence.contact_probs',
            'confidence.token_asym_id', 'confidence.token_has_frame',
            'confidence.token_pair_pae', 'confidence.token_pair_pde'}
    for index, row in enumerate(plan['runs']):
        spec = pinned_read(row['specification'])
        request = pinned_read(spec['request'])
        profile = pinned_read(spec['profile'])
        arm = row['arm']
        if request['hip_device'] != 7 or request['backend'] != ('native' if arm == 'native' else 'fireprot'):
            raise RuntimeError('Unexpected device/backend')
        settings = {k: v for k, v in request.items() if k not in ('run_id', 'output_dir', 'backend')}
        if scientific is None:
            scientific = settings
        if settings != scientific or spec['timeout_seconds'] != 1800:
            raise RuntimeError('Scientific workload or timeout changed')
        flag = profile['environment'].pop('FIREPROT_ENABLE_BASE_N1000_OPTIMIZATIONS')
        if flag != ('0' if arm == 'native' else '1'):
            raise RuntimeError('Unexpected composition flag')
        profiles[arm] = profile
        if pin(row['reference']['path']) != row['reference']:
            raise RuntimeError('Prediction anchor changed')
        inv_path = Path(row['measurement'])/'invocation.json'
        invocation = read(inv_path)
        if invocation['exit_code'] != 0 or invocation['specification'] != spec:
            raise RuntimeError('Failed invocation or mismatched controls')
        elapsed = invocation['elapsed_seconds']
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError('Invalid E2E timing')
        out = Path(row['output'])
        if pin(out/'accepted.json') != invocation['accepted']:
            raise RuntimeError('Acceptance pin differs from measured invocation')
        checks = read(study/f'exact-{index:03d}.json')
        if set(checks) != keys or not all(v['bitwise_equal'] and v['passed'] for v in checks.values()):
            raise RuntimeError('An arm did not reproduce its qualified prediction anchor')
        exact_checks.append(dict(index=index, report=pin(study/f'exact-{index:03d}.json'), arrays=checks))
        predictions = read(out/'predictions.json')
        construction = read(out/'construction-runtime.json')
        if (not construction['enabled'] or not construction['installed']
                or construction['unavailable'] is not None
                or construction['counts']['selected'] != 1
                or construction['counts']['recovered'] != 0
                or construction['counts']['errors'] != 0):
            raise RuntimeError('Shared fast initialization was not actually selected')
        if arm == 'optimized':
            context = read(out/'base-n1000-optimizations.json')
            if context['error'] is not None or context['implementation'] != 'amd-gfx90a-base-n1000-cached-v1':
                raise RuntimeError('Incorrect maintained context')
        groups[arm].append(dict(pair=row['pair'], run_id=request['run_id'],
            e2e_seconds=elapsed, model_seconds=predictions['model_seconds'],
            invocation=pin(inv_path), accepted=pin(out/'accepted.json')))
    if profiles['native'] != profiles['optimized']:
        raise RuntimeError('Undeclared profile differences')
    for rows in groups.values():
        rows.sort(key=lambda row: row['pair'])
    spec = dict(schema_version=1,
        treatment_description='Native Torch versus maintained Base N1000 composition; shared fast initialization',
        expected_changes={
            'backend': dict(baseline='native', candidate='fireprot'),
            'launch.environment.FIREPROT_ENABLE_BASE_N1000_OPTIMIZATIONS': dict(baseline='0', candidate='1'),
            'invocation.profile_sha256': dict(baseline=pin(study/'native-profile.json')['sha256'],
                                            candidate=pin(study/'optimized-profile.json')['sha256'])})
    for side, arm in [('baseline', 'native'), ('candidate', 'optimized')]:
        spec[side] = dict(schema_version=1, scope='end_to_end',
            cache_state='fresh processes; unflushed filesystem; shared fast initialization; three AB/BA/AB pairs',
            runs=[r['accepted'] for r in groups[arm]], invocations=[r['invocation'] for r in groups[arm]])
    spec_path, comparison_path = study/f'{tag}-compare.json', study/f'{tag}-legacy-comparison.json'
    if spec_path.exists() or comparison_path.exists():
        raise RuntimeError('Use a fresh report tag; previous evidence is never overwritten')
    spec_path.write_text(json.dumps(spec, indent=2)+'\n')
    process = subprocess.run([str(executable), 'compare-runs', str(spec_path)], capture_output=True, text=True)
    comparison_path.write_text(process.stdout)
    (study/f'{tag}-legacy-stderr.log').write_text(process.stderr)
    comparison = json.loads(process.stdout)  # A condition/authentication failure is not a numerical report.
    if process.returncode not in (0, 1) or len(comparison['pairs']) != 3:
        raise RuntimeError('Rust archive/condition comparison failed')
    aligned = []
    for pair_index, pair in enumerate(comparison['pairs']):
        if not pair['cif_atom_identities_checked'] or len(pair['samples']) != 1:
            raise RuntimeError('Missing atom identities or sample coverage')
        sample = pair['samples'][0]
        confidence = sample['confidence']
        if (confidence['numeric_leaf_count'] <= 0 or not math.isfinite(confidence['max_abs_delta'])
                or confidence['max_abs_delta'] > 1e-3 or confidence['missing_from_candidate']
                or confidence['extra_in_candidate'] or not all(sample['exact_arrays'].values())):
            raise RuntimeError('Unchanged confidence or metadata criterion failed')
        drift = read(study/f'aligned-{pair_index}.json')
        if (not drift['coordinate_gate_passed'] or not drift['atom_identities_exact']
                or not drift['npz_cif_binding_checked']
                or abs(drift['all_atom']['aligned_rmsd_angstrom']-
                       sample['coordinates']['aligned_rmsd_angstrom']) > 1e-12):
            raise RuntimeError('Aligned coordinate checks disagree')
        aligned.append(dict(pair=pair_index, report=pin(study/f'aligned-{pair_index}.json'),
            all_atom=drift['all_atom'], ca=drift['ca'], confidence=confidence,
            floating_confidence_diagnostics=sample['floating_confidence_diagnostics']))
    medians = {arm: statistics.median(r['e2e_seconds'] for r in rows) for arm, rows in groups.items()}
    report = dict(scope=plan['scope'], order=plan['order'], scientific=scientific,
        policy='Separate user-authorized aligned RMSD <=0.1 A; unchanged summary-confidence <=1e-3; prediction regression, not experimental accuracy',
        runs=groups, e2e_medians=medians, median_speedup=medians['native']/medians['optimized'],
        paired_speedups=[a['e2e_seconds']/b['e2e_seconds'] for a, b in zip(groups['native'], groups['optimized'])],
        model_medians={arm: statistics.median(r['model_seconds'] for r in rows) for arm, rows in groups.items()},
        exact_within_arm=exact_checks, pair_numerics=aligned, user_policy_passed=True,
        legacy_exit_code=process.returncode, legacy_comparison=comparison,
        study_plan=pin(study/'study-plan.json'), driver_status=pin(study/'driver-status.json'))
    destination = study/f'{tag}-summary.json'
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(path=str(destination), e2e_medians=medians, median_speedup=report['median_speedup'])))


if __name__ == '__main__':
    summarize(*sys.argv[1:])
