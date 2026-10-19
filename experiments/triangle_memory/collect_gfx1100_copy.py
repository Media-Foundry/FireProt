"""Publish only the independently audited three-pair gfx1100 copy evidence."""
import json
from pathlib import Path
from statistics import median
import sys

from audit_gfx1100_copy import checked, pin, read


repo, root, destination = (Path(v).resolve() for v in sys.argv[1:])
audit = read(root/'audit-final.json')
assert audit['all_passed'] and len(audit['records']) == 7
benchmark = read(root/'paired/benchmark.json')
assert benchmark['phase'] == 'succeeded'
comparison = read(checked(benchmark['comparison'], root))
assert comparison['correctness_passed'] and len(comparison['pairs']) == 3
for item in benchmark['checks']:
    assert read(checked(item, root))['correctness_passed']
records = [r for r in audit['records'] if '/paired/' in r['invocation']['path']]
assert len(records) == 6
forward = {arm: median(r['model_seconds'] for r in records if r['enabled'] == flag)
           for arm, flag in [('baseline', False), ('candidate', True)]}
forward['speedup'] = forward['baseline']/forward['candidate']
report = dict(schema_version=1, host='Precision', architecture='gfx1100',
    device='W7900, HIP0, PCI 0000:53:00.0',
    workload=dict(model='OpenFold3/OpenBind-0', length=1000, precision='strict deterministic FP32',
                  recycles=3, trunk_passes=4, diffusion_steps=200, samples=1, seed=101,
                  msa='query only', templates='none; internal padded slots retained'),
    preparation=pin(root/'preparation.json'), final_audit=pin(root/'audit-final.json'),
    collector=pin(Path(__file__).resolve()), benchmark=pin(root/'paired/benchmark.json'),
    comparison=benchmark['comparison'], native_reference=audit['reference'],
    operator_screen=pin(root/'screen.json'), operator_results=read(root/'screen.json')['results'],
    all_11_arrays_exact=True, full_cif_exact=True,
    paired_runs=records, diagnostic_cycle=[r for r in audit['records'] if '/paired/' not in r['invocation']['path']],
    end_to_end_performance=comparison['performance'], model_forward=forward,
    implementation='Native arithmetic retained; tiled uint32 HIP transpose for N1000/chunk256 and tail232, explicit native BMM operand packing and pre-normalization layout materialization.',
    limitations=[
        'Incremental comparison against qualified gfx1100 softmax/pair/atom/template reuse, not direct native E2E.',
        'Single N1000 development input; no independent biological accuracy evaluation or N1536 qualification.',
        'Fresh workers, unflushed caches, prebuilt kernels, no concurrent GPU jobs or asset transfers.',
        'Clocks unlocked; hardware telemetry is sampled, not a controlled-frequency study.',
        'Three-sample p95 is the observed maximum, not a confidence bound.',
        'Allocator peaks cover forward only; not hardware/whole-invocation memory peaks.',
        'Frozen experiment; inherited environment inventory and maintained route promotion remain open.'])
for name in ('gpu-cycle.json', 'gpu-after.json'):
    if (root/name).exists():
        report[name] = pin(root/name)
with destination.open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(end_to_end=report['end_to_end_performance'], forward=forward), indent=2))
