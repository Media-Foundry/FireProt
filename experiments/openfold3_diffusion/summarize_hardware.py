"""Validate and summarize the separate OpenFold3 trace and counter diagnostics."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys


def pin(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def verify(record):
    path = Path(record['path'])
    if pin(path)['sha256'] != record['sha256']:
        raise ValueError(f'Artifact changed: {path}')
    return path


def summarize(root, destination):
    root = Path(root)
    receipts = {}
    for mode in ('trace', 'counters'):
        receipt = read(root / mode / 'collected.json')
        assert receipt['correctness_passed'] and receipt['all11_exact'] and receipt['full_cif_exact']
        assert len(receipt['arrays']) == 11
        assert all(v['passed'] and v['bitwise_equal'] for v in receipt['arrays'].values())
        plan = read(verify(receipt['plan']))
        for ref in receipt['raw'] + [receipt['npz'], receipt['cif']] + plan['profiler_inputs']:
            verify(ref)
        receipts[mode] = receipt
    trace_path = next((root / 'trace/profiler').rglob('*kernel_trace.csv'))
    stats_path = next((root / 'trace/profiler').rglob('*kernel_stats.csv'))
    counter_path = next((root / 'counters/profiler').rglob('*counter_collection.csv'))
    agents_path = next((root / 'counters/profiler').rglob('*agent_info.csv'))
    agents = list(csv.DictReader(agents_path.open()))
    agent = next(x for x in agents if x['Node_Id'] == '9')
    assert agent['Name'] == 'gfx90a' and int(agent['Location_Id']) == 0x9300
    assert int(agent['Cu_Count']) == 104
    cu_count = int(agent['Cu_Count'])
    intervals = []
    groups = collections.defaultdict(list)
    resources = {}
    for row in csv.DictReader(trace_path.open()):
        assert row['Agent_Id'] == 'Agent 9'
        start, end = int(row['Start_Timestamp']), int(row['End_Timestamp'])
        assert end >= start
        intervals.append((start, end))
        grid = int(row['Grid_Size_X']) * int(row['Grid_Size_Y']) * int(row['Grid_Size_Z'])
        key = row['Kernel_Name'], grid
        groups[key].append(end - start)
        resources[key] = {k: row[k] for k in (
            'LDS_Block_Size', 'VGPR_Count', 'Accum_VGPR_Count', 'SGPR_Count',
            'Workgroup_Size_X', 'Workgroup_Size_Y', 'Workgroup_Size_Z')}
    intervals.sort()
    start, end = intervals[0]
    union = 0
    for a, z in intervals[1:]:
        if a > end:
            union += end - start
            start, end = a, z
        else:
            end = max(end, z)
    union += end - start
    span = max(z for _, z in intervals) - intervals[0][0]
    stats = list(csv.DictReader(stats_path.open()))
    total = sum(sum(ds) for ds in groups.values())
    assert total == sum(int(x['TotalDurationNs']) for x in stats)
    assert len(intervals) == sum(int(x['Calls']) for x in stats)
    dispatches = {}
    metrics = set(read(root / 'counters/plan.json')['metrics'])
    for row in csv.DictReader(counter_path.open()):
        assert row['Agent_Id'] == 'Agent 9'
        d = dispatches.setdefault(row['Dispatch_Id'], dict(
            kernel=row['Kernel_Name'], grid=int(row['Grid_Size']), metrics={},
            duration_ns=int(row['End_Timestamp']) - int(row['Start_Timestamp'])))
        assert row['Counter_Name'] not in d['metrics']
        d['metrics'][row['Counter_Name']] = float(row['Counter_Value'])
    sampled = collections.defaultdict(list)
    for d in dispatches.values():
        assert set(d['metrics']) == metrics and d['duration_ns'] > 0
        m = d['metrics']
        d['derived'] = dict(
            cu_busy_percent=100 * m['SQ_BUSY_CU_CYCLES'] / m['GRBM_COUNT'] / cu_count,
            hbm_GB_s=(m['FetchSize'] + m['WriteSize']) * 1024 / d['duration_ns'])
        sampled[d['kernel'], d['grid']].append(d)
    records = []
    for key, ds in groups.items():
        samples = sampled.get(key, [])
        record = dict(kernel=key[0], grid=key[1], calls=len(ds), seconds=sum(ds)/1e9,
                      median_ms=statistics.median(ds)/1e6, max_ms=max(ds)/1e6,
                      resources=resources[key], counter_samples=len(samples))
        if samples:
            record['sampled_metrics'] = {k: statistics.median(d['metrics'][k] for d in samples) for k in metrics}
            record['sampled_derived'] = {k: statistics.median(d['derived'][k] for d in samples) for k in samples[0]['derived']}
        records.append(record)
    records.sort(key=lambda r: r['seconds'], reverse=True)
    report = dict(
        scope='Strict FP32 N1536 R3/four trunk passes D200/S1 seed101; HIP7 PCI93/SMI5; separate instrumented diagnostics.',
        correctness_passed=True, all11_exact=True, full_cif_exact=True,
        qualifications={m: pin(root / m / 'collected.json') for m in receipts},
        trace=pin(trace_path), stats=pin(stats_path), counters=pin(counter_path), agent=agent,
        metric_definitions=pin('/opt/rocm/share/rocprofiler-sdk/config.yaml'),
        dispatches=len(intervals), kernel_seconds=total/1e9, kernel_union_seconds=union/1e9,
        first_to_last_kernel_seconds=span/1e9, kernel_busy_fraction=union/span,
        counter_dispatches=len(dispatches), counter_groups=len(sampled),
        counter_groups_absent_from_trace=[dict(kernel=k[0], grid=k[1]) for k in sampled if k not in groups],
        trace_seconds_with_matching_counter_group=sum(r['seconds'] for r in records if r['counter_samples']),
        groups=records,
        limits=['Instrumented durations are not E2E measurements; all trace observations, including large stalls, are retained.',
                'Counter occurrence filtering is by kernel name, not by tensor shape. Matching name/grid does not prove identical tensor strides or phase.',
                'Sample coverage does not justify model-wide average utilization or a single model-wide compute/memory-bound label.',
                'CU busy follows SDK SIMD_UTILIZATION times100 with104 CUs and no extra factor4; occupancy is a different metric.',
                'FetchSize and WriteSize are KiB; bandwidth uses decimal GB/s. No new hardware peak calibration is included.'])
    with Path(destination).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return {k: v for k, v in report.items() if k not in ('groups', 'agent')}


if __name__ == '__main__':
    print(json.dumps(summarize(*sys.argv[1:]), indent=2))
