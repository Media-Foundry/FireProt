"""Independent archive and same-grouping reference checks for ESMC B4/B16 E2E."""
import json
from pathlib import Path
import statistics
import sys
import zipfile

from collect_esmc_pointwise_pairs import archive,checked,pin


def main():
    work=Path(sys.argv[1]);report=json.loads((work/'report.json').read_text());assert report['passed'] and len(report['pairs'])==3
    checked(report['input']);assert pin(Path(report['cli']['path']))==report['cli']
    reference=work.parent/'esmc-large-batch-20261016/screen'
    baseline=checked(pin(reference/'report.json'));assert baseline['passed'] and baseline['assets_rehashed_after']
    qualify=checked(report['qualification']);assert qualify['passed'] and qualify['assets_rehashed_after']
    assert pin(Path(report['library']['path']))==report['library']
    rows=[];provenance=None;reference_pins=[]
    for index,pair in enumerate(report['pairs']):
        assert pair['order']==([16,4] if index==1 else [4,16])
        for n in pair['order']:
            row=pair[str(n)];invocation=checked(row['invocation']);assert invocation['exit_code']==0 and invocation['accepted']==row['accepted']
            assert row['e2e_seconds']==invocation['elapsed_seconds']
            out=Path(row['accepted']['path']).parent;prediction,arrays=archive(out);assert len(arrays)==16
            request=json.loads((out/'request.json').read_text());assert request['input']==report['input']
            assert request['seed']==101 and request['hip_device']==6 and request['backend']=='fireprot'
            assert request['workload']==dict(model='plm',model_name='esmc-fold2-6b',precision='native',
                batch=dict(items=16,max_batch_size=n,max_padded_tokens=n*1002,grouping='length_sorted'))
            assert prediction['batch_groups']==[list(range(i,i+n)) for i in range(0,16,n)]
            assert [p['input_id'] for p in prediction['predictions']]==[p['id'] for p in baseline['inputs']]
            for i,actual in enumerate(arrays):
                path=reference/f'native-b{n}-item-{i}.npz'
                with zipfile.ZipFile(path) as f:assert actual=={k:f.read(k) for k in f.namelist()}
                if index==0:reference_pins.append(pin(path))
            actual_provenance=json.loads((out/'provenance.json').read_text())
            if provenance is None:provenance=actual_provenance
            assert actual_provenance==provenance
            launch=json.loads((out/'execution.json').read_text())
            assert launch['environment']['FIREPROT_ESMC_FIXED_GEMM']==('1' if n==4 else '0')
            assert launch['environment']['FIREPROT_AMD_SMI_PROTOCOL']=='snapshot-v1'
            supervisor=json.loads((out/'supervisor.json').read_text());assert supervisor['phase']=='succeeded' and supervisor['device_verified']
            assert supervisor['physical_device']['pci_address']=='0000:8e:00.0' and supervisor['isolation_samples']>=2
            assert row['model_seconds']==prediction['model_seconds']
            rows.append(dict(pair=index,batch=n,invocation=row['invocation'],accepted=row['accepted'],
                e2e_seconds=invocation['elapsed_seconds'],model_seconds=prediction['model_seconds'],all_arrays_exact=True))
    medians={str(n):statistics.median(r['e2e_seconds'] for r in rows if r['batch']==n) for n in (4,16)}
    assert medians==report['medians']['e2e_seconds']
    result=dict(schema_version=1,passed=True,observations=rows,reference_archives=reference_pins,
        measured_report=pin(work/'report.json'),script=pin(Path(__file__)),medians=medians,
        paired_speedups=[p['speedup'] for p in report['pairs']],median_speedup=medians['4']/medians['16'],
        decision='Do not promote B16: complete invocation improvement is negligible and one pair regresses.',
        preserved_direct_native_b4_speedup=1.9783613518957768)
    with (work/'collected.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,medians=medians,paired_speedups=result['paired_speedups']),indent=2))


if __name__=='__main__':main()
