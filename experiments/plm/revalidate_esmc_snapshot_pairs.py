"""Revalidate measured incremental pairs with the final comparison-only guard."""
import json
from pathlib import Path
import subprocess
import sys

from collect_esmc_pointwise_pairs import checked, pin


def main():
    work=Path(sys.argv[1]).resolve()
    final=json.loads((work/'final-cli.json').read_text());assert pin(Path(final['path']))==final
    reports=[]
    for count in (4,64):
        case=work/f'b{count}-pairs'
        measured=json.loads((case/'measurements/benchmark.json').read_text());assert measured['phase']=='succeeded'
        settings=checked(measured['comparison_settings'])
        expected=checked(measured['comparison'])
        spec=dict(schema_version=1,treatment_description=settings['treatment_description'],
                  expected_changes=settings['expected_changes'],monitor_treatment=True)
        for arm in ('baseline','candidate'):
            jobs=[m for m in measured['measurements'] if m['slot']['arm']==arm]
            jobs.sort(key=lambda m:m['slot']['pair'])
            spec[arm]=dict(schema_version=1,scope='end_to_end',cache_state=settings['cache_state'],
                           runs=[m['accepted'] for m in jobs],invocations=[m['invocation'] for m in jobs])
        path=case/'final-policy-spec.json'
        with path.open('x') as f:json.dump(spec,f,indent=2);f.write('\n')
        result=case/'final-policy-report.json'
        with result.open('x') as f:
            subprocess.run([final['path'],'compare-runs',str(path)],stdout=f,check=True)
        actual=json.loads(result.read_text())
        assert actual==expected, 'Final comparison policy changed the accepted report'
        reports.append(dict(count=count,spec=pin(path),report=pin(result),unchanged_report=True))
    with (work/'final-policy.json').open('x') as f:json.dump(dict(passed=True,cli=final,reports=reports),f,indent=2);f.write('\n')
    print('Final policy revalidates both incremental reports without differences',flush=True)


if __name__=='__main__':main()
