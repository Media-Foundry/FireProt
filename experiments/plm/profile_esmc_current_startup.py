"""Instrument the current ESMC B4 route; diagnostic only, never paired evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from collect_esmc_pointwise_pairs import archive, pin


def replace(path, needle, value):
    source=path.read_text();assert source.count(needle)==1,(path,needle)
    path.write_text(source.replace(needle,value))


def main():
    root=Path('/media/IntelSSD/fireprot');work=Path(sys.argv[1]).resolve();work.mkdir(exist_ok=False)
    profile=json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-exact.json').read_text())
    original=Path(profile['workers']).parent;deployment=work/'diagnostic-deployment'
    shutil.copytree(original,deployment)
    main=deployment/'python/fireprot_workers/__main__.py'
    transport=deployment/'python/fireprot_workers/transport.py'
    replace(transport,'import socket\n','import socket\nimport time\n_TIMING_MARKS = {"module_start":time.perf_counter()}\n')
    replace(transport,'        torch.cuda.init()', '        _TIMING_MARKS["hip_init_start"]=time.perf_counter()\n        torch.cuda.init()\n        _TIMING_MARKS["hip_init_end"]=time.perf_counter()')
    replace(transport,"        self.emit(dict(event='ready'", "        _TIMING_MARKS['ready_emit']=time.perf_counter()\n        self.emit(dict(event='ready'")
    needle="            raise RuntimeError('Missing or invalid supervisor start command')"
    replace(transport,needle,needle+'\n        _TIMING_MARKS["start_received"]=time.perf_counter()')
    replace(main,'import traceback\n','import traceback\nimport time\n')
    replace(main,'from .transport import Channel, announce_outputs','from .transport import Channel, announce_outputs, _TIMING_MARKS')
    replace(main,'        import torch\n','        _TIMING_MARKS["torch_import_start"]=time.perf_counter()\n        import torch\n        _TIMING_MARKS["torch_import_end"]=time.perf_counter()\n')
    replace(main,'            def prepare_cpu():\n','            def prepare_cpu():\n                _TIMING_MARKS["upstream_import_start"]=time.perf_counter()\n')
    replace(main,'                from esm.models.esmc.model import EsmcModel','                from esm.models.esmc.model import EsmcModel\n                _TIMING_MARKS["upstream_import_end"]=time.perf_counter()')
    replace(main,'        bundle = predict(request, channel)', '''        from . import plm_loading, plm, plm_batch
        def timed(module, name, label):
            original=getattr(module,name)
            def call(*args, **kwargs):
                before=time.perf_counter()
                result=original(*args, **kwargs)
                after=time.perf_counter()
                _TIMING_MARKS.setdefault(label,[]).append([before,after])
                return result
            setattr(module,name,call)
        timed(plm_loading,'streamed_esmc_state','placement')
        timed(plm,'load_model','load_model')
        timed(plm_batch,'prepare_forward','tokenization')
        from . import transport
        timed(transport,'save_arrays','exports')
        _TIMING_MARKS['predict_start']=time.perf_counter()
        bundle = predict(request, channel)
        _TIMING_MARKS['predict_end']=time.perf_counter()''')
    replace(main,'        announce_outputs(channel, bundle)', '        _TIMING_MARKS["announce_start"]=time.perf_counter()\n        announce_outputs(channel, bundle)\n        _TIMING_MARKS["announce_end"]=time.perf_counter()')
    replace(main,'        channel.socket.close()', '        channel.socket.close()\n        _TIMING_MARKS["main_done"]=time.perf_counter()\n        Path(os.environ["FIREPROT_OUTPUT_DIR"]).joinpath("startup-timing.json").write_text(json.dumps(_TIMING_MARKS,indent=2)+"\\n")')
    for p in (main,transport):compile(p.read_text(),str(p),'exec')
    request=json.loads((root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json').read_text())
    request.update(run_id='esmc-current-startup-diagnostic',output_dir=str(work/'prediction'))
    profile['workers']=str(deployment/'python')
    profile['environment']['FIREPROT_BRIDGE_LIBRARY']=str(deployment/'target/release/lib_fireprot_bridge.so')
    for name,value in (('request',request),('profile',profile)):
        (work/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    cli=root/'target/release/fireprot';cli_pin=pin(cli)
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for name in ('ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):env.pop(name,None)
    before=time.perf_counter()
    with (work/'invocation.log').open('x') as log:
        subprocess.run([str(cli),'esm',str(work/'request.json'),str(work/'profile.json')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    after=time.perf_counter();assert pin(cli)==cli_pin
    out=work/'prediction';prediction,actual=archive(out)
    reference=root/'outputs/experiments/esmc-load-20261016/validation/native-pairs/baseline-1'
    _,expected=archive(reference);assert actual==expected
    supervisor=json.loads((out/'supervisor.json').read_text());assert supervisor['phase']=='succeeded'
    marks=json.loads((out/'startup-timing.json').read_text())
    phases={label:marks[end]-marks[start] for label,start,end in (
        ('torch_import','torch_import_start','torch_import_end'),('hip_init','hip_init_start','hip_init_end'),
        ('ready_to_start','ready_emit','start_received'),('upstream_import','upstream_import_start','upstream_import_end'),
        ('predict','predict_start','predict_end'),('announce','announce_start','announce_end'))}
    phases.update({label:sum(b-a for a,b in marks[label]) for label in ('placement','load_model','tokenization','exports')})
    report=dict(schema_version=1,boundary='Instrumented current-route diagnostic, not a paired E2E result; overlapping/nested phases must not be summed.',
        invocation_seconds=after-before,supervisor_seconds=supervisor['elapsed_seconds'],model_seconds=prediction['model_seconds'],
        phases=phases,marks=marks,all_arrays_exact=True,cli=cli_pin,script=pin(Path(__file__)),
        files=[pin(p) for p in (main,transport,work/'request.json',work/'profile.json',out/'startup-timing.json',out/'accepted.json')])
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(phases=phases,model_seconds=prediction['model_seconds'],invocation_seconds=after-before),indent=2))


if __name__=='__main__':main()
