"""Inclusive event/wall intervals for trained packed contractions; diagnostic only."""
from contextlib import contextmanager
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
import torch


@contextmanager
def measure(output, setup_ms):
    from fireprot.backends.amd import lt_bmm
    original=lt_bmm.packed_bmm
    records=defaultdict(list)
    def call(a,b,outgoing):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter();start.record()
        result=original(a,b,outgoing)
        end.record()
        records['outgoing' if outgoing else 'incoming'].append((start,end,(time.perf_counter()-wall)*1000))
        return result
    lt_bmm.packed_bmm=call
    try:
        yield
    finally:
        lt_bmm.packed_bmm=original
        torch.cuda.synchronize()
        rows={}
        for direction,values in records.items():
            gpu=[a.elapsed_time(b) for a,b,_ in values];cpu=[v for _,_,v in values]
            rows[direction]=dict(count=len(values),inclusive_event_total_ms=sum(gpu),first_event_ms=gpu[0],median_event_ms=statistics.median(gpu),enqueue_wall_total_ms=sum(cpu),first_enqueue_ms=cpu[0],event_ms=gpu,enqueue_ms=cpu)
        Path(output,'base-packed-profile.json').write_text(json.dumps(dict(scope='Instrumented trained contractions; events include launch gaps; not E2E speedup evidence',adapter_setup_ms=setup_ms,rows=rows),indent=2)+'\n')
