"""Uninstrumented tensor worker for the isolated Rust throughput experiment.

This does not issue a maintained inference acceptance receipt. Its complete
prediction arrays and summaries are checked independently by the Rust harness.
"""
import json
import os
from pathlib import Path
import platform
import time


def main():
    if os.environ.get('HIP_VISIBLE_DEVICES') != '7':
        raise RuntimeError('This experiment is restricted to HIP7')
    request = json.loads(Path(os.environ['FIREPROT_REQUEST']).read_text())
    output = Path(os.environ['FIREPROT_OUTPUT_DIR'])
    output.mkdir(exist_ok=False)
    import torch
    torch.cuda.init()
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx90a':
        raise RuntimeError('Expected one visible gfx90a device')

    class Progress:
        def emit(self, message):
            with (output/'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time=time.time(), payload=message))+'\n')

    from fireprot_workers.protenix import predict
    result = predict(request, Progress())
    result['environment'] = dict(torch=str(torch.__version__), hip=str(torch.version.hip),
                                 python=platform.python_version())
    (output/'predictions.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'memory.json').write_text(json.dumps(dict(
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved()))+'\n')


if __name__ == '__main__':
    main()
