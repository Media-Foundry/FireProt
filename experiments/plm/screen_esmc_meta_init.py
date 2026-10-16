"""Cold-process ESMC meta-constructor diagnostic; no GPU or checkpoint reads."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    import torch
    from esm.models.esmc.model import EsmcModel
    from esm.models.esmc.config import EsmcConfig
    from esmc_meta_init import skip_meta_initialization
    config_path,flag,output=map(str,sys.argv[1:4]);enabled=flag=='1';assert flag in ('0','1')
    cfg=EsmcConfig.from_dict(json.loads(Path(config_path).read_text())['esmc_config']);cfg.attn_implementation='sdpa'
    torch.set_num_threads(8);torch.manual_seed(101)
    before=torch.get_rng_state().clone()
    with torch.inference_mode():
        start=time.perf_counter()
        with skip_meta_initialization(enabled) as calls,torch.device('meta'):
            model=EsmcModel(cfg)
        seconds=time.perf_counter()-start
    assert not torch.cuda.is_initialized()
    def tensor(t):
        return dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device),requires_grad=t.requires_grad,
            bytes_sha256=None if t.is_meta else hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    structure=dict(modules={k:type(v).__module__+'.'+type(v).__name__ for k,v in model.named_modules()},
        parameters={k:tensor(v) for k,v in model.named_parameters()},buffers={k:tensor(v) for k,v in model.named_buffers()},
        state_keys=sorted(model.state_dict()),rng_unchanged=torch.equal(before,torch.get_rng_state()),
        rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    assert all(p['device']=='meta' for p in structure['parameters'].values())
    report=dict(schema_version=1,enabled=enabled,constructor_seconds=seconds,calls=calls,structure=structure,
        gpu_initialized=False,seed=101,torch=str(torch.__version__),affinity=sorted(os.sched_getaffinity(0)))
    with Path(output).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(enabled,seconds,calls,flush=True)


if __name__=='__main__':main()
