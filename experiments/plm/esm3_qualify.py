"""Compare the standalone ESM3 adapter with the pinned official model factory."""
from pathlib import Path
import json,sys,time
from unittest.mock import patch
import torch
from fireprot_workers.plm import load_model
from fireprot_workers.atlasfold import read_sequence
from fireprot_workers.transport import save_arrays
request=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
checkpoint=Path(request['checkpoints']['model']['path']);root=checkpoint.parents[2]
torch.set_num_threads(8);torch.manual_seed(request['seed'])
_,sequence=read_sequence(request['input']['path'])
from esm.models.esm3 import ESM3
import esm.pretrained as pretrained
import esm.utils.constants.esm3 as constants
with patch.object(pretrained,'data_root',lambda _:root),patch.object(constants,'data_root',lambda _:root):
    native=ESM3.from_pretrained('esm3_sm_open_v1',device=torch.device('cuda')).eval()
tokens=torch.tensor([native.tokenizers.sequence.encode(sequence)],dtype=torch.long,device='cuda')
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    result=native(sequence_tokens=tokens)
    reference=dict(tokens=tokens,embeddings=result.embeddings,logits=result.sequence_logits)
    torch.cuda.synchronize()
    save_arrays(out/'official.npz',reference)
    expected={k:v.clone() for k,v in reference.items()}
del reference,result,native
torch.cuda.empty_cache()
model,ids,forward=load_model(request,sequence)
with torch.inference_mode():
    result=forward();torch.cuda.synchronize()
    exact={k:torch.equal(v,result[k]) for k,v in expected.items()}
    finite={k:bool(torch.isfinite(v).all()) for k,v in result.items()}
    descriptors=save_arrays(out/'adapter.npz',result)
report=dict(model='esm3-open',length=len(sequence),source=request['source'],checkpoint=request['checkpoints']['model'],
    seed=request['seed'],exact=exact,finite=finite,arrays=descriptors,
    scope='Official ESM3.from_pretrained factory with local data_root routing versus FireProt sequence-only local loader; all native forward inputs, BF16 autocast and output selection matched. No generation or structure/function codec loading.')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if not all(exact.values()) or not all(finite.values()):raise RuntimeError('ESM3 native factory parity failed')
print(json.dumps(report,indent=2),flush=True)
