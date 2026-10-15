"""Tensor boundary for Rust-planned, model-resident sequence batches."""
import json
import os
from pathlib import Path
import time


def prepare_forward(model, name, sequences):
    """Use native tokenization, padding, masks, precision and output heads."""
    import torch
    if name == 'atlaslm-3b':
        tokens = model.encode(sequences)
    elif name == 'esm2-3b':
        _, _, tokens = model.alphabet.get_batch_converter()(
            [(str(i), sequence) for i, sequence in enumerate(sequences)])
        tokens = tokens.cuda()
    elif name == 'esmc-fold2-6b':
        from esm.models.esmc.tokenizer import EsmcTokenizer
        tokens = EsmcTokenizer()(sequences, padding=True, return_tensors='pt')['input_ids'].cuda()
    elif name == 'esm3-open':
        tokenizer = model.tokenizers.sequence
        rows = [torch.tensor(tokenizer.encode(s), device='cuda') for s in sequences]
        tokens = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True,
                                                padding_value=tokenizer.pad_token_id)
    else:
        raise ValueError('Unsupported pLM batch model')
    if tuple(tokens.shape) != (len(sequences), max(map(len, sequences)) + 2):
        raise ValueError('Native batch tokenizer changed the expected shape')

    def forward():
        if name == 'esm2-3b':
            out = model(tokens, repr_layers=[36], return_contacts=False)
            return dict(tokens=tokens, embeddings=out['representations'][36], logits=out['logits'])
        if name == 'esmc-fold2-6b':
            out = model(tokens, output_hidden_states=False, output_attentions=False, compute_sae=False)
            return dict(tokens=tokens, embeddings=out.last_hidden_state)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(tokens, return_logits=True) if name == 'atlaslm-3b' else model(sequence_tokens=tokens)
        return dict(tokens=tokens, embeddings=out.embeddings, logits=out.sequence_logits)
    return forward


def predict(request, channel):
    import torch
    from fireprot.ops._bridge import bridge
    from .atlasfold import snapshot_input
    from .plm import load_model
    from .transport import save_arrays
    from contextlib import nullcontext
    if request['backend'] not in ('native', 'fireprot'):
        raise ValueError('Unsupported pLM batch backend')
    output = Path(os.environ['FIREPROT_OUTPUT_DIR'])
    retained = snapshot_input(request['input']['path'], output)
    plan = json.loads(bridge.plan_plm_batch(retained.read_text(), json.dumps(request['workload']['batch'])))
    torch.set_num_threads(8); torch.manual_seed(request['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    predictions = [None] * len(plan['sequences'])
    runtimes = []
    if request['backend'] == 'fireprot':
        if any(len(item['sequence']) != 1000 for item in plan['sequences']) or any(len(g) not in (1,2,4) for g in plan['groups']):
            raise ValueError('Optimized pLM batches require N1000 with group sizes 1, 2 or 4')
    elapsed = 0.0
    with torch.inference_mode():
        model, tokens, first_forward = load_model(request, plan['sequences'][0]['sequence'])
        del tokens, first_forward
        for group_number, indices in enumerate(plan['groups']):
            channel.emit(dict(event='progress', stage=f'language_model_batch_{group_number}'))
            sequences = [plan['sequences'][i]['sequence'] for i in indices]
            forward = prepare_forward(model, request['workload']['model_name'], sequences)
            scope = nullcontext(None)
            if request['backend'] == 'fireprot':
                from fireprot.adapters.plm.runtime import optimized_scope
                scope = optimized_scope(model, request, 1000, batch_size=len(indices))
            torch.cuda.synchronize(); begin = time.perf_counter()
            with scope as runtime:
                arrays = forward()
            torch.cuda.synchronize(); elapsed += time.perf_counter() - begin
            if runtime is not None:
                runtimes.append(dict(group=group_number, **runtime))
            for row, index in enumerate(indices):
                item = plan['sequences'][index]
                # Export only real tokens, retaining native BOS/EOS and BF16 bits.
                trimmed = {k: v[row:row+1, :len(item['sequence'])+2].contiguous() for k, v in arrays.items()}
                archive = f'sample-{index}.npz'
                descriptors = save_arrays(output / archive, trimmed)
                predictions[index] = dict(sample=index, input_id=item['id'], archive=archive,
                                          arrays=descriptors, confidence=None)
                del trimmed
            del arrays, forward
    if runtimes:
        (output/'operator-runtime.json').write_text(json.dumps(runtimes, indent=2)+'\n')
    return dict(schema_version=6, input_snapshot=retained.name,
                run_id=request['run_id'], seed=request['seed'], workload=request['workload'],
                batch_groups=plan['groups'], predictions=predictions,
                model_seconds=elapsed, request_seconds=time.perf_counter()-started)
