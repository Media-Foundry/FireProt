"""Native resident serial versus tensor batching; diagnostic, not accepted E2E.

Retain every per-item native output and report batch/singleton drift separately.
Use distinct deterministic development sequences, not a biological test set.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from fireprot_workers.atlasfold import read_sequence
from fireprot_workers.plm import load_model
from fireprot_workers.transport import save_arrays


def prepare(model, name, sequences):
    if name == 'atlaslm-3b':
        tokens = model.encode(sequences)
    elif name == 'esm2-3b':
        _, _, tokens = model.alphabet.get_batch_converter()(
            [(str(i), sequence) for i, sequence in enumerate(sequences)])
        tokens = tokens.cuda()
    elif name == 'esmc-fold2-6b':
        from esm.models.esmc.tokenizer import EsmcTokenizer
        tokens = EsmcTokenizer()(sequences, padding=True, return_tensors='pt')['input_ids'].cuda()
    else:
        tokenizer = model.tokenizers.sequence
        rows = [torch.tensor(tokenizer.encode(s), device='cuda') for s in sequences]
        tokens = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True,
                                                padding_value=tokenizer.pad_token_id)

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


def cpu_item(arrays, index, length):
    return {key: value[index:index+1, :length+2].detach().cpu().contiguous()
            for key, value in arrays.items()}


def differences(reference, candidate):
    result = {}
    for key, expected in reference.items():
        actual = candidate[key]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        bitwise = torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        delta = (expected.float() - actual.float()).abs()
        result[key] = dict(bitwise_equal=bitwise, max_abs=float(delta.max()),
                           mean_abs=float(delta.mean()), different_values=int((expected != actual).sum()))
    assert result['tokens']['bitwise_equal'], 'Batch item/token association changed'
    return result


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2]); output.mkdir(exist_ok=False)
    assert os.environ.get('HIP_VISIBLE_DEVICES') == '6'
    assert request['backend'] == 'native'
    torch.set_num_threads(8); torch.manual_seed(request['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, source = read_sequence(request['input']['path'])
    assert len(source) == 1000
    name = request['workload']['model_name']
    report = dict(model=name, seed=request['seed'], precision='native', hip_visible_devices='6',
                  torch=torch.__version__, hip=torch.version.hip,
                  boundary='warm resident forward only; excludes tokenization, loading, export and validation',
                  cache_state='unflushed; two warmups per arm per pair',
                  inputs='deterministic rotations/truncations; development data, not independent test set',
                  request_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(), cases=[])
    with torch.inference_mode():
        model, _, original_forward = load_model(request, source)
        original = cpu_item(original_forward(), 0, 1000)
        rebuilt = cpu_item(prepare(model, name, [source])(), 0, 1000)
        report['singleton_wrapper_check'] = differences(original, rebuilt)
        assert all(v['bitwise_equal'] for v in report['singleton_wrapper_check'].values())
        del original, rebuilt, original_forward
        for lengths in ([1000, 1000], [1000]*4, [600, 800, 900, 1000]):
            label = '-'.join(map(str, lengths))
            directory = output / label; directory.mkdir()
            sequences = [(source[(i*137):] + source[:(i*137)])[:length] for i, length in enumerate(lengths)]
            (directory/'inputs.json').write_text(json.dumps(sequences, indent=2)+'\n')
            singles = [prepare(model, name, [s]) for s in sequences]
            batch = prepare(model, name, sequences)
            expected = [cpu_item(fn(), 0, n) for fn, n in zip(singles, lengths)]
            actual = batch()
            drift = []
            for i, (reference, length) in enumerate(zip(expected, lengths)):
                item = cpu_item(actual, i, length)
                drift.append(differences(reference, item))
                save_arrays(directory/f'single-{i}.npz', reference)
                save_arrays(directory/f'batch-{i}.npz', item)
            del actual, expected, item, reference

            def serial():
                # Do not retain all intermediate GPU outputs during serial timing.
                for fn in singles:
                    value = fn()
                    del value

            pairs = []
            for order in [('serial', 'batch'), ('batch', 'serial'), ('serial', 'batch')]:
                pair = {}
                for arm in order:
                    fn = serial if arm == 'serial' else batch
                    for _ in range(2): fn()
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    observations = []
                    for _ in range(3):
                        torch.cuda.synchronize(); start = time.perf_counter()
                        value = fn(); torch.cuda.synchronize()
                        observations.append(time.perf_counter()-start)
                        del value
                    pair[arm] = dict(seconds=observations, peak_allocated_bytes=torch.cuda.max_memory_allocated())
                pairs.append(pair)
            medians = {arm: statistics.median([t for p in pairs for t in p[arm]['seconds']])
                       for arm in ('serial', 'batch')}
            case = dict(lengths=lengths, singleton_drift=drift, pairs=pairs, medians=medians,
                        forward_throughput_speedup=medians['serial']/medians['batch'],
                        batch_sequences_per_second=len(lengths)/medians['batch'])
            report['cases'].append(case)
            (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(dict(model=name, lengths=lengths, medians=medians,
                                  speedup=case['forward_throughput_speedup'], drift=drift)), flush=True)
            del singles, batch


if __name__ == '__main__':
    main()
