"""Standalone pLM tensor execution. Rust owns pins, supervision and acceptance."""
import json
import os
from pathlib import Path
import time

ESM_COMMIT = '43b4548b86762edfa747b07d5f440aad3c33acee'
ATLAS_COMMIT = '8ab3aca0e18c8b814d5ca6756b2617a07d72c68d'


def load_model(request, sequence):
    """Load local, verified assets only; return native tokens and a forward closure."""
    import torch
    name = request['workload']['model_name']
    source = Path(request['source']['root']).resolve()
    checkpoint = Path(request['checkpoints']['model']['path']).resolve()
    if name == 'atlaslm-3b':
        import atlaslm
        from atlaslm.pretrained import load_model as load_atlas
        if request['source']['commit'] != ATLAS_COMMIT or Path(atlaslm.__file__).resolve().parents[2] != source:
            raise ValueError('AtlasLM imported source differs from its pin')
        model = load_atlas(checkpoint, device='cuda').eval()
        tokens = model.encode([sequence])
        def forward():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(tokens, return_logits=True)
            return dict(tokens=tokens, embeddings=out.embeddings, logits=out.sequence_logits)
    elif name == 'esm2-3b':
        import esm
        from esm.pretrained import load_model_and_alphabet_core
        # fair-esm and Biohub esm have the same import name. Bind every Python
        # file of the installed fair-esm package to independently checked resources.
        root = Path(esm.__file__).resolve().parent
        declared = {Path(v['path']).resolve() for k, v in request['resources'].items() if k.startswith('esm_source_')}
        if declared != {p.resolve() for p in root.rglob('*.py')}:
            raise ValueError('ESM2 requires the complete pinned fair-esm source file set')
        regression = request['resources']['contact_regression']['path']
        # Published fair-esm weights contain an argparse configuration object.
        data = torch.load(checkpoint, map_location='cpu', weights_only=False)
        regress = torch.load(regression, map_location='cpu', weights_only=False)
        if request['backend'] == 'fireprot':
            from esm.pretrained import _load_model_and_alphabet_core_v2
            data['model'].update(regress['model'])
            with torch.device('meta'):
                model, alphabet, state = _load_model_and_alphabet_core_v2(data)
            model.load_state_dict(state, strict=True, assign=True)
            if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
                raise RuntimeError('ESM2 checkpoint does not fully cover meta construction')
        else:
            model, alphabet = load_model_and_alphabet_core('esm2_t36_3B_UR50D', data, regress)
        # Published ESM2 stores FP16 weights; its native constructor loads them
        # into FP32 parameters. Preserve that promotion after meta assignment.
        model = model.float().eval().cuda()
        if model.num_layers != 36 or model.embed_dim != 2560:
            raise ValueError('Expected ESM2 3B architecture')
        _, _, tokens = alphabet.get_batch_converter()([('query', sequence)])
        tokens = tokens.cuda()
        def forward():
            out = model(tokens, repr_layers=[36], return_contacts=False)
            return dict(tokens=tokens, embeddings=out['representations'][36], logits=out['logits'])
    else:
        import esm
        if request['source']['commit'] != ESM_COMMIT or Path(esm.__file__).resolve().parent.parent != source:
            raise ValueError('Biohub ESM imported source differs from its pin')
        if name == 'esmc-fold2-6b':
            from safetensors import safe_open
            from esm.models.esmc.model import EsmcModel
            from esm.models.esmc.config import EsmcConfig
            from esm.models.esmc.checkpoint_layout import published_to_native
            from esm.models.esmc.tokenizer import EsmcTokenizer
            from .esmfold2 import model_directory
            asset_request = dict(request, resources={k:v for k,v in request['resources'].items() if k != 'hip_rope'})
            root = model_directory(asset_request)
            cfg = EsmcConfig.from_dict(json.loads((root / 'config.json').read_text())['esmc_config'])
            if (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads) != (2560, 80, 40):
                raise ValueError('Expected the bundled ESMC 6B architecture')
            cfg.attn_implementation = 'sdpa'
            with torch.device('meta'):
                model = EsmcModel(cfg)
            mapping = json.loads(checkpoint.read_text())['weight_map']
            raw = {}
            for shard in sorted({v for k, v in mapping.items() if k.startswith('esmc.')}):
                with safe_open(root / shard, framework='pt', device='cpu') as handle:
                    for key in handle.keys():
                        if key.startswith('esmc.'):
                            raw[key[5:]] = handle.get_tensor(key)
            model.load_state_dict(published_to_native(raw), strict=True, assign=True)
            del raw
            model = model.to(device='cuda', dtype=torch.bfloat16).eval()
            tokens = EsmcTokenizer()(sequence, return_tensors='pt')['input_ids'].cuda()
            def forward():
                out = model(tokens, output_hidden_states=False, output_attentions=False, compute_sae=False)
                return dict(tokens=tokens, embeddings=out.last_hidden_state)
        elif name == 'esm3-open':
            from types import SimpleNamespace
            from esm.models.esm3 import ESM3
            from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
            tokenizer = EsmSequenceTokenizer()
            def unavailable(*args, **kwargs):
                raise ValueError('Standalone ESM3 entry supports sequence forward only, not structure/function codecs')
            # Native forward only reads the sequence tokenizer. No auxiliary
            # decoder or annotation vocabulary download is needed in this scope.
            from accelerate import init_empty_weights
            # Keep native nonpersistent RoPE buffers on CPU, as upstream does.
            with init_empty_weights():
                model = ESM3(d_model=1536, n_heads=24, v_heads=256, n_layers=48,
                    structure_encoder_fn=unavailable, structure_decoder_fn=unavailable,
                    function_decoder_fn=unavailable, tokenizers=SimpleNamespace(sequence=tokenizer))
            weights = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
            model.load_state_dict(weights, strict=True, assign=True)
            del weights
            model = model.to(device='cuda', dtype=torch.bfloat16).eval()
            tokens = torch.tensor([tokenizer.encode(sequence)], dtype=torch.int64, device='cuda')
            def forward():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(sequence_tokens=tokens)
                return dict(tokens=tokens, embeddings=out.embeddings, logits=out.sequence_logits)
        else:
            raise ValueError('Unsupported standalone pLM')
    model.requires_grad_(False)
    if tuple(tokens.shape) != (1, len(sequence) + 2):
        raise ValueError('Tokenizer changed the sequence length')
    return model, tokens, forward


def predict(request, channel):
    import torch
    from .atlasfold import read_sequence, snapshot_input
    from .transport import save_arrays
    output = Path(os.environ['FIREPROT_OUTPUT_DIR'])
    retained = snapshot_input(request['input']['path'], output)
    _, sequence = read_sequence(retained)
    if not 1 <= len(sequence) <= 1000:
        raise ValueError('Standalone pLM entry currently supports one sequence of 1..1000 residues')
    torch.set_num_threads(8)
    torch.manual_seed(request['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    model, tokens, forward = load_model(request, sequence)
    from contextlib import nullcontext
    scope = nullcontext(None)
    if request['backend'] == 'fireprot':
        from fireprot.adapters.plm.runtime import optimized_scope
        scope = optimized_scope(model, request, len(sequence))
    channel.emit(dict(event='progress', stage='language_model'))
    torch.cuda.synchronize()
    begin = time.perf_counter()
    with torch.inference_mode(), scope as runtime:
        arrays = forward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    if runtime is not None:
        (output / 'operator-runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    descriptors = save_arrays(output / 'sample-0.npz', arrays)
    return dict(schema_version=5, input_snapshot=retained.name,
        run_id=request['run_id'], seed=request['seed'], workload=request['workload'],
        predictions=[dict(sample=0, archive='sample-0.npz', arrays=descriptors, confidence=None)],
        model_seconds=seconds, request_seconds=time.perf_counter() - started)
