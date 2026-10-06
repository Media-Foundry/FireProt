"""Experimental HIP Graph replay of the Base trunk's complete Pairformer stack.

Use only through an isolated, fingerprinted worker with Rust supervision. The
first scientific cycle warms the side stream; the second captures and replays;
remaining cycles replay with fresh static-buffer input copies. No extra model
cycle is evaluated and random sampling outside the trunk remains eager.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import torch


@contextmanager
def install(model, output):
    mode = os.environ.get('FIREPROT_BASE_RECYCLE_GRAPH', 'off')
    if mode not in ('off', 'on'):
        raise ValueError('Expected FIREPROT_BASE_RECYCLE_GRAPH=off/on')
    if mode == 'off':
        yield
        return
    if (model.training or not torch.version.hip
            or not torch.are_deterministic_algorithms_enabled()
            or not torch.utils.deterministic.fill_uninitialized_memory
            or model.configs.model.N_cycle != 10):
        raise RuntimeError('Graph experiment requires strict Base C10 HIP inference')
    stack = model.pairformer_stack
    original = stack.forward
    side = torch.cuda.Stream()
    graph = None
    static = None
    result = None
    contract = None
    records = dict(mode=mode, scope='Base trunk Pairformer only', calls=0,
                   warmup_calls=0, captures=0, replays=0, capture_wall_seconds=0,
                   error=None, output_alias_policy='Valid until next replay; consumed before next cycle')

    def forward(s, z, **kwargs):
        nonlocal graph, static, result, contract
        records['calls'] += 1
        if (torch.is_grad_enabled() or s.dtype != torch.float32 or z.dtype != torch.float32
                or tuple(z.shape) != (1000, 1000, 128)
                or not s.is_cuda or s.device != z.device
                or s.requires_grad or z.requires_grad
                or kwargs.get('pair_mask') is not None
                or kwargs.get('inplace_safe') is not True
                or kwargs.get('triangle_attention') != 'fireprot'
                or kwargs.get('triangle_multiplicative') != 'fireprot'
                or any(isinstance(v, torch.Tensor) for v in kwargs.values())):
            raise RuntimeError('Unsupported Pairformer graph signature')
        signature = (tuple(s.shape), tuple(s.stride()), tuple(z.shape), tuple(z.stride()), kwargs)
        if contract is None:
            contract = signature
        elif signature != contract:
            raise RuntimeError('Pairformer graph signature changed')
        current = torch.cuda.current_stream()
        if records['calls'] == 1:
            side.wait_stream(current)
            with torch.cuda.stream(side):
                warm = original(s, z, **kwargs)
                s.record_stream(side)
                z.record_stream(side)
            current.wait_stream(side)
            records['warmup_calls'] += 1
            return warm
        if static is None:
            static = (s.clone(memory_format=torch.preserve_format),
                      z.clone(memory_format=torch.preserve_format))
        else:
            static[0].copy_(s)
            static[1].copy_(z)
        if graph is None:
            # No random operation is expected in an eval-mode Pairformer stack.
            rng = torch.cuda.get_rng_state()
            cpu_rng = torch.get_rng_state()
            graph = torch.cuda.CUDAGraph()
            side.wait_stream(current)
            start = time.perf_counter()
            with torch.cuda.graph(graph, stream=side):
                result = original(*static, **kwargs)
            records['capture_wall_seconds'] = time.perf_counter() - start
            records['captures'] += 1
            if not torch.equal(rng, torch.cuda.get_rng_state()) or not torch.equal(cpu_rng, torch.get_rng_state()):
                raise RuntimeError('Unexpected RNG use during Pairformer capture')
        graph.replay()
        records['replays'] += 1
        return result

    stack.forward = forward
    try:
        yield
        if records['calls'] != 10 or records['captures'] != 1 or records['replays'] != 9:
            raise RuntimeError('Unexpected scientific cycle count or graph selection')
        torch.cuda.synchronize()
        records['peak_memory_allocated_bytes'] = torch.cuda.max_memory_allocated()
        records['peak_memory_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except BaseException as error:
        records['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        stack.forward = original
        Path(output, 'base-recycle-graph.json').write_text(json.dumps(records, indent=2)+'\n')
