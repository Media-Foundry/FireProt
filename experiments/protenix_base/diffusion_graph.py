"""Experimental HIP Graph for the deterministic denoiser; sampler stays eager."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import torch


def identity(value):
    if isinstance(value, torch.Tensor):
        return ('tensor', id(value), value.data_ptr(), tuple(value.shape),
                tuple(value.stride()), str(value.dtype), str(value.device), value._version)
    if isinstance(value, dict):
        return ('dict', tuple((k, identity(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(identity(v) for v in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value).__name__, value)
    raise TypeError(f'Unsupported denoiser constant {type(value).__name__}')


@contextmanager
def install(model, output):
    mode = os.environ.get('FIREPROT_BASE_DIFFUSION_GRAPH', 'off')
    if mode not in ('on', 'off'):
        raise ValueError('Expected FIREPROT_BASE_DIFFUSION_GRAPH=off/on')
    if mode == 'off':
        yield
        return
    if (model.training or not torch.version.hip
            or not torch.are_deterministic_algorithms_enabled()
            or not torch.utils.deterministic.fill_uninitialized_memory
            or model.configs.sample_diffusion.N_step != 200
            or model.configs.sample_diffusion.N_sample != 1):
        raise RuntimeError('Denoiser graph requires strict Base HIP D200/S1')
    module = model.diffusion_module
    original = module.forward
    side = torch.cuda.Stream()
    graph = static = result = constants = signature = None
    records = dict(mode=mode, calls=0, captures=0, replays=0, warmup_calls=0,
                   scope='Denoiser only; noise generation and sampler remain eager',
                   capture_wall_seconds=0, error=None)

    def forward(**kwargs):
        nonlocal graph, static, result, constants, signature
        records['calls'] += 1
        x = kwargs['x_noisy']; t = kwargs['t_hat_noise_level']
        if (torch.is_grad_enabled() or not x.is_cuda or t.device != x.device
                or x.dtype != torch.float32 or t.dtype != torch.float32
                or x.requires_grad or t.requires_grad
                or tuple(x.shape) != (1, 7979, 3) or tuple(t.shape) != (1,)
                or kwargs['s_trunk'].shape[-2] != 1000
                or kwargs.get('inplace_safe') is not True):
            raise RuntimeError('Unsupported denoiser graph signature')
        fixed = {k:v for k,v in kwargs.items() if k not in ('x_noisy', 't_hat_noise_level')}
        now = (tuple(x.shape), tuple(x.stride()), tuple(t.shape), tuple(t.stride()), identity(fixed))
        if signature is not None and now != signature:
            raise RuntimeError('Denoiser inputs/constants changed identity, version or signature')
        current = torch.cuda.current_stream()
        if records['calls'] == 1:
            side.wait_stream(current)
            with torch.cuda.stream(side):
                warm = original(**kwargs)
                x.record_stream(side); t.record_stream(side)
            current.wait_stream(side)
            records['warmup_calls'] += 1
            # Preserve constant objects through all replays. Reject mutation by
            # subsequent eager/captured calls instead of replaying stale data.
            constants = fixed
            signature = (tuple(x.shape), tuple(x.stride()), tuple(t.shape), tuple(t.stride()), identity(fixed))
            return warm
        if static is None:
            static = (x.clone(memory_format=torch.preserve_format), t.clone(memory_format=torch.preserve_format))
        else:
            static[0].copy_(x); static[1].copy_(t)
        if graph is None:
            rng = torch.cuda.get_rng_state(); cpu_rng = torch.get_rng_state()
            graph = torch.cuda.CUDAGraph()
            side.wait_stream(current)
            start = time.perf_counter()
            with torch.cuda.graph(graph, stream=side):
                result = original(x_noisy=static[0], t_hat_noise_level=static[1], **constants)
            records['capture_wall_seconds'] = time.perf_counter() - start
            records['captures'] += 1
            if identity(constants) != signature[-1]:
                raise RuntimeError('Denoiser capture mutated constant inputs')
            if not torch.equal(rng, torch.cuda.get_rng_state()) or not torch.equal(cpu_rng, torch.get_rng_state()):
                raise RuntimeError('Unexpected RNG use in denoiser graph')
        graph.replay()
        records['replays'] += 1
        return result

    module.forward = forward
    try:
        yield
        if records['calls'] != 200 or records['captures'] != 1 or records['replays'] != 199:
            raise RuntimeError('Denoiser graph did not execute all 200 steps')
        torch.cuda.synchronize()
        records['peak_memory_allocated_bytes'] = torch.cuda.max_memory_allocated()
        records['peak_memory_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except BaseException as error:
        records['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        module.forward = original
        Path(output, 'base-diffusion-graph.json').write_text(json.dumps(records, indent=2)+'\n')
