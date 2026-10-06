"""Isolated Base graph composition, retaining scoped Lt and dual storage.

Not a maintained entry or a qualified optimization. Run in a frozen worker with
Rust supervision. The entire model call uses one side stream in both arms so
MSA, trunk and confidence contractions share ordered workspace ownership.
"""
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import time


class ReplayContract:
    def __init__(self, enabled, group_size):
        if type(enabled) is not bool or group_size not in (4, 12, 48):
            raise ValueError('Expected boolean mode and 4/12/48 blocks per graph')
        self.enabled = enabled
        self.group_size = group_size
        self.guards = []
        self.capturing = False
        self.completed = False
        self.captures = 0
        self.replays = 0

    @property
    def python_contractions(self):
        # 88 outside the trunk; 96 in each of 10 scientific trunk passes.
        # A captured trunk has only one eager pass and one Python capture pass.
        return 88 + 96 * (2 if self.enabled else 10)

    @property
    def logical_contractions(self):
        return 1048

    @contextmanager
    def capture(self):
        if not self.enabled or self.capturing or self.completed:
            raise RuntimeError('Unexpected or nested capture')
        self.capturing = True
        try:
            yield
            self.captures += 1
        finally:
            self.capturing = False

    def before_replay(self):
        if not self.enabled or self.capturing or self.completed or not self.captures:
            raise RuntimeError('Replay outside captured contract')
        for guard in self.guards:
            guard()

    def finish(self, calls):
        groups = 48 // self.group_size
        expected = (groups, 9 * groups) if self.enabled else (0, 0)
        if calls != 10 or self.capturing or (self.captures, self.replays) != expected:
            raise RuntimeError('Incomplete scientific cycles or graph coverage')
        self.completed = True

    def validate_completed(self):
        if not self.completed:
            raise RuntimeError('Graph composition did not complete')


@contextmanager
def _trunk(model, output, contract, side):
    import torch
    from diffusion_graph import identity

    stack = model.pairformer_stack
    if len(stack.blocks) != 48 or stack.training:
        raise RuntimeError('Expected evaluation-mode 48-block Base trunk')
    original = stack.forward
    stamp = identity(dict(stack.named_parameters()))
    buffers = dict(stack.named_buffers())
    buffer_stamp = identity(buffers)
    partitions = []
    signature = None
    pool = torch.cuda.graph_pool_handle() if contract.enabled else None
    record = dict(calls=0, capture_seconds=0.0, first_replay_submit_seconds=0.0,
                  group_size=contract.group_size, enabled=contract.enabled, error=None)

    def forward(s, z, **kwargs):
        nonlocal signature
        record['calls'] += 1
        if (torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.cuda.current_stream().cuda_stream != side.cuda_stream
                or any(t.dtype != torch.float32 or not t.is_cuda or t.requires_grad
                       or t.is_conj() or t.is_neg() for t in (s, z))
                or s.device != z.device or tuple(s.shape) != (1000, 384)
                or tuple(z.shape) != (1000, 1000, 128)
                or kwargs.get('pair_mask') is not None
                or kwargs.get('inplace_safe') is not True
                or kwargs.get('triangle_attention') != 'fireprot'
                or kwargs.get('triangle_multiplicative') != 'fireprot'):
            raise RuntimeError('Unsupported trunk graph inputs or stream')
        now = (tuple(s.stride()), tuple(z.stride()), identity(kwargs))
        if signature is None:
            signature = now
        if now != signature or identity(dict(stack.named_parameters())) != stamp:
            raise RuntimeError('Trunk signature or parameters changed')
        if identity(dict(stack.named_buffers())) != buffer_stamp:
            raise RuntimeError('Trunk buffers changed')
        if not contract.enabled or record['calls'] == 1:
            return original(s, z, **kwargs)
        if record['calls'] == 2:
            prepared = stack._prep_blocks(**kwargs)
            if len(prepared) != 48:
                raise RuntimeError('Unexpected prepared block count')
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            for begin in range(0, 48, contract.group_size):
                static = (s.clone(memory_format=torch.preserve_format),
                          z.clone(memory_format=torch.preserve_format))
                graph = torch.cuda.CUDAGraph()
                start = time.perf_counter()
                with contract.capture():
                    with torch.cuda.graph(graph, pool=pool, stream=side):
                        result = static
                        for block in prepared[begin:begin + contract.group_size]:
                            result = block(*result)
                record['capture_seconds'] += time.perf_counter() - start
                contract.before_replay()
                start = time.perf_counter()
                graph.replay()
                record['first_replay_submit_seconds'] += time.perf_counter() - start
                contract.replays += 1
                partitions.append((static, graph, result))
                s, z = result
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(gpu_rng, torch.cuda.get_rng_state()):
                raise RuntimeError('Trunk capture unexpectedly used RNG')
            return s, z
        for static, graph, result in partitions:
            static[0].copy_(s)
            static[1].copy_(z)
            contract.before_replay()
            graph.replay()
            contract.replays += 1
            s, z = result
        return s, z

    stack.forward = forward
    try:
        yield
        contract.finish(record['calls'])
        torch.cuda.synchronize()
        record.update(captures=contract.captures, replays=contract.replays,
                      python_contractions=contract.python_contractions,
                      logical_contractions=contract.logical_contractions,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        stack.forward = original
        Path(output, 'base-composed-trunk-graph.json').write_text(json.dumps(record, indent=2)+'\n')


@contextmanager
def install(model, output, request, data):
    import torch
    from fireprot.adapters.protenix.base_n1000 import validate_contract
    from fireprot.adapters.protenix.base_n1000 import global_bias, local_bias, transition
    import graph_triangle_lt
    import graph_dual_storage

    atoms = validate_contract(request, data)
    mode = os.environ.get('FIREPROT_BASE_COMPOSED_GRAPH', 'off')
    if mode not in ('off', 'on'):
        raise ValueError('Expected composed graph off/on')
    group_size = int(os.environ.get('FIREPROT_BASE_GRAPH_GROUP_SIZE', '12'))
    contract = ReplayContract(mode == 'on', group_size)
    if (model.training or not torch.version.hip
            or not torch.are_deterministic_algorithms_enabled()
            or not torch.utils.deterministic.fill_uninitialized_memory):
        raise RuntimeError('Strict HIP inference required')
    current = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(current)
    try:
        with torch.cuda.stream(side), ExitStack() as contexts:
            contexts.enter_context(graph_triangle_lt.install(model, output, contract))
            contexts.enter_context(graph_dual_storage.install(model, output, contract))
            contexts.enter_context(global_bias.install(model, output))
            contexts.enter_context(local_bias.install(model, output))
            contexts.enter_context(transition.install(model, output, atoms))
            contexts.enter_context(_trunk(model, output, contract, side))
            yield
    finally:
        # Context teardown synchronizes successful runs before releasing graphs.
        # The worker also synchronizes before exposing prediction arrays.
        current.wait_stream(side)
