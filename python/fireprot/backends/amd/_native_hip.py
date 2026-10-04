"""Environment descriptors and ctypes boundary for Rust-owned direct HIP builds."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from fireprot._paths import REPOSITORY_ROOT
import shutil


def load(package, source, root, *, link_flags=(), libraries=(), extra_files=(), versions=None):
    import torch
    from fireprot.ops._bridge import bridge

    root = Path(root).absolute()
    source = Path(source).absolute()
    guard = Path(os.environ.get('FIREPROT_BUILD_GUARD',
        str(REPOSITORY_ROOT/'target/release/fireprot-compiler-guard'))).absolute()
    compiler = root/'bin/hipcc'
    tools = [compiler, root/'llvm/bin/clang++', root/'llvm/bin/ld.lld', guard,
             root/'lib/libamdhip64.so', Path('/bin/sh'), Path('/bin/bash')]
    libraries = [Path(p).absolute() for p in libraries]
    paths = [source, Path(__file__).absolute(), REPOSITORY_ROOT/'python/fireprot/_paths.py', *tools, *libraries,
             *(Path(p).absolute() for p in extra_files)]
    inspector = shutil.which('ldd')
    if inspector is None:
        raise RuntimeError('ldd is required for the native compiler receipt')
    roots = [root/'include', root/'llvm/lib/clang', Path('/usr/include'),
             Path('/usr/local/include'), Path('/usr/lib/gcc')]
    environment = {key: os.environ.get(key) for key in (
        'PATH','HOME','TMPDIR','CXX','CC','CFLAGS','CXXFLAGS','LDFLAGS','CPATH',
        'CPLUS_INCLUDE_PATH','LIBRARY_PATH','LD_LIBRARY_PATH','ROCM_HOME','HIP_PATH',
        'HIP_CLANG_PATH','HIPCC_COMPILE_FLAGS_APPEND','HIPCC_LINK_FLAGS_APPEND')}
    environment.update(LC_ALL='C', HIP_VISIBLE_DEVICES='')
    spec = dict(build=dict(schema_version=1, package=package, target='gfx90a',
        files=[dict(name=str(p),path=str(p)) for p in paths],
        trees=[dict(name=f'dependency-{i}',path=str(p),suffixes=[]) for i,p in enumerate(roots)],
        dynamic_inputs=dict(inspector=inspector,objects=[str(p) for p in [*tools,*libraries]]),
        arguments=dict(compile=['-O3','-std=c++17','-fPIC','-shared','--offload-arch=gfx90a'],
                       include=[str(root/'include')], link=list(link_flags)),
        environment=environment,
        versions=dict(recipe='hip-cabi-1',abi='c',torch=str(torch.__version__),
                      hip=str(torch.version.hip),**(versions or {}))),
        compiler=str(compiler),guard=str(guard),sources=[str(source)],
        timeout_ms=600_000,lock_timeout_ms=600_000)
    cache=Path(os.environ.get('FIREPROT_EXTENSION_CACHE',
        str(REPOSITORY_ROOT/'runtime/torch_extensions')))/'rust-v1'
    result=json.loads(bridge.build_hip(json.dumps(spec),cache))
    artifacts=result['receipt']['artifacts']
    if len(artifacts)!=1:
        raise RuntimeError('Expected one native library artifact')
    name=next(iter(artifacts));library=Path(result['directory'])/name
    result.update(library=str(library),library_sha256=artifacts[name],
                  dependency_scope='Declared HIP compiler tools and load-time dependencies, ROCm/Clang/system/GCC headers, and explicitly linked libraries; runtime plugins and device-library data are not a complete environment lock')
    return ctypes.CDLL(str(library)),result
