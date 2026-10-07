#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Shared JIT loader for the hand-written CUDA DLR-Net kernels.

Every CUDA kernel used by this package lives as a single ``.cu`` file under
``kernels/cuda/``. Each kernel is written generically and specialised at compile
time through ``-D`` macros (model dimension, low rank, tile sizes, ...), so one
source file serves every supported shape. This helper compiles a given source +
macro combination once and caches the resulting extension module for the process.

Using one loader keeps the individual kernel wrappers small: they only declare
which source file, entry-point symbol, and macro values they need.
"""
import os

_CUDA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")

# Common flags for every kernel: optimise, C++17, target Hopper (sm_90a),
# fast math, and keep line info for profiling.
_BASE_CFLAGS = ["-O3", "-std=c++17", "-arch=sm_90a", "--use_fast_math", "-lineinfo"]

_MODULE_CACHE = {}


def load_kernel(cu_file, symbol, defines, include_paths=()):
    """Compile ``kernels/cuda/<cu_file>`` with the given ``-D`` macros and return
    the loaded extension module (cached per distinct configuration).

    Args:
        cu_file: file name inside ``kernels/cuda/`` (e.g. ``"cooperative.cu"``).
        symbol:  base name for the built extension; the macro values are appended
                 so different configurations get distinct build directories.
        defines: dict of macro -> value, passed as ``-D<macro>=<value>``.
        include_paths: extra ``-I`` directories (e.g. for CUTLASS headers).
    """
    cfg = tuple(sorted(defines.items()))
    key = (cu_file, symbol, cfg, tuple(include_paths))
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]

    from torch.utils.cpp_extension import load

    # Deterministic, readable module name unique to this configuration.
    suffix = "_".join(f"{k[4:] if k.startswith('CFG_') else k}{v}" for k, v in cfg)
    name = f"{symbol}__{suffix}" if suffix else symbol

    mod = load(
        name=name,
        sources=[os.path.join(_CUDA_DIR, cu_file)],
        extra_cuda_cflags=_BASE_CFLAGS + [f"-D{k}={v}" for k, v in cfg],
        extra_include_paths=list(include_paths),
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    _MODULE_CACHE[key] = mod
    return mod
