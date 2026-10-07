#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Shared helpers for loading a pretrained causal LM."""

import torch


def autocast(device, dtype=torch.bfloat16):
    """``torch.autocast`` for whichever backend ``device`` names.

    Keeps the scripts runnable on CPU (useful for smoke tests) instead of
    hardcoding ``device_type="cuda"``. Autocast is not a convenience here: the
    DLR-Net parameters are held in fp32 for optimizer stability while the
    backbone stays in bf16, and autocast is what reconciles the two dtypes at
    every matmul.
    """
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def empty_cache(device):
    """Release cached allocator blocks, if the backend has a cache."""
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def peak_memory_gb(device):
    """Peak allocated memory in GB, or 0 on backends without a tracker."""
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / 1e9
    return 0.0


def reset_peak_memory(device):
    """Reset the peak-memory tracker so a measurement starts from zero."""
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def load_causal_lm(model_name, device="cuda:0", device_map=None,
                   torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                   revision=None):
    """Load a causal LM onto one device, or shard it via ``device_map``.

    ``device_map`` pulls in ``accelerate``, so it is only passed through when
    sharding was actually requested; otherwise this is a plain single-device
    load followed by ``.to(device)``.

    ``revision`` pins the Hugging Face Hub revision -- a branch, tag, or commit
    hash. Leaving it unset resolves to the repository's default branch, whose
    contents can change over time; pin it when you need a run to stay
    reproducible, or when you want to be sure of exactly which weights you are
    loading.
    """
    from transformers import AutoModelForCausalLM

    kwargs = dict(torch_dtype=torch_dtype,
                  attn_implementation=attn_implementation)
    if device_map is not None:
        kwargs["device_map"] = device_map
    # "main" is transformers' own default for revision, so passing it when the
    # caller supplied nothing is behaviourally identical to omitting the
    # argument -- while keeping the pin visible at the call site.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision or "main", **kwargs)
    if device_map is None:
        model = model.to(device)
    return model


def load_tokenizer(model_name, revision=None):
    """Load a tokenizer, ensuring a pad token exists.

    ``revision`` pins the Hub revision; see :func:`load_causal_lm`.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              revision=revision or "main")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
