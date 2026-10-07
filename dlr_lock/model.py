#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Deep low-rank residual network (DLR-Net) and model surgery.

A DLR-Net replaces one transformer MLP residual sub-block

    x' = z + FFN(RMSNorm(z))

with a depth-L stack of low-rank residual updates

    h_0     = z
    h_{i+1} = h_i + alpha_i * U_i @ silu(V_i @ RMSNorm(h_i))
    x'      = h_L

where V_i has shape (r, d), U_i has shape (d, r), and alpha_i is a learnable
ReZero scalar initialized at 0. Each layer costs 2*d*r + d parameters, so at a
fixed budget P matching the SwiGLU MLP the depth is

    L = floor(P / (2*d*r + d)).

A smaller rank buys a deeper network at the same parameter count. RMSNorm
pre-normalization is what makes training stable at L > 100, and its backward
pass is also what forces an attacker to keep an O(d) activation per layer even
when U and V are frozen.
"""

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class DLRNet(nn.Module):
    """Depth-``depth`` low-rank residual network over a ``d``-dimensional state.

    Parameters
    ----------
    d : int
        Hidden dimension of the transformer (the residual stream width).
    r : int
        Low-rank bottleneck dimension, ``r << d``.
    depth : int
        Number of residual layers ``L``. Use :func:`compute_depth` to pick the
        value that matches a given parameter budget.
    alpha_init : float
        Initial value of the ReZero scalars. ``0.0`` makes the network an exact
        identity at initialization, which is the right starting point for
        residual-target distillation.
    eps : float
        RMSNorm epsilon.
    """

    def __init__(self, d, r, depth, alpha_init=0.0, eps=1e-6):
        super().__init__()
        self.d = d
        self.r = r
        self.depth = depth
        self.eps = eps

        # V ~ N(0, 1/d), U ~ N(0, 1/(r*L)): the 1/sqrt(L) factor keeps the
        # summed residual update O(1) as depth grows.
        v_scale = 1.0 / math.sqrt(d)
        u_scale = 1.0 / math.sqrt(r * depth)

        self.Vs = nn.ParameterList(
            [nn.Parameter(torch.randn(r, d) * v_scale) for _ in range(depth)]
        )
        self.Us = nn.ParameterList(
            [nn.Parameter(torch.randn(d, r) * u_scale) for _ in range(depth)]
        )
        self.rms_w = nn.ParameterList(
            [nn.Parameter(torch.ones(d)) for _ in range(depth)]
        )
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(alpha_init))) for _ in range(depth)]
        )

        # Recompute the layer stack in the backward pass instead of storing all
        # L intermediates. Set to False to measure the true Theta(L*d)
        # activation footprint (the memory barrier of Proposition 1).
        self.use_checkpoint = True

    def forward(self, x):
        if self.use_checkpoint and torch.is_grad_enabled() and x.requires_grad:
            return torch_checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        h = x
        for i in range(self.depth):
            ms = h.pow(2).mean(dim=-1, keepdim=True)
            inp = h * self.rms_w[i] * torch.rsqrt(ms + self.eps)
            z = inp @ self.Vs[i].T                    # (..., r)
            dh = F.silu(z) @ self.Us[i].T             # (..., d)
            h = h + self.alphas[i] * dh
        return h


class ResidualDLRNet(nn.Module):
    """Adapter that lets a DLR-Net absorb a whole MLP residual sub-block.

    A HuggingFace decoder layer computes ``h + mlp(post_attention_layernorm(h))``.
    Replacing the norm with ``nn.Identity`` and the MLP with this wrapper makes
    the layer compute ``h + (DLRNet(h) - h) = DLRNet(h)``, so the DLR-Net owns
    the normalization and the residual connection itself. This is the
    "residual-target" formulation used throughout distillation: the student
    regresses the full sub-block map ``z -> z + FFN(RMSNorm(z))``, whose output
    norm is dominated by the residual and is therefore far better conditioned
    than the bare ``FFN`` output.
    """

    def __init__(self, dlrnet):
        super().__init__()
        self.dlrnet = dlrnet

    def forward(self, h):
        return self.dlrnet(h) - h


def dlrnet_param_count(d, r, depth):
    """Parameter count of a :class:`DLRNet`: ``L * (2dr + d + 1)``."""
    return depth * (2 * d * r + d + 1)


def compute_depth(d, d_ff, r, multiplier=1.0):
    """Depth whose parameter count matches ``multiplier`` x the SwiGLU MLP.

    A SwiGLU MLP with gate, up, and down projections costs ``3 * d * d_ff``
    parameters. Each DLR-Net layer costs ``2dr`` for U and V, plus ``d`` for the
    RMSNorm gain and ``1`` for the ReZero scalar.
    """
    target = int(3 * d * d_ff * multiplier)
    per_layer = 2 * d * r + d + 1
    return max(target // per_layer, 2)


def dlrnet_config(d, d_ff, r, multiplier=1.0):
    """JSON-serializable description of the DLR-Net for a given model shape."""
    depth = compute_depth(d, d_ff, r, multiplier)
    return {
        "d": d,
        "d_ff": d_ff,
        "rank": r,
        "multiplier": multiplier,
        "depth": depth,
        "params": dlrnet_param_count(d, r, depth),
        "swiglu_params": 3 * d * d_ff,
    }


def replace_layer_mlp(model, layer_idx, d, rank, depth, state_dict=None,
                      alpha_init=0.0):
    """Swap one decoder layer's MLP for a DLR-Net, in place.

    ``state_dict`` may carry a ``_orig_mod.`` prefix if it was saved from a
    ``torch.compile``-wrapped module; the prefix is stripped automatically.
    """
    dlrnet = DLRNet(d, rank, depth, alpha_init=alpha_init)
    if state_dict is not None:
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v
                          for k, v in state_dict.items()}
        dlrnet.load_state_dict(state_dict)

    layer = model.model.layers[layer_idx]
    ref = next(layer.parameters())
    dlrnet = dlrnet.to(device=ref.device, dtype=ref.dtype)

    layer.post_attention_layernorm = nn.Identity()
    layer.mlp = ResidualDLRNet(dlrnet)
    return dlrnet


def build_locked_model(model_name, rank, multiplier=1.0, state_dicts=None,
                       device="cpu", device_map=None,
                       torch_dtype=torch.bfloat16,
                       attn_implementation="sdpa", alpha_init=0.0,
                       verbose=True, revision=None):
    """Load a causal LM and replace every MLP with a DLR-Net.

    Parameters
    ----------
    model_name : str
        HuggingFace model id or a local path.
    rank : int
        Low-rank bottleneck ``r``. Pick it so ``compute_depth`` lands in the
        140-150 range (r=32 for a 1024-wide model, r=64 for 2048, ...).
    multiplier : float
        Parameter budget relative to the SwiGLU MLP being replaced.
    state_dicts : dict[int, dict] or None
        Per-layer DLR-Net weights, keyed by layer index. ``None`` leaves the
        DLR-Nets randomly initialized (an identity map when ``alpha_init=0``).
    device_map : str or dict or None
        Passed to ``from_pretrained``; takes precedence over ``device``. Use
        ``"auto"`` to shard a large model across GPUs, in which case each
        DLR-Net is placed on the same device as the layer it replaces.
    """
    # Only route through device_map (which needs accelerate) when sharding was
    # actually requested; a single-device load is a plain .to(device).
    from .loading import load_causal_lm

    model = load_causal_lm(model_name, device=device, device_map=device_map,
                           torch_dtype=torch_dtype,
                           attn_implementation=attn_implementation,
                           revision=revision)

    d = model.config.hidden_size
    d_ff = model.config.intermediate_size
    n_layers = len(model.model.layers)
    depth = compute_depth(d, d_ff, rank, multiplier)

    if verbose:
        cfg = dlrnet_config(d, d_ff, rank, multiplier)
        print(f"Replacing {n_layers} MLPs with DLRNet("
              f"d={d}, r={rank}, depth={depth})")
        print(f"  {cfg['params']:,} params per MLP "
              f"({cfg['params'] / cfg['swiglu_params']:.2f}x SwiGLU)")

    for li in range(n_layers):
        sd = state_dicts.get(li) if state_dicts is not None else None
        replace_layer_mlp(model, li, d, rank, depth, state_dict=sd,
                          alpha_init=alpha_init)

    return model


def load_dlrnet_weights(model, checkpoint_dir, n_layers=None):
    """Load per-layer DLR-Net weights saved by the distillation scripts.

    Expects ``{checkpoint_dir}/layer{ii:02d}.pt`` for each layer. Returns the
    number of layers loaded.
    """
    import os

    if n_layers is None:
        n_layers = len(model.model.layers)
    for li in range(n_layers):
        path = os.path.join(checkpoint_dir, f"layer{li:02d}.pt")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model.model.layers[li].mlp.dlrnet.load_state_dict(sd)
    return n_layers


def save_dlrnet_weights(model, checkpoint_dir, n_layers=None):
    """Save every layer's DLR-Net weights as ``layer{ii:02d}.pt``."""
    import os

    os.makedirs(checkpoint_dir, exist_ok=True)
    if n_layers is None:
        n_layers = len(model.model.layers)
    for li in range(n_layers):
        sd = model.model.layers[li].mlp.dlrnet.state_dict()
        torch.save({k: v.cpu() for k, v in sd.items()},
                   os.path.join(checkpoint_dir, f"layer{li:02d}.pt"))
    return n_layers


def set_checkpointing(model, enabled):
    """Toggle the DLR-Nets' internal activation checkpointing.

    Disabling it stores all ``L`` intermediates per layer, which is the
    unmitigated ``Theta(L*d)`` activation footprint the locking argument relies
    on. Enabling it is the adaptive attacker's best mitigation.
    """
    n = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        dlrnet = getattr(mlp, "dlrnet", mlp)
        if isinstance(dlrnet, DLRNet):
            dlrnet.use_checkpoint = enabled
            n += 1
    return n


def write_config(path, d, d_ff, r, multiplier=1.0):
    """Write :func:`dlrnet_config` to ``path`` as JSON."""
    with open(path, "w") as f:
        json.dump(dlrnet_config(d, d_ff, r, multiplier), f, indent=2)
