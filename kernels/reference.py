#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""PyTorch reference for SwiGLU and DLR-Net, plus the weight packer.

The DLR-Net here is numerically identical to ``dlr_lock.model.DLRNet`` and shares
its ``state_dict`` layout, so a distilled checkpoint can be packed and fed
straight to the kernels. It is duplicated rather than imported so that
``kernels/`` stands alone and can be benchmarked without installing the package.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_CONFIGS = {
    "Qwen3-0.6B": {"d": 1024, "d_ff": 3072, "r": 32},
    "Qwen3-1.7B": {"d": 2048, "d_ff": 6144, "r": 64},
    "Qwen3-8B":   {"d": 4096, "d_ff": 12288, "r": 128},
    "Qwen3-32B":  {"d": 5120, "d_ff": 25600, "r": 256},
}

DTYPE = torch.bfloat16
EPS = 1e-6


def compute_depth(d, d_ff, r):
    swiglu_params = 3 * d * d_ff
    params_per_layer = 2 * d * r + d + 1
    return max(swiglu_params // params_per_layer, 2)


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj   = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DLRNet(nn.Module):
    def __init__(self, d, r, depth, alpha_init=0.01):
        super().__init__()
        self.d, self.r, self.depth = d, r, depth
        u_scale = 1.0 / math.sqrt(r * depth)
        v_scale = 1.0 / math.sqrt(d)
        self.Us     = nn.ParameterList([nn.Parameter(torch.randn(d, r) * u_scale) for _ in range(depth)])
        self.Vs     = nn.ParameterList([nn.Parameter(torch.randn(r, d) * v_scale) for _ in range(depth)])
        self.rms_w  = nn.ParameterList([nn.Parameter(torch.ones(d)) for _ in range(depth)])
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(float(alpha_init))) for _ in range(depth)])

    def forward(self, x):
        h = x
        for i in range(self.depth):
            ms  = h.pow(2).mean(dim=-1, keepdim=True)
            inp = h * self.rms_w[i] * torch.rsqrt(ms + EPS)
            z   = inp @ self.Vs[i].T
            dh  = F.silu(z) @ self.Us[i].T
            h   = h + self.alphas[i] * dh
        return h


def pack_dlrnet_weights(dlrnet: DLRNet, device="cuda", premultiply=True):
    """Pack weights into contiguous tensors; folds RMSNorm gain into V and alpha into U."""
    L, D, R = dlrnet.depth, dlrnet.d, dlrnet.r
    Vs_list, Us_list, Ws_list, As_list = [], [], [], []
    for v, u, w, a in zip(dlrnet.Vs, dlrnet.Us, dlrnet.rms_w, dlrnet.alphas):
        V_dr = v.detach().to(device, DTYPE).T.contiguous()
        U_rd = u.detach().to(device, DTYPE).T.contiguous()
        w_d  = w.detach().to(device, DTYPE)
        a_f  = a.detach().to(device, torch.float32)
        if premultiply:
            V_dr = (V_dr.to(torch.float32) * w_d.to(torch.float32).unsqueeze(-1)).to(DTYPE).contiguous()
            U_rd = (U_rd.to(torch.float32) * a_f.to(torch.float32)).to(DTYPE).contiguous()
        Vs_list.append(V_dr)
        Us_list.append(U_rd)
        Ws_list.append(w_d)
        As_list.append(a_f)
    return {
        "Vs": torch.stack(Vs_list).contiguous(),
        "Us": torch.stack(Us_list).contiguous(),
        "Ws": torch.stack(Ws_list).contiguous(),
        "As": torch.stack(As_list).contiguous(),
        "L": L, "D": D, "R": R,
        "premultiplied": premultiply,
    }


def build_models(name="Qwen3-0.6B", seed=42):
    cfg = MODEL_CONFIGS[name]
    d, d_ff, r = cfg["d"], cfg["d_ff"], cfg["r"]
    depth = compute_depth(d, d_ff, r)
    torch.manual_seed(seed)
    swiglu = SwiGLU(d, d_ff).to(DTYPE).cuda()
    torch.manual_seed(seed)
    dlrnet = DLRNet(d, r, depth).to(DTYPE).cuda()
    return swiglu, dlrnet, {"d": d, "d_ff": d_ff, "r": r, "depth": depth}
