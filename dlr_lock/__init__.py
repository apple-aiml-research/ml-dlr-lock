#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""DLR-Lock: making pretrained LLM weights hard to fine-tune.

Each SwiGLU MLP in a pretrained transformer is replaced by a deep low-rank
residual network (DLR-Net) of comparable parameter count, distilled to match the
original module's behavior. The replacement preserves the forward map while
making backpropagation through the model substantially more expensive.
"""

from .model import (
    DLRNet,
    ResidualDLRNet,
    build_locked_model,
    compute_depth,
    dlrnet_config,
    dlrnet_param_count,
    load_dlrnet_weights,
    replace_layer_mlp,
    save_dlrnet_weights,
    set_checkpointing,
    write_config,
)

__version__ = "0.1.0"

__all__ = [
    "DLRNet",
    "ResidualDLRNet",
    "build_locked_model",
    "compute_depth",
    "dlrnet_config",
    "dlrnet_param_count",
    "load_dlrnet_weights",
    "replace_layer_mlp",
    "save_dlrnet_weights",
    "set_checkpointing",
    "write_config",
]
