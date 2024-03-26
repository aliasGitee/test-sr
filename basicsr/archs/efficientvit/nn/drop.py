# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import numpy as np
import torch
import torch.nn as nn
from basicsr.archs.efficientvit.nn.ops import IdentityLayer, ResidualBlock
from basicsr.archs.efficientvit.utils import build_kwargs_from_config

__all__ = ["apply_drop_func"]


def apply_drop_func(network: nn.Module, drop_config: dict[str, any] or None) -> None:
    if drop_config is None:
        return

    drop_lookup_table = {
        "droppath": apply_droppath,
    }

    drop_func = drop_lookup_table[drop_config["name"]]
    drop_kwargs = build_kwargs_from_config(drop_config, drop_func)

    drop_func(network, **drop_kwargs)


def apply_droppath(
    network: nn.Module,
    drop_prob: float,
    linear_decay=True,
    scheduled=True,
    skip=0,
) -> None:
    all_valid_blocks = []
    for m in network.modules():
        for name, sub_module in m.named_children():
            if isinstance(sub_module, ResidualBlock) and isinstance(sub_module.shortcut, IdentityLayer):
                all_valid_blocks.append((m, name, sub_module))
    all_valid_blocks = all_valid_blocks[skip:]
    for i, (m, name, sub_module) in enumerate(all_valid_blocks):
        prob = drop_prob * (i + 1) / len(all_valid_blocks) if linear_decay else drop_prob
        new_module = DropPathResidualBlock(
            sub_module.main,
            sub_module.shortcut,
            sub_module.post_act,
            sub_module.pre_norm,
            prob,
            scheduled,
        )
        m._modules[name] = new_module
