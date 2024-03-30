import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.safmn.SAFMN import safmn as model
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class SAFMN(nn.Module):
    def __init__(self,dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=2):
        super(SAFMN, self).__init__()
        self.model = model(dim, n_blocks, ffn_scale, upscaling_factor)
    def forward(self, x):
        return self.model(x)