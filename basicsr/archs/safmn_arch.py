import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.safmn.SAFMN import SAFMN as model

class SAFMN(nn.Module):
    def __init__(self):
        super(SAFMN, self).__init__()
        self.model = model(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=2)
    def forward(self, x):
        return self.model(x)