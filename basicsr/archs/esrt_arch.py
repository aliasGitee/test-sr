import torch
from torch import nn as nn

from basicsr.archs.esrt.esrt import ESRT as esrt
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ESRT(nn.Module):
    def __init__(self, upscaling_factor):
        super(ESRT, self).__init__()

        self.model = esrt(upscale = upscaling_factor)

    def forward(self, x):
        return self.model(x)