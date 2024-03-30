from basicsr.archs.myarchs import myfix2
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DefaultArch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = myfix2()
    def forward(self,x):
        return self.model(x)