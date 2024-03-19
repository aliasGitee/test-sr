from basicsr.archs.bsrn.BSRN import BSRN2
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class TESTSR01(nn.Module):
    def __init__(self):
        super(TESTSR01,self).__init__()
        self.arch_all = BSRN2(upscale=2)
    def forward(self,x):
        x = self.arch_all(x)
        return x