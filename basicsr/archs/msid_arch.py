import torch.nn.functional as F
from torch import nn
from basicsr.archs.msid.MSID import MSID as myarchs
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MSID(nn.Module):
    def __init__(self,upscaling_factor):
        super(MSID, self).__init__()
        self.model = myarchs(upscale=upscaling_factor)
    def forward(self,img_lr):
        img_sr = self.model(img_lr)
        return img_sr