import torch.nn.functional as F
from torch import nn
from basicsr.archs.myarchs.myfix6 import myfix6 as myarchs
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MyFix6(nn.Module):
    def __init__(self,upscale=2):
        super(MyFix6, self).__init__()

        self.model = myarchs(upscale=upscale)

    def forward(self,img_lr):
        img_sr = self.model(img_lr)
        return img_sr