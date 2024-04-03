import torch.nn.functional as F
from torch import nn
from basicsr.archs.myarchs.myfix3 import myfix3 as myarchs
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MyFix3(nn.Module):
    def __init__(self,dim,n_blocks,upscaling_factor):
        super(MyFix3, self).__init__()

        self.model = myarchs(n_blocks=n_blocks,dim=dim,upscaling_factor=upscaling_factor)

    def forward(self,img_lr):
        img_sr = self.model(img_lr)
        return img_sr