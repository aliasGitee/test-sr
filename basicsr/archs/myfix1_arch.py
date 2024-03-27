import torch.nn.functional as F
from torch import nn
from basicsr.archs.myarchs.myfix1 import myfix1 as myarchs
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MyFix1(nn.Module):
    def __init__(self,upscaling_factor):
        super(MyFix1, self).__init__()

        self.model = myarchs(num_feat=64,num_block=5,upscale=upscaling_factor)

    def forward(self,img_lr):
        img_sr = self.model(img_lr)
        return img_sr