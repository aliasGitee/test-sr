import torch.nn.functional as F
from torch import nn
from basicsr.archs.myarchs.myfix5 import myfix5 as myarchs
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MyFix5(nn.Module):
    def __init__(self,upscale=2):
        super(MyFix5, self).__init__()

        self.model = myarchs(upscale=2)

    def forward(self,img_lr):
        img_sr = self.model(img_lr)
        return img_sr