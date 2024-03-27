import torch.nn.functional as F
from torch import nn
#from basicsr.archs.myarchs.myfix1 import SAFMN as myarchs
from basicsr.utils.registry import ARCH_REGISTRY


#@ARCH_REGISTRY.register()
# class MyFix1(nn.Module):
#     def __init__(self,upscaling_factor):
#         super(MyFix1, self).__init__()

#         self.model = myarchs(dim=36, n_blocks=4, ffn_scale=2.0, upscaling_factor=upscaling_factor)

#     def forward(self,img_lr):
#         img_sr = self.model(img_lr)
#         return img_sr