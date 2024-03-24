import torch.nn.functional as F
from torch import nn
from basicsr.archs.nafnet.hparams import hparams
from basicsr.archs.nafnet.denoise_modulers import DenoiseNet as denoise_fn
from basicsr.archs.nafnet.diffusion import GaussianDiffusion
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class RestorDiff(nn.Module):
    def __init__(self):
        super(RestorDiff, self).__init__()

        self.model = GaussianDiffusion(
            denoise_fn=denoise_fn(),
            timesteps=1000,
            sampling_timesteps=100,
            loss_type='l1')

        self.scale_factor = hparams['sr_scale']
        self.sample_type = 'ddim'

    def sample(self,img_lr):
        img_lr_up = F.interpolate(img_lr,scale_factor=self.scale_factor,mode='bicubic')
        shape = img_lr_up.shape
        if self.sample_type == 'ddim':
            img =  self.model.sample_ddim(img_lr_up, shape)
        else:
            img =  self.model.sample(img_lr_up, shape)
        return img

    def forward(self,img_hr, img_lr, t=None):
        img_lr_up = F.interpolate(img_lr,scale_factor=self.scale_factor,mode='bicubic')
        loss = self.model(img_hr, img_lr_up, t)
        return loss