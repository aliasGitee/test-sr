import torch.nn.functional as F
from torch import nn
from basicsr.archs.restormer.hparams import hparams
from basicsr.archs.restormer.restormer import Restormer as denoise_fn
from basicsr.archs.restormer.diffusion import GaussianDiffusion
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class RestorDiff(nn.Module):
    def __init__(self):
        super(RestorDiff, self).__init__()

        self.model = GaussianDiffusion(
            denoise_fn=denoise_fn(
                inp_channels=3,
                out_channels=3,
                dim = 16,
                num_blocks = [2,4,4,4],
                num_refinement_blocks = 4,
                heads = [1,2,4,8],
                ffn_expansion_factor = 2.66
            ),
            timesteps=2000,
            sampling_timesteps=2000,
            loss_type='l1')

        self.scale_factor = hparams['sr_scale']
        self.sample_type = 'ddpm'

    def sample(self,img_lr):
        #img_lr_up = F.interpolate(img_lr,scale_factor=self.scale_factor,mode='bicubic')
        shape = img_lr.shape
        if self.sample_type == 'ddim':
            img =  self.model.ddim_sample(img_lr, shape)
        else:
            img =  self.model.sample(img_lr, shape)
        return img

    def forward(self,img_hr, img_lr, t=None):
        #img_lr_up = F.interpolate(img_lr,scale_factor=self.scale_factor,mode='bicubic')
        loss = self.model(img_hr, img_lr, t)
        return loss
