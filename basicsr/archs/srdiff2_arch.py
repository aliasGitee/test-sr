import torch.nn.functional as F
from torch import nn
import torch
from basicsr.archs.srdiff2.hparams import hparams
from basicsr.archs.srdiff2.diffsr_modules import Unet, RRDBNet
from basicsr.archs.srdiff2.diffusion import GaussianDiffusion
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class SRDiff2(nn.Module):
    def __init__(self):
        super(SRDiff2, self).__init__()

        hidden_size = hparams['hidden_size']
        dim_mults = hparams['unet_dim_mults']
        dim_mults = [int(x) for x in dim_mults.split('|')]
        denoise_fn = Unet(
            hidden_size, out_dim=3, cond_dim=hparams['rrdb_num_feat'], dim_mults=dim_mults)
        if hparams['use_rrdb']:
            rrdb = RRDBNet(3, 3, hparams['rrdb_num_feat'], hparams['rrdb_num_block'],
                           hparams['rrdb_num_feat'] // 2)
            rrdb_params = torch.load(hparams['load_ckpt'])['state_dict']
            rrdb.load_state_dict(rrdb_params)
        else:
            rrdb = None
        self.model = GaussianDiffusion(
            denoise_fn=denoise_fn,
            rrdb_net=rrdb,
            timesteps=hparams['timesteps'],
            loss_type=hparams['loss_type']
        )
        self.scale_factor = hparams['sr_scale']
        self.sample_type = 'ddpm'

    def sample(self,img_lr):
        img_lr_up = F.interpolate(img_lr,scale_factor=self.scale_factor,mode='bicubic')
        b,c,h,w = img_lr.shape
        shape = (b,c,h*self.scale_factor, w*self.scale_factor)
        if self.sample_type == 'ddim':
            img =  self.model.ddim_sample(img_lr, shape)
        else:
            img =  self.model.sample(img_lr, shape)

        '''
        for res
        '''
        img = img.clamp(-1, 1)
        img = img / 2 + img_lr_up
        ''''''

        return img

    def forward(self,img_hr, img_lr, t=None):
        img_lr_up = F.interpolate(img_lr,scale_factor=self.scale_factor,mode='bicubic')
        '''
        for res
        '''
        img_hr = ((img_hr-img_lr_up)*2).clamp(-1, 1)
        ''''''
        loss = self.model(img_hr, img_lr, t)
        return loss