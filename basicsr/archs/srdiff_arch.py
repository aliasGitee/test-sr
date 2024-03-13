import torch
import torch.nn.functional as F
from torch import nn
from basicsr.archs.srdiff.hparams import hparams
from basicsr.archs.srdiff.diffsr_modules import Unet, RRDBNet
from basicsr.archs.srdiff.diffusion import GaussianDiffusion
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class SRDiff(nn.Module):
    def __init__(self):
        super(SRDiff, self).__init__()
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
        self.sample_type = 'ddim'

    def sample(self,img_lr):
        img_lr_up = F.interpolate(img_lr,scale_factor=self.scale_factor,mode='bicubic')
        shape = img_lr_up.shape
        if self.sample_type == 'ddim':
            img, rrdb_out =  self.model.sample_ddim(img_lr,img_lr_up, shape, sampling_timesteps=50)
        else:
            img, rrdb_out =  self.model.sample(img_lr, img_lr_up, shape)
        return img, rrdb_out

    def forward(self,img_hr, img_lr, t=None):
        img_lr_up = F.interpolate(img_lr,scale_factor=self.scale_factor,mode='bicubic')
        ret, (x_tp1, x_t_gt, x_t), t = self.model(img_hr, img_lr, img_lr_up, t)
        return ret, (x_tp1, x_t_gt, x_t), t