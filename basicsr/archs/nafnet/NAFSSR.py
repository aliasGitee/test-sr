# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
NAFSSR: Stereo Image Super-Resolution Using NAFNet

@InProceedings{Chu2022NAFSSR,
  author    = {Xiaojie Chu and Liangyu Chen and Wenqing Yu},
  title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  booktitle = {CVPRW},
  year      = {2022},
}
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from basicsr.archs.nafnet.NAFNet import LayerNorm2d, NAFBlock
from basicsr.archs.nafnet.local import Local_Base

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class MySequential(nn.Sequential):
    def forward(self, t,*inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(t,*inputs)
            else:
                inputs = module(t,inputs)
        return inputs

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c,time_emb_dim):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.time_mlp = nn.Sequential(
                Mish(),
                nn.Linear(time_emb_dim, c)
            )

    def forward(self, t, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta + self.time_mlp(t)[:, :, None, None]
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma + self.time_mlp(t)[:, :, None, None]

        return x_l + F_r2l, x_r + F_l2r

class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, t,*feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(t,*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats

class NAFBlockSR(nn.Module):
    '''
    NAFBlock for Super-Resolution
    '''
    def __init__(self, c, time_emb_dim,fusion=False, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c,time_emb_dim) if fusion else None

    def forward(self, t,*feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(t,*feats)
        return feats

class NAFNetSR(nn.Module):
    '''
    NAFNet for Super-Resolution
    '''
    def __init__(self, time_emb_dim=32,up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=1000, dual=True):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.body = MySequential(
            *[DropPath(
                drop_path_rate,
                NAFBlockSR(
                    width,
                    time_emb_dim,
                    fusion=(fusion_from <= i and i <= fusion_to),
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        )

        # time_embed
        self.time_pos_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # self.up = nn.Sequential(
        #     nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        #     nn.PixelShuffle(up_scale)
        # )
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        )
        self.up_scale = up_scale

        '''
        new
        '''
        self.conv_last = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=1)
        ''''''

    def forward(self, x_t, t, cond):
        inp = torch.concat([x_t,cond],dim=1)
        t = self.time_pos_emb(t)
        # inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        # if self.dual:
        #     inp = inp.chunk(2, dim=1)
        # else:
        #     inp = (inp, )
        # feats = [self.intro(x) for x in inp]
        # feats = self.body(*feats)
        # out = torch.cat([self.up(x) for x in feats], dim=1)
        # out = out + inp_hr
        #inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        inp_cpoy = inp
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        feats = [self.intro(x) for x in inp]
        feats = self.body(t,*feats)
        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + inp_cpoy

        out = self.conv_last(out)
        return out


if __name__ == '__main__':
    num_blks = 16
    width = 48
    droppath=0.1
    train_size = (1, 6, 30, 90)

    net = NAFNetSR(up_scale=4, width=width, num_blks=num_blks, drop_path_rate=droppath)

    x = torch.randn(1,6,192,192)
    print(net(x[:,0:4,:,:],torch.tensor([1]),x[:,4:,:,:]).shape)
    # import thop
    # total_ops, total_params = thop.profile(net,(x[:,0:4,:,:],torch.tensor([1]),x[:,4:,:,:],))
    # print(total_ops,' ',total_params)





