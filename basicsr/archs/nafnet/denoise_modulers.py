from basicsr.archs.nafnet.NAFSSR import NAFNetSR as dnmodel
import torch
import torch.nn as nn


class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_blks = 16
        width = 48
        droppath=0.1
        self.denoise_fn = dnmodel(up_scale=4, width=width, num_blks=num_blks, drop_path_rate=droppath)
    def forward(self, x_t,t,cond):
        noise = self.denoise_fn(x_t,t,cond)
        return noise