import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from basicsr.archs.msid import Upsamplers as Upsamplers
from basicsr.archs.efficientvit.EFVIT import EfficientVITStages


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# SE
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


# Channel MLP: Conv1*1 -> Conv1*1
class ChannelMLP(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mlp(x)


# MBConv: Conv1*1 -> DW Conv3*3 -> [SE] -> Conv1*1
class MBConv(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mbconv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mbconv(x)


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )
        # self.ccm = nn.Sequential(
        #     nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=1,stride=1,groups=1),
        #     nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3,stride=1,padding=1,groups=dim),
        #     nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3,stride=1,padding=2,dilation=2,groups=dim),
        #     nn.GELU()
        # )
        # self.pw = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_dim,out_channels=dim,kernel_size=1,stride=1,groups=1),
        #     nn.GELU()
        #     )

    def forward(self, x):
        #x = self.pw(x + self.ccm(x))

        return self.ccm(x)


# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels

        # Spatial Weighting
        self.efvit = EfficientVITStages(in_channels=dim,dim=dim,down=True,num_stage=2,expand_ratio=3)

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim*3, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        xc = self.efvit(x)
        out = self.aggr(xc)
        out = self.act(out) * x
        return out

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.safm = SAFM(dim)
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x

class myfix1(nn.Module):
    def __init__(self, dim=16, n_blocks=3, ffn_scale=2.0, upscaling_factor=2):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x



if __name__== '__main__':
    import thop
    x = torch.randn(1, 3, 48, 48)
    model = myfix1(dim=16)
    total_ops, total_params = thop.profile(model,(x,))
    print(total_ops, ' ',total_params)
    #print(model(x).shape)