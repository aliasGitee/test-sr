import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.efficientvit.fix.ops_fix import EfficientViTBlock3 as EFTB

'''
514825632.0   226668.0
'''

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

    def forward(self, x):
        return self.ccm(x)

# MFAAB
class MFAAB(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.d1 = nn.Conv2d(in_channels=dim,out_channels=dim//3,kernel_size=3,padding=2,dilation=2)
        self.d2 = nn.Conv2d(in_channels=dim,out_channels=dim//3,kernel_size=3,padding=3,dilation=3)
        self.d3 = nn.Conv2d(in_channels=dim,out_channels=dim//3,kernel_size=3,padding=4,dilation=4)
        self.dw5 = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=5,padding=2,groups=dim)
        self.dw7d1 = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=7,padding=6,dilation=2,groups=dim)
        self.pw = nn.Conv2d(dim,dim,1)
    def forward(self,x):
        d1 = self.d1(x)
        d2 = self.d2(x)
        d3 = self.d3(x)
        d = torch.cat([d1,d2,d3],dim=1)
        d2 = self.dw7d1(self.dw5(self.pw(x)))
        d = d+d2
        d3 = F.sigmoid(F.adaptive_avg_pool2d(d, (1,1)))
        d = d*d3
        return x+d

# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

# SAFM
class ESAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        #self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        self.mfr = nn.ModuleList([EFTB(in_channels=chunk_dim,
                dim=chunk_dim//3,
                expand_ratio=4,
                norm="ln2d",
                act_func="hswish") for _ in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

class EAttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.safm = ESAFM(dim)
        # Feedforward layer
        self.ccm = MFAAB(dim)#CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x))
        return x

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.safm = SAFM(dim)
        # Feedforward layer
        self.ccm = MFAAB(dim)#CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x))
        return x

class myfix6(nn.Module):
    def __init__(self, dim=36, n_blocks=4, ffn_scale=2.0, upscaling_factor=2):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[
            nn.Sequential(
                AttBlock(dim, ffn_scale),
                EAttBlock(dim, ffn_scale)) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x



if __name__ == '__main__':
    import thop
    model = myfix6()
    #model = MFAAB(dim=36)
    #model = CCM(dim=36)
    x = torch.randn(1,3,48,48)
    total_ops, total_params = thop.profile(model, (x,))
    print(total_ops,' ',total_params)
    print(model(x).shape)
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))