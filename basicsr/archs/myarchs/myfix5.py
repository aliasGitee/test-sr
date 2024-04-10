import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.efficientvit.fix.ops_fix import EfficientViTBlock3 as EFTB
from basicsr.archs.msnlan.common_fix import My_Block,default_conv
from basicsr.archs.msnlan.common import CAer as CA


# 458853984.0   254316.0

# 458853984.0   254316.0 / 514825632.0   226668.0
#    Set5
#        psnr: 38.9888/37.9654  ssim: 0.9610/0.9611
#    Set14
#        psnr: 33.5966/33.5346  ssim: 0.9183/0.9176
#    B100
#        psnr: 32.1683/32.1559  ssim: 0.9003/0.9001
#    Urban100
#        psnr: 31.9756/31.8995  ssim: 0.9261/0.9258
#    Manga109
#	    psnr: 38.6541/38.5548  ssim: 0.9772/0.9770

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

class CCCM(nn.Module):
    def __init__(self,dim, growth_rate=2.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,padding=1,groups=dim//2)
        self.conv = nn.Conv2d(in_channels=dim*2,out_channels=dim,kernel_size=1)
        #self.ca = CA(channel=dim,reduction=9)
    def forward(self,x):
        x1 = self.conv1(x)
        #x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_out = torch.cat([x1,x3],dim=1)
        x_out = self.conv(F.gelu(x_out))
        return x_out

# SAFM
class SAFM(nn.Module):
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

# DAFM
class DAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr1 = nn.ModuleList([
                nn.Conv2d(chunk_dim, chunk_dim, kernel_size=3, padding=i+1, dilation=i+1,groups=chunk_dim)
                for i in range(self.n_levels)])
        self.mfr2 = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

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
                s = self.mfr1[i](s)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr2[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr1[i](xc[i])
                s = self.mfr2[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

class SAFMBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.safm = SAFM(dim)
        # Feedforward layer
        self.ccm = CCCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x

class DAFMBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.dafm = DAFM(dim)
        # Feedforward layer
        self.ccm = CCCM(dim, ffn_scale)

    def forward(self, x):
        x = self.dafm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x

class myfix5(nn.Module):
    def __init__(self, dim=36, n_blocks=4, ffn_scale=2.0, upscaling_factor=2):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[
            nn.Sequential(SAFMBlock(dim, ffn_scale),
                          DAFMBlock(dim,ffn_scale))
            for _ in range(n_blocks)])

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
    model = myfix5()
    x = torch.randn(1,3,48,48)
    total_ops, total_params = thop.profile(model, (x,))
    print(total_ops,' ',total_params)

    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
