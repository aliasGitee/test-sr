import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from basicsr.archs.msid import Upsamplers as Upsamplers
from basicsr.archs.vmamba.mamba_sys import VSSBlock, PatchEmbed2D, PatchExpand
from basicsr.archs.safmn.SAFMN import SAFM,LayerNorm

'''
X2_DIV2K
1280 * 720

thop:
    x2:
        64445976000.0   282140.0
    x3:
        29370789000.0   289715.0
    x4:
        17156563200.0   300320.0

fvcore:
    x2:
        params: 543212
        | module                               | #parameters or shape   | #flops     | #activations   |
        |:-------------------------------------|:-----------------------|:-----------|:---------------|
        | model                                | 0.543M                 | 0.106T     | 2.506G         |
    x3:
        params: 550787
        | module                               | #parameters or shape   | #flops     | #activations   |
        |:-------------------------------------|:-----------------------|:-----------|:---------------|
        | model                                | 0.551M                 | 47.907G    | 1.114G         |
    x4:
        params: 561392
        | module                               | #parameters or shape   | #flops     | #activations   |
        |:-------------------------------------|:-----------------------|:-----------|:---------------|
        | model                                | 0.561M                 | 27.6G      | 0.629G         |

2024-04-12 11:50:01,132 INFO: Testing Set5...
2024-04-12 11:50:02,259 INFO: Validation Set5
         # psnr: 38.0752        Best: 38.0752 @ test_myfix6_x2 iter
         # ssim: 0.9612 Best: 0.9612 @ test_myfix6_x2 iter

2024-04-12 11:50:02,260 INFO: Testing Set14...
2024-04-12 11:50:05,173 INFO: Validation Set14
         # psnr: 33.8424        Best: 33.8424 @ test_myfix6_x2 iter
         # ssim: 0.9200 Best: 0.9200 @ test_myfix6_x2 iter

2024-04-12 11:50:05,173 INFO: Testing B100...
2024-04-12 11:50:16,163 INFO: Validation B100
         # psnr: 32.2436        Best: 32.2436 @ test_myfix6_x2 iter
         # ssim: 0.9011 Best: 0.9011 @ test_myfix6_x2 iter

2024-04-12 11:50:16,164 INFO: Testing Urban100...
2024-04-12 11:51:19,885 INFO: Validation Urban100
         # psnr: 32.5007        Best: 32.5007 @ test_myfix6_x2 iter
         # ssim: 0.9315 Best: 0.9315 @ test_myfix6_x2 iter

2024-04-12 11:51:19,891 INFO: Testing Manga109...
2024-04-12 11:52:42,363 INFO: Validation Manga109
         # psnr: 39.0092        Best: 39.0092 @ test_myfix6_x2 iter
         # ssim: 0.9780 Best: 0.9780 @ test_myfix6_x2 iter
'''


'''
x2:
| module                               | #parameters or shape   | #flops     | #activations   |
|:-------------------------------------|:-----------------------|:-----------|:---------------|
| model                                | 0.543M                 | 0.106T     | 2.506G         |
|  fea_conv                            |  0.728K                |  0.155G    |  25.805M       |
|   fea_conv.pw                        |   0.168K               |   38.707M  |   12.902M      |
|    fea_conv.pw.weight                |    (56, 3, 1, 1)       |            |                |
|   fea_conv.dw                        |   0.56K                |   0.116G   |   12.902M      |
|    fea_conv.dw.weight                |    (56, 1, 3, 3)       |            |                |
|    fea_conv.dw.bias                  |    (56,)               |            |                |
|  B1                                  |  50.12K                |  9.661G    |  0.244G        |
|   B1.mamba                           |   46.256K              |   8.89G    |   0.227G       |
|    B1.mamba.ln_1                     |    0.112K              |    64.512M |    0           |
|    B1.mamba.self_attention           |    46.144K             |    8.825G  |    0.227G      |
|   B1.safm                            |   3.752K               |   0.771G   |   17.186M      |
|    B1.safm.mfr                       |    0.56K               |    38.556M |    4.284M      |
|    B1.safm.aggr                      |    3.192K              |    0.723G  |    12.902M     |
|   B1.norm                            |   0.112K               |   0        |   0            |
|    B1.norm.weight                    |    (56,)               |            |                |
|    B1.norm.bias                      |    (56,)               |            |                |
|  B2                                  |  50.12K                |  9.661G    |  0.244G        |
|   B2.mamba                           |   46.256K              |   8.89G    |   0.227G       |
|    B2.mamba.ln_1                     |    0.112K              |    64.512M |    0           |
|    B2.mamba.self_attention           |    46.144K             |    8.825G  |    0.227G      |
|   B2.safm                            |   3.752K               |   0.771G   |   17.186M      |
|    B2.safm.mfr                       |    0.56K               |    38.556M |    4.284M      |
|    B2.safm.aggr                      |    3.192K              |    0.723G  |    12.902M     |
|   B2.norm                            |   0.112K               |   0        |   0            |
|    B2.norm.weight                    |    (56,)               |            |                |
|    B2.norm.bias                      |    (56,)               |            |                |
|  B3                                  |  50.12K                |  9.661G    |  0.244G        |
|   B3.mamba                           |   46.256K              |   8.89G    |   0.227G       |
|    B3.mamba.ln_1                     |    0.112K              |    64.512M |    0           |
|    B3.mamba.self_attention           |    46.144K             |    8.825G  |    0.227G      |
|   B3.safm                            |   3.752K               |   0.771G   |   17.186M      |
|    B3.safm.mfr                       |    0.56K               |    38.556M |    4.284M      |
|    B3.safm.aggr                      |    3.192K              |    0.723G  |    12.902M     |
|   B3.norm                            |   0.112K               |   0        |   0            |
|    B3.norm.weight                    |    (56,)               |            |                |
|    B3.norm.bias                      |    (56,)               |            |                |
|  B4                                  |  50.12K                |  9.661G    |  0.244G        |
|   B4.mamba                           |   46.256K              |   8.89G    |   0.227G       |
|    B4.mamba.ln_1                     |    0.112K              |    64.512M |    0           |
|    B4.mamba.self_attention           |    46.144K             |    8.825G  |    0.227G      |
|   B4.safm                            |   3.752K               |   0.771G   |   17.186M      |
|    B4.safm.mfr                       |    0.56K               |    38.556M |    4.284M      |
|    B4.safm.aggr                      |    3.192K              |    0.723G  |    12.902M     |
|   B4.norm                            |   0.112K               |   0        |   0            |
|    B4.norm.weight                    |    (56,)               |            |                |
|    B4.norm.bias                      |    (56,)               |            |                |
|  B5                                  |  50.12K                |  9.661G    |  0.244G        |
|   B5.mamba                           |   46.256K              |   8.89G    |   0.227G       |
|    B5.mamba.ln_1                     |    0.112K              |    64.512M |    0           |
|    B5.mamba.self_attention           |    46.144K             |    8.825G  |    0.227G      |
|   B5.safm                            |   3.752K               |   0.771G   |   17.186M      |
|    B5.safm.mfr                       |    0.56K               |    38.556M |    4.284M      |
|    B5.safm.aggr                      |    3.192K              |    0.723G  |    12.902M     |
|   B5.norm                            |   0.112K               |   0        |   0            |
|    B5.norm.weight                    |    (56,)               |            |                |
|    B5.norm.bias                      |    (56,)               |            |                |
|  B6                                  |  50.12K                |  9.661G    |  0.244G        |
|   B6.mamba                           |   46.256K              |   8.89G    |   0.227G       |
|    B6.mamba.ln_1                     |    0.112K              |    64.512M |    0           |
|    B6.mamba.self_attention           |    46.144K             |    8.825G  |    0.227G      |
|   B6.safm                            |   3.752K               |   0.771G   |   17.186M      |
|    B6.safm.mfr                       |    0.56K               |    38.556M |    4.284M      |
|    B6.safm.aggr                      |    3.192K              |    0.723G  |    12.902M     |
|   B6.norm                            |   0.112K               |   0        |   0            |
|    B6.norm.weight                    |    (56,)               |            |                |
|    B6.norm.bias                      |    (56,)               |            |                |
|  B7                                  |  50.12K                |  9.661G    |  0.244G        |
|   B7.mamba                           |   46.256K              |   8.89G    |   0.227G       |
|    B7.mamba.ln_1                     |    0.112K              |    64.512M |    0           |
|    B7.mamba.self_attention           |    46.144K             |    8.825G  |    0.227G      |
|   B7.safm                            |   3.752K               |   0.771G   |   17.186M      |
|    B7.safm.mfr                       |    0.56K               |    38.556M |    4.284M      |
|    B7.safm.aggr                      |    3.192K              |    0.723G  |    12.902M     |
|   B7.norm                            |   0.112K               |   0        |   0            |
|    B7.norm.weight                    |    (56,)               |            |                |
|    B7.norm.bias                      |    (56,)               |            |                |
|  B8                                  |  50.12K                |  9.661G    |  0.244G        |
|   B8.mamba                           |   46.256K              |   8.89G    |   0.227G       |
|    B8.mamba.ln_1                     |    0.112K              |    64.512M |    0           |
|    B8.mamba.self_attention           |    46.144K             |    8.825G  |    0.227G      |
|   B8.safm                            |   3.752K               |   0.771G   |   17.186M      |
|    B8.safm.mfr                       |    0.56K               |    38.556M |    4.284M      |
|    B8.safm.aggr                      |    3.192K              |    0.723G  |    12.902M     |
|   B8.norm                            |   0.112K               |   0        |   0            |
|    B8.norm.weight                    |    (56,)               |            |                |
|    B8.norm.bias                      |    (56,)               |            |                |
|  B9                                  |  50.12K                |  9.661G    |  0.244G        |
|   B9.mamba                           |   46.256K              |   8.89G    |   0.227G       |
|    B9.mamba.ln_1                     |    0.112K              |    64.512M |    0           |
|    B9.mamba.self_attention           |    46.144K             |    8.825G  |    0.227G      |
|   B9.safm                            |   3.752K               |   0.771G   |   17.186M      |
|    B9.safm.mfr                       |    0.56K               |    38.556M |    4.284M      |
|    B9.safm.aggr                      |    3.192K              |    0.723G  |    12.902M     |
|   B9.norm                            |   0.112K               |   0        |   0            |
|    B9.norm.weight                    |    (56,)               |            |                |
|    B9.norm.bias                      |    (56,)               |            |                |
|  B10                                 |  50.12K                |  9.661G    |  0.244G        |
|   B10.mamba                          |   46.256K              |   8.89G    |   0.227G       |
|    B10.mamba.ln_1                    |    0.112K              |    64.512M |    0           |
|    B10.mamba.self_attention          |    46.144K             |    8.825G  |    0.227G      |
|   B10.safm                           |   3.752K               |   0.771G   |   17.186M      |
|    B10.safm.mfr                      |    0.56K               |    38.556M |    4.284M      |
|    B10.safm.aggr                     |    3.192K              |    0.723G  |    12.902M     |
|   B10.norm                           |   0.112K               |   0        |   0            |
|    B10.norm.weight                   |    (56,)               |            |                |
|    B10.norm.bias                     |    (56,)               |            |                |
|  c1                                  |  31.416K               |  7.225G    |  12.902M       |
|   c1.weight                          |   (56, 560, 1, 1)      |            |                |
|   c1.bias                            |   (56,)                |            |                |
|  c2                                  |  3.696K                |  0.839G    |  25.805M       |
|   c2.pw                              |   3.136K               |   0.723G   |   12.902M      |
|    c2.pw.weight                      |    (56, 56, 1, 1)      |            |                |
|   c2.dw                              |   0.56K                |   0.116G   |   12.902M      |
|    c2.dw.weight                      |    (56, 1, 3, 3)       |            |                |
|    c2.dw.bias                        |    (56,)               |            |                |
|  upsampler.upsampleOneStep.0         |  6.06K                 |  1.393G    |  2.765M        |
|   upsampler.upsampleOneStep.0.weight |   (12, 56, 3, 3)       |            |                |
|   upsampler.upsampleOneStep.0.bias   |   (12,)                |            |                |
|  norm                                |  0.112K                |  0         |  0             |
|   norm.weight                        |   (56,)                |            |                |
|   norm.bias                          |   (56,)                |            |                |
'''

class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_ch,
                bias=bias,
                padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'silu':
        layer = nn.SiLU(inplace)
    elif act_type == 'gelu':
        layer = nn.GELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class SLKA(nn.Module):
    def __init__(self, n_feats, k=21, d=3, shrink=0.5, scale=2):
        super().__init__()
        f = int(n_feats*shrink)
        self.head = nn.Conv2d(n_feats, f, 1)
        self.proj_2 = nn.Conv2d(f, f, kernel_size=1)
        self.activation = nn.GELU()
        self.LKA = nn.Sequential(
            nn.Conv2d(f, f, 1),
            conv_layer(f, f, k // d, dilation=1, groups=f),
            self.activation,
            nn.Conv2d(f, f, 1),
            conv_layer(f, f, 2*d-1, groups=f),
            self.activation,
            conv_layer(f, f, kernel_size=1),
        )
        self.tail = nn.Conv2d(f, n_feats, 1)
        self.scale = scale

    def forward(self, x):
        c1 = self.head(x)
        c2 = F.max_pool2d(c1, kernel_size=self.scale * 2 + 1, stride=self.scale)
        c2 = self.LKA(c2)
        c3 = F.interpolate(c2, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        a = self.tail(c3 + self.proj_2(c1))
        a = F.sigmoid(a)
        return x * a


class AFD2(nn.Module):
    def __init__(self, in_channels, conv=nn.Conv2d, attn_shrink=0.25, act_type='silu', attentionScale=2):
        super(AFD, self).__init__()

        kwargs = {'padding': 1}
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = conv_block(in_channels, self.dc, 1, act_type=act_type)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)
        self.act = activation(act_type)

        self.c2_d = conv_block(self.remaining_channels, self.dc, 1, act_type=act_type)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3,  **kwargs)

        self.c3 = conv(self.remaining_channels, self.rc, kernel_size=5,  **{'padding': 2})

        self.c3_d = conv_block(self.remaining_channels, self.dc, 1, act_type=act_type)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=7, **{'padding': 3})

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)

        self.esa = SLKA(in_channels, k=21, d=3, shrink=attn_shrink, scale=attentionScale)


    def forward(self, input):
        distilled_c1 = self.c1_d(input)
        r_c1 = self.act(self.c1_r(input))

        distilled_c2 = self.c2_d(r_c1)
        r_c2 = self.act(self.c2_r(r_c1))
        r_c3 = self.act(self.c3(r_c2))

        distilled_c3 = self.c3_d(r_c3)
        r_c4 = self.act(self.c3_r(r_c3))
        r_c5 = self.act(self.c4(r_c4))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c5], dim=1)
        out = self.c5(out)
        out_fused = self.esa(out)
        out_fused = out_fused + input

        return out_fused

class AFD(nn.Module):
    def __init__(self, in_channels, conv=nn.Conv2d, attn_shrink=0.25, act_type='silu', attentionScale=2):
        super(AFD, self).__init__()
        self.mamba = VSSBlock(hidden_dim=in_channels)
        self.safm = SAFM(dim=in_channels)
        self.norm = LayerNorm(in_channels)
        #self.esa = SLKA(in_channels, k=21, d=3, shrink=attn_shrink, scale=attentionScale)


    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.mamba(x)
        x = x.permute(0,3,1,2)
        x = x + self.safm(self.norm(x))
        return x

class myfix6(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=56, num_block=10, num_out_ch=3, upscale=3,
                 conv='BSConvU', upsampler='pixelshuffledirect', attn_shrink=0.25, act_type='gelu'):
        super(myfix6, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'DepthWiseConv':
            self.conv = DepthWiseConv
        elif conv == 'BSConvU':
            self.conv = BSConvU
        else:
            self.conv = nn.Conv2d
        self.fea_conv = self.conv(num_in_ch, num_feat, kernel_size=3, **kwargs)

        self.B1 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B2 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B3 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B4 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B5 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=3)
        self.B6 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=3)
        self.B7 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=3)
        self.B8 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=4)
        self.B9 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=4)
        self.B10 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type,attentionScale=4)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()
        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)

        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))
        self.norm = LayerNorm(num_feat)
    def forward(self, input):
        #input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)
        out_B9 = self.B9(out_B8)
        out_B10 = self.B10(out_B9)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(self.norm(out_B))
        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

if __name__ == '__main__':
    import thop
    x = torch.randn(1, 3, 48, 48).cuda()
    model = myfix6(num_feat=56).cuda()
    total_ops, total_params = thop.profile(model,(x,))
    print(total_ops, ' ',total_params)
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))