import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.msnlan.utils.tools import extract_image_patches, \
    reduce_mean, reduce_sum, same_padding
from basicsr.archs.msnlan.common import stdv_channels


def default_conv(in_channels, out_channels, kernel_size, group_num=1, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, groups=group_num,
        padding=(kernel_size//2), stride=stride, bias=bias)

class My_Block(nn.Module):
    def __init__(self, conv, n_feats, bias=True, act=nn.GELU(), res_scale=1):
        super(My_Block, self).__init__()
        self.res_scale = res_scale
        ms_body1 = []
        ms_body1.append(conv(n_feats, n_feats, 3, group_num=n_feats, bias=bias))
        ms_body1.append(conv(n_feats, n_feats, 1, bias=bias))
        ms_body1.append(act)

        ms_body2 = []
        ms_body2.append(conv(n_feats, n_feats, 3, group_num=n_feats, bias=bias))
        ms_body2.append(conv(n_feats, n_feats, 1, bias=bias))
        ms_body2.append(act)

        ms_body3=[]
        ms_body3.append(conv(n_feats, n_feats, 3, group_num=n_feats, bias=bias))
        ms_body3.append(conv(n_feats, n_feats, 1, bias=bias))
        ms_body3.append(act)

        self.branch1 = nn.Sequential(*ms_body1)
        self.branch2 = nn.Sequential(*ms_body2)
        self.branch3 = nn.Sequential(*ms_body3)
        self.fusion = conv(n_feats * 3, n_feats, 1, bias=bias)
        # self.act = act
        # self.CA = CAer(n_feats)
        # self.esa = ESA(n_feats, nn.Conv2d)
        self.CCA = CCALayer(n_feats)

    def forward(self, x):
        res = x
        x1 = self.branch1(x)
        x2 =self.branch2(x1)
        x3 =self.branch3(x2)
        bag = torch.cat([x1, x2, x3], dim=1)
        bag1 = self.fusion(bag)   # 1*1 Conv
        out = self.CCA(bag1)
        # out = self.esa(out1)
        output = res + out
        # output = out.mul(self.res_scale)
        # output = self.fusion(output)
        # output = self.act(output)

        # res = self.SA(res)*self.res_scale+res
        return output


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x*y

if __name__ == '__main__':
    x = torch.randn(1,36,64,64)
    model = My_Block(n_feats=36,conv=default_conv)
    print(model(x).shape)