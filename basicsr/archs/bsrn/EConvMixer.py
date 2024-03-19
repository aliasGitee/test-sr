import torch.nn as nn
import math


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

class EConvMixer(nn.Module):
    def __init__(self,in_c,out_c, kernel_size=9):
        super().__init__()

        self.pw = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.dw = nn.Conv2d(
            in_channels=out_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            padding="same",
            groups=out_c,
            bias=True,
        )
        self.pw2 = nn.Conv2d(
            in_channels=out_c,
            out_channels=out_c,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.ddw = nn.Conv2d(
            in_channels=out_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            padding="same",
            dilation=3,
            groups=out_c,
            bias=True,
        )
        self.act = nn.GELU()

    def forward(self,x):
        x = self.pw(x)
        x = self.dw(x)
        x = self.pw2(x)
        x = self.act(x)
        x = self.ddw(x)
        x = self.act(x)
        return x

