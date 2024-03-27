import torch
import torch.nn as nn
from basicsr.archs.efficientvit.nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
    UpSampleLayer
)
from basicsr.archs.efficientvit.backbone import efficientvit_backbone_b0

class EFTBlocks(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, dim=32,
                 expand_ratio=4, # 作用于build_local_block，用于选择 MBConv or DSConv
                 norm="bn2d", act_func="hswish",fewer_norm=True,d=1,down=False):
        super().__init__()
        self.MBConv = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2 if down else 1,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False) if fewer_norm else False,
            norm=(None, None, norm) if fewer_norm else norm,
            act_func=(act_func, act_func, None),
        )
        self.EfficientVITBlocks  = nn.ModuleList([
            EfficientViTBlock(
                in_channels=out_channels,
                dim=dim,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func) for _ in range(d)]
        )
    def forward(self, x):
        x = self.MBConv(x)
        for block in self.EfficientVITBlocks:
            x = block(x)
        return x

class EFVIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientvit_backbone_b0()

    def fea_forward(self,x):
        x_dict = self.backbone(x)
        for k,v in x_dict.items():
            print(k,v.shape)
        return 0
    def forward(self,x):
        return self.fea_forward(x)

if __name__ == '__main__':
    model = EFTBlocks(in_channels=64,out_channels=32)
    x = torch.randn(1,64,192,192)
    print(model(x).shape)