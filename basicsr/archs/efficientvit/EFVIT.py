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

# x -> (MBConv -> EfficientVIT Module) -> y
class EfficientVITStage(nn.Module):
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

# x -> n * (MBConv -> EfficientVIT Module) -> upsample -> y
class EfficientVITStages(nn.Module):
    def __init__(self,in_channels=32, dim=32,
                 expand_ratio=4,norm="bn2d", act_func="hswish",fewer_norm=True,d=1,down=True, num_stage=2):
        super().__init__()
        self.in_c = in_channels
        self.stage_list = []
        width_list = [in_channels*2, in_channels*4]
        for i in width_list:
            self.stage_list.append(
                EfficientVITStage(in_channels, i,
                          dim,expand_ratio, norm, act_func,fewer_norm,d,down))
            in_channels = i
        self.Stages = nn.ModuleList(self.stage_list)
        #self.conv1 = ConvLayer(in_channels=self.in_c*4, out_channels=self.in_c)

        self.conv1_stage2 = ConvLayer(in_channels=self.in_c, out_channels=self.in_c,norm='ln')
        self.conv1_stage3 = ConvLayer(in_channels=self.in_c*2, out_channels=self.in_c,norm='ln')
        self.conv1_stage4 = ConvLayer(in_channels=self.in_c*4, out_channels=self.in_c,norm='ln')
        self.upsample_stage3 = UpSampleLayer(factor=2**(num_stage-1))
        self.upsample_stage4 = UpSampleLayer(factor=2**num_stage)
    def forward(self, x):
        out_list = [x]
        for stage in self.Stages:
            x = stage(x)
            out_list.append(x)
        out_stage2 = self.conv1_stage2(out_list[0])
        out_stage3 = self.upsample_stage3(self.conv1_stage3(out_list[1]))
        out_stage4 = self.upsample_stage4(self.conv1_stage4(out_list[2]))
        out = torch.cat([out_stage2,out_stage3,out_stage4],dim=1)
        return out

if __name__ == '__main__':
    import thop
    model2 = EfficientViTBlock(
                in_channels=128,
                dim=32,
                expand_ratio=4,
                norm="bn2d", act_func="hswish")
    x2 = torch.randn(1,128,12,12)


    model = EfficientVITStages(in_channels=16,dim=16,down=True,num_stage=2)

    x = torch.randn(1,16,48,48)
    total_ops, total_params = thop.profile(model, (x,))
    #print(model(x).shape)
    print(total_ops,' ', total_params)
