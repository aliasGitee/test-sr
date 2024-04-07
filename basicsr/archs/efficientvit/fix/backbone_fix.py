import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.efficientvit.fix.ops_fix import EfficientViTBlock as EFTB
from basicsr.archs.efficientvit.fix.ops_fix import MBConv, OpSequential,ResidualBlock, IdentityLayer, ConvLayer, DSConv

# width_list=[8, 16, 32, 64, 128],
# depth_list=[1, 2, 2, 2, 2]
class backbone_fix(nn.Module):
    def __init__(self,in_c, w_list, d_list):
        super().__init__()

        in_channels = in_c
        self.dim=dim=32
        width_list=[8, 16, 32, 64, 128] if not w_list else w_list
        depth_list=[1, 2, 2, 2, 2] if not d_list else w_list
        self.stages = []

        # input stem
        #   conv(in=3,out=8,k=3,s=2,p=same) -> norm -> act
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                kernel_size=3,
                stride=2,
                norm="bn2d",
                act_func="hswish")]
        #   dwconv(8,8,3,1,1) -> pwconv(8,8,1,1,0) -> x_copy +
        for _ in range(depth_list[0]):
            block = DSConv(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                use_bias=False,
                norm="bn2d",
                act_func=("hswish", None))
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)

        # stage1,2
        #   MBConv: conv(8,8*4,1,1,0) -> dwconv(8*4,8*4,k=3,s=2,p=same) -> pwconv(8*4, 16, 1, 1, 0) 图像缩小2倍
        #   MBConv: conv(16,16*4,1,1,0) -> dwconv(16*4,16*4,3,1,1) -> pwconv(16*4, 16, 1,1,0)
        #   MBConv: conv(16,16*4,1,1,0) -> dwconv(16*4,16*4,k=3,s=2,p=same) -> pwconv(16*4, 32, 1,1,0) 图像缩小2倍
        #   MBConv: conv(32,32*4,1,1,0) -> dwconv(32*4,32*4,3,1,1) -> pwconv(32*4, 32, 1,1,0)
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = MBConv(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride, # stride
                    expand_ratio=4,
                    use_bias=False,
                    norm="bn2d",
                    act_func=("hswish", "hswish", None))
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))


        # stage3,4
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage=[]
            block = MBConv(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=4,
                use_bias=(True, True, False),
                norm=(None, None, "bn2d"),
                act_func=("hswish", "hswish", None))
            stage.append(block)
            in_channels = w
            for _ in range(d):
                stage.append(
                    EFTB(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=4,
                        norm="bn2d",
                        act_func="hswish"))
            self.stages.append(OpSequential(stage))

        self.stages = nn.ModuleList(self.stages)
    def forward(self,x):
        '''
        x: [1, 3, 64, 64]
        torch.Size([1, 8, 32, 32])
        torch.Size([1, 16, 16, 16])
        torch.Size([1, 32, 8, 8])
        torch.Size([1, 64, 4, 4])
        torch.Size([1, 128, 2, 2])
        '''
        x = self.input_stem(x)
        print(x.shape)
        output_list = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            print(x.shape)
            if idx>0:
                output_list.append(x)
        return output_list

if __name__ == '__main__':
    model = backbone_fix(in_c=3,d_list=[1,1,1,2,2],w_list=[8,16,32,64,64])
    x = torch.randn(1,3,64,64)
    model(x)