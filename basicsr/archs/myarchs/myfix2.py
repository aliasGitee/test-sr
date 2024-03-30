from basicsr.archs.emt.emt import MixedTransformerBlock,Swish
import torch
import torch.nn as nn

class myfix2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(*[MixedTransformerBlock(dim=60, num_layer=6, num_heads=3, num_GTLs=2,
                                    window_list=[ [32, 8],[8, 32] ], shift_list=[ [16, 4],[4, 16]],
                                    mlp_ratio=2, act_layer=Swish) for _ in range(6)])
        self.conv = nn.Conv2d(in_channels=3,out_channels=60,kernel_size=3,padding=1)
        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels=60,out_channels=12,kernel_size=3,padding=1),
            nn.PixelShuffle(upscale_factor=2))
    def forward(self,x):
        x = self.conv_last(self.block(self.conv(x)))
        return x

if __name__ == '__main__':
    model = myfix2()
    x = torch.randn(1,3,48,48)
    print(model(x).shape)