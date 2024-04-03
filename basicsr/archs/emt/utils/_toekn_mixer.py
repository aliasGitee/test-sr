import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange
import math
from basicsr.archs.emt.utils import MeanShift, Conv2d1x1, Conv2d3x3, ShiftConv2d1x1

__all__ = ['PixelMixer', 'SWSA']
class PixelMixer(nn.Module):
    def __init__(self, planes: int, mix_margin: int = 1) -> None:
        super(PixelMixer, self).__init__()

        assert planes % 5 == 0

        self.planes = planes
        self.mix_margin = mix_margin  # 像素的偏移量
        self.mask = nn.Parameter(torch.zeros((self.planes, 1, mix_margin * 2 + 1, mix_margin * 2 + 1)),
                                 requires_grad=False)

        self.mask[3::5, 0, 0, mix_margin] = 1.
        self.mask[2::5, 0, -1, mix_margin] = 1.
        self.mask[1::5, 0, mix_margin, 0] = 1.
        self.mask[0::5, 0, mix_margin, -1] = 1.
        self.mask[4::5, 0, mix_margin, mix_margin] = 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mix_margin
        x = f.conv2d(input=f.pad(x, pad=(m, m, m, m), mode='circular'),
                     weight=self.mask, bias=None, stride=(1, 1), padding=(0, 0),
                     dilation=(1, 1), groups=self.planes)
        return x

class SWSA(nn.Module):
    r"""
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        attn_layer (list): Layers used to calculate attn
        proj_layer (list): Layers used to proj output
        window_list (tuple): List of window sizes. Input will be equally divided
            by channel to use different windows sizes
        shift_list (tuple): list of shift sizes
        return_attns (bool): Returns attns or not

    Returns:
        b c h w -> b c h w
    """

    def __init__(self, dim: int,
                 num_heads: int,
                 attn_layer: list = None,
                 proj_layer: list = None,
                 window_list: tuple = ((8, 8),),
                 shift_list: tuple = None,
                 return_attns: bool = False,
                 ) -> None:
        super(SWSA, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.return_attns = return_attns

        self.window_list = window_list
        if shift_list is not None:
            assert len(shift_list) == len(window_list)
            self.shift_list = shift_list
        else:
            self.shift_list = ((0, 0),) * len(window_list)

        self.attn = nn.Sequential(*attn_layer if attn_layer is not None else [nn.Identity()])
        self.proj = nn.Sequential(*proj_layer if proj_layer is not None else [nn.Identity()])

    @staticmethod
    def check_image_size(x: torch.Tensor, window_size: tuple) -> torch.Tensor:
        _, _, h, w = x.size()
        windows_num_h = math.ceil(h / window_size[0])
        windows_num_w = math.ceil(w / window_size[1])
        mod_pad_h = windows_num_h * window_size[0] - h
        mod_pad_w = windows_num_w * window_size[1] - w
        return f.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    '''
    SWSA(dim=dim, num_heads=num_heads,
        attn_layer=[Conv2d1x1(dim, dim * 2),
                    nn.BatchNorm2d(dim * 2)],


        proj_layer=[Conv2d1x1(dim, dim)],
        window_list=[ [32, 8],[8, 32] ],
        shift_list=shift_list if (_ + 1) % 2 == 0 else None) [ [16, 4],[4, 16]]
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor or tuple:
        r"""
        Args:
            x: b c h w

        Returns:
            b c h w -> b c h w
        """
        # calculate qkv
        qkv = self.attn(x)  #通道数x2
        _, C, _, _ = qkv.size()

        # split channels #通道分割 1/2，组成list，表示不同的窗口划分方式
        qkv_list = torch.split(qkv, [C // len(self.window_list)] * len(self.window_list), dim=1)


        output_list = list()
        if self.return_attns:
            attn_list = list()

        # 输入，窗口划分方式
        for attn_slice, window_size, shift_size in zip(qkv_list, self.window_list, self.shift_list):
            _, _, h, w = attn_slice.size()

            # 填充图像，使其h，w能被window_size整除
            attn_slice = self.check_image_size(attn_slice, window_size)
            #print(attn_slice.shape) # b, 60, 32, 32

            # roooll! #图像内区域位置移动
            if shift_size != (0, 0):
                attn_slice = torch.roll(attn_slice, shifts=shift_size, dims=(2, 3))

            # cal attn # 通道维度按注意力头和qv分割，head=3，num_qv=2 -> (q or v).C = 60/(2*3) = 10
            _, _, H, W = attn_slice.size()
            q, v = rearrange(attn_slice, 'b (qv head c) (nh ws1) (nw ws2) -> qv (b head nh nw) (ws1 ws2) c',
                             qv=2, head=self.num_heads,
                             ws1=window_size[0], ws2=window_size[1])
            #print(q.shape)
            attn = (q @ q.transpose(-2, -1))
            attn = f.softmax(attn, dim=-1)
            if self.return_attns:
                attn_list.append(attn.reshape(self.num_heads, -1,
                                              window_size[0] * window_size[1],
                                              window_size[0] * window_size[1]))  # noqa
            output = rearrange(attn @ v, '(b head nh nw) (ws1 ws2) c -> b (head c) (nh ws1) (nw ws2)',
                               head=self.num_heads,
                               nh=H // window_size[0], nw=W // window_size[1],
                               ws1=window_size[0], ws2=window_size[1])

            # roooll back!
            if shift_size != (0, 0):
                output = torch.roll(output, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))

            output_list.append(output[:, :, :h, :w])

        # proj output
        output = self.proj(torch.cat(output_list, dim=1))

        if self.return_attns:
            return output, attn_list
        else:
            return output

if __name__ == '__main__':
    # dim = 60
    # window_list=[ [4, 2],[2, 4] ]
    # shift_list=[ [4, 2],[2, 4]]

    # model = SWSA(dim=dim, num_heads=3,
    #     attn_layer=[Conv2d1x1(dim, dim * 2),
    #                 nn.BatchNorm2d(dim * 2)],
    #     proj_layer=[Conv2d1x1(dim, dim)],
    #     window_list=window_list,
    #     shift_list=shift_list)
    # x = torch.randn(1,60,32,32)
    # print(model(x).shape)
    x = torch.randn(1,36,12,12)
    model = PixelMixer(planes=40)
    print(model(x).shape)