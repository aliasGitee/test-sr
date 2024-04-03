import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from basicsr.archs.efficientvit.nn.ops import ConvLayer,LiteMLA
from basicsr.archs.efficientvit.nn.act import build_act
from basicsr.archs.efficientvit.nn.norm import build_norm
from basicsr.archs.efficientvit.utils import get_same_padding, list_sum, resize, val2list, val2tuple

from basicsr.archs.biformer.bra_legacy import BiLevelRoutingAttention as BA

# in_c:9, dim:9//3
class LiteMLAFix(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(LiteMLAFix, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio) # in_c//dim * 1.0 = 4
        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        # conv1*1: (b,c,h,w) -> (b,3*c,h,w)
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        # 首先聚合每个序列空间位置上相邻的信息
        # 然后对每个dim做一次独立的全连接运算
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

        self.BA = BA(dim=3*total_dim, n_win=4, num_heads=heads)
        '''
        x = x.permute(0, 2, 3, 1)
    model = BiLevelRoutingAttention(dim=36,n_win=4,num_heads=4)
        '''
    @autocast(enabled=False)
    # 假设in_c=12，乘以3就是36，假设dim=3，dim表示划分qkv、划分头后的向量维度，那heads就是4
    # 输入(b, 36, h, w)，输出(b, 36, h, w)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        # qkv : (b, 36, h, w)
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        # (b, heads:4, 3*dim, len_seq:h*w)
        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        # (b, heads:4, len_seq:h*w, 3*dim)
        qkv = torch.transpose(qkv, -1, -2)

        # q,k,v: (b, -1, h*w, dim)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )

        # lightweight linear attention
        # q -> relu(q)
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        # (q@k.T)@v
        trans_k = k.transpose(-1, -2)
        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        # 恢复形状
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        # (b, 36, h, w)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        # conv1*1: (b,c,h,w) -> (b,3*c,h,w)
        qkv = self.qkv(x)

        multi_scale_qkv = [self.BA(qkv.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))

        out_list = []
        for elem in multi_scale_qkv:
            out = self.relu_linear_att(elem)
            out_list.append(out)

        # cat(qkv, dwconv(qkv)) -> (b,6*c,h,w)
        out = torch.cat(out_list, dim=1)
        out = self.proj(out)
        return out

if __name__ == '__main__':
    x = torch.randn(1,12,12,12)
    model1 = LiteMLAFix(in_channels=12,out_channels=12,dim=3,scales=(5,3))
    model2 = LiteMLA(in_channels=12,out_channels=12,dim=3,scales=(5,3))
    import thop
    total_ops, total_params = thop.profile(model1, (x,))
    print(total_ops,' ', total_params)
    total_ops, total_params = thop.profile(model2, (x,))
    print(total_ops,' ', total_params)
    #print(model1(x)==model2(x))