import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.archs.biformer.bra_legacy import TopkRouting, KVGather
from basicsr.archs.efficientvit.fix.ops_fix import ConvLayer,ResidualBlock,MBConv,IdentityLayer

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)
    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        return q, kv

def WinProj(x,proj):
    if proj == 'mean':
        return x.mean([2,3])
    elif proj == 'max':
        y = rearrange(x, "b n_hw hs ws c -> b n_hw (hs ws) c")
        y = y.max(dim=2).values
        return y

class BA(nn.Module):
    def __init__(self,dim,n_win,topk,num_heads,lepe=None):
        super().__init__()
        self.n_win = n_win
        self.topk = topk
        self.dim=dim
        self.qk_dim = dim
        self.num_heads = num_heads # 必须能被dim整除
        self.scale = None or self.qk_dim ** -0.5
        self.qkv = QKVLinear(self.dim,self.qk_dim) # v的dim与qk的有区别
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=False,
                                  param_routing=False)
        self.kv_gather = KVGather(mul_weight='none')
        self.mlp = nn.Linear(dim, dim)
        self.use_lepe = lepe
        self.lepe = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=5//2, groups=dim) if lepe else nn.Identity()
        self.win_proj = 'max'

        self.qkv2 = ConvLayer(dim,3 * dim,1,use_bias=False,norm=None,act_func=None)
        # 首先聚合每个序列空间位置上相邻的信息
        # 然后对每个dim做一次独立的全连接运算
        self.aggreg = nn.ModuleList(
            [ nn.Sequential(
                    nn.Conv2d(3 * dim,3 * dim, scale,padding=scale//2,groups=3 * dim,bias=False),
                    nn.Conv2d(3 * dim, 3 * dim, 1, groups=3 * num_heads, bias=False))
                for scale in (3,5,)])
        self.proj = ConvLayer(dim * (1 + 2), dim, 1, use_bias=False, norm="ln2d",act_func=None)
    def forward(self, x):
        qkv = self.qkv2(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        out_list = []
        for elem in multi_scale_qkv:
            b,c,h,w = elem.shape
            n_w, n_h=self.n_win, self.n_win
            hs,ws = h//n_h, w//n_w
            elem = elem.permute(0,2,3,1) # (C,H,W) -> (H, W, C)
            elem = rearrange(elem, "b (n_h hs) (n_w ws) c -> b (n_h n_w) hs ws c", n_h=n_h, n_w=n_w) # (H,W,C) -> (nh*nw, hs, ws, C)
            q, kv = elem[...,:self.dim], elem[...,self.dim:]
            out = self.qkv_forward(q, kv, n_w, n_h, hs,ws)
            out_list.append(out.permute(0,3,1,2))
        out = torch.cat(out_list, dim=1)
        out = self.proj(out)
        return out

    def qkv_forward(self, q, kv, n_w, n_h, hs,ws):
        q_pix = rearrange(q, "b n_hw hs ws c -> b n_hw (hs ws) c") # (nh*nw, hs, ws, C) -> (nh*nw, hs*ws, C)
        kv_pix = rearrange(kv, "b n_hw hs ws c -> b n_hw (hs ws) c") # (nh*nw, hs, ws, C) -> (nh*nw, hs*ws, C)

        # 窗口内通道维度平均
        #   q_win: (nh*nw, q_C)
        #   k_win: (nh*nw, k_C)
        #q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])
        q_win, k_win = WinProj(q,self.win_proj), WinProj(kv[..., 0:self.qk_dim], self.win_proj)

        # 窗口间自注意力，获得attn矩阵的topk分数和索引
        r_weight, r_idx = self.router(q_win, k_win)
        # k、v聚合: (b, nh*nw, topk, hs*ws, k_C+v_C)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(b, nh*nw, topk, hs*ws, k_C+v_C)
        # (b, nh*nw, topk, hs*ws, k_C)
        # (b, nh*nw, topk, hs*ws, v_C)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        # (b*nh*nw, heads, topk*hs*ws, k_C//heads)
        # (b*nh*nw, heads, topk*hs*ws, v_C//heads)
        # (b*nh*nw, heads, hs*ws, q_C//heads)
        k_pix_sel = rearrange(k_pix_sel, 'b n_hw topk s_hw (heads c) -> (b n_hw) heads c (topk s_hw)', heads=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'b n_hw topk s_hw (heads c) -> (b n_hw) heads (topk s_hw) c', heads=self.num_heads)
        q_pix = rearrange(q_pix, 'b n_hw s_hw (heads c) -> (b n_hw) heads s_hw c', heads=self.num_heads)

        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = F.softmax(attn_weight, dim=-1)
        out = attn_weight @ v_pix_sel
        out = rearrange(out, '(b n_h n_w) heads (hs ws) c -> b (n_h hs) (n_w ws) (heads c)', n_h=n_h,n_w=n_w, hs=hs,ws=ws)

        out = self.mlp(out)

        if self.use_lepe:
            # 单独对v进行一次卷积变换
            lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'b (n_h n_w) hs ws c -> b c (n_h hs) (n_w ws)', n_h=n_h, n_w=n_w).contiguous())
            lepe = rearrange(lepe, 'b c (n_h hs) (n_w ws) -> b (n_h hs) (n_w ws) c', n_h=n_h, n_w=n_w)
            out = out + lepe
        return out
    def main_forward(self, x):
        b,c,h,w = x.shape
        n_w, n_h=self.n_win, self.n_win
        hs,ws = h//n_h, w//n_w

        x = x.permute(0,2,3,1) # (C,H,W) -> (H, W, C)
        x = rearrange(x, "b (n_h hs) (n_w ws) c -> b (n_h n_w) hs ws c", n_h=n_h, n_w=n_w) # (H,W,C) -> (nh*nw, hs, ws, C)
        # q: (nh*nw, hs, ws, q_C)
        # kv: (nh*nw, hs, ws, k_C+v_C), q_C == k_C
        q, kv = self.qkv(x)

        q_pix = rearrange(q, "b n_hw hs ws c -> b n_hw (hs ws) c") # (nh*nw, hs, ws, C) -> (nh*nw, hs*ws, C)
        kv_pix = rearrange(kv, "b n_hw hs ws c -> b n_hw (hs ws) c") # (nh*nw, hs, ws, C) -> (nh*nw, hs*ws, C)

        # 窗口内通道维度平均
        #   q_win: (nh*nw, q_C)
        #   k_win: (nh*nw, k_C)
        #q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])
        q_win, k_win = WinProj(q,self.win_proj), WinProj(kv[..., 0:self.qk_dim], self.win_proj)

        # 窗口间自注意力，获得attn矩阵的topk分数和索引
        r_weight, r_idx = self.router(q_win, k_win)
        # k、v聚合: (b, nh*nw, topk, hs*ws, k_C+v_C)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(b, nh*nw, topk, hs*ws, k_C+v_C)
        # (b, nh*nw, topk, hs*ws, k_C)
        # (b, nh*nw, topk, hs*ws, v_C)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        # (b*nh*nw, heads, topk*hs*ws, k_C//heads)
        # (b*nh*nw, heads, topk*hs*ws, v_C//heads)
        # (b*nh*nw, heads, hs*ws, q_C//heads)
        k_pix_sel = rearrange(k_pix_sel, 'b n_hw topk s_hw (heads c) -> (b n_hw) heads c (topk s_hw)', heads=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'b n_hw topk s_hw (heads c) -> (b n_hw) heads (topk s_hw) c', heads=self.num_heads)
        q_pix = rearrange(q_pix, 'b n_hw s_hw (heads c) -> (b n_hw) heads s_hw c', heads=self.num_heads)

        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = F.softmax(attn_weight, dim=-1)
        out = attn_weight @ v_pix_sel
        out = rearrange(out, '(b n_h n_w) heads (hs ws) c -> b (n_h hs) (n_w ws) (heads c)', n_h=n_h,n_w=n_w, hs=hs,ws=ws)

        out = self.mlp(out)

        if self.use_lepe:
            # 单独对v进行一次卷积变换
            lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'b (n_h n_w) hs ws c -> b c (n_h hs) (n_w ws)', n_h=n_h, n_w=n_w).contiguous())
            lepe = rearrange(lepe, 'b c (n_h hs) (n_w ws) -> b (n_h hs) (n_w ws) c', n_h=n_h, n_w=n_w)
            out = out + lepe
        return out

class BABlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expand_ratio: float = 4,
        norm="ln2d",
        act_func="hswish",
    ):
        super(BABlock, self).__init__()
        self.context_module = ResidualBlock(
            BA(dim=in_channels,topk=3,n_win=8,num_heads=4),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x

if __name__ == '__main__':
    model = BABlock(in_channels=36)#BA(dim=36,topk=3,n_win=4,num_heads=4)
    x = torch.randn(1,36,48,48)
    import thop
    total_ops, total_params = thop.profile(model,(x,))
    print(total_ops,' ',total_params)