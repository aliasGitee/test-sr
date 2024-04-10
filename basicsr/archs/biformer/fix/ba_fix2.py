'''
划分区域：(C,H,W) -> (S^2, WH/S^2, C)
x = torch.randn(1,3,6,6)
b,c,h,w = x.shape
n_w,n_h=3,3 # h和w方向上的窗口数量
x = F.unfold(x,kernel_size=(h//n_h, w//n_w),stride=(h//n_h, w//n_w))
x = x.transpose(1,2).reshape(1,n_w*n_h,c,(h*w)//(n_h*n_w)).transpose(2,3)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.archs.biformer.bra_legacy import TopkRouting, KVGather

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
    def forward(self, x):
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

if __name__ == '__main__':
    model = BA(dim=36,topk=3,n_win=4,num_heads=4)
    x = torch.randn(1,36,48,48)
    print(model(x).shape)