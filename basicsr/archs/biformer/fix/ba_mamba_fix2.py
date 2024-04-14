import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.archs.biformer.bra_legacy import TopkRouting, KVGather
from basicsr.archs.efficientvit.fix.ops_fix import ConvLayer,ResidualBlock,MBConv,IdentityLayer
from basicsr.archs.vmamba.fix.mamba_sys_fix import SS2D
from basicsr.archs.safmn.SAFMN import LayerNorm

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
        self.win_proj = 'mean'

        self.ss2d = SS2D(d_model=dim*topk)
        self.ss2d_k = SS2D(d_model=dim*topk)
        self.norm = LayerNorm(dim,data_format="channels_last")
        self.act = nn.GELU()

    def ss2d_forward(self,v_pix_sel,ss2d, b,n_h,n_w,hs,ws):
        '''
        ss2d
        '''
        # (b, nh*nw, hs*ws, topk*v_C)
        v_pix_sel = v_pix_sel.transpose(2,3).reshape(b, n_h*n_w, hs*ws, -1)
        v_pix_sel = rearrange(v_pix_sel, "b (n_h n_w) (hs ws) c -> b (n_h hs) (n_w ws) c", n_h=n_h,n_w=n_w, hs=hs,ws=ws)
        v_pix_sel = ss2d(v_pix_sel)
        v_pix_sel = rearrange(v_pix_sel, "b (n_h hs) (n_w ws) c -> b (n_h n_w) (hs ws) c ", n_h=n_h,n_w=n_w, hs=hs,ws=ws)
        v_pix_sel = v_pix_sel.reshape(b, n_h*n_w, hs*ws, self.topk, -1)
        # (b, nh*nw, topk, hs*ws, v_C)
        v_pix_sel = v_pix_sel.transpose(2,3)
        return v_pix_sel

    def forward(self, x):
        b,c,h,w = x.shape
        n_w, n_h=self.n_win, self.n_win
        hs,ws = h//n_h, w//n_w

        x = x.permute(0,2,3,1) # (C,H,W) -> (H, W, C)
        x = rearrange(x, "b (n_h hs) (n_w ws) c -> b (n_h n_w) hs ws c", n_h=n_h, n_w=n_w) # (H,W,C) -> (nh*nw, hs, ws, C)
        print('x: ',x.shape)
        # q: (nh*nw, hs, ws, q_C)
        # kv: (nh*nw, hs, ws, k_C+v_C), q_C == k_C
        q, kv = self.qkv(x)
        print('q: ',q.shape,' ','kv: ',kv.shape)

        # 窗口内通道维度平均
        #   q_win: (nh*nw, q_C)
        #   k_win: (nh*nw, k_C)
        #q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])
        q_win, k_win = WinProj(q,self.win_proj), WinProj(kv[..., 0:self.qk_dim], self.win_proj)
        print('q_win: ',q_win.shape)
        # 窗口间自注意力，获得attn矩阵的topk分数和索引
        r_weight, r_idx = self.router(q_win, k_win)
        print(r_weight.shape)
        #print(r_idx)
        #print(r_weight)
        #print(r_weight.sort(dim=2).indices)

        q_pix = rearrange(q, "b n_hw hs ws c -> b n_hw (hs ws) c") # (nh*nw, hs, ws, C) -> (nh*nw, hs*ws, C)
        kv_pix = rearrange(kv, "b n_hw hs ws c -> b n_hw (hs ws) c") # (nh*nw, hs, ws, C) -> (nh*nw, hs*ws, C)

        # k、v聚合: (b, nh*nw, topk, hs*ws, k_C+v_C)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(b, nh*nw, topk, hs*ws, k_C+v_C)

        # (b, nh*nw, topk, hs*ws, k_C)
        # (b, nh*nw, topk, hs*ws, v_C)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1) # 使用v_pix_sel

        '''
        ss2d
        '''
        v_pix_sel = self.ss2d_forward(v_pix_sel, self.ss2d, b, n_h, n_w, hs, ws)
        k_pix_sel = self.ss2d_forward(k_pix_sel, self.ss2d_k, b, n_h, n_w, hs, ws)


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

        if self.use_lepe:
            # 单独对v进行一次卷积变换
            lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'b (n_h n_w) hs ws c -> b c (n_h hs) (n_w ws)', n_h=n_h, n_w=n_w).contiguous())
            lepe = rearrange(lepe, 'b c (n_h hs) (n_w ws) -> b (n_h hs) (n_w ws) c', n_h=n_h, n_w=n_w)
            out = out + lepe

        out = self.mlp(out)
        return self.act(self.norm(out))

if __name__ == '__main__':
    model = BA(dim=16, topk=4, n_win=8, num_heads=4)
    x = torch.randn(1,16,48,48)
    # import thop
    # total_ops, total_params = thop.profile(model,(x,))
    # print(total_ops,' ',total_params)
    print(model(x).shape)