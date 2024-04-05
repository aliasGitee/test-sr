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

class BA(nn.Module):
    def __init__(self,in_c,n_win,topk):
        super().__init__()
        self.n_win = n_win
        self.topk = topk
        #self.linear = nn.Conv2d(in_channels=in_c,out_channels=in_c*3,kernel_size=1)
        self.linear = nn.Linear(in_features=in_c,out_features=in_c*3)
    def forward(self, x):
        b,c,h,w = x.shape
        n_w, n_h=self.n_win, self.n_win
        hs,ws = h//n_h, w//n_w
        #(C,H,W) -> (S^2, WH/S^2, C)
            # (B, C, H, W) -> (B, hs*ws*C, n_h*n_w)
        x = F.unfold(x,kernel_size=(hs, ws),stride=(hs, ws))
            # (B, hs*ws*C, n_h*n_w) -> (B, n_h*n_w, hs*ws*C)
        x = x.transpose(1,2)
            # (B, n_h*n_w, hs*ws*C) -> (B, n_h*n_w, C, hs*ws)
        x = x.reshape(1, n_w*n_h,c, hs*ws)
            # (B, n_h*n_w, C, hs*ws) -> (B, n_h*n_w, hs*ws, C)
        x = x.transpose(2,3)

        # linear: C ->3*C
        x = self.linear(x)
        # q,k,v (S^2, WH/S^2, C)
        q,k,v =  x[:,:,:,:c], x[:,:,:,c:2*c], x[:,:,:,2*c:3*c]
        # (S^2, WH/S^2, C) -> (S^2, C)
        q_r,k_r = q.mean(dim=2),k.mean(dim=2)
        # Ar
        a_r = q_r@(k_r.transpose(1,2))
        a_r = F.relu(a_r)
        # Ir, topk
        # ir = torch.topk(ar, k=self.topk, dim=2)
        # ir_index = ir.indices
        k,q = (a_r @ k.reshape(b, n_w*n_h, -1)).reshape(b,n_w*n_h,-1,c),(a_r @ q.reshape(b, n_w*n_h, -1)).reshape(b,n_w*n_h,-1,c)
        o = F.relu((q@k.transpose(-1,-2)))@v

        # 复原
        o = o.transpose(2,3).reshape(b, n_h*n_w, hs*ws*c).transpose(1,2)
        o = F.fold(o,output_size=(h,w),kernel_size=(hs,ws),stride=(hs,ws))

        return o#ir_index#k_g#ir.values

class BAmutil(nn.Module):
    def __init__(self,in_c,n_win,topk,heads):
        super().__init__()
        self.n_win = n_win
        self.topk = topk
        self.heads = heads
        self.c_perh = in_c//heads
        #self.linear = nn.Conv2d(in_channels=in_c,out_channels=in_c*3,kernel_size=1)
        self.linear = nn.Linear(in_features=in_c,out_features=in_c*3)
    def forward(self, x):
        b,c,h,w = x.shape
        n_w, n_h=self.n_win, self.n_win
        hs,ws = h//n_h, w//n_w
        #(C,H,W) -> (S^2, WH/S^2, C)
            # (B, C, H, W) -> (B, hs*ws*C, n_h*n_w)
        x = F.unfold(x,kernel_size=(hs, ws),stride=(hs, ws))
            # (B, hs*ws*C, n_h*n_w) -> (B, n_h*n_w, hs*ws*C)
        x = x.transpose(1,2)
            # (B, n_h*n_w, hs*ws*C) -> (B, n_h*n_w, C, hs*ws)
        x = x.reshape(b, n_w*n_h,c, hs*ws)
            # (B, n_h*n_w, C, hs*ws) -> (B, n_h*n_w, hs*ws, C)
        x = x.transpose(2,3)

        # linear: C ->3*C
        x = self.linear(x)
        # q,k,v (S^2, WH/S^2, C) -> (S^2, WH/S^2, heads, C//heads) -> (heads, S^2, WH/S^2, C//heads)
        q,k,v =  x[:,:,:,:c], x[:,:,:,c:2*c], x[:,:,:,2*c:3*c]
        q = q.reshape(b,n_h*n_w, hs*ws, self.heads, self.c_perh).permute(0,3,1,2,4)
        k = k.reshape(b,n_h*n_w, hs*ws, self.heads, self.c_perh).permute(0,3,1,2,4)
        v = v.reshape(b,n_h*n_w, hs*ws, self.heads, self.c_perh).permute(0,3,1,2,4)

        # (heads, S^2, WH/S^2, C//heads) -> (heads, S^2, C//heads)
        q_r,k_r = q.mean(dim=3),k.mean(dim=3)
        # Ar: (heads, S^2, S^2)
        a_r = F.relu(q_r)@(F.relu(k_r.transpose(2,3)))
        #a_r = F.softmax(F.relu(a_r),dim=3)
        # Ir, topk
        # ir = torch.topk(ar, k=self.topk, dim=2)
        # ir_index = ir.indices
        k = (a_r @ k.reshape(b, self.heads,n_w*n_h, -1)).reshape(b,self.heads, n_w*n_h,-1,self.c_perh)
        q = (a_r @ q.reshape(b, self.heads,n_w*n_h, -1)).reshape(b,self.heads, n_w*n_h,-1,self.c_perh)

        o = (F.relu(q)@F.relu(k.transpose(-1,-2)))@v

        # 复原
        o = o.transpose(2,3).permute(0,2,3,1,4).reshape(b, n_h*n_w, hs*ws, c).reshape(b, n_h*n_w, hs*ws*c).transpose(1,2)
        o = F.fold(o,output_size=(h,w),kernel_size=(hs,ws),stride=(hs,ws))

        return o#ir_index#k_g#ir.values
if __name__ == '__main__':
    x = torch.randn(1,36,64,64)
    # model = BA(36,16,3)
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    model = BAmutil(in_c=36,n_win=16,topk=None,heads=4)
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
