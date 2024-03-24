import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# 最开始使用
class TimeEmbedHead(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.time_embed = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, t):
        return self.mlp(self.time_embed(t))

# 中间block使用
class TimeMlpMid(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.mlp = nn.Sequential(
                Mish(),
                nn.Linear(dim, dim_out)
            )
    def forward(self,time_embed):
        return self.mlp(time_embed)
