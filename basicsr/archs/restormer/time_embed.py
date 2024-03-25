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

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedHead_LWTDM(nn.Module):
    def __init__(self, inner_channel):
        super().__init__()
        self.time_embed = TimeEmbedding(inner_channel)
        self.mlp = nn.Sequential(
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
    def forward(self, t):
        return self.mlp(self.time_embed(t))

class TimeMlpMid_LWTDM(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.mlp = nn.Sequential(
                Swish(),
                nn.Linear(dim, dim_out)
            )
    def forward(self,time_embed):
        return self.mlp(time_embed)


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

