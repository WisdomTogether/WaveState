import torch
import torch.nn as nn
from timm.models.layers import DropPath

from mamba_ssm import Mamba


class PSSLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba = Mamba(
            d_model=input_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.mamba2 = Mamba(
            d_model=input_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, L, C = x.shape

        x_flat = x.transpose(1, 2)  # Shape: [B, C, L]

        if x_flat.size(-1) != self.input_dim:
            x_flat = x_flat.transpose(1, 2)  # [B, L, C] -> [B, C, L] to match input_dim

        x_norm = self.norm(x_flat)

        alpha = torch.sigmoid(x_norm.mean(dim=-1, keepdim=True)) * self.skip_scale

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba2(self.mamba(x1)) + alpha * x1
        x_mamba2 = self.mamba2(self.mamba(x2)) + alpha * x2
        x_mamba3 = self.mamba2(self.mamba(x3)) + alpha * x3
        x_mamba4 = self.mamba2(self.mamba(x4)) + alpha * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)

        out = x_mamba.transpose(1, 2).reshape(B, self.output_dim, L)
        return out

class PSSLayer2(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba = Mamba(
            d_model=input_dim // 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.mamba2 = Mamba(
            d_model=input_dim // 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, L, C = x.shape

        x_flat = x.transpose(1, 2)  # Shape: [B, C, L]

        if x_flat.size(-1) != self.input_dim:
            x_flat = x_flat.transpose(1, 2)  # [B, L, C] -> [B, C, L] to match input_dim

        x_norm = self.norm(x_flat)

        alpha = torch.sigmoid(x_norm.mean(dim=-1, keepdim=True)) * self.skip_scale

        x1, x2 = torch.chunk(x_norm, 2, dim=2)
        x_mamba1 = self.mamba2(self.mamba(x1)) + alpha * x1
        x_mamba2 = self.mamba2(self.mamba(x2)) + alpha * x2
        x_mamba = torch.cat([x_mamba1, x_mamba2], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)

        out = x_mamba.transpose(1, 2).reshape(B, self.output_dim, L)
        return out
    

class PSSLayer1(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba = Mamba(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.mamba2 = Mamba(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, L, C = x.shape

        x_flat = x.transpose(1, 2)  # Shape: [B, C, L]

        if x_flat.size(-1) != self.input_dim:
            x_flat = x_flat.transpose(1, 2)  # [B, L, C] -> [B, C, L] to match input_dim

        x_norm = self.norm(x_flat)

        alpha = torch.sigmoid(x_norm.mean(dim=-1, keepdim=True)) * self.skip_scale

        x_mamba = self.mamba2(self.mamba(x_norm)) + alpha * x_norm


        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)

        out = x_mamba.transpose(1, 2).reshape(B, self.output_dim, L)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PSSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.mtsc = PSSLayer1(input_dim=dim, output_dim=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        shortcut = x

        # 归一化和注意力机制
        x = self.norm1(x)
        x = self.mtsc(x)
        x = shortcut + self.drop_path(x).permute(0, 2, 1)

        # 归一化和 MLP 层
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# 测试代码
# x = torch.randn(16, 2, 8).to('cuda:0')  # 输入形状为 (B, L, C)
# block = MLLABlock(dim=8, num_heads=2).to('cuda:0')
# output = block(x)
# print(output.shape)  # 输出形状应为 (16, 2, 8)