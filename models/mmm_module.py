import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MMMLayer(nn.Module):
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
