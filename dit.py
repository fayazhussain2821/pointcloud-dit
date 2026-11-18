"""DiT (Diffusion Transformer, Peebles & Xie 2023) for point clouds.

Differences from the image DiT:
  - No patches: every point is one token. A linear layer turns its (x, y) into a vector.
  - No position embeddings. A point cloud is a set: shuffling its points gives the same shape, so
    the model must not depend on their order. Without position embeddings, attention and the MLPs
    treat every token the same way, so shuffling the input points shuffles the output the same way
    (permutation equivariance; check.py tests this).
  - The output is one 2D vector per point instead of a patch of pixels.
  - Unconditional: no class labels, only the time t.
Everything else is as in DiT: sine-cosine time embedding, adaLN-Zero blocks, a conditioned final layer.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim=256, max_period=10000):
    """Sine/cosine features of t, as in DiT. DiT's t runs 0-999 and ours runs 0-1, so scale it up."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    angles = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [part.reshape(B, N, self.num_heads, -1).transpose(1, 2) for part in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    """Attention + MLP. The conditioning vector c sets a shift, scale and gate for each of the two parts."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c)[:, None].chunk(6, dim=-1)
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1)
        x = x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)
        return x


class PointDiT(nn.Module):
    def __init__(self, point_dim=2, dim=384, depth=12, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.embed = nn.Linear(point_dim, dim)
        self.time_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth))
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.final_linear = nn.Linear(dim, point_dim)
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for layer in (self.time_mlp[0], self.time_mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        # The "Zero" in adaLN-Zero: every block and the final layer start out doing nothing.
        for layer in [block.adaLN[-1] for block in self.blocks] + [self.final_adaLN[-1], self.final_linear]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, t):
        """x: (B, num_points, 2) noisy clouds, t: (B,) times. Returns (B, num_points, 2)."""
        tokens = self.embed(x)
        c = self.time_mlp(timestep_embedding(t))
        for block in self.blocks:
            tokens = block(tokens, c)
        shift, scale = self.final_adaLN(c)[:, None].chunk(2, dim=-1)
        return self.final_linear(self.final_norm(tokens) * (1 + scale) + shift)
