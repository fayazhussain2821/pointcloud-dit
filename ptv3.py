"""PointTransformerV3 (Wu et al., CVPR 2024) for flow matching on small 2D point clouds.

Kept from PTv3:
  - Serialization: points are sorted along space-filling curves (Z-order and Hilbert, each also with
    x and y swapped), so points that are close in space end up close in the list.
  - Patch attention: attention runs inside groups of consecutive points in that list, not over all pairs.
  - Order shuffling: successive blocks use different curves, so patches differ from block to block.
  - A U-Net shape: an encoder that reduces the number of points and a decoder with skip connections.
  - A conditional position encoding (CPE) before attention in every block.

Changed for this project:
  - Pooling merges neighbouring pairs along the curve instead of merging points in a grid cell. Every
    cloud then keeps the same number of points at each stage (256, 128, 64, 32), so clouds can be batched.
  - The CPE is a 1D convolution along the curve instead of a sparse 3D convolution on a grid.
  - The time t conditions every block with adaLN-Zero, exactly as in dit.py, so that the comparison
    with the DiT is about the backbone and not about how time enters the network.

Shuffling the input points shuffles the output in the same way, because the curves depend only on the
point positions (check_ptv3.py tests this).
"""

import torch
import torch.nn as nn

from dit import Attention, timestep_embedding

GRID_BITS = 16  # each axis is divided into 2^16 cells, so two points almost never share a cell


def spread_bits(v):
    """Insert a zero between the bits of a 16-bit integer: abcd -> 0a0b0c0d."""
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    return (v | (v << 1)) & 0x55555555


def z_order(x, y):
    """Position along the Z-order (Morton) curve of integer grid cells."""
    return spread_bits(x) | (spread_bits(y) << 1)


def hilbert(x, y, bits=GRID_BITS):
    """Position along the Hilbert curve of integer grid cells."""
    size = 1 << bits
    code = torch.zeros_like(x)
    for k in range(bits - 1, -1, -1):
        s = 1 << k
        rx, ry = (x & s) > 0, (y & s) > 0
        code = code + s * s * ((3 * rx.long()) ^ ry.long())
        flip = rx & ~ry
        x, y = torch.where(flip, size - 1 - x, x), torch.where(flip, size - 1 - y, y)
        x, y = torch.where(~ry, y, x), torch.where(~ry, x, y)
    return code


@torch.no_grad()
def serial_orders(coords):
    """Four sort orders of the points (B, N, 2), one per curve. Each is a (B, N) tensor of indices."""
    low = coords.min(dim=1, keepdim=True).values
    extent = (coords.max(dim=1, keepdim=True).values - low).max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    cells = (((coords - low) / extent) * ((1 << GRID_BITS) - 1)).long().clamp(0, (1 << GRID_BITS) - 1)
    x, y = cells[..., 0], cells[..., 1]
    codes = [z_order(x, y), hilbert(x, y), z_order(y, x), hilbert(y, x)]
    return [code.argsort(dim=1, stable=True) for code in codes]


def take(x, order):
    """Reorder the points of x (B, N, D) by order (B, N)."""
    return x.gather(1, order[..., None].expand(-1, -1, x.shape[-1]))


def undo(x, order):
    """Put points that were reordered with `take` back in their original places."""
    return take(x, order.argsort(dim=1))


class PTv3Block(nn.Module):
    """CPE, then patch attention and an MLP. As in DiT, c sets a shift, scale and gate for both."""

    def __init__(self, dim, num_heads, cond_dim, patch_size, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.cpe = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))

    def patch_attention(self, x):
        B, N, D = x.shape
        patch = min(self.patch_size, N)
        return self.attn(x.reshape(B * (N // patch), patch, D)).reshape(B, N, D)

    def forward(self, x, order, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c)[:, None].chunk(6, dim=-1)
        x = take(x, order)
        x = x + self.cpe(x.transpose(1, 2)).transpose(1, 2)
        x = x + gate1 * self.patch_attention(self.norm1(x) * (1 + scale1) + shift1)
        x = x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)
        return undo(x, order)


class PointTransformerV3(nn.Module):
    def __init__(self, point_dim=2, channels=(96, 192, 384, 384), enc_depths=(2, 2, 4, 4), dec_depths=(2, 2, 2),
                 num_heads=(2, 4, 8, 8), patch_size=64, cond_dim=384, mlp_ratio=4.0):
        super().__init__()
        if not len(channels) == len(enc_depths) == len(num_heads) == len(dec_depths) + 1:
            raise ValueError("channels, enc_depths and num_heads need one entry per stage, dec_depths one less")
        self.num_stages = len(channels)
        self.patch_size = patch_size
        self.embed = nn.Linear(point_dim, channels[0])
        self.time_mlp = nn.Sequential(nn.Linear(256, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        def stage(index, depth):
            return nn.ModuleList(PTv3Block(channels[index], num_heads[index], cond_dim, patch_size, mlp_ratio)
                                 for _ in range(depth))

        self.encoder = nn.ModuleList(stage(i, depth) for i, depth in enumerate(enc_depths))
        self.decoder = nn.ModuleList(stage(i, depth) for i, depth in enumerate(dec_depths))
        self.pool = nn.ModuleList(nn.Linear(2 * channels[i], channels[i + 1]) for i in range(self.num_stages - 1))
        self.unpool = nn.ModuleList(nn.Linear(channels[i + 1], channels[i]) for i in range(self.num_stages - 1))

        self.final_norm = nn.LayerNorm(channels[0], elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * channels[0]))
        self.final_linear = nn.Linear(channels[0], point_dim)
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for layer in (self.time_mlp[0], self.time_mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        # As in DiT: every block and the final layer start out doing nothing.
        blocks = [block for stage in [*self.encoder, *self.decoder] for block in stage]
        for layer in [block.adaLN[-1] for block in blocks] + [block.cpe for block in blocks] + \
                [self.final_adaLN[-1], self.final_linear]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, t):
        """x: (B, num_points, 2) noisy clouds, t: (B,) times. Returns (B, num_points, 2)."""
        B, N, _ = x.shape
        if N % 2 ** (self.num_stages - 1):
            raise ValueError(f"{N} points cannot be halved {self.num_stages - 1} times")
        c = self.time_mlp(timestep_embedding(t))
        feats, coords = self.embed(x), x

        skips = []
        for index, stage in enumerate(self.encoder):
            orders = serial_orders(coords)
            for depth, block in enumerate(stage):
                feats = block(feats, orders[depth % len(orders)], c)
            if index < self.num_stages - 1:
                skips.append((feats, orders))
                # Merge neighbouring pairs along the first curve: half as many points, wider features.
                count = feats.shape[1]
                feats = self.pool[index](take(feats, orders[0]).reshape(B, count // 2, -1))
                coords = take(coords, orders[0]).reshape(B, count // 2, 2, -1).mean(dim=2)

        for index in reversed(range(self.num_stages - 1)):
            skip, orders = skips[index]
            # Each merged point goes back to the two points it came from, then joins the skip connection.
            feats = skip + undo(self.unpool[index](feats).repeat_interleave(2, dim=1), orders[0])
            for depth, block in enumerate(self.decoder[index]):
                feats = block(feats, orders[depth % len(orders)], c)

        shift, scale = self.final_adaLN(c)[:, None].chunk(2, dim=-1)
        return self.final_linear(self.final_norm(feats) * (1 + scale) + shift)
