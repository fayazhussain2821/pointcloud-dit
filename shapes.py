"""Toy dataset of 2D point clouds: triangles, squares and circles.

Every cloud has --num_points points (default 256) placed at random along the OUTLINE of one shape.
Before any transform, every shape fits exactly in the unit circle (circumradius 1): the circle has
radius 1, and the corners of the triangle and the square lie on it.

Each cloud is then (each step can be switched off):
  1. rotated by a random angle in [0, 360) degrees,
  2. scaled by a random factor in [scale_min, scale_max] (default 0.5 to 1.0),
  3. shifted by a random offset in [-shift, shift] along x and along y (default 0.5).

So every coordinate lies in [-1.5, 1.5] with the defaults. Shapes take turns (triangle, square,
circle, triangle, ...), so every split is balanced. The model never sees the labels; they are only
used to describe what it generates.

    python shapes.py      saves dataset_examples.png with a few clouds of each shape
"""

import argparse
import math
from pathlib import Path

import torch

SHAPES = ["triangle", "square", "circle"]


def polygon_corners(shape):
    """Corners of a regular polygon with circumradius 1: a triangle pointing up, a square with flat sides."""
    sides, first = {"triangle": (3, math.pi / 2), "square": (4, math.pi / 4)}[shape]
    angles = first + 2 * math.pi * torch.arange(sides) / sides
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


def sample_outline(shape, num_points, generator):
    """num_points points spread uniformly at random along the outline of one untransformed shape."""
    if shape == "circle":
        angles = 2 * math.pi * torch.rand(num_points, generator=generator)
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    corners = polygon_corners(shape)
    # All sides of a regular polygon are equally long, so: pick a side at random, then a spot on it.
    side = torch.randint(len(corners), (num_points,), generator=generator)
    where = torch.rand(num_points, 1, generator=generator)
    start, end = corners[side], corners[(side + 1) % len(corners)]
    return start + where * (end - start)


def random_transform(points, generator, rotate=True, scale_min=0.5, scale_max=1.0, shift=0.5):
    if rotate:
        angle = 2 * math.pi * torch.rand(1, generator=generator).item()
        c, s = math.cos(angle), math.sin(angle)
        points = points @ torch.tensor([[c, s], [-s, c]])
    scale = scale_min + (scale_max - scale_min) * torch.rand(1, generator=generator).item()
    points = points * scale
    if shift > 0:
        points = points + (2 * torch.rand(1, 2, generator=generator) - 1) * shift
    return points


def make_dataset(num_clouds, num_points=256, seed=0, rotate=True, scale_min=0.5, scale_max=1.0, shift=0.5):
    """Returns clouds (num_clouds, num_points, 2) and labels (num_clouds,) with label i = SHAPES[i]."""
    generator = torch.Generator().manual_seed(seed)
    clouds, labels = [], []
    for i in range(num_clouds):
        label = i % len(SHAPES)
        points = sample_outline(SHAPES[label], num_points, generator)
        clouds.append(random_transform(points, generator, rotate, scale_min, scale_max, shift))
        labels.append(label)
    return torch.stack(clouds), torch.tensor(labels)


def plot_clouds(clouds, path, titles=None, columns=8, limit=1.6):
    """Scatter plots of clouds in a grid, all with the same axes so sizes and positions compare."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clouds = clouds.detach().cpu()
    columns = min(columns, len(clouds))
    rows = math.ceil(len(clouds) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.6, rows * 1.6), squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
    for i, cloud in enumerate(clouds):
        ax = axes.flat[i]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=1.5, color="black")
        if titles is not None:
            ax.set_title(titles[i], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Save example clouds from the toy dataset.")
    p.add_argument("--num", type=int, default=24)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "dataset_examples.png"))
    args = p.parse_args()
    clouds, labels = make_dataset(args.num)
    plot_clouds(clouds, args.out, [SHAPES[l] for l in labels])
    print(f"Saved {args.num} example clouds to {args.out}")
