"""Quick checks that the pieces fit together. Runs in under a minute on the CPU, needs no GPU.

    python check.py
"""

import math

import torch

from dit import PointDiT
from flow_matching import add_noise, flow_matching_loss, generate, to_velocity
from metrics import chamfer, generation_scores, pairwise_chamfer
from shapes import SHAPES, make_dataset, polygon_corners, sample_outline


def distance_to_outline(points, corners):
    """Distance from each point to the closest side of a polygon."""
    best = torch.full((len(points),), float("inf"))
    for k in range(len(corners)):
        a, b = corners[k], corners[(k + 1) % len(corners)]
        along = ((points - a) @ (b - a) / ((b - a) @ (b - a))).clamp(0, 1)
        best = torch.minimum(best, (points - (a + along[:, None] * (b - a))).norm(dim=1))
    return best


def main():
    g = torch.Generator().manual_seed(0)

    # Shapes
    circle = sample_outline("circle", 500, g)
    assert ((circle.norm(dim=1) - 1).abs() < 1e-5).all()
    for shape in ("triangle", "square"):
        assert distance_to_outline(sample_outline(shape, 500, g), polygon_corners(shape)).max() < 1e-5, shape
    square = sample_outline("square", 500, g)
    assert ((square.abs().max(dim=1).values - 1 / math.sqrt(2)).abs() < 1e-5).all()
    print("OK  points lie on the outline of each shape (square with flat sides)")

    clouds, labels = make_dataset(300, num_points=256, seed=0)
    assert clouds.shape == (300, 256, 2)
    assert torch.bincount(labels).tolist() == [100, 100, 100]
    assert clouds.abs().max() <= 1.5 + 1e-5
    still, _ = make_dataset(3, seed=0, rotate=False, scale_min=1.0, scale_max=1.0, shift=0.0)
    assert (still[2].norm(dim=1) - 1).abs().max() < 1e-5  # third cloud: an untouched unit circle
    print(f"OK  dataset: {tuple(clouds.shape)}, {len(SHAPES)} shapes balanced, coordinates within [-1.5, 1.5]")

    # Chamfer distance
    a, b = clouds[0], clouds[1]
    assert chamfer(a, a) < 1e-8
    assert chamfer(a, a[torch.randperm(len(a), generator=g)]) < 1e-8
    assert chamfer(a, a + 0.1) > 1e-3
    assert abs(chamfer(a, b) - chamfer(b, a)) < 1e-6
    matrix = pairwise_chamfer(clouds[:5], clouds[5:9], chunk=2)
    assert abs(matrix[3, 2] - chamfer(clouds[3], clouds[7])) < 1e-6
    print("OK  Chamfer distance: 0 for the same cloud in any order, symmetric, chunks agree")

    other, other_labels = make_dataset(60, seed=1)
    real = generation_scores(clouds[:60], other, other_labels, SHAPES)
    noise_scores = generation_scores(torch.randn(60, 256, 2, generator=g), other)
    assert noise_scores["nna"] > 0.9 and real["nna"] < noise_scores["nna"]
    print(f"OK  1-NNA: real vs real {real['nna']:.2f} (near 0.5 is good), noise vs real {noise_scores['nna']:.2f}")

    # Model
    torch.manual_seed(0)
    model = PointDiT(dim=64, depth=2, num_heads=4)
    x = clouds[:4]
    t = torch.rand(4)
    out = model(x, t)
    assert out.shape == x.shape
    assert out.abs().max() < 1e-6
    print("OK  the model's output has the same shape as its input, and is zero before training (adaLN-Zero)")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):  # adaLN-Zero starts at zero, so the output only changes after a few updates
        optimizer.zero_grad()
        flow_matching_loss(model, clouds[:8], "v").backward()
        optimizer.step()
    order = torch.randperm(256, generator=g)
    out = model(x, t)
    assert (model(x[:, order], t) - out[:, order]).abs().max() < 1e-4
    print("OK  shuffling the input points shuffles the output the same way (order does not matter)")
    early, late = model(x, torch.full((4,), 0.1)), model(x, torch.full((4,), 0.9))
    assert (early - late).abs().max() > 1e-6
    print("OK  the output depends on t")

    # Flow matching
    noise = torch.randn_like(x)
    assert torch.allclose(add_noise(x, noise, torch.zeros(4)), noise)
    assert torch.allclose(add_noise(x, noise, torch.ones(4)), x)
    t = 0.1 + 0.8 * torch.rand(4)
    x_t = add_noise(x, noise, t)
    for prediction, perfect in [("v", x - noise), ("x", x), ("eps", noise)]:
        assert (to_velocity(perfect, x_t, t, prediction) - (x - noise)).abs().max() < 1e-4, prediction
    print("OK  mixing (t=0 noise, t=1 cloud), and x, eps and v all convert to the correct velocity")
    for method in ("euler", "heun"):
        arrived = generate(lambda z, t: x - noise, noise, "v", steps=10, method=method)
        assert (arrived - x).abs().max() < 1e-4, method
    print("OK  Euler and Heun sampling with a perfect velocity reach the cloud")
    print("All checks passed.")


if __name__ == "__main__":
    main()
