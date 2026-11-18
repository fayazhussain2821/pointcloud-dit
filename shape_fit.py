"""Shape-fit check: which shape is each cloud, and how cleanly does it lie on that shape's outline?

We know the true shapes, so for every cloud we fit a perfect triangle, square and circle (position,
size and rotation) and keep the one that fits best. Real clouds go through the same check, which shows
whether the check names shapes correctly and what a perfect generator would score.

    python shape_fit.py | tee runs/shape_fit.txt

Columns:
    triangle/square/circle %  share of clouds named as that shape (the dataset has 33.3 % of each)
    distance                  average distance from a point to the fitted outline
    off outline %             points further than 0.02 from the fitted outline (stray points)
    covered %                 parts of the outline with a point within 0.05 (gaps lower this)
    clean %                   clouds that fit as well as 99 % of the real clouds
    size                      average fitted size (the dataset is uniform in 0.5 to 1, so about 0.75)
"""

import argparse
import json
import math
from argparse import Namespace
from pathlib import Path

import torch

from flow_matching import generate
from shapes import SHAPES, make_dataset, polygon_corners
from train import build_model, data_settings

OFF_OUTLINE = 0.02
COVER_RADIUS = 0.05
COLUMNS = [("triangle", "triangle %", 1), ("square", "square %", 1), ("circle", "circle %", 1),
           ("distance", "distance", 5), ("off", "off outline %", 2), ("covered", "covered %", 2),
           ("clean", "clean %", 1), ("size", "size", 3)]


def outline_template(shape, num_points, device):
    """Evenly spaced points along the outline of the untransformed shape."""
    if shape == "circle":
        angles = 2 * math.pi * torch.arange(num_points).float() / num_points
        return torch.stack([angles.cos(), angles.sin()], dim=1).to(device)
    corners = polygon_corners(shape)
    place = torch.arange(num_points).float() * len(corners) / num_points
    side = place.long()
    where = (place - side)[:, None]
    start, end = corners[side], corners[(side + 1) % len(corners)]
    return (start + where * (end - start)).to(device)


def distance_to_unit_outline(points, shape):
    """Exact distance from points (B, N, 2) to the outline of the untransformed shape. Returns (B, N)."""
    if shape == "circle":
        return (points.pow(2).sum(-1).add(1e-12).sqrt() - 1).abs()
    start = polygon_corners(shape).to(points.device)
    side = start.roll(-1, dims=0) - start
    from_start = points[:, :, None] - start
    where = ((from_start * side).sum(-1) / side.pow(2).sum(-1)).clamp(0, 1)
    gap = from_start - where[..., None] * side
    return gap.pow(2).sum(-1).min(dim=-1).values.add(1e-12).sqrt()


def to_unit(clouds, center, log_size, angle):
    """Undo the shift, rotation and size, so a perfect cloud lands on the untransformed outline."""
    shifted = clouds - center[:, None]
    c, s = angle.cos()[:, None], angle.sin()[:, None]
    x = c * shifted[..., 0] + s * shifted[..., 1]
    y = -s * shifted[..., 0] + c * shifted[..., 1]
    return torch.stack([x, y], dim=-1) / log_size.exp()[:, None, None]


def from_unit(outline, center, log_size, angle):
    """Place the outline template (M, 2) at each fitted pose. Returns (B, M, 2)."""
    c, s = angle.cos()[:, None], angle.sin()[:, None]
    x = c * outline[:, 0] - s * outline[:, 1]
    y = s * outline[:, 0] + c * outline[:, 1]
    return torch.stack([x, y], dim=-1) * log_size.exp()[:, None, None] + center[:, None]


def fit_errors(clouds, shape, outline, center, log_size, angle):
    """Distances from points to the fitted outline (B, N) and from the outline to the nearest point (B, M)."""
    unit = to_unit(clouds, center, log_size, angle)
    to_outline = distance_to_unit_outline(unit, shape) * log_size.exp()[:, None]
    placed = from_unit(outline, center, log_size, angle)
    squared = (placed[:, :, None] - clouds[:, None]).pow(2).sum(-1)
    to_cloud = squared.min(dim=2).values.add(1e-12).sqrt()
    return to_outline, to_cloud


def fit_shape(clouds, shape, num_angles=36, steps=150, lr=0.01, outline_points=256):
    """Fit one shape to every cloud: a grid search over rotation, then gradient descent on the pose."""
    outline = outline_template(shape, outline_points, clouds.device)
    center = clouds.mean(dim=1)
    radius = (clouds - center[:, None]).norm(dim=-1).mean(dim=1)
    log_size = (radius / outline.norm(dim=-1).mean()).clamp(min=1e-3).log()
    angle = torch.zeros(len(clouds), device=clouds.device)
    if shape != "circle":
        # Turning a regular polygon by one corner gives the same polygon, so this range is enough.
        period = 2 * math.pi / len(polygon_corners(shape))
        grid = torch.arange(num_angles, device=clouds.device).float() * period / num_angles
        trials = []
        for value in grid:
            unit = to_unit(clouds, center, log_size, value.expand(len(clouds)))
            trials.append(distance_to_unit_outline(unit, shape).pow(2).mean(dim=1))
        angle = grid[torch.stack(trials).argmin(dim=0)]

    pose = [p.clone().requires_grad_(True) for p in (center, log_size, angle)]
    optimizer = torch.optim.Adam(pose, lr=lr)
    for _ in range(steps):
        to_outline, to_cloud = fit_errors(clouds, shape, outline, *pose)
        loss = (to_outline.pow(2).mean(dim=1) + to_cloud.pow(2).mean(dim=1)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        to_outline, to_cloud = fit_errors(clouds, shape, outline, *pose)
    score = to_outline.pow(2).mean(dim=1) + to_cloud.pow(2).mean(dim=1)
    return {"score": score, "to_outline": to_outline, "to_cloud": to_cloud, "size": pose[1].detach().exp()}


def fit_clouds(clouds, chunk=100):
    """Fit all shapes to all clouds and describe each cloud with the shape that fits it best."""
    parts = []
    for batch in clouds.split(chunk):
        fits = [fit_shape(batch, shape) for shape in SHAPES]
        label = torch.stack([fit["score"] for fit in fits], dim=1).argmin(dim=1)
        pick = lambda key: torch.stack([fits[k][key][i] for i, k in enumerate(label.tolist())])
        to_outline, to_cloud, size, score = pick("to_outline"), pick("to_cloud"), pick("size"), pick("score")
        parts.append({"label": label, "distance": to_outline.mean(dim=1),
                      "off": (to_outline > OFF_OUTLINE).float().mean(dim=1),
                      "covered": (to_cloud <= COVER_RADIUS).float().mean(dim=1),
                      "relative": score / size.pow(2), "size": size})
    return {key: torch.cat([part[key] for part in parts]).cpu() for key in parts[0]}


def summary(fit, threshold):
    row = {name: 100 * (fit["label"] == k).float().mean().item() for k, name in enumerate(SHAPES)}
    row.update(distance=fit["distance"].mean().item(), off=100 * fit["off"].mean().item(),
               covered=100 * fit["covered"].mean().item(), size=fit["size"].mean().item(),
               clean=100 * (fit["relative"] <= threshold).float().mean().item())
    return row


def average(rows):
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def spread(rows):
    return {key: torch.tensor([row[key] for row in rows]).std().item() if len(rows) > 1 else 0.0 for key in rows[0]}


def print_row(name, row):
    cells = " ".join(f"{row[key]:>{len(title)}.{digits}f}" for key, title, digits in COLUMNS)
    print(f"{name:<22} {cells}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="*", default=["runs/x/ckpt.pt", "runs/eps/ckpt.pt", "runs/v/ckpt.pt"])
    p.add_argument("--method", default="heun", choices=["euler", "heun"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seeds", type=int, default=3, help="noise seeds per model (the same as evaluate.py)")
    p.add_argument("--real_sets", type=int, default=10, help="fresh sets of real clouds")
    p.add_argument("--first_real_seed", type=int, default=2000)
    p.add_argument("--out", default="runs/shape_fit.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = args.device

    settings = Namespace(**torch.load(args.ckpts[0], map_location="cpu")["args"])
    num_clouds = settings.num_eval

    print(f"Fitting {args.real_sets} sets of {num_clouds} real clouds...", flush=True)
    real_fits, correct = [], []
    for seed in range(args.first_real_seed, args.first_real_seed + args.real_sets):
        clouds, labels = make_dataset(num_clouds, seed=seed, **data_settings(settings))
        fit = fit_clouds(clouds.to(device))
        real_fits.append(fit)
        correct.append(100 * (fit["label"] == labels).float().mean().item())
    threshold = torch.quantile(torch.cat([fit["relative"] for fit in real_fits]), 0.99).item()
    real_rows = [summary(fit, threshold) for fit in real_fits]

    results = {"real": average(real_rows), "real_spread": spread(real_rows),
               "real_named_correctly": sum(correct) / len(correct), "models": {}}

    rows = {}
    for path in args.ckpts:
        ckpt = torch.load(path, map_location="cpu")
        a = Namespace(**ckpt["args"])
        model = build_model(a).to(device)
        model.load_state_dict(ckpt["ema"])
        model.eval()
        print(f"Generating and fitting pred={a.pred} ({args.method}/{args.steps}, {args.seeds} seeds)...", flush=True)
        seed_rows = []
        for seed in range(args.seeds):
            noise = torch.randn(num_clouds, a.num_points, 2,
                                generator=torch.Generator().manual_seed(seed)).to(device)
            samples = generate(model, noise, a.pred, args.steps, args.method, a.clip)
            seed_rows.append(summary(fit_clouds(samples), threshold))
        rows[a.pred] = average(seed_rows)
        results["models"][a.pred] = {"average": rows[a.pred], "spread": spread(seed_rows)}

    print(f"\nReal clouds named as the correct shape: {results['real_named_correctly']:.1f} %\n")
    print(f"{'':<22} " + " ".join(title for _, title, _ in COLUMNS))
    print_row("real (the limit)", results["real"])
    print_row("real chance spread", results["real_spread"])
    for name, row in rows.items():
        print_row(f"{name} ({args.method}/{args.steps})", row)

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
