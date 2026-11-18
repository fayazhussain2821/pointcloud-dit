"""Score trained checkpoints with different samplers, without retraining.

    python evaluate.py runs/x/ckpt.pt runs/eps/ckpt.pt runs/v/ckpt.pt
    python evaluate.py runs/v/ckpt.pt --methods euler heun --steps 10 50 200 --seeds 3

Uses the EMA weights. For each sampler and number of steps it generates --num_eval clouds from
several noise seeds and prints the scores from metrics.py, averaged over the seeds. Heun calls the
model twice per step, so heun/50 costs about as much as euler/100.
"""

import argparse
from argparse import Namespace

import torch

from flow_matching import generate
from metrics import generation_scores, overfit_scores
from shapes import SHAPES, make_dataset
from train import build_model, data_settings


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--methods", nargs="+", default=["euler", "heun"])
    p.add_argument("--steps", nargs="+", type=int, default=[50, 200])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--num_eval", type=int, default=None, help="default: the value used in training")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    device = args.device

    for path in args.ckpts:
        ckpt = torch.load(path, map_location="cpu")
        a = Namespace(**ckpt["args"])
        num_eval = args.num_eval or a.num_eval
        model = build_model(a).to(device)
        model.load_state_dict(ckpt["ema"])
        model.eval()
        if a.overfit:
            train_clouds, _ = make_dataset(a.overfit, seed=a.seed, **data_settings(a))
            train_clouds = train_clouds.to(device)
            print(f"\n{path} (step {ckpt['step']}, pred {a.pred}, overfitting {a.overfit} clouds)")
            print(f"{'sampler':>10} {'nearest CD':>11} {'worst':>9} {'covered':>8}")
        else:
            reference, reference_labels = make_dataset(num_eval, seed=a.seed + 1000, **data_settings(a))
            reference = reference.to(device)
            print(f"\n{path} (step {ckpt['step']}, pred {a.pred})")
            print(f"{'sampler':>10} {'1-NNA':>7} {'MMD':>9} {'COV':>6} {'nearest CD':>11}   shapes")

        for method in args.methods:
            for steps in args.steps:
                runs = []
                for seed in range(args.seeds):
                    noise = torch.randn(num_eval, a.num_points, 2,
                                        generator=torch.Generator().manual_seed(seed)).to(device)
                    samples = generate(model, noise, a.pred, steps, method, a.clip)
                    if a.overfit:
                        runs.append(overfit_scores(samples, train_clouds))
                    else:
                        runs.append(generation_scores(samples, reference, reference_labels, SHAPES))
                mean = lambda k: sum(r[k] for r in runs) / len(runs)
                name = f"{method}/{steps}"
                if a.overfit:
                    print(f"{name:>10} {mean('nearest_cd'):11.5f} {mean('worst_nearest_cd'):9.5f} {mean('covered'):8.1f}",
                          flush=True)
                else:
                    shapes = {s: sum(r["shapes"][s] for r in runs) / len(runs) for s in SHAPES}
                    shapes = "  ".join(f"{s} {n:.0f}" for s, n in shapes.items())
                    print(f"{name:>10} {mean('nna'):7.3f} {mean('mmd'):9.5f} {mean('cov'):6.2f} "
                          f"{mean('nearest_cd'):11.5f}   {shapes}", flush=True)


if __name__ == "__main__":
    main()
