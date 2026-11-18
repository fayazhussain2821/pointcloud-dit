"""Train flow matching with a point-cloud DiT on the toy shapes, for one prediction type.

    python train.py --pred v                    full training on the toy dataset
    python train.py --pred x --overfit 3        overfit 3 clouds (one of each shape) as a sanity check

Every --sample_every steps it generates --num_eval clouds from the same fixed noise, with both the
EMA weights and the raw weights, and scores them (see metrics.py):
  full training: MMD, COV and 1-NNA against --num_eval held-out real clouds, plus shape counts
  --overfit N:   Chamfer distance to the closest of the N training clouds, and how many are covered

Outputs in --out (default runs/<pred>, or runs/overfit_<pred>):
    samples/step_02000.png   24 clouds generated with the EMA weights
    reference.png            real clouds for comparison (the training clouds when overfitting)
    metrics.json             settings, training loss of every step, scores at every checkpoint
    final_samples.pt         the last 24 generated clouds
    ckpt.pt                  model, EMA model, optimizer and settings (overwritten at every checkpoint)
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from dit import PointDiT
from flow_matching import flow_matching_loss, generate
from metrics import generation_scores, overfit_scores
from shapes import SHAPES, make_dataset, plot_clouds

HERE = Path(__file__).resolve().parent


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", default="v", choices=["x", "eps", "v"], help="what the model outputs")
    p.add_argument("--overfit", type=int, default=0, help="train on only this many clouds (0 = full dataset)")
    # Data (see shapes.py)
    p.add_argument("--num_train", type=int, default=10000)
    p.add_argument("--num_points", type=int, default=256)
    p.add_argument("--rotate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--scale_min", type=float, default=0.5, help="set scale_min = scale_max = 1 to turn scaling off")
    p.add_argument("--scale_max", type=float, default=1.0)
    p.add_argument("--shift", type=float, default=0.5, help="largest shift along x and y (0 = off)")
    # Model (default: DiT-S size, 12 blocks, width 384, 6 heads)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    # Training
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0, help="clip the gradient norm to this (0 = off)")
    p.add_argument("--ema", type=float, default=0.999, help="EMA decay; 0.999 averages roughly the last 1000 steps")
    p.add_argument("--clip", type=float, default=0.05, help="smallest denominator in the x/eps conversion")
    p.add_argument("--seed", type=int, default=0)
    # Sampling and scoring
    p.add_argument("--sampler", default="euler", choices=["euler", "heun"])
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--num_eval", type=int, default=300, help="clouds generated (and real clouds compared) per checkpoint")
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    if args.out is None:
        args.out = str(HERE / "runs" / (f"overfit_{args.pred}" if args.overfit else args.pred))
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def data_settings(args):
    return dict(num_points=args.num_points, rotate=args.rotate, scale_min=args.scale_min,
                scale_max=args.scale_max, shift=args.shift)


def build_model(args):
    return PointDiT(point_dim=2, dim=args.dim, depth=args.depth, num_heads=args.heads)


@torch.no_grad()
def ema_update(ema, model, decay):
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.lerp_(p, 1 - decay)


def main(argv=None):
    args = get_args(argv)
    device = args.device
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}   output: {out}", flush=True)

    train_clouds, train_labels = make_dataset(args.num_train if not args.overfit else args.overfit,
                                              seed=args.seed, **data_settings(args))
    train_clouds = train_clouds.to(device)
    if args.overfit:
        reference, reference_labels = train_clouds, train_labels
    else:
        reference, reference_labels = make_dataset(args.num_eval, seed=args.seed + 1000, **data_settings(args))
        reference = reference.to(device)
    plot_clouds(reference[:24], out / "reference.png", [SHAPES[l] for l in reference_labels[:24]])
    print(f"Training clouds: {tuple(train_clouds.shape)}   "
          f"{'overfitting' if args.overfit else 'held-out reference clouds: ' + str(len(reference))}", flush=True)

    torch.manual_seed(args.seed)  # same starting weights for x, eps and v
    model = build_model(args).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters   pred={args.pred}   "
          f"sampler={args.sampler}/{args.sample_steps}   batch={args.batch_size}   lr={args.lr}   "
          f"ema={args.ema}   grad_clip={args.grad_clip or 'off'}   steps={args.steps}", flush=True)

    # The same starting noise at every checkpoint (and for x, eps and v), so samples are comparable.
    noise = torch.randn(args.num_eval, args.num_points, 2,
                        generator=torch.Generator().manual_seed(args.seed)).to(device)
    record = {"args": vars(args), "losses": [], "checkpoints": []}

    def checkpoint(step):
        scores = {"step": step}
        for name, net in (("ema", ema), ("raw", model)):
            net.eval()
            samples = generate(net, noise, args.pred, args.sample_steps, args.sampler, args.clip)
            if args.overfit:
                scores[name] = overfit_scores(samples, train_clouds)
            else:
                scores[name] = generation_scores(samples, reference, reference_labels, SHAPES)
            if name == "ema":
                plot_clouds(samples[:24], out / "samples" / f"step_{step:06d}.png")
                torch.save(samples[:24].cpu(), out / "final_samples.pt")
        model.train()
        record["checkpoints"].append(scores)
        e, r = scores["ema"], scores["raw"]
        if args.overfit:
            print(f"  [checkpoint {step}] nearest CD: EMA {e['nearest_cd']:.5f}  raw {r['nearest_cd']:.5f}   "
                  f"covered (EMA): {e['covered']}/{args.overfit}", flush=True)
        else:
            print(f"  [checkpoint {step}] EMA: 1-NNA {e['nna']:.3f}  MMD {e['mmd']:.5f}  COV {e['cov']:.2f}  "
                  f"shapes {e['shapes']}   raw: 1-NNA {r['nna']:.3f}  MMD {r['mmd']:.5f}", flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "args": vars(args)}, out / "ckpt.pt")
        (out / "metrics.json").write_text(json.dumps(record, indent=1))

    model.train()
    start = time.time()
    for step in range(1, args.steps + 1):
        chosen = torch.randint(len(train_clouds), (args.batch_size,), device=device)
        loss = flow_matching_loss(model, train_clouds[chosen], args.pred, args.clip)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        ema_update(ema, model, args.ema)
        record["losses"].append(loss.item())

        if step % args.log_every == 0 or step == 1:
            recent = record["losses"][-args.log_every:]
            ms = (time.time() - start) / step * 1000
            print(f"[{args.pred}] step {step:>6}/{args.steps}  loss {sum(recent) / len(recent):.4f}  {ms:.0f} ms/step",
                  flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            checkpoint(step)

    print(f"Done in {(time.time() - start) / 60:.1f} min. Results in {out}", flush=True)


if __name__ == "__main__":
    main()
