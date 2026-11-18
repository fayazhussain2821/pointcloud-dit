"""Compare finished runs (for example x, eps and v) in charts and one table.

    python compare.py runs/x runs/eps runs/v
    python compare.py runs/overfit_x runs/overfit_eps runs/overfit_v --out runs/overfit_comparison

Outputs in --out (default runs/comparison):
    scores_vs_step.png    full training: 1-NNA, MMD and COV at every checkpoint (solid = EMA, dashed = raw)
                          overfitting: nearest Chamfer distance at every checkpoint
    loss_curves.png       training loss, smoothed over 50 steps
    final_samples.png     the last generated clouds of each run, one row per run
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen on the lab machines: save figures to files only
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run folders written by train.py")
    p.add_argument("--out", default=str(HERE / "runs" / "comparison"))
    p.add_argument("--per_run", type=int, default=8, help="clouds shown per run in final_samples.png")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = {Path(r).name: json.loads((Path(r) / "metrics.json").read_text()) for r in args.runs}
    first = next(iter(runs.values()))
    overfit = bool(first["args"]["overfit"])
    colors = dict(zip(runs, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    if overfit:
        panels = [("nearest_cd", "Nearest Chamfer distance (lower = better)", True)]
    else:
        panels = [("nna", "1-NNA (0.5 = best)", False), ("mmd", "MMD-CD (lower = better)", True),
                  ("cov", "COV-CD (higher = better)", False)]

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, (key, label, log) in zip(axes[0], panels):
        for name, rec in runs.items():
            steps = [c["step"] for c in rec["checkpoints"]]
            ax.plot(steps, [c["ema"][key] for c in rec["checkpoints"]], "o-", color=colors[name], label=f"{name} (EMA)")
            ax.plot(steps, [c["raw"][key] for c in rec["checkpoints"]], "x--", color=colors[name], alpha=0.6,
                    label=f"{name} (raw)")
        if key == "nna":
            ax.axhline(0.5, color="gray", linestyle=":", label="best possible")
        ax.set(xlabel="Training step", ylabel=label, yscale="log" if log else "linear")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "scores_vs_step.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rec in runs.items():
        losses = torch.tensor(rec["losses"])
        window = min(50, len(losses))
        ax.plot(losses.unfold(0, window, 1).mean(dim=1), color=colors[name], label=name)
    ax.set(xlabel="Training step", ylabel="Loss (velocity space)", yscale="log", title="Training loss (smoothed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close(fig)

    rows = [(name, torch.load(Path(r) / "final_samples.pt")[:args.per_run]) for name, r in zip(runs, args.runs)]
    fig, axes = plt.subplots(len(rows), args.per_run, figsize=(args.per_run * 1.5, len(rows) * 1.6), squeeze=False)
    for r, (name, clouds) in enumerate(rows):
        for c in range(args.per_run):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.set_aspect("equal")
            if c < len(clouds):
                ax.scatter(clouds[c][:, 0], clouds[c][:, 1], s=1.5, color="black")
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center")
    fig.suptitle("Final generated clouds (EMA weights)")
    fig.tight_layout()
    fig.savefig(out / "final_samples.png", dpi=150)
    plt.close(fig)

    keys = ["nearest_cd", "covered"] if overfit else ["nna", "mmd", "cov"]
    print("EMA scores per checkpoint:")
    print(f"{'step':>7}" + "".join(f"{name:>34}" for name in runs))
    for i, c in enumerate(first["checkpoints"]):
        cells = []
        for rec in runs.values():
            if i < len(rec["checkpoints"]):
                e = rec["checkpoints"][i]["ema"]
                cells.append("  ".join(f"{k} {e[k]:.4g}" for k in keys))
            else:
                cells.append("-")
        print(f"{c['step']:>7}" + "".join(f"{cell:>34}" for cell in cells))
    print(f"Charts saved in {out}")


if __name__ == "__main__":
    main()
