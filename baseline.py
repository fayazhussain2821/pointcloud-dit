"""Real-vs-real baseline: the scores a perfect generator would get.

A perfect generator produces clouds from exactly the same distribution as the dataset. So we stand in
for it with fresh real clouds (new seeds) and score them against the same 300 reference clouds that
train.py and evaluate.py use. This shows:
  - the best MMD and COV we can expect with 300 clouds,
  - how much the scores change by chance alone (the spread over the repeats),
  - how often "shape of the nearest reference cloud" gets the true shape right.

    python baseline.py | tee runs/baseline.txt
"""

import argparse

import torch

from metrics import generation_scores, pairwise_chamfer
from shapes import SHAPES, make_dataset


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_eval", type=int, default=300, help="clouds per set, as in training")
    p.add_argument("--repeats", type=int, default=10, help="how many fresh sets of real clouds to score")
    p.add_argument("--reference_seed", type=int, default=1000, help="train.py uses seed + 1000 = 1000")
    p.add_argument("--first_seed", type=int, default=2000, help="seeds of the fresh sets start here")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()
    reference, reference_labels = make_dataset(args.num_eval, seed=args.reference_seed)
    reference = reference.to(args.device)

    print(f"Real vs real: {args.repeats} fresh sets of {args.num_eval} clouds against the reference set")
    print(f"{'seed':>6} {'1-NNA':>7} {'MMD':>9} {'COV':>6} {'nearest CD':>11} {'labels right':>13}   shapes")
    rows = []
    for seed in range(args.first_seed, args.first_seed + args.repeats):
        clouds, labels = make_dataset(args.num_eval, seed=seed)
        clouds = clouds.to(args.device)
        scores = generation_scores(clouds, reference, reference_labels, SHAPES)

        distances = pairwise_chamfer(clouds, reference)
        closest = reference_labels.to(distances.device)[distances.argmin(dim=1)]
        scores["labels"] = (closest == labels.to(distances.device)).float().mean().item()

        rows.append(scores)
        shapes = "  ".join(f"{name} {count}" for name, count in scores["shapes"].items())
        print(f"{seed:>6} {scores['nna']:>7.3f} {scores['mmd']:>9.5f} {scores['cov']:>6.2f} "
              f"{scores['nearest_cd']:>11.5f} {scores['labels']:>13.3f}   {shapes}", flush=True)

    print("\nAverage +- spread (standard deviation) over the repeats:")
    for key, name, digits in [("nna", "1-NNA", 3), ("mmd", "MMD", 5), ("cov", "COV", 3),
                              ("nearest_cd", "nearest CD", 5), ("labels", "labels right", 3)]:
        values = torch.tensor([row[key] for row in rows])
        spread = values.std().item() if len(values) > 1 else 0.0
        print(f"  {name:<13} {values.mean().item():.{digits}f} +- {spread:.{digits}f}")


if __name__ == "__main__":
    main()
