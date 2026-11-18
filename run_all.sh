#!/usr/bin/env bash
# Full training on the toy dataset with x, eps and v (default settings in train.py),
# then the sampler comparison and the charts. Extra options are passed to train.py,
# e.g.  bash run_all.sh --steps 5000
set -euo pipefail
cd "$(dirname "$0")"

python -u check.py

for pred in x eps v; do
  python -u train.py --pred "$pred" "$@"
done

python -u evaluate.py runs/x/ckpt.pt runs/eps/ckpt.pt runs/v/ckpt.pt --methods euler heun --steps 50 200 \
  | tee runs/evaluation.txt
python -u compare.py runs/x runs/eps runs/v --out runs/comparison
