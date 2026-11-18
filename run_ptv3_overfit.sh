#!/usr/bin/env bash
# Sanity check for PTv3: overfit 3 clouds (one triangle, one square, one circle) with x, eps and v.
set -euo pipefail
cd "$(dirname "$0")"

python -u check_ptv3.py

for pred in x eps v; do
  python -u train.py --model ptv3 --pred "$pred" --overfit 3 --steps 3000 --lr 3e-4 --ema 0.99 \
    --sample_every 500 --num_eval 30
done

python -u compare.py runs/ptv3_overfit_x runs/ptv3_overfit_eps runs/ptv3_overfit_v --out runs/ptv3_overfit_comparison
