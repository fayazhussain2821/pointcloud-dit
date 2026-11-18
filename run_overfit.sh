#!/usr/bin/env bash
# Sanity check: overfit 3 clouds (one triangle, one square, one circle) with x, eps and v.
# Every generated cloud should become an exact copy of one of the 3, and all 3 should appear.
set -euo pipefail
cd "$(dirname "$0")"

python -u check.py
python -u shapes.py

for pred in x eps v; do
  python -u train.py --pred "$pred" --overfit 3 --steps 3000 --lr 3e-4 --ema 0.99 \
    --sample_every 500 --num_eval 30
done

python -u compare.py runs/overfit_x runs/overfit_eps runs/overfit_v --out runs/overfit_comparison
