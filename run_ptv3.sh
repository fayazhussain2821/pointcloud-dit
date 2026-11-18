#!/usr/bin/env bash
# Full PTv3 training with x, eps and v (same settings as the DiT runs), then the same evaluation.
# Extra options are passed to train.py, e.g.  bash run_ptv3.sh --steps 5000
set -euo pipefail
cd "$(dirname "$0")"

python -u check_ptv3.py

for pred in x eps v; do
  python -u train.py --model ptv3 --pred "$pred" "$@"
done

python -u evaluate.py runs/ptv3_x/ckpt.pt runs/ptv3_eps/ckpt.pt runs/ptv3_v/ckpt.pt \
  --methods euler heun --steps 50 200 | tee runs/ptv3_evaluation.txt
python -u compare.py runs/ptv3_x runs/ptv3_eps runs/ptv3_v --out runs/ptv3_comparison
python -u shape_fit.py runs/ptv3_x/ckpt.pt runs/ptv3_eps/ckpt.pt runs/ptv3_v/ckpt.pt \
  --out runs/ptv3_shape_fit.json | tee runs/ptv3_shape_fit.txt
