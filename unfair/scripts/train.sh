#! /bin/bash
#
# Execute the entire training process, starting with raw data.
#
# Usage:
#     ./train.sh <experiment data directory> <output directory>
#
# Features are generated into the experiment data directory because they may be
# very large. Sampled training data and the resulting models are stored in the
# output directory.

set -o errexit

exp_dir="$1"
out_dir="$2"
unfair_dir="$(cd "$(dirname "$0")"/.. && pwd)"
workspace_dir="$(dirname "$unfair_dir")"
export PYTHONPATH="$workspace_dir:$PYTHONPATH"

# python "$unfair_dir/model/gen_features.py" \
#     --exp-dir="$exp_dir" \
#     --untar-dir="$exp_dir" \
#     --out-dir="$exp_dir" \
#     --parallel=20
# python "$unfair_dir/model/prepare_data.py" \
#     --data-dir="$exp_dir" \
#     --out-dir="$out_dir" \
#     --train-split=70 \
#     --val-split=0 \
#     --test-split=30 \
#     --warmup-percent=5 \
#     --sample-percent=5
python "$unfair_dir/model/train.py" \
    --out-dir="$out_dir" \
    --data-dir="$out_dir" \
    --model=HistGbdtSklearn \
    --sample-percent=100 \
    --no-rand \
    --conf-trials=1 \
    --max-iter=100

    #  \
    # --analyze-features \
    # --clusters=20 \
    # --features-to-pick=10 \
    # --permutation-importance-repeats=2

    # --balance \
    # --drop-popular \