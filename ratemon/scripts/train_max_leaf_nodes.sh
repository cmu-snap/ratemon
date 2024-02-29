#! /bin/bash
#
# Execute the entire training process, starting with raw data.
#
# Usage:
#     ./train.sh <experiment data directory> <output directory> <tag>
#
# Features are generated into the experiment data directory because they may be
# very large. Sampled training data and the resulting models are stored in the
# output directory. The tag is attached to the model filename to help
# differentiate it.

set -o errexit

exp_dir="$1"
out_dir="$2"
tag="$3"
ratemon_dir="$(cd "$(dirname "$0")"/.. && pwd)"
workspace_dir="$(dirname "$ratemon_dir")"
export PYTHONPATH="$workspace_dir:$PYTHONPATH"

# python "$ratemon_dir/model/gen_features.py" \
#     --exp-dir="$exp_dir" \
#     --untar-dir="$exp_dir" \
#     --out-dir="$exp_dir" \
#     --parallel=20
# python "$ratemon_dir/model/prepare_data.py" \
#     --data-dir="$exp_dir" \
#     --out-dir="$out_dir" \
#     --model=HistGbdtSklearn \
#     --train-split=70 \
#     --val-split=0 \
#     --test-split=30 \
#     --warmup-percent=5 \
#     --sample-percent=20
# 100 250 500 1000 250 5000 10000 31 -1
# 25000 50000 100000 200000
for max_leaf_nodes in 250; do
    OMP_NUM_THREADS=10 python "$ratemon_dir/model/train.py" \
        --out-dir="${out_dir}/vary_max_leaf_nodes/${max_leaf_nodes}" \
        --data-dir="${out_dir}" \
        --model=HistGbdtSklearn \
        --sample-percent=40 \
        --no-rand \
        --conf-trials=1 \
        --max-iter=10000 \
        --tag="${tag}_${max_leaf_nodes}" \
        --max-leaf-nodes="${max_leaf_nodes}" \
        --early-stop
    # --analyze-features \
    # --clusters=10 \
    # --features-to-pick=20 \
    # --permutation-importance-repeats=1
    # --balance \
    # --drop-popular
done
