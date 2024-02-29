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
rwnd_dir="$(cd "$(dirname "$0")"/.. && pwd)"
workspace_dir="$(dirname "$rwnd_dir")"
export PYTHONPATH="$workspace_dir:$PYTHONPATH"

# python "$rwnd_dir/model/gen_features.py" \
#     --exp-dir="$exp_dir" \
#     --untar-dir="$exp_dir" \
#     --out-dir="$exp_dir" \
#     --parallel=20

prepare() {
    # Usage: prepare <other CCA name>
    cur_exp_dir="$exp_dir/cubic-$1"
    tmp_dir="/tmp/prepare_data/cubic-$1"
    mkdir -p "$tmp_dir"
    python "$rwnd_dir/model/prepare_data.py" \
        --data-dir="$cur_exp_dir" \
        --out-dir="$tmp_dir" \
        --model=HistGbdtSklearn \
        --train-split=70 \
        --val-split=0 \
        --test-split=30 \
        --warmup-percent=5 \
        --sample-percent=10
    pushd "$tmp_dir"
    mv -vf ./* "$cur_exp_dir"
    popd
}

for cca in "highspeed" "illinois" "vivace"; do
    prepare "$cca"
done

# python "$rwnd_dir/model/train.py" \
#     --out-dir="$out_dir" \
#     --data-dir="$out_dir"\
#     --model=HistGbdtSklearn \
#     --sample-percent=15 \
#     --no-rand \
#     --conf-trials=1 \
#     --max-iter=10000 \
#     --tag="$tag" \
#     --early-stop \
#     --analyze-features \
#     --clusters=10 \
#     --features-to-pick=20 \
#     --permutation-importance-repeats=1
    # --balance \
    # --drop-popular
