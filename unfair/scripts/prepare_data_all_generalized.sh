#!/bin/bash

set -eou pipefail

base_dir="$HOME/fawnstore2/out/cloudlab/2021-5-12"

source "$HOME/src/unfair/.venv/bin/activate"

for dir in "$base_dir"/*; do
    pair="$(echo "$dir" | cut -d'/' -f 8)"
    if [ "$pair" == "composite" ] || [ "$pair" == "cubic-bbr-fewer-features-more-data" ]; then
        continue
    fi
    out_dir="$dir/new_splits"
    echo "Preparing data for: $dir -> $out_dir"
    PYTHONPATH="$HOME/src/unfair" python "$HOME/src/unfair/unfair/model/prepare_data.py" \
        --data-dir="$dir" \
        --train-split=70 \
        --val-split=0 \
        --test-split=30 \
        --warmup-percent=5 \
        --out-dir="$out_dir" \
        --sample-percent=0.1 \
        --disjoint-splits
done

deactivate
