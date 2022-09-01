#!/bin/bash

set -eou pipefail

if [ "$#" -ne 13 ]; then
    echo "Illegal number of parameters".
    echo "Usage: ./train.sh <model tag> <train_data_dir> <full_models_dir>" \
        "<small_models_dir> <sample_percent> <max_iter> <max_leaf_nodes>" \
        "<max_depth> <min_samples_leaf> <feature_selection_percent>" \
        "<num_clusters> <num_features_to_pick> <venv_dir>"
    exit 1
fi

model_tag="$1"
train_data_dir="$2"
full_models_dir="$3"
small_models_dir="$4"
sample_percent="$5"
max_iter="$6"
max_leaf_nodes="$7"
max_depth="$8"
min_samples_leaf="$9"
feature_selection_percent="${10}"
num_clusters="${11}"
num_features_to_pick="${12}"
venv="${13}"

unfair_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/../.."

pushd /tmp

source "$venv/bin/activate"

# Step 4: Select features from initial model
echo "Training model with all features. Progress: tail -f /tmp/train.log"
PYTHONPATH="$unfair_dir" python "$unfair_dir/unfair/model/train.py" \
    --no-rand \
    --tag="$model_tag" \
    --data-dir="$train_data_dir" \
    --out-dir="$full_models_dir" \
    --model=HistGbdtSklearn \
    --balance \
    --sample-percent="$sample_percent" \
    --max-iter="$max_iter" \
    --max-leaf-nodes="$max_leaf_nodes" \
    --max-depth="$max_depth" \
    --min-samples-leaf="$min_samples_leaf" \
    --early-stop \
    --analyze-features \
    --feature-selection-type="perm" \
    --feature-selection-percent="$feature_selection_percent" \
    --clusters="$num_clusters" \
    --num-features-to-pick="$num_features_to_pick" \
    --permutation-importance-repeats=1 ||
    {
        echo 'Error encountered during full model training and feature selection, quitting!'
        exit 4
    }
mv "/tmp/train.log" "$full_models_dir/${model_tag}_train.log"

# Step 5: Train model with selected features
echo "Training model with selected features. Progress: tail -f /tmp/train.log"
PYTHONPATH="$unfair_dir" python "$unfair_dir/unfair/model/train.py" \
    --no-rand \
    --tag="${model_tag}_selected-features" \
    --data-dir="$train_data_dir" \
    --out-dir="$small_models_dir" \
    --model=HistGbdtSklearn \
    --balance \
    --sample-percent="$sample_percent" \
    --max-iter="$max_iter" \
    --max-leaf-nodes="$max_leaf_nodes" \
    --max-depth="$max_depth" \
    --min-samples-leaf="$min_samples_leaf" \
    --early-stop \
    --selected-features="$(ls "$full_models_dir"/model_*-"$model_tag"-selected_features.json)" ||
    {
        echo 'Error encountered while training with selected features, quitting!'
        exit 5
    }
mv "/tmp/train.log" "$small_models_dir/${model_tag}_selected-features_train.log"

deactivate

popd
