#!/usr/bin/env bash

set -eou pipefail

if [ "$#" -ne 16 ]; then
    echo "Illegal number of parameters."
    echo "Usage: ./train.sh <model tag> <train_data_dir> <full_models_dir>" \
        "<small_models_dir> <sample_percent> <max_iter> <max_leaf_nodes>" \
        "<max_depth> <min_samples_leaf> <lr> <val_frac> <val_tol>" \
        "<n_iters_no_change> <feature_selection_percent>" \
        "<num_features_to_pick> <venv_dir>"
    exit 255
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
lr="${10}"
val_frac="${11}"
val_tol="${12}"
n_iter_no_change="${13}"
feature_selection_percent="${14}"
num_features_to_pick="${15}"
venv_dir="${16}"

echo "train.sh:"
printf "\tmodel_tag: %s\n" "$model_tag"
printf "\ttrain_data_dir: %s\n" "$train_data_dir"
printf "\tfull_models_dir: %s\n" "$full_models_dir"
printf "\tsmall_models_dir: %s\n" "$small_models_dir"
printf "\tsample_percent: %s\n" "$sample_percent"
printf "\tmax_iter: %s\n" "$max_iter"
printf "\tmax_leaf_nodes: %s\n" "$max_leaf_nodes"
printf "\tmax_depth: %s\n" "$max_depth"
printf "\tmin_samples_leaf: %s\n" "$min_samples_leaf"
printf "\tlr: %s\n" "$lr"
printf "\tval_frac: %s\n" "$val_frac"
printf "\tval_tol: %s\n" "$val_tol"
printf "\tn_iter_no_change: %s\n" "$n_iter_no_change"
printf "\tfeature_selection_percent: %s\n" "$feature_selection_percent"
printf "\tnum_features_to_pick: %s\n" "$num_features_to_pick"
printf "\tvenv_dir: %s\n" "$venv_dir"

set +u
# shellcheck disable=SC1091
source "$venv_dir/bin/activate"
set -u

ratemon_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/../.."
pushd /tmp

# Step 4: Select features from initial model
echo "Training model with all features. Progress: tail -f /tmp/train.log"
bash -x -c "PYTHONPATH='$ratemon_dir' python '$ratemon_dir/ratemon/model/train.py' \
    --no-rand \
    --tag='$model_tag' \
    --data-dir='$train_data_dir' \
    --out-dir='$full_models_dir' \
    --model=HistGbdtSklearn \
    --balance \
    --sample-percent='$sample_percent' \
    --max-iter='$max_iter' \
    --max-leaf-nodes='$max_leaf_nodes' \
    --max-depth='$max_depth' \
    --min-samples-leaf='$min_samples_leaf' \
    --hgbdt-lr='$lr' \
    --validation-fraction='$val_frac' \
    --validation-tolerance='$val_tol' \
    --n-iter-no-change='$n_iter_no_change' \
    --early-stop \
    --analyze-features \
    --feature-selection-type='perm' \
    --feature-selection-percent='$feature_selection_percent' \
    --clusters='$num_features_to_pick' \
    --permutation-importance-repeats=1" ||
    {
        echo "Error encountered during full model training and feature" \
            "selection, quitting!"
        exit 4
    }
mv -fv "/tmp/train.log" "$full_models_dir/${model_tag}_train.log"

# Step 5: Train model with selected features
echo "Training model with selected features. Progress: tail -f /tmp/train.log"
bash -x -c "PYTHONPATH='$ratemon_dir' python '$ratemon_dir/ratemon/model/train.py' \
    --no-rand \
    --tag='${model_tag}_selected-features' \
    --data-dir='$train_data_dir' \
    --out-dir='$small_models_dir' \
    --model=HistGbdtSklearn \
    --balance \
    --sample-percent='$sample_percent' \
    --max-iter='$max_iter' \
    --max-leaf-nodes='$max_leaf_nodes' \
    --max-depth='$max_depth' \
    --min-samples-leaf='$min_samples_leaf' \
    --hgbdt-lr='$lr' \
    --validation-fraction='$val_frac' \
    --validation-tolerance='$val_tol' \
    --n-iter-no-change='$n_iter_no_change' \
    --early-stop \
    --selected-features='$(ls -t "$full_models_dir"/model_*-"$model_tag"-selected_features.json | head -n 1)'" ||
    {
        echo "Error encountered while training with selected features," \
            "quitting!"
        exit 5
    }
mv -fv "/tmp/train.log" \
    "$small_models_dir/${model_tag}_selected-features_train.log"

deactivate
popd
