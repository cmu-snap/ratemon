#!/usr/bin/env bash
#
# Parses the output of run_many_for_validation.sh, from rware?cctestbed, to
# validate the CloudLab parallel testbed.
#
# Usage: ./parse_validation_exps.sh \
#     <cmu-snap:unfair repo dir> <iterations dir> <scratch (untar) dir>

set -o errexit
set -o nounset

UNFAIR_DIR=$1
ITERS_DIR=$2
UNTAR_DIR=$3
mkdir -p "$UNTAR_DIR"

NUM_ITERS=10

for (( ITER=1; ITER<=NUM_ITERS; ITER++ ))
do
    for BATCH_SIZE in 1 5 10 15 20:
    do
        EXP_DIR="$ITERS_DIR/iter_$ITER/batchsize_$BATCH_SIZE"

        python "$UNFAIR_DIR/model/parse_cloudlab.py" --exp-dir "$EXP_DIR" \
               --untar-dir "$UNTAR_DIR" --out-dir "$EXP_DIR" \
               --skip-smoothed-features
    done
done

echo "Done"
