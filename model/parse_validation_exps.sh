#!/usr/bin/env bash
#
# Parses the output of run_many_for_validation.sh, from rware?cctestbed, to
# validate the CloudLab parallel testbed.
#
# Usage: ./parse_validation_exps.sh <iterations dir> <scratch (untar) dir>

set -o errexit
set -o nounset

ITERS_DIR=$1
UNTAR_DIR=$2
mkdir -p "$UNTAR_DIR"

NUM_ITERS=10
NUM_EXPS=10

for (( ITER=1; ITER<=NUM_ITERS; ITER++ ))
do
    for BATCH_SIZE in 1 $NUM_EXPS
    do
        EXP_DIR="$ITERS_DIR/iter_$ITER/batchsize_$BATCH_SIZE"

        python "$HOME/src/unfair/model/parse_cloudlab.py" --exp-dir "$EXP_DIR" \
               --untar-dir "$UNTAR_DIR" --out-dir "$EXP_DIR"
    done
done

echo "Done"
