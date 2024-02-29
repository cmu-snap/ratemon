#!/usr/bin/env bash
#
# Parses the output of run_many_for_validation.sh, from rware:cctestbed, to
# validate the CloudLab parallel testbed.
#
# Usage: ./parse_validation_exps.sh \
#     <cmu-snap:rwnd repo dir> <iterations dir> <scratch (untar) dir> \
#     <direction to parse (forward or reverse)>

set -o errexit
set -o nounset

RWND_DIR=$1
ITERS_DIR=$2
UNTAR_DIR=$3
DIRECTION=$4
mkdir -p "$UNTAR_DIR"

NUM_ITERS=10


parse_iter_batchsize () {
    ITER=$1
    BATCH_SIZE=$2
    EXP_DIR="$ITERS_DIR/iter_$ITER/batchsize_$BATCH_SIZE"
    python "$RWND_DIR/model/parse_cloudlab.py" --exp-dir "$EXP_DIR" \
           --untar-dir "$UNTAR_DIR" --out-dir "$EXP_DIR" \
           --skip-smoothed-features
}


if [ "$DIRECTION" = "forward" ]
then
    for (( ITER=1; ITER<=NUM_ITERS; ITER++ ))
    do
        for BATCH_SIZE in 1 5 10 20
        do
            parse_iter_batchsize "$ITER" "$BATCH_SIZE"
        done
    done
elif [ "$DIRECTION" = "reverse" ]
then
    for (( ITER=NUM_ITERS; ITER>=1; ITER-- ))
    do
        for BATCH_SIZE in 20 10 5 1
        do
            parse_iter_batchsize "$ITER" "$BATCH_SIZE"
        done
    done
else
    echo "Unknown direction: $DIRECTION"
fi

echo "Done"
