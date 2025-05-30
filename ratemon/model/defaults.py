"""Default values."""

import math
import struct
from enum import IntEnum

# Parameter defaults.
DEFAULTS = {
    "data_dir": ".",
    "warmup_percent": 0,
    "sample_percent": 100,
    "num_exps": None,
    "exps": [],
    # Cannot determine this dynamically because of import loop.
    "model": "HistGbdtSklearn",
    "features": [],
    "epochs": 100,
    "num_gpus": 0,
    "train_batch": None,
    "test_batch": None,
    "lr": 0.001,
    "momentum": 0.09,
    "kernel": "linear",
    "degree": 3,
    "penalty": "l1",
    "max_iter": 10000,
    "rfe": "None",
    "folds": 2,
    "graph": False,
    "standardize": True,
    "early_stop": False,
    "val_patience": 10,
    "val_improvement_thresh": 0.1,
    "conf_trials": 1,
    "max_attempts": 10,
    "no_rand": True,
    "timeout_s": 0,
    "out_dir": ".",
    "tmp_dir": None,
    "regen_data": False,
    "sync": False,
    "n_estimators": 100,
    "balance": False,
    "drop_popular": True,
    "analyze_features": False,
    "feature_selection_type": "perm",
    "feature_selection_percent": 100,
    "l2_regularization": 0,
    "clusters": 30,
    "num_fets_to_pick": None,
    "perm_imp_repeats": 10,
    "tag": None,
    "max_leaf_nodes": -1,
    "max_depth": -1,
    "min_samples_leaf": 20,
    "selected_features": None,
    "balance_weighted": False,
    "hgbdt_lr": 0.1,
    "validation_fraction": 0.1,
    "validation_tolerance": 1e-7,
    "n_iter_no_change": 10,
}
# When converting an arguments dictionary to a string, ignore arguments that do
# not impact model training.
ARGS_TO_IGNORE_MODEL = [
    "data_dir",
    "out_dir",
    "tmp_dir",
    "sims",
    "features",
    "exps",
    "analyze_features",
    "sync",
    "graph",
    "test_batch",
    "regen_data",
    "clusters",
    "num_fets_to_pick",
    "perm_imp_repeats",
    "tag",
    "selected_features",
]
ARGS_TO_IGNORE_DATA = ARGS_TO_IGNORE_MODEL + ["max_iter"]
# String to prepend to processed train/val/test data saved on disk.
DATA_PREFIX = "data_"
# String to prepend to trained models saved on disk.
MODEL_PREFIX = "model_"
# The maximum number of epochs when using early stopping.
EPCS_MAX = 10_000
# Whether to execute synchronously or in parallel.
SYNC = False
# The random seed.
SEED = 1337
# Name to use for lock files.
LOCK_FLN = "lock"
# The maximum number of times to try finding a cluster threshold.
CLUSTER_ATTEMPTS = 1000
# Defines the region around fair in which a flow is considered fair. In the
# range [0, 1].
FAIR_THRESH = 0.1
# The window size to use for the ground truth.
CHOSEN_WIN = 8
# The type format of the Copa header, which is the beginning of the UDP payload.
# See https://github.com/venkatarun95/genericCC/blob/master/tcp-header.hh
#     int seq_num;
#     int flow_id;
#     int src_id;
#     double sender_timestamp;  // milliseconds
#     double receiver_timestamp;  // milliseconds
COPA_HEADER_FMT = "iiidd"
# "iiidd" is ordinarily 28 bytes. However, when in a C struct, it is padded
# to enforce memory alignment. Therefore, it somehow ends up being 32 bytes.
COPA_HEADER_SIZE_B = struct.calcsize(COPA_HEADER_FMT)


class Class(IntEnum):
    """Classes for three-class models.

    Flow throughput is below, near, or above the target rate.
    """

    # The order must be *below, near, above* to maintain compatibility with already
    # trained models.
    BELOW_TARGET = 0
    NEAR_TARGET = 1
    ABOVE_TARGET = 2
    NO_CLASS = 3

    @staticmethod
    def ratio_to_class(ratio):
        """
        Converts a ratio of actual throughput to throughput fair share into
        a fairness class.
        """
        # An entry may be either a tuple containing a single value or a
        # single value.
        if isinstance(ratio, tuple):
            assert len(ratio) == 1, "Should be only one column."
            ratio = ratio[0]

        if ratio < 1 - FAIR_THRESH:
            cls = Class.BELOW_TARGET
        elif ratio <= 1 + FAIR_THRESH:
            cls = Class.NEAR_TARGET
        elif ratio > 1 + FAIR_THRESH:
            cls = Class.ABOVE_TARGET
        else:
            raise RuntimeError("This case should never be reached.")
        return cls


class Decision(IntEnum):
    """Pacing decisions.

    Either paced or not paced.
    """

    PACED = 0
    NOT_PACED = 1


# Mathis model constant.
MATHIS_C = math.sqrt(3 / 2)

# The smallest RWND value to allow.
MIN_RWND_B = 2000

# Assume this is the TCP MSS.
MSS_B = 1448

# Assume this is the total packet size.
PACKET_LEN_B = 1514
