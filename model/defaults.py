""" Default values. """

import sys


# Parameter defaults.
DEFAULTS = {
    "data_dir": ".",
    "warmup_percent": 0,
    "keep_percent": 100,
    "num_exps": sys.maxsize,
    "exps": [],
    "model": "HistGbdtSklearn",
    "features": [],
    "epochs": 100,
    "num_gpus": 0,
    "train_batch": sys.maxsize,
    "test_batch": sys.maxsize,
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
    "no_rand": False,
    "timeout_s": 0,
    "out_dir": ".",
    "tmp_dir": None,
    "regen_data": False,
    "sync": False,
    "cca": "bbr",
    "n_estimators": 100,
    "max_depth": 10,
    "balance": False,
    "drop_popular": True,
    "analyze_features": False,
    "l2_regularization": 0,
    "cluster_threshold": 1,
    "fets_to_pick": 20,
    "perm_imp_repeats": 10
}
# Arguments to ignore when converting an arguments dictionary to a
# string.
ARGS_TO_IGNORE = ["data_dir", "out_dir", "tmp_dir", "sims", "features", "exps"]
# The maximum number of epochs when using early stopping.
EPCS_MAX = 10_000
# Whether to execute synchronously or in parallel.
SYNC = False
# The random seed.
SEED = 1337
# Name to use for lock files.
LOCK_FLN = "lock"
