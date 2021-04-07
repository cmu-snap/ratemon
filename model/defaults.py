""" Default values. """


# Parameter defaults.
DEFAULTS = {
    "data_dir": ".",
    "warmup_percent": 0,
    "sample_percent": 100,
    "num_exps": None,
    "exps": [],
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
    "no_rand": False,
    "timeout_s": 0,
    "out_dir": ".",
    "tmp_dir": None,
    "regen_data": False,
    "sync": False,
    "n_estimators": 100,
    "max_depth": 10,
    "balance": False,
    "drop_popular": True,
    "analyze_features": False,
    "l2_regularization": 0,
    "clusters": 30,
    "fets_to_pick": None,
    "perm_imp_repeats": 10
}
# When converting an arguments dictionary to a string, ignore arguments that do
# not impact model training.
ARGS_TO_IGNORE = [
    "data_dir", "out_dir", "tmp_dir", "sims", "features", "exps",
    "analyze_features", "sync", "graph", "test_batch", "regen_data"]
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
