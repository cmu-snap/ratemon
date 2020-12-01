""" Default values. """

import sys

import models

# Parameter defaults.
DEFAULTS = {
    "data_dir": ".",
    "warmup_percent": 0,
    "keep_percent": 100,
    "num_sims": sys.maxsize,
    "sims": [],
    "model": models.MODEL_NAMES[0],
    "features": [],
    "epochs": 100,
    "num_gpus": 0,
    "train_batch": sys.maxsize,
    "test_batch": sys.maxsize,
    "learning_rate": 0.001,
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
    "sync": False
}
# The maximum number of epochs when using early stopping.
EPCS_MAX = 10_000
# Whether to execute synchronously or in parallel.
SYNC = False
# Cloudlab experiments server and client ports
CL_PORT_START_CLIENT = 5555
CL_PORT_START_SERVER = 5201
