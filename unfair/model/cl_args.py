""" Common command line arguments. """

import os
from os import path

from unfair.model import defaults, models


def add_out(psr, psr_verify=lambda args: args):
    """
    Adds an "out-dir" argument to the provided ArgumentParser, and returns it.
    """
    def verify(args):
        out_dir = args.out_dir
        if not path.exists(out_dir):
            os.makedirs(out_dir)
        return args

    psr.add_argument(
        "--out-dir", default=".",
        help="The directory in which to store output files.", type=str)
    return psr, lambda args: verify(psr_verify(args))


def add_warmup(psr, psr_verify=lambda args: args):
    """
    Adds a "warmup-percent" argument to the provided ArgumentParser, and returns
    it.
    """
    def verify(args):
        warmup_prc = args.warmup_percent
        assert 0 <= warmup_prc < 100, \
            ("\"warmup-percent\" must be in the range [0, 100), but is: "
             f"{warmup_prc}")
        return args

    psr.add_argument(
        "--warmup-percent", default=defaults.DEFAULTS["warmup_percent"],
        help=("The percent of each experiment's datapoint to drop from the "
              "beginning."),
        type=float)
    return psr, lambda args: verify(psr_verify(args))


def add_num_exps(psr, psr_verify=lambda args: args):
    """
    Adds a "num-exps" argument to the provided ArgumentParser, and returns it.
    """
    def verify(args):
        num_exps = args.num_exps
        assert num_exps is None or num_exps >= 0, \
            f"\"num-exps\" cannot be negative, but is: {num_exps}"
        return args

    psr.add_argument(
        "--num-exps", default=defaults.DEFAULTS["num_exps"],
        help="The number of experiments to consider.", required=False, type=int)
    return psr, lambda args: verify(psr_verify(args))


def add_standardize(psr, psr_verify=lambda args: args):
    """
    Adds a "standardize" argument to the provided ArgumentParser, and returns
    it.
    """
    psr.add_argument(
        "--standardize", action="store_true",
        help=("Standardize the data so that it has a mean of 0 and a variance "
              "of 1. Otherwise, data will be rescaled to the range [0, 1]."))
    # "standardize" does not require verification.
    return psr, psr_verify


def add_sample_percent(psr, psr_verify=lambda args: args):
    """
    Adds a "sample-kercent" argument to the provided ArgumentParser, and returns
    it.
    """
    def verify(args):
        sample_prc = args.sample_percent
        assert 0 < sample_prc <= 100, \
            ("\"sample-percent\" must be in the range (0, 100], but is: "
             f"{sample_prc}")
        return args

    psr.add_argument(
        "--sample-percent", default=defaults.DEFAULTS["sample_percent"],
        required=False, type=float)
    return psr, lambda args: verify(psr_verify(args))


def add_training(psr, psr_verify=lambda args: args):
    """
    Adds training-related arguments to the provided ArgumentParser, and returns
    it.
    """
    def verify(args):
        if args.early_stop:
            args.epochs = defaults.EPCS_MAX
        degree = args.degree
        assert degree >= 0, \
            ("\"degree\" must be an integer greater than or equal to 0, but "
             f"is: {degree}")
        max_iter = args.max_iter
        assert max_iter > 0, \
            f"\"max-iter\" must be greater than 0, but is: {max_iter}"
        folds = args.folds
        assert folds >= 2, f"\"folds\" must be at least 2, but is: {folds}"
        return args

    psr, psr_verify = add_sample_percent(
        *add_out(*add_standardize(psr, psr_verify)))
    psr.add_argument(
        "--data-dir",
        help=("The path to a directory containing the "
              "training/validation/testing data (required)."),
        required=True, type=str)
    psr.add_argument(
        "--no-rand", action="store_true", help="Use a fixed random seed.")
    psr.add_argument(
        "--model", choices=models.MODEL_NAMES,
        default=defaults.DEFAULTS["model"], help="The model to use.", type=str)
    psr.add_argument(
        "--epochs", default=defaults.DEFAULTS["epochs"],
        help="The number of epochs to train for.", type=int)
    psr.add_argument(
        "--num-gpus", default=defaults.DEFAULTS["num_gpus"],
        help="The number of GPUs to use.", type=int)
    psr.add_argument(
        "--train-batch", default=defaults.DEFAULTS["train_batch"],
        help="The batch size to use during training.", type=int)
    psr.add_argument(
        "--test-batch", default=defaults.DEFAULTS["test_batch"],
        help="The batch size to use during validation and testing.", type=int)
    psr.add_argument(
        "--lr", default=defaults.DEFAULTS["lr"],
        help="Learning rate for SGD training.", type=float)
    psr.add_argument(
        "--momentum", default=defaults.DEFAULTS["momentum"],
        help="Momentum for SGD training.", type=float)
    psr.add_argument(
        "--kernel", default=defaults.DEFAULTS["kernel"],
        choices=["linear", "poly", "rbf", "sigmoid"],
        help=(f"If the model is of type \"{models.SvmSklearnWrapper().name}\", "
              "then use this type kernel. Ignored otherwise."),
        type=str)
    psr.add_argument(
        "--degree", default=defaults.DEFAULTS["degree"],
        help=(f"If the model is of type \"{models.SvmSklearnWrapper().name}\" "
              "and \"--kernel=poly\", then this is the degree of the "
              "polynomial that will be fit. Ignored otherwise."),
        type=int)
    psr.add_argument(
        "--penalty", default=defaults.DEFAULTS["penalty"], choices=["l1", "l2"],
        help=(f"If the model is of type \"{models.SvmSklearnWrapper().name}\", "
              "then use this type of regularization. Ignored otherwise."))
    psr.add_argument(
        "--max-iter", default=defaults.DEFAULTS["max_iter"],
        help=("If the model is an sklearn model, then this is the maximum "
              "number of iterations to use during the fitting process. Ignored "
              "otherwise."),
        type=int)
    psr.add_argument(
        "--rfe", choices=["None", "rfe", "rfecv"], default="None",
        help=(f"If the model is of type \"{models.LrSklearnWrapper().name}\" "
              f"or \"{models.LrCvSklearnWrapper().name}\", then this is the "
              "type of recursive feature elimination to use. Ignored "
              "otherwise."),
        type=str)
    psr.add_argument(
        "--folds", default=defaults.DEFAULTS["folds"],
        help=(f"If the model is of type \"{models.LrCvSklearnWrapper().name}\","
              " then use this number of cross-validation folds."),
        type=int)
    psr.add_argument(
        "--early-stop", action="store_true", help="Enable early stopping.")
    psr.add_argument(
        "--val-patience", default=defaults.DEFAULTS["val_patience"],
        help=("The number of times that the validation loss can increase "
              "before training is automatically aborted."),
        type=int)
    psr.add_argument(
        "--val-improvement-thresh",
        default=defaults.DEFAULTS["val_improvement_thresh"],
        help="Threshold for percept improvement in validation loss.",
        type=float)
    psr.add_argument(
        "--conf-trials", default=defaults.DEFAULTS["conf_trials"],
        help="The number of trials to run.", type=int)
    psr.add_argument(
        "--max-attempts", default=defaults.DEFAULTS["max_attempts"],
        help="The maximum number of failed training attempts to survive.",
        type=int)
    psr.add_argument(
        "--timeout-s", default=defaults.DEFAULTS["timeout_s"],
        help="Automatically stop training after this amount of time (seconds).",
        type=float)
    return psr, lambda args: verify(psr_verify(args))


def add_running(psr, psr_verify=lambda args: args):
    """
    Adds model execution--related arguments to the provided ArgumentParser, and
    returns it.
    """
    def verify(args):
        scl_prms_flp = args.scale_params
        assert path.exists(scl_prms_flp), \
            f"Scale parameters file does not exist: {scl_prms_flp}"
        return args

    psr, psr_verify = add_out(*add_standardize(*add_warmup(psr, psr_verify)))
    psr.add_argument(
        "--scale-params", help="The path to the input scaling parameters.",
        required=True, type=str)
    return psr, lambda args: verify(psr_verify(args))
