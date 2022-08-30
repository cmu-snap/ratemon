#!/usr/bin/env python3
"""Creates unified training data from separate experiments.

Reads parsed experiment files created by gen_features.py and creates unified
training, validation, and test sets.
"""

import argparse
import json
import logging
import math
import os
from os import path
import time
import random
import sys

import numpy as np

from unfair.model import cl_args, defaults, features, models, utils


SPLIT_NAMES = ["train", "val", "test"]


class Split:
    """Represents either the training, validation, or test split."""

    def __init__(
        self, name, split_frac, sample_frac, out_dir, dtype, num_pkts_tot, shuffle
    ):
        self.name = name
        self.frac = split_frac * sample_frac
        self.shuffle = shuffle
        self.fets = [name for name, typ in dtype]

        if self.frac == 0:
            logging.info("Skipping split %s because it will select no packets.", name)
            self.finished = True
            return

        flp = utils.get_split_data_flp(out_dir, name)
        if split_exists(out_dir, name):
            raise Exception(f"Split already exists: {flp}")
        logging.info(
            '\tInitializing split "%s" (%d%%, sampling %d%%%s) at: %s',
            self.name,
            split_frac * 100,
            sample_frac * 100,
            ", shuffled" if self.shuffle else "",
            flp,
        )

        # Track where this Split has been finalized, in which case it
        # cannot have methods called on it.
        self.finished = False

        num_pkts = math.ceil(num_pkts_tot * self.frac)
        if num_pkts == 0:
            self.dat = None
        else:
            # Create an empty file for each split. Features values that cannot
            # be computed are replaced with -1. When reading the splits later,
            # we can detect incomplete feature values by looking for -1s.
            self.dat = np.memmap(flp, dtype=dtype, mode="w+", shape=(num_pkts,))

        # The next available index in self.dat.
        self.idx = 0
        # # List of all available indices in this split. This set is
        # # reduced as the split is populated.
        # self.dat_available_idxs = set(range(num_pkts))

        # Save this Split's metadata so that its data file can be read later.
        utils.save_split_metadata(out_dir, self.name, dat=(num_pkts, dtype))

    def take(self, exp_dat, exp_available_idxs):
        """Bring additional samples into this Split.

        Takes this Split's specified fraction of data from exp_dat,
        choosing from exp_available_idxs. Removes the chosen indices from
        exp_available_idxs and returns the modified version.
        """
        assert not self.finished, "Trying to call a method on a finished Split."
        if self.dat is None:
            # This split is not supposed to select any packets.
            return exp_available_idxs

        num_exp_pkts = exp_dat.shape[0]
        num_new = math.floor(num_exp_pkts * self.frac)
        # Verify that if a split fraction was nonzero, then at least
        # one packet was selected. This is a common-case heuristic
        # rather than an invariant, since it is reasonable for zero
        # packets to be selected if the experiment has either very few
        # packets or the split fraction is very low.
        assert num_new > 0 or self.frac == 0, (
            f"Selecting 0 of {num_exp_pkts} packets, but fraction is: " f"{self.frac}"
        )
        # Randomly select the packets to pull into this split. This must be
        # random to capture a diverse set of situations.
        exp_new_idxs = random.sample(exp_available_idxs, num_new)
        exp_available_idxs -= set(exp_new_idxs)

        # Identify the indices in the merged array.
        start_idx = self.idx
        self.idx += num_new
        assert self.idx <= self.dat.shape[0], (
            f'Index {self.idx} into "{self.name}" split does not fit '
            f"within shape {self.dat.shape}"
        )
        # dat_new_idxs = list(range(start_idx, self.idx))

        self.dat[start_idx : self.idx] = exp_dat[exp_new_idxs]
        # self.dat_available_idxs -= set(dat_new_idxs)
        return exp_available_idxs

    def finish(self):
        """Finalize this split, and maybe shuffle it.

        A finalized Split cannot have methods called on it.
        """
        if self.finished:
            return

        self.finished = True
        if self.dat is None:
            # This split does not contain any packets.
            return

        # Mark any unused indices as invalid by filling their values with -1.
        # self.dat[list(self.dat_available_idxs)].fill(-1)
        self.dat[self.idx :].fill(-1)

        # Shuffle in-place at the end. This is only okay because I plan to
        # always store the output in a tmpfs.
        if self.shuffle:
            logging.info('Shuffling split "%s"...', self.name)
            tim_srt_s = time.time()
            np.random.default_rng().shuffle(self.dat)
            logging.info(
                'Done shuffling split "%s" (took %0.2f seconds)',
                self.name,
                time.time() - tim_srt_s,
            )

        self.dat.flush()


def survey(exp_flps, warmup_frac):
    """Determine total packets and dtype of valid experiments.

    Surveys the provided experiments to determine their total packets and dtype,
    which are returned. Also returns the list of experiment filepaths filtered
    to remove those whose output file is invalid.
    """
    num_exps = len(exp_flps)
    logging.info("Surveying %d experiments...", num_exps)
    assert exp_flps, "Must provide at least one experiment."

    # Produces a nested list of the form:
    #     [ list of tuples, each element corresponding to an experiment:
    #         (
    #             experiment filepath,
    #             [ list of tuples, each element corresponding to a flow:
    #               tuple of the form: (name, shape, dtype)
    #             ]
    #         )
    #     ]
    num_exps_original = len(exp_flps)
    exp_headers = [(exp_flp, utils.get_npz_headers(exp_flp)) for exp_flp in exp_flps]
    # Remove experiments whose headers could not be read.
    exp_headers = [(exp_flp, headers) for exp_flp, headers in exp_headers if headers]
    # Extract the filepaths of the experiments whose headers could be read. This
    # list will be returned.
    exp_flps = list(zip(*exp_headers))[0]
    num_exps_invalid = num_exps_original - len(exp_flps)
    if num_exps_invalid:
        logging.info("Warning: Removed %d invalid experiments!", num_exps_invalid)
    assert exp_headers, "Error: No valid experiments!"

    # Drop experiments that do not contain any flows.
    to_drop = set()
    for idx, (exp_flp, flw_headers) in enumerate(exp_headers):
        if not flw_headers:
            logging.info('Experiment "%s" does not contain any flows!', exp_flp)
            to_drop.add(idx)
    exp_headers = [
        exp_header for idx, exp_header in enumerate(exp_headers) if idx not in to_drop
    ]

    # Extract the dtypes and verify that all dtypes are the same.
    dtype = exp_headers[0][1][0][2]
    for exp_flp, flw_headers in exp_headers:
        for flw_name, _, flw_dtype in flw_headers:
            assert flw_dtype == dtype, (
                f'Experiment "{exp_flp}" flow "{flw_name}" dtype does not '
                "match target detype."
            )

    # Count the total number of packets by looking at the header shapes.
    save_frac = 1 - warmup_frac
    num_pkts = sum(
        math.ceil(flw_shape[0] * save_frac)
        for _, flw_headers in exp_headers
        for _, flw_shape, _ in flw_headers
    )

    return exp_flps, num_pkts, dtype


def merge(exp_flps, out_dir, num_pkts, dtype, split_fracs, warmup_frac, sample_frac):
    """Merge the provided experiments into training, validation, and test splits.

    Uses the defined by the percents in split_fracs. Stores the resulting files
    in out_dir. The experiments contain a total of num_pkts packets and have the
    provided dtype.
    """
    logging.info("Preparing split files...")
    splits = {
        name: Split(
            name,
            split_frac,
            sample_frac,
            out_dir,
            dtype,
            num_pkts,
            # Shuffle all splits so that later attempts to select a range of packets
            # from the beginning are random.
            shuffle=True,
        )
        for name, split_frac in split_fracs.items()
    }
    # Keep track of the number of packets that do not get selected for
    # any of the splits.
    pkts_forgotten = 0
    num_exps = len(exp_flps)
    dtype_names = [name for name, typ in dtype]
    for idx, exp_flp in enumerate(exp_flps):
        exp, dat = utils.load_exp(
            exp_flp, msg=f"{idx + 1:{f'0{len(str(num_exps))}'}}/{num_exps}"
        )
        if dat is None:
            logging.info("\tError loading %s", exp_flp)
            continue

        # Combine flows.
        dat_combined = None
        for flw in range(exp.tot_flws):
            dat_flw = dat[flw]
            if warmup_frac != 0:
                # Remove a percentage of packets from the beginning of the
                # experiment.
                dat_flw = dat_flw[math.floor(dat_flw.shape[0] * warmup_frac) :]
            if dat_combined is None:
                dat_combined = dat_flw
            else:
                dat_combined = np.concatenate((dat_combined, dat_flw))
        dat = dat_combined

        # Select only the features that the model requires.
        dat = dat[dtype_names]

        # Start with the list of all indices. Each split selects some
        # indices for itself, then removes them from this set.
        all_idxs = set(range(dat.shape[0]))
        # For each split, take a fraction of the experiment packets.
        for split in splits.values():
            if not split.finished:
                all_idxs = split.take(dat, all_idxs)
        # Record how many packets are not being moved to one of the
        # merged files.
        pkts_forgotten += len(all_idxs)
    logging.info(
        "Forgot %d/%d packets (%0.2f%%)",
        pkts_forgotten,
        num_pkts,
        pkts_forgotten / num_pkts * 100,
    )

    for split in splits.values():
        split.finish()

    # Delete the splits to force their data to be written to
    # disk. Note that this is not needed for correctness. Since the
    # process of flushing the tables to disk (which occurs when the
    # memory-mapped ndarray object inside each Split object is
    # deleted) may take a long time, we explicitly do so here.
    logging.info("Flushing splits to disk...")
    del splits


def splits_exist(out_dir):
    """Check if all splits exist in out_dir."""
    for name in SPLIT_NAMES:
        if not split_exists(out_dir, name):
            return False
    return True


def split_exists(out_dir, split_name):
    """Check if a particular split exists in out_dir."""
    exists = path.exists(utils.get_split_data_flp(out_dir, split_name)) and path.exists(
        utils.get_split_metadata_flp(out_dir, split_name)
    )
    print("Split %s exists: %s" % (split_name, exists))
    return exists


def parse_args():
    psr = argparse.ArgumentParser(
        description=(
            "Merges parsed experiment files into unified training, validation, "
            "and test data."
        )
    )
    psr.add_argument(
        "--data-dir",
        help="The path to a directory containing the experiment files.",
        required=True,
        type=str,
    )
    psr.add_argument(
        "--train-split",
        default=50,
        help="Training data fraction",
        required=False,
        type=float,
    )
    psr.add_argument(
        "--val-split",
        default=20,
        help="Validation data fraction",
        required=False,
        type=float,
    )
    psr.add_argument(
        "--test-split",
        default=30,
        help="Test data fraction",
        required=False,
        type=float,
    )
    psr.add_argument(
        "--model",
        choices=models.MODEL_NAMES,
        help="Optionally select only the features relevant to this model.",
        required=False,
        type=str,
    )
    psr.add_argument(
        "--disjoint-splits",
        action="store_true",
        help="If set, splits will be taken from disjoint sets of experiments.",
    )
    psr.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If set, assume that splits already found on disk are complete and do "
            "not signify an error or need to be regenerated."
        ),
    )
    psr, psr_verify = cl_args.add_sample_percent(
        *cl_args.add_out(*cl_args.add_warmup(*cl_args.add_num_exps(psr)))
    )
    args = psr_verify(psr.parse_args())

    split_fracs = {
        "train": args.train_split / 100,
        "val": args.val_split / 100,
        "test": args.test_split / 100,
    }
    tot_split = sum(split_fracs.values())
    assert tot_split == 1, (
        "The sum of the training, validation, and test splits must equal 100, "
        f"not {tot_split * 100}"
    )

    assert not splits_exist(
        args.out_dir
    ), f"Not regenerating splits because they already exist in: {args.out_dir}"

    return args, split_fracs


def _main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG,
    )
    utils.set_rand_seed()
    args, split_fracs = parse_args()

    tim_srt_s = time.time()
    # Determine the experiment filepaths.
    exps_dir = args.data_dir
    exp_flps = [
        path.join(exps_dir, fln)
        for fln in os.listdir(exps_dir)
        if not fln.startswith(defaults.DATA_PREFIX) and fln.endswith(".npz")
    ]

    random.shuffle(exp_flps)
    num_exps = len(exp_flps) if args.num_exps is None else args.num_exps
    exp_flps = exp_flps[:num_exps]
    logging.info("Selected %d experiments", num_exps)

    if args.disjoint_splits:
        # Split the experiments into disjoint sets of experiments. Split based on the
        # number of experiments and not the number of packets because the number of
        # experiments is directly tied to the diversisty of a set of experiments.
        logging.info("Building disjoint splits")

        # Determine the number of experiments in each split. Since we are using ceil(),
        # these values may sum to more than the total number of experiments.
        num_exps_tot = len(exp_flps)
        num_exps_per_split = {
            split: math.ceil(num_exps_tot * split_frac)
            for split, split_frac in split_fracs.items()
        }
        for split, num_exps in num_exps_per_split.items():
            assert 0 <= num_exps < num_exps_tot, (
                f"Number of exps per split must be in the range [0, {num_exps_tot}], "
                f"but for split {split} is: {num_exps}"
            )

        # Shuffle the experiments and divide them between the splits. Select test,
        # validation, and training experiments in that order so that any border
        # conditions from using ceil() end up decreasing the number of training
        # experiments only.
        random.shuffle(exp_flps)
        exp_flps_per_split = {
            "test": exp_flps[: num_exps_per_split["test"]],
            "val": exp_flps[
                num_exps_per_split["test"] : num_exps_per_split["test"]
                + num_exps_per_split["val"]
            ],
            "train": exp_flps[num_exps_per_split["test"] + num_exps_per_split["val"] :],
        }

        # Do the actual splitting.
        for split_name in SPLIT_NAMES:
            if args.resume and split_exists(args.out_dir, split_name):
                # If we are resuming, do not attempt to regenerate splits that
                # already exist.
                logging.info("Skipping split %s", split_name)
                continue
            logging.info("Creating disjoint %s split", split_name)
            do_prepare(
                args,
                {
                    split_name_: (1 if split_name_ == split_name else 0)
                    for split_name_ in SPLIT_NAMES
                },
                exp_flps_per_split[split_name],
            )
    else:
        exp_flps_per_split = {"all": exp_flps}
        do_prepare(
            args,
            {
                split_name: split_frac
                for split_name, split_frac in split_fracs
                # If we are resuming, do not attempt to regenerate splits that already
                # exist.
                if not args.resume or not split_exists(args.out_dir, split_name)
            },
            exp_flps,
        )

    # Record which experiments are used for each split so that we do not reuse any
    # of these for evaluation later.
    with open(path.join(args.out_dir, "used_exps.json"), "w", encoding="utf-8") as fil:
        json.dump(
            # Extract the filename from each filepath.
            {
                key: [path.basename(fln) for fln in val]
                for key, val in exp_flps_per_split.items()
            },
            fil,
            indent=4,
        )

    logging.info("Finished - time: %0.2f seconds", time.time() - tim_srt_s)
    return 0


def do_prepare(args, split_fracs, exp_flps):
    if not exp_flps:
        logging.info("No experiments found.")
        return

    warmup_frac = args.warmup_percent / 100
    sample_frac = args.sample_percent / 100
    exp_flps, num_pkts, dtype = survey(exp_flps, warmup_frac)
    logging.info(
        "Total packets: %d\nAll found features (%d):\n\t%s",
        num_pkts,
        len(dtype.names),
        "\n\t".join(sorted(dtype.names)),
    )

    # Assemble the minimum dtype.
    dtype = dtype.descr
    if args.model is not None:
        model = models.MODELS[args.model]()
        # We need to keep the model's input and output features, as well as the extra
        # features that are used for analysis.
        required = set(model.in_spc) | set(model.out_spc) | set(features.EXTRA_FETS)
        new_dtype = []
        for field in dtype:
            if field[0] in required:
                new_dtype.append(field)
                required -= {field[0]}
        assert (
            len(required) == 0
        ), f"Did not find all required features in surveyed files. Missing: {required}"
        dtype = new_dtype
    logging.info(
        "Using minimum dtype: \n\t%s", "\n\t".join(sorted(str(fet) for fet in dtype))
    )

    # Create the merged training, validation, and test files.
    merge(
        exp_flps, args.out_dir, num_pkts, dtype, split_fracs, warmup_frac, sample_frac
    )


if __name__ == "__main__":
    sys.exit(_main())
