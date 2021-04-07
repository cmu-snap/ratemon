#!/usr/bin/env python3
"""
Reads parsed experiment files created by gen_features.py and creates unified
training, validation, and test sets.
"""

import argparse
import math
import os
from os import path
import time
import random

import numpy as np

import cl_args
import utils


class Split:
    """ Represents either the training, validation, or test split. """

    def __init__(self, name, split_frac, sample_frac, out_dir, dtype,
                 num_pkts_tot, shuffle):
        self.name = name
        self.frac = split_frac * sample_frac
        self.shuffle = shuffle
        self.fets = dtype.names

        flp = utils.get_split_data_flp(out_dir, name)
        print(
            f"\tInitializing split \"{self.name}\" "
            f"({split_frac * 100}%, sampling {sample_frac * 100}%"
            f"{', shuffled' if self.shuffle else ''}) at: {flp}")

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
        utils.save_split_metadata(
            out_dir, self.name, dat=(num_pkts, dtype.descr))

    def take(self, exp_dat, exp_available_idxs):
        """
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
        assert num_new > 0 or self.frac == 0, \
            (f"Selecting 0 of {num_exp_pkts} packets, but fraction is: "
             f"{self.frac}")
        # Randomly select the packets to pull into this split. This must be
        # random to capture a diverse set of situations.
        exp_new_idxs = random.sample(exp_available_idxs, num_new)
        exp_available_idxs -= set(exp_new_idxs)

        # Identify the indices in the merged array.
        start_idx = self.idx
        self.idx += num_new
        assert self.idx <= self.dat.shape[0], \
            (f"Index {self.idx} into \"{self.name}\" split does not fit "
             f"within shape {self.dat.shape}")
        # dat_new_idxs = list(range(start_idx, self.idx))

        self.dat[start_idx:self.idx] = exp_dat[exp_new_idxs]
        # self.dat_available_idxs -= set(dat_new_idxs)
        return exp_available_idxs

    def finish(self):
        """
        Finalize this split. A finalized Split cannot have methods called on it.
        """
        self.finished = True
        if self.dat is None:
            # This split does not contain any packets.
            return

        # Mark any unused indices as invalid by filling their values with -1.
        #self.dat[list(self.dat_available_idxs)].fill(-1)
        self.dat[self.idx:].fill(-1)

        # Shuffle in-place at the end. This is only okay because I plan to
        # always store the output in a tmpfs.
        if self.shuffle:
            print(f"Shuffling split {self.name}...")
            tim_srt_s = time.time()
            np.random.default_rng().shuffle(self.dat)
            print(
                f"Done shuffling split {self.name} "
                f"(took {time.time() - tim_srt_s:.2f} seconds)")

        self.dat.flush()


def survey(exp_flps, warmup_frac):
    """
    Surveys the provided experiments to determine their total packets and dtype,
    which are returned.
    """
    num_exps = len(exp_flps)
    print(f"Surveying {num_exps} experiments...")
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
    exp_headers = [
        (exp_flp, utils.get_npz_headers(exp_flp)) for exp_flp in exp_flps]

    # Drop experiments that do not contain any flows.
    to_drop = set()
    for idx, (exp_flp, flw_headers) in enumerate(exp_headers):
        if not flw_headers:
            print(f"Experiment \"{exp_flp}\" does not contain any flows!")
            to_drop.add(idx)
    exp_headers = [
        exp_header for idx, exp_header in enumerate(exp_headers)
        if idx not in to_drop]

    # Extract the dtypes and verify that all dtypes are the same.
    dtype = exp_headers[0][1][0][2]
    for exp_flp, flw_headers in exp_headers:
        for flw_name, _, flw_dtype in flw_headers:
            assert flw_dtype == dtype, \
                (f"Experiment \"{exp_flp}\" flow \"{flw_name}\" dtype does not "
                 "match target detype.")

    # Count the total number of packets by looking at the header shapes.
    save_frac = 1 - warmup_frac
    num_pkts = sum(
        math.ceil(flw_shape[0] * save_frac)
        for _, flw_headers in exp_headers for _, flw_shape, _ in flw_headers)

    return num_pkts, dtype


def merge(exp_flps, out_dir, num_pkts, dtype, split_fracs, warmup_frac,
          sample_frac):
    """
    Merges the provided experiments into training, validation, and
    test splits as defined by the percents in split_fracs. Stores the
    resulting files in out_dir. The experiments contain a total of
    num_pkts packets and have the provided dtype.
    """
    print("Preparing split files...")
    splits = {
        name: Split(
            name, split_frac, sample_frac, out_dir, dtype, num_pkts,
            shuffle=name == "train")
        for name, split_frac in split_fracs.items()}
    # Keep track of the number of packets that do not get selected for
    # any of the splits.
    pkts_forgotten = 0
    num_exps = len(exp_flps)
    for idx, exp_flp in enumerate(exp_flps):
        exp, dat = utils.load_exp(
            exp_flp, msg=f"{idx + 1:{f'0{len(str(num_exps))}'}}/{num_exps}")
        if dat is None:
            print(f"\tError loading {exp_flp}")
            continue

        # Combine flows.
        dat_combined = None
        for flw in range(exp.tot_flws):
            dat_flw = dat[flw]
            if warmup_frac != 0:
                # Remove a percentage of packets from the beginning of the
                # experiment.
                dat_flw = dat_flw[math.floor(dat_flw.shape[0] * warmup_frac):]
            if dat_combined is None:
                dat_combined = dat_flw
            else:
                dat_combined = np.concatenate((dat_combined, dat_flw))
        dat = dat_combined

        # Start with the list of all indices. Each split selects some
        # indices for itself, then removes them from this set.
        all_idxs = set(range(dat.shape[0]))
        # For each split, take a fraction of the experiment packets.
        for split in splits.values():
            all_idxs = split.take(dat, all_idxs)
        # Record how many packets are not being moved to one of the
        # merged files.
        pkts_forgotten += len(all_idxs)
    print(
        f"Forgot {pkts_forgotten}/{num_pkts} packets "
        f"({pkts_forgotten / num_pkts:.2f}%)")

    for split in splits.values():
        split.finish()

    # Delete the splits to force their data to be written to
    # disk. Note that this is not needed for correctness. Since the
    # process of flushing the tables to disk (which occurs when the
    # memory-mapped ndarray object inside each Split object is
    # deleted) may take a long time, we explicitly do so here.
    print("Flushing splits to disk...")
    del splits


def main():
    """ This program's entrypoint. """
    utils.set_rand_seed()

    psr = argparse.ArgumentParser(
        description=(
            "Merges parsed experiment files into unified training, validation, "
            "and test data."))
    psr.add_argument(
        "--data-dir",
        help="The path to a directory containing the experiment files.",
        required=True, type=str)
    psr.add_argument(
        "--train-split", default=50, help="Training data fraction",
        required=False, type=float)
    psr.add_argument(
        "--val-split", default=20, help="Validation data fraction",
        required=False, type=float)
    psr.add_argument(
        "--test-split", default=30, help="Test data fraction",
        required=False, type=float)
    psr, psr_verify = cl_args.add_sample_percent(*cl_args.add_out(
        *cl_args.add_warmup(*cl_args.add_num_exps(psr))))
    args = psr_verify(psr.parse_args())

    split_fracs = {
        "train": args.train_split / 100, "val": args.val_split / 100,
        "test": args.test_split / 100}
    tot_split = sum(split_fracs.values())
    assert tot_split == 1, \
        ("The sum of the training, validation, and test splits must equal 100, "
         f"not {tot_split * 100}")

    tim_srt_s = time.time()
    # Determine the experiment filepaths.
    exps_dir = args.data_dir
    exp_flps = [
        path.join(exps_dir, fln) for fln in os.listdir(exps_dir)
        if fln.endswith(".npz")]
    random.shuffle(exp_flps)
    num_exps = len(exp_flps) if args.num_exps is None else args.num_exps
    exp_flps = exp_flps[:num_exps]
    print(f"Selected {num_exps} experiments")
    warmup_frac = args.warmup_percent / 100
    sample_frac = args.sample_percent / 100
    num_pkts, dtype = survey(exp_flps, warmup_frac)
    print(
        f"Total packets: {num_pkts}\nFeatures ({len(dtype.names)}):\n\t" +
        "\n\t".join(sorted(dtype.names)))

    # Create the merged training, validation, and test files.
    merge(
        exp_flps, args.out_dir, num_pkts, dtype, split_fracs, warmup_frac,
        sample_frac)
    print(f"Finished - time: {time.time() - tim_srt_s:.2f} seconds")
    return 0


if __name__ == "__main__":
    main()
