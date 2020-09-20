#!/usr/bin/env python3
"""
Reads parsed simulation files created by parse_dumbell.py and
creates unified training, validation, and test sets.
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

    def __init__(self, name, prc, out_dir, dtype, num_pkts_tot, shuffle):
        self.name = name
        print(f"Initializing split \"{self.name}\"...")
        self.frac = prc / 100
        self.shuffle = shuffle
        if self.shuffle:
            print(f"Split \"{self.name}\" will be shuffled")
        # Track where this Split has been finalized, in which case it
        # cannot have methods called on it.
        self.finished = False

        num_pkts = math.ceil(num_pkts_tot * self.frac)
        # Create an empty file for each split, and set all entries
        # to -1. This matches the behavior of parse_dumbbell.py:
        # Features values that cannot be computed are replaced
        # with -1. When reading the splits later, we can detect
        # incomplete feature values by looking for -1s.
        flp = path.join(out_dir, f"{self.name}.npy")
        self.dat = np.memmap(
            flp, dtype=(dtype + [("num_flws", "int32")]), mode="w+",
            shape=(num_pkts,))

        # The next available index in self.dat. Used if self.shuffle == False.
        self.idx = 0
        # List of all available indices in this split. This set is
        # reduced as the split is populated.
        self.dat_available_idxs = set(range(num_pkts))

        # Save this Split's metadata so that its data file can be !read later.
        utils.save_split_metadata(
            out_dir, self.name, dat=(num_pkts, self.dat.dtype.descr))


    def take(self, sim_dat, sim_available_idxs, sim):
        """
        Takes this Split's specified fraction of data from sim_dat,
        choosing from sim_available_idxs. Removes the chosen indices from
        sim_available_idxs and returns the modified version.
        """
        # Need to append a column for the number of flows

        assert not self.finished, "Trying to call a method on a finished Split."
        num_sim_pkts = sim_dat.shape[0]
        num_new = math.floor(num_sim_pkts * self.frac)
        # Verify that if a split fraction was nonzero, then at least
        # one packet was selected. This is a common-case heuristic
        # rather than an invariant, since it is reasonable for zero
        # packets to be selected if the simulation has either very few
        # packets or the split fraction is very low.
        assert num_new > 0 or self.frac == 0, \
            (f"Selecting 0 of {num_sim_pkts} packets, but fraction is: "
             f"{self.frac}")
        # Randomly select the packets to pull into this split.
        sim_new_idxs = random.sample(sim_available_idxs, num_new)
        sim_available_idxs -= set(sim_new_idxs)

        if self.shuffle:
            num_slots_remaining = len(self.dat_available_idxs)
            assert num_slots_remaining >= num_new, \
                (f"Trying to find locations for {num_new} new packets when "
                 f"there are only {num_slots_remaining} packet slots "
                 "available!")
            dat_new_idxs = random.sample(self.dat_available_idxs, num_new)
            selected_rows = self.dat[dat_new_idxs]
        else:
            start_idx = self.idx
            self.idx = self.idx + num_new
            assert self.idx <= self.dat.shape[0], \
                (f"Index {self.idx} into \"{self.name}\" split does not fit "
                 f"within shape {self.dat.shape}")
            dat_new_idxs = range(start_idx, self.idx)
            selected_rows = self.dat[start_idx:self.idx]
        # Assign the feature values.
        num_cols = len(self.dat.dtype.names)
        selected_rows[:num_cols - 1] = sim_dat[sim_new_idxs]
        # Assign the number of flows.
        selected_rows[num_cols].fill(
            sim.unfair_flws + sim.fair_flws)

        self.dat_available_idxs -= set(dat_new_idxs)
        return sim_available_idxs

    def finish(self):
        """
        Finalize this split. A finalized Split cannot have methods called on it.
        """
        self.finished = True
        # Mark any unused indices as invalid by filling their values
        # with -1.
        self.dat[list(self.dat_available_idxs)].fill(-1)


def survey(sim_flps, warmup_frac):
    """
    Surveys the provided simulations to determine their total packets and dtype,
    which are returned.
    """
    num_sims = len(sim_flps)
    print(f"Surveying {num_sims} simulations...")
    assert sim_flps, "Must provide at least one simulation."

    # Extract the header from each simulation. Reshape to remove a
    # middle dimension of size 1.
    headers = np.array(
        [utils.get_npz_headers(flp) for flp in sim_flps]).reshape((num_sims, 3))
    # Count the total number of packets by looking at the header shapes.
    save_frac = 1 - warmup_frac
    num_pkts = sum(math.ceil(shape[0] * save_frac) for shape in headers[:, 1])
    # Extract the dtype and verify that all dtypes are the same.
    dtype = headers[0][2]
    assert (headers[1:][:, 2] == dtype).all(), \
        "Not all simulations agree on dtype!"
    return num_pkts, dtype


def merge(sim_flps, out_dir, num_pkts, dtype, split_prcs, warmup_frac):
    """
    Merges the provided simulations into training, validation, and
    test splits as defined by the percents in split_prcs. Stores the
    resulting files in out_dir. The simulations contain a total of
    num_pkts packets and have the provided dtype.
    """
    print("Preparing split files...")
    splits = {
        name: Split(
            name, prc, out_dir, dtype, num_pkts,
            shuffle=name == "train")
        for name, prc in split_prcs.items()}
    # Keep track of the number of packets that do not get selected for
    # any of the splits.
    pkts_forgotten = 0
    num_sims = len(sim_flps)
    for idx, sim_flp in enumerate(sim_flps):
        # Load the simulation.
        sim, dat = utils.load_sim(
            sim_flp, msg=f"{idx + 1:{f'0{len(str(num_sims))}'}}/{num_sims}")
        if dat is None:
            continue
        # Remove a percentage of packets from the beginning of the
        # simulation.
        dat = dat[math.floor(dat.shape[0] * warmup_frac):]
        # Start with the list of all indices. Each split select some
        # indices for itself, then removes them from this set.
        all_idxs = set(range(dat.shape[0]))
        # For each split, take a fraction of the simulation packets.
        for split in splits.values():
            all_idxs = split.take(dat, all_idxs, sim)
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
            "Merges parsed simulation files into unified training, validation, "
            "and test data."))
    psr.add_argument(
        "--data-dir",
        help="The path to a directory containing the simulation files.",
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
    psr, psr_verify = cl_args.add_out(
        *cl_args.add_warmup(*cl_args.add_num_sims(psr)))
    args = psr_verify(psr.parse_args())

    split_prcs = {
        "train": args.train_split, "val": args.val_split,
        "test": args.test_split}
    tot_split = sum(split_prcs.values())
    assert tot_split == 100, \
        ("The sum of the training, validation, and test splits must equal 100, "
         f"not {tot_split}")

    tim_srt_s = time.time()
    # Determine the simulation filepaths.
    sims_dir = args.data_dir
    sim_flns = os.listdir(sims_dir)
    random.shuffle(sim_flns)
    num_sims = args.num_sims
    num_sims = len(sim_flns) if num_sims is None else num_sims
    print(f"Selected {num_sims} simulations")
    sim_flps = [
        path.join(sims_dir, sim_fln) for sim_fln in sim_flns[:num_sims]]
    warmup_frac = args.warmup_percent / 100
    num_pkts, dtype = survey(sim_flps, warmup_frac)
    fets = dtype.names
    print(
        f"Total packets: {num_pkts}\nFeatures:\n    " +
        "\n    ".join(sorted(fets)))

    # Create the merged training, validation, and test files.
    merge(sim_flps, args.out_dir, num_pkts, dtype, split_prcs, warmup_frac)
    print(f"Finished - time: {time.time() - tim_srt_s:.2f} seconds")
    return 0


if __name__ == "__main__":
    main()
