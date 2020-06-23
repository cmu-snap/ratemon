#! /usr/bin/env python3
""" Utility functions. """

from os import path

import numpy as np
import scapy.utils
import scapy.layers.l2
import scapy.layers.inet
import scapy.layers.ppp
import torch


class Dataset(torch.utils.data.Dataset):
    """ A simple Dataset that wraps arrays of input and output features. """

    def __init__(self, dat_in, dat_out):
        """
        dat_out is assumed to have only a single practical dimension (e.g.,
        dat_out should be of shape (X,), or (X, 1)).
        """
        super(Dataset).__init__()
        shp_in = dat_in.shape
        shp_out = dat_out.shape
        assert shp_in[0] == shp_out[0], \
            "Mismatched dat_in ({shp_in}) and dat_out ({shp_out})!"
        # Convert the numpy arrays to Torch tensors.
        self.dat_in = torch.tensor(dat_in, dtype=torch.float)
        # Reshape the output into a 1D array first, because
        # CrossEntropyLoss expects a single value. The dtype must be
        # long because the loss functions expect longs.
        self.dat_out = torch.tensor(
            dat_out.reshape(shp_out[0]), dtype=torch.long)

    def to(self, dev):
        """ Move the entire dataset to the target device. """
        try:
            # This will fail if there is insufficient memory.
            self.dat_in = self.dat_in.to(dev)
            self.dat_out = self.dat_out.to(dev)
        except RuntimeError:
            print(f"Warning:: Unable to move dataset to device: {dev}")
            # In case the input data was moved successfully but there
            # was insufficient device memory for the output data, move
            # the input data back to main memory.
            self.dat_in = self.dat_in.to(torch.device("cpu"))

    def __len__(self):
        """ Returns the number of items in this Dataset. """
        return len(self.dat_in)

    def __getitem__(self, idx):
        """ Returns a specific (input, output) pair from this Dataset. """
        assert torch.utils.data.get_worker_info() is None, \
            "This Dataset does not support being loaded by multiple workers!"
        return self.dat_in[idx], self.dat_out[idx]

    def raw(self):
        """ Returns the raw data underlying this dataset. """
        return self.dat_in, self.dat_out


class BalancedSampler:
    """
    A batching sampler that creates balanced batches. The batch size
    must be evenly divided by the number of classes. This does not
    inherit from any of the existing Torch Samplers because it does
    not require any of their functionalty. Instead, this is a wrapper
    for many other Samplers, one for each class.
    """

    def __init__(self, dataset, batch_size, drop_last):
        assert isinstance(dataset, Dataset), \
            "Dataset must be an instance of utils.Dataset."
        # Determine the unique classes.
        _, dat_out = dataset.raw()
        clss = set(dat_out.tolist())
        num_clss = len(clss)
        assert batch_size >= num_clss, \
            (f"The batch size ({batch_size}) must be at least as large as the "
             f"number of classes ({num_clss})!")
        assert batch_size % num_clss == 0, \
            (f"The number of classes ({num_clss}) must evenly divide the batch "
             f"size ({batch_size})!")

        print("Balancing classes...")
        # Find the indices for each class.
        clss_idxs = {cls: torch.where(dat_out == cls)[0] for cls in clss}
        # Determine the number of examples in the most populous class.
        max_examples = max(len(cls_idxs) for cls_idxs in clss_idxs.values())
        # Generate new samples to fill in under-represented classes.
        for cls, cls_idxs in clss_idxs.items():
            num_examples = len(cls_idxs)
            # If this class has insufficient examples...
            if num_examples < max_examples:
                new_examples = max_examples - num_examples
                # Duplicate existing examples to make this class balanced.
                # Append the duplicated examples to the true examples.
                clss_idxs[cls] = torch.cat(
                    (cls_idxs,
                     torch.multinomial(
                         # Sample from the existing examples using a uniform
                         # distribution.
                         torch.ones((num_examples,)),
                         num_samples=new_examples,
                         # Sample with replacement in case the number of new
                         # examples is greater than the number of existing
                         # examples.
                         replacement=True)),
                    dim=0)
                print(f"Added {new_examples} examples to class {cls}!")
        # Create a BatchSampler iterator for each class.
        examples_per_cls = batch_size // num_clss
        self.samplers = {
            cls: torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(cls_idxs, replacement=False),
                examples_per_cls, drop_last)
            for cls, cls_idxs in clss_idxs.items()}
        # After __iter__() is called, this will contain an iterator for each
        # class.
        self.iters = {}
        self.num_batches = max_examples // examples_per_cls

    def __iter__(self):
        # Create an iterator for each class.
        self.iters = {
            cls: iter(sampler) for cls, sampler in self.samplers.items()}
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        # Pull examples from each class and merge them into a single list.
        return [idx for it in self.iters.values() for idx in next(it)]


class Sim():
    """ Describes the parameters of a simulation. """

    def __init__(self, sim):
        if "/" in sim:
            sim = path.basename(sim)
        self.name = sim
        toks = sim.split("-")
        if sim.endswith(".npz"):
            # 8Mbps-9000us-489p-1unfair-4other-9000,9000,9000,9000,9000us-1380B-80s-2rttW.npz
            toks = toks[:-1]
        # 8Mbps-9000us-489p-1unfair-4other-9000,9000,9000,9000,9000us-1380B-80s
        (bw_Mbps, btl_delay_us, queue_p, unfair_flws, other_flws, edge_delays,
         payload_B, dur_s) = toks

        # Link bandwidth (Mbps).
        self.bw_Mbps = float(bw_Mbps[:-4])
        # Bottleneck router delay (us).
        self.btl_delay_us = float(btl_delay_us[:-2])
        # Queue size (packets).
        self.queue_p = float(queue_p[:-1])
        # Number of unfair flows
        self.unfair_flws = int(unfair_flws[:-6])
        # Number of other flows
        self.other_flws = int(other_flws[:-5])
        # Edge delays
        self.edge_delays = [int(del_us) for del_us in edge_delays[:-2].split(",")]
        # Packet size (bytes)
        self.payload_B = float(payload_B[:-1])
        # Experiment duration (s).
        self.dur_s = float(dur_s[:-1])


def parse_packets_endpoint(flp, packet_size_B):
    """
    Takes in a file path and returns (seq, timestamp).
    """
    # Not using parse_time_us for efficiency purpose
    return [
        (scapy.layers.ppp.PPP(pkt_dat)[scapy.layers.inet.TCP].seq,
         pkt_mdat.sec * 1e6 + pkt_mdat.usec)
        for pkt_dat, pkt_mdat in scapy.utils.RawPcapReader(flp)
        # Ignore non-data packets.
        if pkt_mdat.wirelen >= packet_size_B
    ]


def parse_packets_router(flp, packet_size_B):
    """
    Takes in a file path and returns (sender, timestamp).
    """
    # Not using parse_time_us for efficiency purpose
    return [
        # Parse each packet as a PPP packet.
        (int(scapy.layers.ppp.PPP(pkt_dat)[
            scapy.layers.inet.IP].src.split(".")[2]),
         pkt_mdat.sec * 1e6 + pkt_mdat.usec)
        for pkt_dat, pkt_mdat in scapy.utils.RawPcapReader(flp)
        # Ignore non-data packets.
        if pkt_mdat.wirelen >= packet_size_B
    ]


def scale(val, min_in, max_in, min_out, max_out):
    """
    Scales val, which is from the range [min_in, max_in], to the range
    [min_out, max_out].
    """
    assert min_in != max_in, "Divide by zero!"
    return min_out + (val - min_in) * (max_out - min_out) / (max_in - min_in)


def load_sim(flp, msg=None):
    """
    Loads one simulation results file (generated by parse_dumbbell.py). Returns
    a tuple of the form: (total number of flows, results matrix).
    """
    print(f"{'' if msg is None else f'{msg} - '}Parsing: {flp}")
    with np.load(flp) as fil:
        assert len(fil.files) == 1 and "1" in fil.files, \
            "More than one unfair flow detected!"
        return Sim(flp), fil["1"]


def clean(arr):
    """
    "Cleans" the provided numpy array by removing its column names. I.e., this
    converts a structured numpy array into a regular numpy array. Assumes that
    dtypes can be converted to float. If the
    """
    assert arr.dtype.names is not None, \
        f"The provided array is not structured. dtype: {arr.dtype.descr}"
    num_dims = len(arr.shape)
    assert num_dims == 1, \
        (f"Only 1D structured arrays are supported, but this one has {num_dims} "
         "dims!")

    num_cols = len(arr.dtype.descr)
    new = np.empty((arr.shape[0], num_cols), dtype=float)
    for col in range(num_cols):
        new[:, col] = arr[arr.dtype.names[col]]
    return new
