""" Utility functions. """

import collections
import json
import math
import os
from os import path
import pickle
import random
import sys
import time
import zipfile

from matplotlib import pyplot as plt
import numpy as np
import scapy
import scapy.layers.l2
import scapy.layers.inet
import scapy.utils
from scipy import stats
from scipy import cluster
from sklearn import ensemble
from sklearn import feature_selection
from sklearn import inspection
import torch

import defaults


class Dataset(torch.utils.data.Dataset):
    """ A simple Dataset that wraps arrays of input and output features. """

    def __init__(self, fets, dat_in, dat_out, dat_extra):
        """
        fets: List of input feature names, corresponding to the columns of
            dat_in.
        dat_in: Numpy array of input data.
        dat_out: Numpy array of output data. Assumed to have a single practical
            dimension only (e.g., dat_out should be of shape (X,), or (X, 1)).
        dat_extra: Numpy array of extra data.
        """
        super(Dataset).__init__()
        shp_in = dat_in.shape
        shp_out = dat_out.shape
        assert shp_in[0] == shp_out[0], \
            f"Mismatched dat_in ({shp_in}) and dat_out ({shp_out})!"
        num_fets = len(fets)
        assert shp_in[1] == num_fets, \
            f"Mismatched dat_in ({shp_in}) and fets (len: {num_fets})"

        self.fets = fets
        # Convert the numpy arrays to Torch tensors.
        self.dat_in = torch.tensor(dat_in, dtype=torch.float)
        # Reshape the output into a 1D array first, because
        # CrossEntropyLoss expects a single value. The dtype must be
        # long because the loss functions expect longs.
        self.dat_out = torch.tensor(
            dat_out.reshape(shp_out[0]), dtype=torch.long)
        # Do not convert dat_extra to a Torch tensor because it will
        # not interact with models.
        self.dat_extra = dat_extra

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
        return self.dat_in.size()[0]

    def __getitem__(self, idx):
        """ Returns a specific (input, output) pair from this Dataset. """
        assert torch.utils.data.get_worker_info() is None, \
            "This Dataset does not support being loaded by multiple workers!"
        return self.dat_in[idx], self.dat_out[idx]

    def raw(self):
        """ Returns the raw data underlying this dataset. """
        return self.fets, self.dat_in, self.dat_out, self.dat_extra


class BalancedSampler:
    """
    A batching sampler that creates balanced batches. The batch size
    must be evenly divided by the number of classes. This does not
    inherit from any of the existing Torch Samplers because it does
    not require any of their functionalty. Instead, this is a wrapper
    for many other Samplers, one for each class.
    """

    def __init__(self, dataset, batch_size, drop_last, drop_popular):
        assert isinstance(dataset, Dataset), \
            "Dataset must be an instance of utils.Dataset."
        _, _, dat_out, _ = dataset.raw()
        assert_tensor(dat_out=dat_out)

        # Determine the unique classes.
        clss = set(dat_out.tolist())
        num_clss = len(clss)
        assert batch_size >= num_clss, \
            (f"The batch size ({batch_size}) must be at least as large as the "
             f"number of classes ({num_clss})!")
        # If we set the batch size, then it must be evenly divisible by the
        # number of classes.
        assert (batch_size == sys.maxsize) or (batch_size % num_clss == 0), \
            (f"The number of classes ({num_clss}) must evenly divide the batch "
             f"size ({batch_size})!")

        print("Balancing classes...")
        # Find the indices for each class.
        clss_idxs = {cls: torch.where(dat_out == cls)[0] for cls in clss}

        if drop_popular:
            # Determine the number of examples in the least populous class.
            target_examples = min(
                cls_idxs.size()[0] for cls_idxs in clss_idxs.values())
            # Remove samples from the popular classes.
            for cls, cls_idxs in clss_idxs.items():
                num_examples = cls_idxs.size()[0]
                # If this class has too many examples...
                if num_examples > target_examples:
                    # Select a subset of the samples.
                    clss_idxs[cls] = cls_idxs[torch.multinomial(
                        # Sample from the existing examples using a uniform
                        # distribution.
                        torch.ones((num_examples,)),
                        num_samples=target_examples,
                        # Do not sample with replacement because num_samples is
                        # guaranteed to be greater than or equal to
                        # target_samples.
                        replacement=False)]
                    print(
                        f"\tRemoved {num_examples - target_examples} examples "
                        f"from class {cls}.")

        else:
            # Determine the number of examples in the most populous class.
            target_examples = max(
                cls_idxs.size()[0] for cls_idxs in clss_idxs.values())
            # Generate new samples to fill in under-represented classes.
            for cls, cls_idxs in clss_idxs.items():
                num_examples = cls_idxs.size()[0]
                # If this class has insufficient examples...
                if num_examples < target_examples:
                    new_examples = target_examples - num_examples
                    # Duplicate existing examples to make this class balanced.
                    # Append the duplicated examples to the true examples.
                    clss_idxs[cls] = torch.cat(
                        (cls_idxs,
                         cls_idxs[torch.multinomial(
                             # Sample from the existing examples using a uniform
                             # distribution.
                             torch.ones((num_examples,)),
                             num_samples=new_examples,
                             # Sample with replacement in case the number of new
                             # examples is greater than the number of existing
                             # examples.
                             replacement=True)]),
                        dim=0)
                    print(f"\tAdded {new_examples} examples to class {cls}.")

        # Create a BatchSampler iterator for each class.
        examples_per_cls = batch_size // num_clss
        self.samplers = {
            cls: torch.utils.data.BatchSampler(
                torch.utils.data.SubsetRandomSampler(cls_idxs),
                examples_per_cls, drop_last)
            for cls, cls_idxs in clss_idxs.items()}
        # After __iter__() is called, this will contain an iterator for each
        # class.
        self.iters = {}
        self.num_batches = target_examples // examples_per_cls

    def __iter__(self):
        # Create an iterator for each class.
        self.iters = {
            cls: iter(sampler) for cls, sampler in self.samplers.items()}
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        # Pull examples from each class and merge them into a single list.
        idxs = [idx for it in self.iters.values() for idx in next(it)]
        random.shuffle(idxs)
        return idxs


class Exp():
    """ Describes the parameters of a simulation. """

    def __init__(self, sim):
        if "/" in sim:
            sim = path.basename(sim)
        self.name = sim
        toks = sim.split("-")
        if sim.endswith(".tar.gz"):
            # unfair-pcc-cubic-8bw-30rtt-64q-1pcc-1cubic-100s-20201118T114242.tar.gz
            # Remove ".tar.gz" from the last token.
            toks[-1] = toks[-1][:-7]
            # Update sim.name.
            self.name = self.name[:-7]
        elif sim.endswith(".npz"):
            # Remove ".npz" from the last token.
            toks[-1] = toks[-1][:-4]
            # Update sim.name.
            self.name = self.name[:-4]
        # unfair-pcc-cubic-8bw-30rtt-64q-1pcc-1cubic-100s-20201118T114242
        (_, self.cca_1_name, self.cca_2_name, bw_Mbps, rtt_ms, queue_p,
         cca_1_flws, cca_2_flws, end_time, _) = toks
        # Link bandwidth (Mbps).
        self.bw_Mbps = float(bw_Mbps[:-2])
        # Bottleneck router delay (us).
        self.rtt_us = float(rtt_ms[:-3]) * 1000
        # Bandwidth-delay product (bits).
        self.bdp_b = self.bw_Mbps * self.rtt_us
        # Queue size (packets).
        self.queue_p = float(queue_p[:-1])
        # Queue size (multiples of the BDP).
        self.queue_bdp = self.queue_p / (self.bdp_b / 8 / 1514)
        # Number of CCA 1 flows.
        self.cca_1_flws = int(cca_1_flws[:-(len(self.cca_1_name))])
        # Number of CCA 2 flows.
        self.cca_2_flws = int(cca_2_flws[:-(len(self.cca_2_name))])
        # The total number of flows.
        self.tot_flws = self.cca_1_flws + self.cca_2_flws
        # Experiment duration (s).
        self.dur_s = int(end_time[:-1])
        # Largest RTT that this experiment should experiment, based on the size
        # of the bottleneck queue and the RTT.
        self.calculated_max_rtt_us = (self.queue_bdp + 1) * self.rtt_us
        # Fair share bandwidth for each flow.
        self.target_per_flow_bw_Mbps = (
            self.bw_Mbps / (self.cca_1_flws + self.cca_2_flws))


def args_to_str(args, order):
    """
    Converts the provided arguments dictionary to a string, using the
    keys in order to determine the order of arguments in the
    string.
    """
    for key in order:
        assert key in args, f"Key {key} not in args: {args}"
    return "-".join(
        [str(args[key]) for key in order if key not in defaults.ARGS_TO_IGNORE])


def str_to_args(args_str, order):
    """
    Converts the provided string of arguments to a dictionary, using
    the keys in order to determine the identity of each argument in the
    string.
    """
    # Remove extension and split on "-".
    toks = ".".join(args_str.split(".")[:-1]).split("-")
    # Remove elements of order that args_to_str() does not use when
    # encoding strings.
    order = [key for key in order if key not in defaults.ARGS_TO_IGNORE]
    num_toks = len(toks)
    num_order = len(order)
    assert num_toks == num_order, \
        (f"Mismatched tokens ({num_toks}) and order ({num_order})! "
         "tokens: {toks}, order: {order}")
    parsed = {}
    for arg, tok in zip(order, toks):
        try:
            parsed_val = float(tok)
        except ValueError:
            parsed_val = tok
        parsed[arg] = parsed_val
    return parsed


def parse_packets(flp, flws_ports):
    """
    Parses a PCAP file. Considers packets between a specified client and server
    using specified ports only.

    Returns a dictionary mapping flow to a tuple containing two lists, one for
    data packets and one for ACK packets:
        {
              (client port, server port) :
                  ([ list of data packets ], [ list of ACK packets ])
        }

    Each packet is a tuple of the form:
         (sequence number, timestamp (us),
          TCP timestamp option TSval, TCP timestamp option TSecr,
          TCP payload size (B), total packet size (B))
    """
    print(f"\tParsing PCAP: {flp}")
    # Use list() to read the pcap file all at once (minimize seeks).
    pkts = list(enumerate(scapy.utils.RawPcapReader(flp)))
    num_pkts = len(pkts)

    def make_empty():
        """ Make an empty numpy array to store the packets. """
        return np.full((num_pkts, 6), -1, dtype="int64")

    def remove_unused_rows(arr):
        """
        Returns a filtered array with unused rows removed. A row is unused if
        all of its entries are -1. As an optimization, we check the first entry
        in each row only because we always set a full row at once.
        """
        return arr[arr[:,0] != -1]

    # Format described above. In this form, the arrays will be sparse. Unused
    # rows will be removed later.
    flw_to_pkts = {
        flw_ports: (make_empty(), make_empty()) for flw_ports in flws_ports}
    for idx, (pkt_dat, pkt_mdat) in pkts:
        ether = scapy.layers.l2.Ether(pkt_dat)
        # Assume that this is a TCP/IP packet.
        ip = ether[scapy.layers.inet.IP]
        tcp = ether[scapy.layers.inet.TCP]
        # Determine this packet's direction. Assume that the client IP address
        # if 192.0.0.4 and the server IP address is 192.0.0.2. Assume that all
        # packets are between the client and server.
        if ip.src[-1] == "4":
            dir_idx = 0
            flw = (tcp.sport, tcp.dport)
        else:
            dir_idx = 1
            flw = (tcp.dport, tcp.sport)
        # Assume that the packets are between the relevent machines. Only check
        # the ports.
        if flw in flw_to_pkts:
            # Decode the TCP Timestamp option.
            if len(tcp.options) >= 3 and tcp.options[2][0] == "Timestamp":
                # Fast path: it is usually the third option.
                ts = tcp.options[2][1]
            else:
                # Slow path: check all of the options.
                for option_name, option in tcp.options:
                    if option_name == "Timestamp":
                        ts = option
                        break
                else:
                    ts = (-1, -1)
            flw_to_pkts[flw][dir_idx][idx] = (
                # Sequence number.
                tcp.seq,
                # Timestamp. Not using parse_time_us for efficiency purpose. Use
                # 1000000 instead of 1e6 to avoid converting floats.
                pkt_mdat.sec * 1000000 + pkt_mdat.usec,
                # Timestamp option.
                ts[0],
                ts[1],
                # TCP payload. Length of the IP packet minus the length of the
                # IP header minus the length of the TCP header.
                ip.len - ip.ihl - (tcp.dataofs * 4),
                # Total packet size. Length of the IP packet plus the size of
                # the Ethernet header.
                ip.len + 14)

    # Remove unused rows.
    for flw in flw_to_pkts.keys():
        data, ack = flw_to_pkts[flw]
        flw_to_pkts[flw] = (remove_unused_rows(data), remove_unused_rows(ack))

    # Verify packet count.
    tot_pkts = sum(sum(
        ((dat_pkts.shape[0], ack_pkts.shape[0])
         for dat_pkts, ack_pkts in flw_to_pkts.values()),
        ()))
    assert tot_pkts <= num_pkts, \
        f"Found more packets than exist ({tot_pkts} > {num_pkts}): {flp}"
    discarded_pkts = num_pkts - tot_pkts
    print(
        f"\tDiscarded packets: {discarded_pkts} "
        f"({discarded_pkts / num_pkts * 100:.2f}%)")

    for flw, (dat_pkts, ack_pkts) in flw_to_pkts.items():
        print(flw, dat_pkts.shape, ack_pkts.shape)

    return flw_to_pkts


def parse_q_stats(line):
    """
    Parses a "stats" line of a BESS queue log. Line should be of the form:
        ( "stats", src port, enqueued, dequeued, dropped )
    """
    return (
        ("stats",) +
        tuple(
            int(tok, 16) if tok.startswith("0x") else int(tok)
            for tok in line.split(":")[1].split(",")))


def parse_q_enq_deq(line):
    """
    Parses a packet log line of a BESS queue log. Line should be of the form:
        ( "enq" or "deq", time ns, src port, seq, payload B, qsize, dropped,
          queued, batch size )
    """
    (event, time_ns, src_port, seq, payload_B, qsize, dropped, queued,
     batch_size) = [
         int(tok, 16) if tok.startswith("0x") else int(tok)
         for tok in line.split(",")]

    event_options = {0, 1}
    assert event in event_options, f"Event \"{event}\" not in {event_options}"
    if event == 0:
        event = "enq"
    else:
        event = "deq"

    return (
        event, time_ns / 1e3, src_port, seq, payload_B, qsize, dropped, queued,
        batch_size)


def parse_queue_log(flp):
    """
    Parses the BESS queue log. Returns a list of tuples. See parse_q_stats() and
    parse_q_enq_deq() for details on the tuple format.
    """
    print(f"\tParsing queue log: {flp}")
    with open(flp, "r") as fil:
        q_log = list(fil)
    return [
        parse_q_stats(line) if line.startswith("stats")
        else parse_q_enq_deq(line)
        for line in q_log if line.strip() != ""]


def scale(val, min_in, max_in, min_out, max_out):
    """
    Scales val, which is from the range [min_in, max_in], to the range
    [min_out, max_out].
    """
    assert min_in != max_in, "Divide by zero!"
    return min_out + (val - min_in) * (max_out - min_out) / (max_in - min_in)


def scale_all(dat, scl_prms, min_out, max_out, standardize):
    """
    Uses the provided scaling parameters to scale the columns of
    dat. If standardize is False, then the values are rescaled to the
    range [min_out, max_out].
    """
    dat_dtype = dat.dtype
    fets = dat_dtype.names
    num_scl_prms = len(scl_prms)
    assert len(fets) == num_scl_prms, \
        (f"Mismatching dtype ({fets}) and number of scale parameters "
         f"({num_scl_prms})!")
    new = np.empty(dat.shape, dtype=dat_dtype)
    for idx, fet in enumerate(fets):
        prm_1, prm_2 = scl_prms[idx]
        new[fet] = (
            (dat[fet] - prm_1) / prm_2 if standardize else
            scale(
                dat[fet], min_in=prm_1, max_in=prm_2, min_out=min_out,
                max_out=max_out))
    return new


def load_exp(flp, msg=None):
    """
    Loads one experiment results file (as generated by gen_features.py).
    """
    print(f"{'' if msg is None else f'{msg} - '}Parsing: {flp}")
    exp = Exp(flp)
    try:
        with np.load(flp, allow_pickle=True) as fil:
            num_files = len(fil.files)
            # Make sure that the results have the correct number of flows.
            if num_files == exp.tot_flws:
                dat = [fil[flw] for flw in fil.files]
            else:
                print(
                    f"\tThe number of subfiles ({num_files}) does not match "
                    f"the number of flows ({exp.tot_flws}): {flp}")
                dat = None

    except zipfile.BadZipFile:
        print(f"Bad simulation file: {flp}")
        dat = None
    return exp, dat


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
        ("Only 1D structured arrays are supported, but this one has "
         f"{num_dims} dims!")

    num_cols = len(arr.dtype.names)
    new = np.empty((arr.shape[0], num_cols), dtype=float)
    for col in range(num_cols):
        new[:, col] = arr[arr.dtype.names[col]]
    return new


def visualize_classes(net, dat):
    """ Prints statistics about the classes in dat. """
    # Visualaize the ground truth data.
    clss = net.get_classes()
    # Handles the cases where dat is a torch tensor, numpy unstructured array,
    # or numpy structured array containing a column named "class".
    dat = (
        dat if isinstance(dat, torch.Tensor) or dat.dtype.names is None
        else dat["class"])
    tots = [(dat == cls).sum() for cls in clss]
    # The total number of class labels extracted in the previous line.
    tot = sum(tots)
    print("\n".join(
        [f"\t{cls}: {tot_cls} examples ({tot_cls / tot * 100:.2f}%)"
         for cls, tot_cls in zip(clss, tots)]))
    tot_actual = dat.size
    assert tot == tot_actual, \
        f"Error visualizing ground truth! {tot} != {tot_actual}"


def safe_mathis_label(tput_true, tput_mathis):
    """
    Returns the Mathis model label based on the true throughput and
    Mathis model fair throughput. If either component value is -1
    (unknown), then the resulting label is -1 (unknown).
    """
    return (
        -1 if tput_true == -1 or tput_mathis == -1 else
        int(tput_true > tput_mathis))


def safe_min(val1, val2):
    """
    Safely computes the min of two values. If either value is -1 or 0,
    then that value is discarded and the other value becomes the
    min. If both values are discarded, then the min is -1 (unknown).
    """
    unsafe = {-1, 0}
    return (
        -1 if val1 in unsafe and val2 in unsafe else (
            val2 if val1 in unsafe else (
                val1 if val2 in unsafe else (
                    min(val1, val2)))))


def safe_mul(val1, val2):
    """
    Safely multiplies two values. If either value is -1, then the
    result is -1 (unknown).
    """
    return -1 if val1 == -1 or val2 == -1 else val1 * val2


def safe_sub(val1, val2):
    """
    Safely subtracts two values. If either value is -1, then the
    result is -1 (unknown).
    """
    return -1 if val1 == -1 or val2 == -1 else val1 - val2


def safe_div(num, den):
    """
    Safely divides two values. If either value is -1 or the denominator is 0,
    then the result is -1 (unknown).
    """
    return -1 if num == -1 or den in {-1, 0} else num / den


def safe_np_div(num_arr, den):
    """
    Safely divides a 1D numpy array by a scalar. If an entry in the numerator
    array is -1 (unknown), then that entry in the output array is -1. If the
    denominator scalar is -1, then all entries in the output array are -1.
    """
    assert num_arr.size == num_arr.shape[0], \
        f"Array is not 1D: {num_arr.shape}"

    out = np.full_like(num_arr, -1)
    if den == -1:
        return out
    # Popular known entries.
    mask = num_arr == -1
    out[mask] = num_arr[mask] / den
    return out


def safe_sqrt(val):
    """
    Safely calculates the square root of a value. If the value is -1 (unknown),
    then the result is -1 (unknown).
    """
    return -1 if val == -1 else math.sqrt(val)


def safe_abs(val):
    """
    Safely calculates the absolute value of a value. If the value is -1
    (unknown), then the result is -1 (unknown).
    """
    return -1 if val == -1 else abs(val)


def get_safe(dat, start_idx=None, end_idx=None):
    """
    Returns a filtered window between the two specified indices, with all
    unknown values (-1) removed.
    """
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = 0 if dat.shape[0] == 0 else dat.shape[0] - 1
    # Extract the window.
    dat_win = dat[start_idx:end_idx + 1]
    # Eliminate values that are -1 (unknown).
    return dat_win[dat_win != -1]


def safe_sum(dat, start_idx=None, end_idx=None):
    """
    Safely calculates a sum over a window. Any values that are -1
    (unknown) are discarded. The sum of an empty window is -1 (unknown).
    """
    dat_safe = get_safe(dat, start_idx, end_idx)
    # If the window is empty, then the mean is -1 (unknown).
    return -1 if dat_safe.shape[0] == 0 else np.sum(dat_safe)


def safe_mean(dat, start_idx=None, end_idx=None):
    """
    Safely calculates a mean over a window. Any values that are -1
    (unknown) are discarded. The mean of an empty window is -1
    (unknown).
    """
    dat_safe = get_safe(dat, start_idx, end_idx)
    # If the window is empty, then the mean is -1 (unknown).
    return -1 if dat_safe.shape[0] == 0 else np.mean(dat_safe)


def safe_update_ewma(prev_ewma, new_val, alpha):
    """
    Safely updates an exponentially weighted moving average. If the
    previous EWMA is -1 (unknown), then the new EWMA is assumed to be
    the unweighted new value.
    """
    return (
        new_val if prev_ewma == -1 else
        alpha * new_val + (1 - alpha) * prev_ewma)


def filt(dat_in, dat_out, dat_extra, scl_grps, num_sims, prc):
    """
    Filters parsed data based on a desired number of simulations and percent of
    results from each simulation. Each dat_* is a Python list, where each entry
    is a Numpy array containing the results of one simulation.
    """
    assert (
        len(dat_in) >= num_sims and
        len(dat_out) == len(dat_in) and
        len(dat_extra) == len(dat_in)), \
        "Arguments contain the wrong number of experiments!"
    # Pick the desired number of simulations.
    dat_in = dat_in[:num_sims]
    dat_out = dat_out[:num_sims]
    dat_extra = dat_extra[:num_sims]
    scl_grps = scl_grps[:num_sims]
    # From each simulation, pick the desired fraction.
    if prc != 100:
        for idx in range(num_sims):
            num_rows = dat_in[idx].shape[0]
            idxs = np.random.random_integers(
                0, num_rows - 1, math.ceil(num_rows * prc / 100))
            dat_in[idx] = dat_in[idx][idxs]
            dat_out[idx] = dat_out[idx][idxs]
            dat_extra[idx] = dat_extra[idx][idxs]
    return dat_in, dat_out, dat_extra, scl_grps


def save(flp, dat_in, dat_out, dat_extra):
    """
    Saves parsed data. Each dat_* is a list where each entry is the results of
    one simulation.
    """
    print(f"Saving data: {flp}")
    np.savez_compressed(
        flp, dat_in=dat_in, dat_out=dat_out, dat_extra=dat_extra)


def load_parsed_data(flp):
    """ Loads parsed data. """
    print(f"Loading data: {flp}")
    with np.load(flp) as fil:
        dat_in = fil["dat_in"]
        dat_out = fil["dat_out"]
        dat_extra = fil["dat_extra"]
    dat_in_shape = dat_in.shape
    dat_out_shape = dat_out.shape
    dat_extra_shape = dat_extra.shape
    assert dat_out_shape[0] == dat_in_shape[0], \
        ("Data has invalid shapes (first dimension should agree)! "
         f"in: {dat_in_shape}, out: {dat_out_shape}")
    assert dat_extra_shape[0] == dat_in_shape[0], \
        ("Data has invalid shapes (first dimension should agree)! "
         f"in: {dat_in_shape}, extra: {dat_extra_shape}")
    return dat_in, dat_out, dat_extra


def save_tmp_file(flp, dat_in, dat_out, dat_extra, scl_grps):
    """ Saves a single-simulation temporary results file. """
    print(f"Saving temporary data: {flp}")
    np.savez_compressed(
        flp, dat_in=dat_in, dat_out=dat_out, dat_extra=dat_extra,
        scl_grps=scl_grps)


def load_tmp_file(flp):
    """ Loads and deletes a single-experiment temporary results file. """
    print(f"Loading temporary data: {flp}")
    with np.load(flp) as fil:
        dat_in = fil["dat_in"]
        dat_out = fil["dat_out"]
        dat_extra = fil["dat_extra"]
        scl_grps = fil["scl_grps"]
    os.remove(flp)
    return dat_in, dat_out, dat_extra, scl_grps


def get_lock_flp(out_dir):
    """ Returns the path to a lock file in out_dir. """
    return path.join(out_dir, defaults.LOCK_FLN)


def create_lock_file(out_dir):
    """ Creates a lock file in out_dir. """
    lock_flp = get_lock_flp(out_dir)
    if not path.exists(lock_flp):
        with open(lock_flp, "w") as _:
            pass


def check_lock_file(out_dir):
    """ Checks whether a lock file exists in out_dir. """
    return path.exists(get_lock_flp(out_dir))


def remove_lock_file(out_dir):
    """ Remove a lock file from out_dir. """
    try:
        os.remove(get_lock_flp(out_dir))
    except FileNotFoundError:
        pass


def get_npz_headers(flp):
    """
    Takes a path to an .npz file, which is a Zip archive of .npy files, and
    returns a list of tuples of the form:
        (name, shape, np.dtype)

    Adapted from: https://stackoverflow.com/a/43223420
    """
    def decode_header(archive, name):
        """ Decodes the header information of a single NPY file. """
        npy = archive.open(name)
        version = np.lib.format.read_magic(npy)
        shape, _, dtype = np.lib.format._read_array_header(npy, version)
        return [name[:-4], shape, dtype]

    with zipfile.ZipFile(flp) as archive:
        return [
            decode_header(archive, name) for name in archive.namelist()
            if name.endswith(".npy")]


def set_rand_seed(seed=defaults.SEED):
    """ Sets the Python, numpy, and Torch random seeds to seed. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_tensor(**kwargs):
    """
    Asserts that the provided values (given as keyword args) are Torch tensors.
    """
    for name, val in kwargs.items():
        assert isinstance(val, torch.Tensor), \
            (f"\"{name}\" is of type \"{type(val)}\" when it should be of type "
             "\"torch.Tensor\"")


def has_non_finite(arr):
    """ Returns whether the provided array contains any NaNs or Infs. """
    for fet in arr.dtype.names:
        if not np.isfinite(arr[fet]).all():
            return True
    return False


def bdp_B(bw_Mbps, rtt_us):
    """ Calculates the BDP in bytes. """
    return (bw_Mbps / 8. * 1e6) * (rtt_us / 1e6)


def get_split_data_flp(split_dir, name):
    """
    Returns the path to the data for a Split with the provided name, which
    stores its data in the provided directory.
    """
    return path.join(split_dir, f"{name}.npy")


def get_split_metadata_flp(split_dir, name):
    """
    Returns the path to the metadata for a Split with the provided name, which
    stores its data in the provided directory.
    """
    return path.join(split_dir, f"{name}_metadata.pickle")


def save_split_metadata(split_dir, name, dat):
    """
    Saves the metadata associated with a Split with the provided name that
    stores its data in the provided directory.
    """
    with open(get_split_metadata_flp(split_dir, name), "wb") as fil:
        pickle.dump(dat, fil)


def load_split_metadata(split_dir, name):
    """
    Loads metadata for a Split with the provided name that stores its data in
    the provided directory.
    """
    with open(get_split_metadata_flp(split_dir, name), "rb") as fil:
        return pickle.load(fil)


def get_scl_prms_flp(out_dir):
    """
    Returns the path to a scaling parameters file in the provided directoy.
    """
    return path.join(out_dir, "scale_params.json")


def save_scl_prms(out_dir, scl_prms):
    """ Saves scaling parameters in the provided directory. """
    scl_prms_flp = get_scl_prms_flp(out_dir)
    print(f"Saving scaling parameters: {scl_prms_flp}")
    with open(scl_prms_flp, "w") as fil:
        json.dump(scl_prms.tolist(), fil)


def load_scl_prms(out_dir):
    """ Loads scaling parameters from the provided directory. """
    with open(get_scl_prms_flp(out_dir), "r") as fil:
        return json.load(fil)


def load_split(split_dir, name):
    """
    Loads training, validation, and test split raw data from disk. Returns a
    tuple of (training, validation, test) data.
    """
    num_pkts, dtype = load_split_metadata(split_dir, name)
    return np.memmap(
        get_split_data_flp(split_dir, name), dtype=dtype, mode="r",
        shape=(num_pkts,))


def get_feature_analysis_flp(out_dir):
    """
    Returns the path to the feature analysis log file in the provided directory.
    """
    return path.join(out_dir, "feature_analysis.txt")


def log_feature_analysis(out_dir, msg):
    """
    Prints a feature analysis log statement while also writing it to a file.
    """
    print(msg)
    with open(get_feature_analysis_flp(out_dir), "a+") as fil:
        fil.write(msg)


def analyze_feature_correlation(net, out_dir, dat_in, dat_out, dat_extra):
    """ Analyzes correlation among features of a net. """
    # Feature analysis.
    fets = np.asarray(net.in_spc)
    corr = stats.spearmanr(dat_in).correlation
    # corr = stats.pearsonr(dat_in_cleaned, dat_out_cleaned).correlation
    # corr = np.corrcoef(dat_in_cleaned, dat_out_cleaned, rowvar=False)
    corr_linkage = cluster.hierarchy.ward(corr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
    # Dendrogram.
    dendro = cluster.hierarchy.dendrogram(
        corr_linkage, labels=fets, ax=ax1, leaf_rotation=90)
    # Heatmap.
    dendro_idx = np.arange(0, len(dendro['ivl']))
    heatmap = ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical', fontsize=5)
    ax2.set_yticklabels(dendro['ivl'], fontsize=5)
    # Create colorbar.
    ax2.figure.colorbar(heatmap, ax=ax2)
    fig.tight_layout()
    plt.savefig(path.join(out_dir, f"dendrogram_{net.name}.pdf"))

    # Determine which cluster each feature belongs to.
    #
    # Maps cluster index to a list of the indices of features in that cluster.
    cluster_to_fets = collections.defaultdict(list)
    for feature_idx, cluster_idx in enumerate(cluster.hierarchy.fcluster(
            corr_linkage, 0.2, criterion='distance')):
        cluster_to_fets[cluster_idx].append(fets[feature_idx])
    # Print the clusters.
    log_feature_analysis(
        out_dir,
        "Feature clusters:\n" + "\n".join(
            (f"\t{cluster_id}:\n\t\t" + "\n\t\t".join(fets))
            for cluster_id, fets in sorted(cluster_to_fets.items())))
    # Print the first feature in every cluster.
    log_feature_analysis(
        out_dir,
        "Naively-selected features:" +
        ", ".join(cluster_fets[0] for cluster_fets in cluster_to_fets.values()))


def analyze_feature_importance(net, out_dir, dat_in, dat_out_classes):
    """ Analyzes the importance of features to a trained net. """
    # Analyze feature coefficients. The underlying model's .coef_
    # attribute may not exist.
    print("Analyzing feature importances...")
    tim_srt_s = time.time()
    fets = net.in_spc
    try:
        if isinstance(
                net.net,
                (feature_selection.RFE,
                 feature_selection.RFECV)):
            # Since the model was trained using RFE, display all
            # features. Sort the features alphabetically.
            best_fets = sorted(
                zip(np.array(fets)[(net.net.ranking_ == 1).nonzero()],
                    net.net.estimator_.coef_[0]),
                key=lambda p: p[0])
            log_feature_analysis(
                out_dir, f"Number of features selected: {len(best_fets)}")
            qualifier = "All"
        else:
            qualifier = "Best"
            if isinstance(
                    net.net, ensemble.HistGradientBoostingClassifier):
                imps = inspection.permutation_importance(
                    net.net, dat_in, dat_out_classes, n_repeats=10,
                    random_state=0).importances_mean
            else:
                imps = net.net.coef_[0]

            # First, sort the features by the absolute value of the
            # importance and pick the top 20. Then, sort the features
            # alphabetically.
            # best_fets = sorted(
            #     sorted(
            #         zip(fets, imps),
            #         key=lambda p: abs(p[1]))[-20:],
            #     key=lambda p: p[0])
            best_fets = list(reversed(
                sorted(zip(fets, imps), key=lambda p: abs(p[1]))[-20:]))
        log_feature_analysis(
            out_dir,
            f"----------\n{qualifier} features ({len(best_fets)}):\n" +
            "\n".join(f"\t{fet}: {coef:.4f}" for fet, coef in best_fets) +
            "\n----------")

        # Graph feature coefficients.
        if net.graph:
            names, coefs = zip(*best_fets)
            num_fets = len(names)
            y_vals = list(range(num_fets))
            plt.figure(figsize=(7, 0.2 * num_fets))
            plt.barh(y_vals, coefs, align="center")
            plt.yticks(y_vals, names)
            plt.ylim((-1, num_fets))
            plt.xlabel("Feature coefficient")
            plt.ylabel("Feature name")
            plt.tight_layout()
            plt.savefig(path.join(out_dir, f"features_{net.name}.pdf"))
            plt.close()
    except AttributeError:
        # Coefficients are only available with a linear kernel.
        log_feature_analysis(
            out_dir, "Warning: Unable to extract coefficients!")
    log_feature_analysis(
        out_dir,
        ("Finished analyzing feature importance - time: "
         f"{time.time()- tim_srt_s:.2f} seconds"))


def check_fets(fets, in_spc):
    """
    Verifies that the processed features that emerge from the data processing
    pipeline are the same as a net's in_spc. fets is intended to come from the
    data processing pipeline (i.e., the actual features after all data
    processing). in_spc is intended to come from a net (i.e., the net's original
    feature specification).
    """
    assert fets == in_spc, \
        ("Provided features do not agreed with in_spc."
        f"\n\tProvided fets ({len(fets)}): {fets}"
         f"\n\tin_spc ({len(in_spc)}): {in_spc}")
