""" Utility functions. """

import math
import os
from os import path
import random
import zipfile

import numpy as np
import torch


# Arguments to ignore when converting an arguments dictionary to a
# string.
ARGS_TO_IGNORE = ["data_dir", "out_dir", "tmp_dir", "sims", "features"]
# The random seed.
SEED = 1337
# Name to use for lock files.
LOCK_FLN = "lock"


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
        return len(self.dat_in)

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

    def __init__(self, dataset, batch_size, drop_last):
        assert isinstance(dataset, Dataset), \
            "Dataset must be an instance of utils.Dataset."
        # Determine the unique classes.
        _, _, dat_out, _ = dataset.raw()
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
                print(f"    Added {new_examples} examples to class {cls}.")
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
        [str(args[key]) for key in order if key not in ARGS_TO_IGNORE])


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
    order = [key for key in order if key not in ARGS_TO_IGNORE]
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


def parse_packets(flp, client_port, server_port, direction="data",
                  extra_filter=None):
    """
    Parses a PCAP file. Returns a list of tuples of the form:
         (sequence number, flow index, timestamp (us), TCP timestamp option,
          payload size (B))
    with one entry for every packet. Considers only packets in either the "ack"
    or "data" direction.
    """
    dir_opts = ["ack", "data"]
    assert direction in dir_opts, \
        f"\"direction\" must be one of {dir_opts}, but is: {direction}"

    if direction == "data":
        if extra_filter:
            filter_s = (
                f"\"tcp.srcport == {client_port} && tcp.dstport == {server_port} && "
                f"tcp.len >= 1000 && {extra_filter}\"")
        else:
            filter_s = (
                f"\"tcp.srcport == {client_port} && tcp.dstport == {server_port} && "
                "tcp.len >= 1000\"")
    else:
        if extra_filter:
            filter_s = (
                f"\"tcp.srcport == {server_port} && tcp.dstport == {client_port} && "
                f"{extra_filter}\"")
        else:
            filter_s = (
                f"\"tcp.srcport == {server_port} && tcp.dstport == {client_port}\"")

    # Strip off the ".pcap" extension and append "_tmp.txt".
    tmp_flp = f"{flp[:-5]}_tmp.txt"

    cmd = " ".join(
        ["tshark", "-n",
         # The AMQP protocol uses port 5672, which our flows may use. tshark
         # tries to parse those packets as AMQP packets, which gives a warning.
         "--disable-protocol", "amqp",
         "--disable-protocol", "hzlcst",
         "-o", "tcp.relative_sequence_numbers:false",
         "-o", "tcp.analyze_sequence_numbers:false",
         "-r", flp, filter_s, ">>", tmp_flp])
    print(f"Running: {cmd}")
    os.system(cmd)

    # Each item is a tuple of the form:
    #     (sequence number, timestamp (us), TCP timestamp option,
    #      payload size (B))
    pkts = []
    with open(tmp_flp, "r") as fil:
        for line in fil:
            array = line.split()
            seq_idx = -1
            tsval_idx = -1
            tsecr_idx = -1
            len_idx = -1
            for i in range(7, len(array)):
                if array[i].startswith("Seq"):
                    seq_idx = i
                elif array[i].startswith("TSval"):
                    tsval_idx = i
                elif array[i].startswith("TSecr"):
                    tsecr_idx = i
                elif array[i].startswith("Len"):
                    len_idx = i
            if (seq_idx != -1 and tsecr_idx != -1 and tsecr_idx != -1 and
                    len_idx != -1):
                tsecr_substring = array[tsecr_idx][6:]
                tsecr_end = (
                    tsecr_substring.index("[")
                    if "[" in tsecr_substring else len(tsecr_substring))

                pkts.append((
                    int(array[seq_idx][4:]),
                    float(array[1]) * 1e6,
                    (int(array[tsval_idx][6:]),
                     int(tsecr_substring[:tsecr_end])),
                    int(array[len_idx][4:])))
    # Delete the temporary file.
    os.remove(tmp_flp)
    return pkts


def parse_q_stats(line):
    """
    Parses a "stats" line of a BESS queue log. Line should be of the form:
        stats:time ns,src port,enqueued,dequeued,dropped
    """
    return (
        ("stats",) +
        tuple(
            int(tok, 16) if tok.startswith("0x") else int(tok)
            for tok in line.split(":")[1].split(",")))


def parse_q_enq_deq(line):
    """
    Parses a packet log line of a BESS queue log. Line should be of the form:
        0 or 1,time ns,src port,seq,payload B,qsize,dropped,queued,batch size
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

    """
    print(f"Parsing queue log: {flp}")
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


def load_sim(flp, msg=None):
    """
    Loads one simulation results file (generated by parse_dumbbell.py). Returns
    a tuple of the form: (total number of flows, results matrix).
    """
    print(f"{'' if msg is None else f'{msg} - '}Parsing: {flp}")
    try:
        with np.load(flp) as fil:
            # assert len(fil.files) == 1 and "1" in fil.files, \
            #     "More than one flow detected!"
            dat = fil["1"]
    except zipfile.BadZipFile:
        print(f"Bad simulation file: {flp}")
        dat = None
    return Sim(flp), dat


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
    # Assumes that dat is a structured numpy array containing a
    # column named "class".
    tots = [
        ((dat if dat.dtype.names is None else dat["class"]) == cls).sum()
        for cls in clss]
    # The total number of class labels extracted in the previous line.
    tot = sum(tots)
    print("Classes:\n" + "\n".join(
        [f"    {cls}: {tot_cls} examples ({tot_cls / tot * 100:.2f}%)"
         for cls, tot_cls in zip(clss, tots)]))
    tot_actual = np.prod(np.array(dat.shape))
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
    Safely divides two values. If either value is -1 or the
    denominator is 0, then the result is -1 (unknown).
    """
    return -1 if num == -1 or den in {-1, 0} else num / den


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
        len(dat_extra) == len(dat_in)), "Malformed arguments!"

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


def load(flp):
    """
    Loads parsed data. Each returned item is a list where each entry is the
    results of one simulation.
    """
    print(f"Loading data: {flp}")
    with np.load(flp) as fil:
        return fil["dat_in"], fil["dat_out"], fil["dat_extra"]


def save_tmp_file(flp, dat_in, dat_out, dat_extra, scl_grps):
    """ Saves a single-simulation temporary results file. """
    print(f"Saving temporary data: {flp}")
    np.savez_compressed(
        flp, dat_in=dat_in, dat_out=dat_out, dat_extra=dat_extra,
        scl_grps=scl_grps)


def load_tmp_file(flp):
    """ Loads and deletes a single-simulation temporary results file. """
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
    return path.join(out_dir, LOCK_FLN)


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
        return name[:-4], shape, dtype

    with zipfile.ZipFile(flp) as archive:
        return [
            decode_header(archive, name) for name in archive.namelist()
            if name.endswith(".npy")]


def set_rand_seed(seed=SEED):
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
