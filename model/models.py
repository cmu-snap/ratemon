#!/usr/bin/env python3
"""
Models of the form:
  (current throughput, current RTT, target throughput) -> (ACK pacing interval)
"""

import functools
import math
import random

import numpy as np
import torch


class Model(torch.nn.Module):
    """ A wrapper class for PyTorch models. """

    # The specification of the input tensor format.
    in_spc = []
    # The specification of the output tensor format.
    out_spc = []
    # The number of output classes.
    num_clss = 0
    # Whether this model is an LSTM. LSTMs require slightly different
    # data handling.
    is_lstm = False
    # The loss function to use during training.
    los_fnc = None
    # The optimizer to use during training.
    opt = None

    def __init__(self):
        super(Model, self).__init__()

    def check(self):
        """ Verifies that this Model instance has been initialized properly. """
        assert self.in_spc, "Empty in_spc!"
        assert self.out_spc, "Empty out_spc!"
        assert self.num_clss is None or self.num_clss > 0, \
    "Invalid number of output classes!"
        assert self.los_fnc is not None, "No loss function!"
        assert self.opt is not None, "No optimizer!"

    def init_hidden(self, batch_size):
        """
        If this model is an LSTM, then this method returns the initialized
        hidden state. Otherwise, returns None.
        """
        return None

    @staticmethod
    def convert_to_class(sim, dat_out):
        """ Converts real-valued feature values into classes. """
        return dat_out

    def modify_data(self, sim, dat_in, dat_out):
        """ Performs an arbitrary transformation on the data. """
        return dat_in, dat_out

    def forward(self, x, hidden):
        raise Exception(("Attempting to call \"forward()\" on the Model base "
                         "class itself."))


class BinaryDnn(Model):
    """ A simple binary classifier neural network. """

    # in_spc = ["arrival time", "loss rate"]
    in_spc = ["arrival time"]
    # in_spc = ["inter-arrival time", "loss rate"]
    # in_spc = ["inter-arrival time"]
    out_spc = ["queue occupancy"]
    num_clss = 2
    los_fnc = torch.nn.CrossEntropyLoss
    opt = torch.optim.SGD

    def __init__(self, win=100, rtt_buckets=True, sqrt_loss=False, disp=False):
        super(BinaryDnn, self).__init__()
        self.check()

        self.win = win
        self.rtt_buckets = rtt_buckets
        if self.rtt_buckets:
            assert "arrival time" in self.in_spc, \
                "When bucketizing packets, \"arrival time\" must be a feature."
        self.sqrt_loss = sqrt_loss

        # Determine layer dimensions. If we are bucketizing packets
        # based on arrival time, then there will be one input feature
        # for every bucket (self.win) plus one input feature for every
        # entry in self.in_spc *except* for "arrival time". If we are
        # not bucketizing packets, then there will be one input
        # feature for each entry in self.in_spc for each packet
        # (self.win).
        num_ins = (self.win + len(self.in_spc) - 1 if self.rtt_buckets
                   else len(self.in_spc) * self.win)
        dim_1 = min(num_ins, 64)
        dim_2 = min(dim_1, 32)
        dim_3 = min(dim_2, 16)
        # We must store these as indivual class variables (instead of
        # just storing them in self.fcs) because PyTorch looks at the
        # class variables to determine the model's trainable weights.
        self.fc0 = torch.nn.Linear(num_ins, dim_1)
        self.fc1 = torch.nn.Linear(dim_1, dim_2)
        self.fc2 = torch.nn.Linear(dim_2, dim_3)
        self.fc3 = torch.nn.Linear(dim_3, self.num_clss)
        self.fcs = [self.fc0, self.fc1, self.fc2, self.fc3]
        self.sg = torch.nn.Sigmoid()
        if disp:
            print(f"BinaryDnn - win: {self.win}, fc layers: {len(self.fcs)}"
                  "\n    " + "\n    ".join(
                      [f"Linear: {lay.in_features}x{lay.out_features}"
                       for lay in self.fcs]) +
                  "\n    Sigmoid")

    def forward(self, x, hidden=None):
        for fc in self.fcs:
            x = torch.nn.functional.relu(fc(x))
        return self.sg(x), hidden

    @staticmethod
    def convert_to_class(sim, dat_out):
        # Verify that the output features consist of exactly one column.
        assert len(dat_out.dtype.names) == 1, "Should be only one column."
        # Map a conversion function across all entries. Note that
        # here an entry is an entire row, since each row is a
        # single tuple value.
        clss = np.vectorize(
            functools.partial(
                # Compare each queue occupancy percent with the fair
                # percent. prc[0] assumes a single column.
                lambda prc, fair: prc[0] > fair,
                fair=1. / (sim.unfair_flws + sim.other_flws)),
            # Convert to integers.
            otypes=[int])(dat_out)
        clss_str = np.empty((clss.shape[0],), dtype=[("class", "int")])
        clss_str["class"] = clss
        return clss_str

    @staticmethod
    def bucketize(dat_in, dat_in_start_idx, dat_in_end_idx, dat_in_new,
                  dat_in_new_idx, dur_us, num_buckets):
        """
        Uses dat_in["arrival time"] to divide the arriving packets from
        the range [dat_in_start_idx, dat_in_end_idx] into num_buckets
        intervals. Each interval has duration dur_us / num_buckets.
        Stores the resulting histogram in dat_in_new, at the row
        indicated by dat_in_new_idx. Also updates dat_in_new with the
        values of the other (i.e., non-bucket) features. Returns
        nothing.
        """
        fets = dat_in.dtype.names
        assert "arrival time" in fets, f"Missing \"arrival time\": {fets}"
        arr_times = dat_in["arrival time"][dat_in_start_idx:dat_in_end_idx + 1]
        num_pkts = arr_times.shape[0]
        assert num_pkts > 0, "Need more than 0 packets!"

        # We are turning the arrival times into buckets, but there are
        # other features that must be preserved.
        other_fets = [col for col in dat_in.dtype.descr
                      if col[0] != "arrival time" and col[0] != ""]
        # The duration of each interval.
        interval_us = dur_us / num_buckets
        # The arrival time of the first packet, and therefore the
        # start of the first interval.
        start_time_us = arr_times[0]

        for arr_time in arr_times:
            diff = arr_time - start_time_us
            raw_interval = diff / interval_us
            interval_idx = np.floor(raw_interval).astype(int)
            # # Convert the arrival times to interval indices and loop over them.
            # for interval_idx in np.floor(
            #         (arr_times - start_time_us) / interval_us).astype(int):
            assert 0 <= interval_idx < num_buckets, \
                (f"Invalid idx ({interval_idx}) for the number of buckets ({num_buckets})!\n"
                 f"arr_times: {arr_times}\n"
                 f"arr_time: {arr_time}\n"
                 f"start_time_us: {start_time_us}\n"
                 f"diff: {diff}\n"
                 f"raw_interval: {raw_interval}\n")
            dat_in_new[dat_in_new_idx][interval_idx] += 1
        # Set the values of the other features based on the last packet in this
        # window.
        for fet, _ in other_fets:
            dat_in_new[fet][dat_in_new_idx] = dat_in[fet][dat_in_end_idx]

        # Check that the bucket features reflect all of the packets.
        bucketed_pkts = sum(dat_in_new[dat_in_new_idx].tolist()[:num_buckets])
        assert bucketed_pkts == num_pkts, \
            (f"Error building counts! Bucketed {bucketed_pkts} of {num_pkts} "
             "packets!")

    def create_buckets(self, sim, dat_in, dat_out):
        """
        Divides dat_in into windows and divides each window into self.win
        buckets, which each defines a temporal interval. The value of
        each bucket is the number of packets that arrived during that
        interval. The output value for each window is the output of
        the last packet in the window.
        """
        fets = dat_in.dtype.names
        assert "arrival time" in fets, f"Missing \"arrival time\": {fets}"
        arr_times = dat_in["arrival time"]

        # 100x the min RTT (as determined by the simulation parameters).
        dur_us = 100 * 2 * (sim.edge_delays[0] * 2 + sim.btl_delay_us)
        # Determine the safe starting index. Do not pick indices
        # between 0 and start_idx to make sure that all windows ending
        # on the chosen index fit within the simulation.
        first_arr_time = None
        start_idx = None
        for idx, arr_time in enumerate(arr_times):
            if first_arr_time is None:
                first_arr_time = arr_time
                continue
            if arr_time - first_arr_time >= dur_us:
                start_idx = idx
                break
        num_pkts = dat_in.shape[0]
        assert start_idx is not None and 0 <= start_idx < num_pkts, \
            f"Invalid start index: {start_idx}"

        # The number of windows is the number of times the window
        # durations fits within the simulation.
        num_wins = 10 * math.floor(sim.dur_s * 1e6 / dur_us)
        # Select random intervals from this simulation to create the
        # new input data. win_idx is the index into
        # dat_in_new. pkt_idx is the index of the last packet in this
        # window.
        pkt_idxs = random.choices(range(start_idx, num_pkts), k=num_wins)
        # Records the number of packets that arrived during each
        # interval. This is a structured numpy array where each column
        # is named based on its bucket index. We add extra columns for
        # the non--arrival time features.
        dat_in_new = np.zeros(
            (num_wins,),
            dtype=([(f"bucket_{bkt}", "float") for bkt in range(self.win)] +
                   [col for col in dat_in.dtype.descr
                    if col[0] != "arrival time" and col[0] != ""]))

        for win_idx, pkt_idx in enumerate(pkt_idxs):
            # Find the first packet in this window (whose index will
            # be win_start_idx).
            cur_arr_time = arr_times[pkt_idx]
            win_start_idx = None
            # Move backwards from the last packet in the window until
            # we find a packet whose arrival time is more that dur_us
            # in the past.
            for arr_time_idx in range(pkt_idx, -1, -1):
                if cur_arr_time - arr_times[arr_time_idx] > dur_us:
                    # This packet is the first packet that is too far
                    # in the past, so we will start the window on the
                    # next packet.
                    win_start_idx = arr_time_idx + 1
                    break
            assert (
                win_start_idx is not None and 0 <= win_start_idx <= pkt_idx), \
                ("Problem finding beginning of window! Are there insufficient "
                 "packets?")
            self.bucketize(
                dat_in, win_start_idx, pkt_idx, dat_in_new, win_idx, dur_us,
                self.win)

        # Verify that we selected at least as many windows as we intended to.
        num_selected_wins = len(dat_in_new)
        assert num_selected_wins >= num_wins, \
            f"Insufficient windows: {num_selected_wins} < {num_wins}"

        return (
            dat_in_new,
            # As an output feature, select only the final ground truth
            # value. I.e., the final ground truth value for this window
            # becomes the ground truth for the entire window.
            np.take(dat_out, pkt_idxs),
            # The buckets all share a scaling group. Each other
            # feature is part of its own group.
            [0] * self.win +
            list(range(1, len(dat_in_new.dtype.names) - self.win + 1)))

    def create_windows(self, dat_in, dat_out):
        """
        Divides dat_in into windows of self.win packets. Flattens the
        features of the packets in a window. The output value for each
        window is the output of the last packet in the window.
        """
        num_pkts = dat_in.shape[0]
        num_wins = math.ceil(num_pkts / self.win)
        fets = [(name, typ) for name, typ in dat_in.dtype.descr if name != ""]
        # Select random intervals from this simulation to create the
        # new input data. Do not pick indices between 0 and self.win
        # to make sure that all windows ending on the chosen index fit
        # within the simulation.
        pkt_idxs = random.choices(range(self.win, num_pkts), k=num_wins)
        # The new data format consists of self.win copies of the
        # existing input features. All copies of a particular feature
        # share the same scaling group.
        scl_grps, dtype = zip(
            *[(scl_grp, (f"{name}_{idx}", typ))
              for idx in range(self.win)
              for scl_grp, (name, typ) in enumerate(fets)])
        dat_in_new = np.zeros((num_wins,), dtype=list(dtype))

        for win_idx, end_idx in enumerate(pkt_idxs):
            # This could be done on a single line with a range select
            # and a generator, but this version is preferable because
            # it removes intermediate data copies and guarantees that
            # the resulting row is properly ordered.
            for fet_idx, pkt_idx in enumerate(
                    range(end_idx - self.win + 1, end_idx + 1)):
                for name, _ in fets:
                    dat_in_new[f"{name}_{fet_idx}"][win_idx] = (
                        dat_in[pkt_idx][name])

        # Verify that we selected at least as many windows as we intended to.
        num_selected_wins = len(dat_in_new)
        assert num_selected_wins >= num_wins, \
            f"Insufficient windows: {num_selected_wins} < {num_wins}"

        # As an output feature, select only the final ground truth
        # value. I.e., the final ground truth value for this window
        # becomes the ground truth for the entire window.
        return dat_in_new, np.take(dat_out, pkt_idxs), scl_grps

    def modify_data(self, sim, dat_in, dat_out):
        """
        Extracts from the set of simulations many separate intervals. Each
        interval becomes a training example.
        """
        dat_in_new, dat_out_new, scl_grps = (
            self.create_buckets(sim, dat_in, dat_out) if self.rtt_buckets
            else self.create_windows(dat_in, dat_out))

        # If configured to do so, compute 1 / sqrt(x) for any features
        # that contain "loss rate". We cannot simply look for a single
        # feature named exactly "loss rate" because if
        # self.rtt_buckets == False, then there may exist many features of the
        # form "loss rate_X".
        if self.sqrt_loss:
            for fet in dat_in_new.dtype.names:
                if "loss rate" in fet:
                    dat_in_new[fet] = np.reciprocal(np.sqrt(dat_in_new[fet]))

        return dat_in_new, dat_out_new, scl_grps



class SVM(BinaryDnn):
    """ A simple SVM binary classifier. """

    in_spc = ["inter-arrival time", "loss rate"]
    out_spc = ["queue occupancy"]
    num_clss = None
    los_fnc = torch.nn.HingeEmbeddingLoss
    opt = torch.optim.SGD

    def __init__(self, win=20, rtt_buckets=True, disp=False):
        super(BinaryDnn, self).__init__()
        self.check()

        self.win = win
        self.rtt_buckets = rtt_buckets
        self.fc0 = torch.nn.Linear(self.win if self.rtt_buckets else len(BinaryDnn.in_spc) * self.win, 1)
        self.sg = torch.nn.Sigmoid()
        if (disp):
            print(f"SVM - win: {self.win}, fc layers: 1\n    " +
                  f"Linear: {self.fc0.in_features}x{self.fc0.out_features}" +
                  "\n    Sigmoid")

    def forward(self, x, hidden=None):
        return self.fc0(x), hidden

    def modify_data(self, sim, dat_in, dat_out, **kwargs):
        new_dat_in, new_dat_out = super(SVM, self).modify_data(sim, dat_in, dat_out, **kwargs)
        for i in range(len(new_dat_out)):
            new_dat_out[i][0] = new_dat_out[i][0] * 2 - 1 # Map [0,1] to [-1, 1]
        return new_dat_in, new_dat_out


class Lstm(Model):
    """ An LSTM that classifies a flow into one of five fairness categories. """

    # For now, we do not use RTT ratio because our method of estimating
    # it cannot be performed by a general receiver.
    # in_spc = ["inter-arrival time", "RTT ratio", "loss rate"]
    in_spc = ["inter-arrival time", "loss rate"]
    out_spc = ["queue occupancy"]
    num_clss = 5
    is_lstm = True
    # Cross-entropy loss is designed for multi-class classification tasks.
    los_fnc = torch.nn.CrossEntropyLoss
    opt = torch.optim.Adam

    def __init__(self, hid_dim=32, num_lyrs=1, out_dim=5, disp=False):
        super(Lstm, self).__init__()
        self.check()

        self.in_dim = len(self.in_spc)
        self.hid_dim = hid_dim
        self.num_lyrs = num_lyrs
        self.out_dim = out_dim
        self.lstm = torch.nn.LSTM(self.in_dim, self.hid_dim)
        self.fc = torch.nn.Linear(self.hid_dim, self.out_dim)
        self.sg = torch.nn.Sigmoid()
        if disp:
            print(f"Lstm - in_dim: {self.in_dim}, hid_dim: {self.hid_dim}, "
                  f"num_lyrs: {self.num_lyrs}, out_dim: {self.out_dim}")

    def forward(self, x, hidden):
        # The LSTM itself, which also takes as input the previous hidden state.
        out, hidden = self.lstm(x, hidden)
        # Select the last piece as the LSTM output.
        out = out.contiguous().view(-1, self.hid_dim)
        # Classify the LSTM output.
        out = self.fc(out)
        out = self.sg(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.num_lyrs, batch_size, self.hid_dim).zero_(),
                weight.new(self.num_lyrs, batch_size, self.hid_dim).zero_())

    @staticmethod
    def convert_to_class(sim, dat_out):
        assert len(dat_out.dtype.names) == 1, "Should be only one column."

        def percent_to_class(prc, fair):
            """ Convert a queue occupancy percent to a fairness class. """
            assert len(prc) == 1, "Should be only one column."
            prc = prc[0]

            # Threshold between fair and unfair.
            tsh_fair = 0.1
            # Threshold between unfair and very unfair.
            tsh_unfair = 0.4

            dif = (fair - prc) / fair
            if dif < -1 * tsh_unfair:
                # We are much higher than fair.
                cls = 4
            elif -1 * tsh_unfair <= dif < -1 * tsh_fair:
                # We are not that much higher than fair.
                cls = 3
            elif -1 * tsh_fair <= dif <= tsh_fair:
                # We are fair.
                cls = 2
            elif tsh_fair < dif <= tsh_unfair:
                # We are not that much lower than fair.
                cls = 1
            elif tsh_unfair < dif:
                # We are much lower than fair.
                cls = 0
            else:
                assert False, "This should never happen."
            return cls

        # Map a conversion function across all entries. Note that here
        # an entry is an entire row, since each row is a single tuple
        # value.
        return np.vectorize(
            functools.partial(
                percent_to_class, fair=1. / (sim.unfair_flws + sim.other_flws)),
            otypes=[int])(dat_out)


#######################################################################
#### Old Models. Present for archival purposes only. These are not ####
#### guaranteed to function.                                       ####
#######################################################################


class SimpleOne(torch.nn.Module):
    """ A simple linear model. """

    in_spc = ["average_throughput_after_bps"]
    out_spc = ["ack_period_us"]

    def __init__(self):
        super(SimpleOne, self).__init__()
        assert len(SimpleOne.in_spc) == 1
        assert len(SimpleOne.out_spc) == 1
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


class FcOne(torch.nn.Module):
    """ An NN with one fully-connected layer. """

    in_spc = ["average_throughput_before_bps", "average_throughput_after_bps",
              "rtt_us"]
    out_spc = SimpleOne.out_spc

    def __init__(self):
        super(FcOne, self).__init__()
        assert len(FcOne.in_spc) == 3
        assert len(FcOne.out_spc) == 1
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)


class FcTwo(torch.nn.Module):
    """ An NN with two fully-connected layers. """

    in_spc = FcOne.in_spc
    out_spc = SimpleOne.out_spc

    def __init__(self):
        super(FcTwo, self).__init__()
        assert len(FcTwo.in_spc) == 3
        assert len(FcTwo.out_spc) == 1
        self.fc1 = torch.nn.Linear(3, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return x


class FcThree(torch.nn.Module):
    """ An NN with three fully-connected layers. """

    in_spc = FcOne.in_spc
    out_spc = SimpleOne.out_spc

    def __init__(self):
        super(FcThree, self).__init__()
        assert len(FcThree.in_spc) == 3
        assert len(FcThree.out_spc) == 1
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        return x


class FcFour(torch.nn.Module):
    """ An NN with four fully-connected layers. """

    in_spc = FcOne.in_spc
    out_spc = SimpleOne.out_spc

    def __init__(self):
        super(FcFour, self).__init__()
        assert len(FcFour.in_spc) == 3
        assert len(FcFour.out_spc) == 1
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        return x


#######################################################################


MODELS = {
    "BinaryDnn": BinaryDnn,
    "Lstm": Lstm,
    "SVM": SVM,
    "SimpleOne": SimpleOne,
    "FcOne": FcOne,
    "FcTwo": FcTwo,
    "FcThree": FcThree,
    "FcFour": FcFour
}
