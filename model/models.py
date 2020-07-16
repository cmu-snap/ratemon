#!/usr/bin/env python3
""" Models. """

import functools
import math
import os
from os import path
import random

from matplotlib import pyplot
import numpy as np
from sklearn import linear_model
from sklearn import svm
import torch


class PytorchModelWrapper:
    """ A wrapper class for PyTorch models. """

    # The name of this model.
    name = None
    # The specification of the input tensor format.
    in_spc = []
    # The specification of the output tensor format.
    out_spc = []
    # The number of output classes.
    num_clss = 0
    # The loss function to use during training.
    los_fnc = None
    # The optimizer to use during training.
    opt = None
    # Model-specific parameters. Each model may use these differently.
    params = []

    def __init__(self):
        self.net = None
        self.graph = False
        self.__check()

    def __check(self):
        """
        Verifies that this PytorchModel instance has been initialized properly.
        """
        assert self.name is not None, "Empty name!"
        assert self.in_spc, "Empty in_spc!"
        assert self.out_spc, "Empty out_spc!"
        assert self.num_clss > 0, "Invalid number of output classes!"
        if not isinstance(self, SvmSklearnWrapper):
            assert self.los_fnc is not None, "No loss function!"
            assert self.opt is not None, "No optimizer!"

    def init_hidden(self, batch_size):
        """
        If this model is an LSTM, then this method returns the initialized
        hidden state. Otherwise, returns None.
        """
        return torch.zeros(()), torch.zeros(())

    @staticmethod
    def convert_to_class(sim, dat_out):
        """ Converts real-valued feature values into classes. """
        return dat_out

    def modify_data(self, sim, dat_in, dat_out):
        """ Performs an arbitrary transformation on the data. """
        return dat_in, dat_out, list(range(len(dat_in.dtype.names)))

    def check_output(self, out, target):
        """
        Returns the number of examples from out that were classified correctly,
        according to target.
        """
        size_out = out.size()
        size_target = target.size()
        assert size_target
        assert size_out[0] == size_target[0], \
            ("Output and target have different batch sizes (first dimension): "
             f"{size_out} != {size_target}")
        # Transform the output into classes.
        out = self._check_output_helper(out)
        size_out = out.size()
        assert size_out == size_target, \
            f"Output and target sizes do not match: {size_out} != {size_target}"
        # eq(): Compare the outputs to the labels.
        # type(): Cast the resulting bools to ints.
        # sum(): Sum them up to get the total number of correct predictions.
        return out.eq(target).type(torch.int).sum().item()

    def _check_output_helper(self, out):
        """ Convert the raw network output into classes. """
        # Assume a one-hot encoding of class probabilities. The class
        # is the index of the output entry with greatest value (i.e.,
        # highest probability). Set dim=1 because the first dimension
        # is the batch.
        size_out = out.size()
        assert size_out[1] == self.num_clss, \
            (f"Expecting one-hot encoding for {self.num_clss} classes, but "
             f"found size: {size_out}")
        return torch.argmax(out, dim=1)

    def new(self):
        """ Returns a new instance of the underlying torch.nn.Module. """
        raise Exception(
            ("Attempting to call \"new()\" on the PytorchModelWrapper base class "
             "itself."))


class BinaryModelWrapper(PytorchModelWrapper):
    """ A base class for binary classification models. """

    num_clss = 2

    def __init__(self, win=100, rtt_buckets=True):
        super(BinaryModelWrapper, self).__init__()

        self.win = win
        self.rtt_buckets = rtt_buckets
        if self.rtt_buckets:
            assert "arrival time" in self.in_spc, \
                "When bucketizing packets, \"arrival time\" must be a feature."
        # Determine layer dimensions. If we are bucketizing packets
        # based on arrival time, then there will be one input feature
        # for every bucket (self.win) plus one input feature for every
        # entry in self.in_spc *except* for "arrival time". If we are
        # not bucketizing packets, then there will be one input
        # feature for each entry in self.in_spc for each packet
        # (self.win).
        self.num_ins = (self.win + len(self.in_spc) - 1 if self.rtt_buckets
                        else len(self.in_spc) * self.win)

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
    def __bucketize(dat_in, dat_in_start_idx, dat_in_end_idx, dat_in_new,
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
        # Convert the arrival times to interval indices and loop over them.
        for interval_idx in np.floor(
                (arr_times - start_time_us) / interval_us).astype(int):
            if interval_idx == num_buckets:
                print(f"Warning: Interval is {interval_idx} when it should be "
                      f"in the range [0, {num_buckets}]. Fixing interval...")
                interval_idx -= 1
            assert 0 <= interval_idx < num_buckets, \
                (f"Invalid idx ({interval_idx}) for the number of buckets "
                 f"({num_buckets})!")
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

    def __create_buckets(self, sim, dat_in, dat_out, dat_out_raw,
                         dat_out_oracle, sequential):
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

        if sequential:
            # Select all valid windows, in order.
            num_wins = num_pkts - start_idx
            pkt_idxs = list(range(start_idx, num_pkts))
        else:
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
            self.__bucketize(
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
            np.take(dat_out_raw, pkt_idxs),
            np.take(dat_out_oracle, pkt_idxs),
            # The buckets all share a scaling group. Each other
            # feature is part of its own group.
            [0] * self.win +
            list(range(1, len(dat_in_new.dtype.names) - self.win + 1)))

    def __create_windows(self, dat_in, dat_out, sequential):
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

    def modify_data(self, sim, dat_in, dat_out, dat_out_raw, dat_out_oracle,
                    sequential):
        """
        Extracts from the set of simulations many separate intervals. Each
        interval becomes a training example.
        """
        dat_in, dat_out, dat_out_raw, dat_out_oracle, scl_grps = (
            self.__create_buckets(
                sim, dat_in, dat_out, dat_out_raw, dat_out_oracle, sequential)
            if self.rtt_buckets
            else self.__create_windows(dat_in, dat_out, sequential))
        return dat_in, dat_out, dat_out_raw, dat_out_oracle, scl_grps


class BinaryDnnWrapper(BinaryModelWrapper):
    """ Wraps BinaryDnn. """

    name = "BinaryDnn"
    # in_spc = ["arrival time", "loss rate"]
    in_spc = ["arrival time"]
    # in_spc = ["inter-arrival time", "loss rate"]
    # in_spc = ["inter-arrival time"]
    out_spc = ["queue occupancy"]
    los_fnc = torch.nn.CrossEntropyLoss
    opt = torch.optim.SGD
    params = ["lr", "momentum"]

    def new(self):
        self.net = BinaryDnn(self.num_ins, self.num_clss)
        return self.net


class BinaryDnn(torch.nn.Module):
    """ A simple binary classifier neural network. """

    def __init__(self, num_ins, num_outs):
        super(BinaryDnn, self).__init__()
        dim_1 = min(num_ins, 64)
        dim_2 = min(dim_1, 32)
        dim_3 = min(dim_2, 16)
        # We must store these as indivual class variables (instead of
        # just storing them in a list) because PyTorch looks at the
        # class variables to determine the model's trainable weights.
        self.fc0 = torch.nn.Linear(num_ins, dim_1)
        self.fc1 = torch.nn.Linear(dim_1, dim_2)
        self.fc2 = torch.nn.Linear(dim_2, dim_3)
        self.fc3 = torch.nn.Linear(dim_3, num_outs)
        self.sg = torch.nn.Sigmoid()
        print("BinaryDnn:\n    " +
              "\n    ".join(
                  [f"Linear: {lay.in_features}x{lay.out_features}"
                   for lay in [self.fc0, self.fc1, self.fc2, self.fc3]]) +
              "\n    Sigmoid")

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc0(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        return self.sg(x)


class SvmWrapper(BinaryModelWrapper):
    """ Wraps Svm. """

    name = "Svm"
    in_spc = ["arrival time", "loss rate"]
    out_spc = ["queue occupancy"]
    los_fnc = torch.nn.HingeEmbeddingLoss
    opt = torch.optim.SGD
    params = ["lr", "momentum"]

    def new(self):
        self.net = Svm(self.num_ins)
        return self.net

    def modify_data(self, sim, dat_in, dat_out, dat_out_raw, dat_out_oracle,
                    sequential):
        dat_in, dat_out, dat_out_raw, dat_out_oracle, scl_grps = (
            super(SvmWrapper, self).modify_data(
                sim, dat_in, dat_out, dat_out_raw, dat_out_oracle, sequential))
        # Map [0,1] to [-1, 1]
        fet = dat_out.dtype.names[0]
        dat_out[fet] = dat_out[fet] * 2 - 1
        return dat_in, dat_out, dat_out_raw, dat_out_oracle, scl_grps

    def _check_output_helper(self, out):
        # Remove a trailing dimension of size 1.
        out = torch.reshape(out, (out.size()[0],))
        # Transform the output to report classes -1 and 1.
        out[torch.where(out < 0)] = -1
        out[torch.where(out >= 0)] = 1
        return out


class Svm(torch.nn.Module):
    """ A simple SVM binary classifier. """

    def __init__(self, num_ins):
        super(Svm, self).__init__()
        self.fc = torch.nn.Linear(num_ins, 1)
        print(f"SVM:\n    Linear: {self.fc.in_features}x{self.fc.out_features}")

    def forward(self, x):
        return self.fc(x)


class SvmSklearnWrapper(SvmWrapper):
    """ Wraps an sklearn SVM. """

    name = "SvmSklearn"
    in_spc = [
        "arrival time",
        "inter-arrival time",
        "true RTT ratio",
        "loss event rate",
        "loss event rate sqrt",
        "mathis model throughput p/s",
        "mathis model label",
        "inter-arrival time ewma-alpha0.1",
        "inter-arrival time ewma-alpha0.2",
        "inter-arrival time ewma-alpha0.3",
        "inter-arrival time ewma-alpha0.4",
        "inter-arrival time ewma-alpha0.5",
        "inter-arrival time ewma-alpha0.6",
        "inter-arrival time ewma-alpha0.7",
        "inter-arrival time ewma-alpha0.8",
        "inter-arrival time ewma-alpha0.9",
        "inter-arrival time ewma-alpha1.0",
        "throughput p/s ewma-alpha0.1",
        "throughput p/s ewma-alpha0.2",
        "throughput p/s ewma-alpha0.3",
        "throughput p/s ewma-alpha0.4",
        "throughput p/s ewma-alpha0.5",
        "throughput p/s ewma-alpha0.6",
        "throughput p/s ewma-alpha0.7",
        "throughput p/s ewma-alpha0.8",
        "throughput p/s ewma-alpha0.9",
        "throughput p/s ewma-alpha1.0",
        "loss rate ewma-alpha0.1",
        "loss rate ewma-alpha0.2",
        "loss rate ewma-alpha0.3",
        "loss rate ewma-alpha0.4",
        "loss rate ewma-alpha0.5",
        "loss rate ewma-alpha0.6",
        "loss rate ewma-alpha0.7",
        "loss rate ewma-alpha0.8",
        "loss rate ewma-alpha0.9",
        "loss rate ewma-alpha1.0",
        "average inter-arrival time windowed-minRtt1",
        "average inter-arrival time windowed-minRtt2",
        "average inter-arrival time windowed-minRtt4",
        "average inter-arrival time windowed-minRtt8",
        "average inter-arrival time windowed-minRtt16",
        "average inter-arrival time windowed-minRtt32",
        "average inter-arrival time windowed-minRtt64",
        "average inter-arrival time windowed-minRtt128",
        "average inter-arrival time windowed-minRtt256",
        "average inter-arrival time windowed-minRtt512",
        "average inter-arrival time windowed-minRtt1024",
        "average throughput p/s windowed-minRtt1",
        "average throughput p/s windowed-minRtt2",
        "average throughput p/s windowed-minRtt4",
        "average throughput p/s windowed-minRtt8",
        "average throughput p/s windowed-minRtt16",
        "average throughput p/s windowed-minRtt32",
        "average throughput p/s windowed-minRtt64",
        "average throughput p/s windowed-minRtt128",
        "average throughput p/s windowed-minRtt256",
        "average throughput p/s windowed-minRtt512",
        "average throughput p/s windowed-minRtt1024",
        "loss rate windowed-minRtt1",
        "loss rate windowed-minRtt2",
        "loss rate windowed-minRtt4",
        "loss rate windowed-minRtt8",
        "loss rate windowed-minRtt16",
        "loss rate windowed-minRtt32",
        "loss rate windowed-minRtt64",
        "loss rate windowed-minRtt128",
        "loss rate windowed-minRtt256",
        "loss rate windowed-minRtt512",
        "loss rate windowed-minRtt1024"
    ]
    out_spc = ["queue occupancy ewma-alpha0.5"]
    los_fnc = None
    opt = None
    params = ["kernel", "degree", "penalty", "max_iter", "graph"]

    def new(self, **kwargs):
        self.graph = kwargs["graph"]
        kernel = kwargs["kernel"]
        max_iter = kwargs["max_iter"]
        # Automatically set the class weights based on the class
        # popularity in the training data. Change the maximum number
        # of iterations.
        self.net = (
            # Use manually-configured regularization. Since the number
            # of samples is greater than the number of features, solve
            # the primal optimization problem instead of its dual.
            svm.LinearSVC(
                penalty=kwargs["penalty"], dual=False, class_weight="balanced",
                verbose=1, max_iter=max_iter)
            if kernel == "linear" else
            # Supports L2 regularization only. The degree parameter is
            # used only if kernel == "poly".
            svm.SVC(
                kernel=kernel, degree=kwargs["degree"], class_weight="balanced",
                verbose=1, max_iter=max_iter))
        return self.net

    def train(self, dat_in, dat_out):
        """ Fits this model to the provided dataset. """
        self.net.fit(dat_in, dat_out)

    def __evaluate(self, preds, labels, raw, fair, flp, x_lim=None, sort_by_unfairness=True):
        # Compute the distance from fair, then divide by fair to
        # compute the relative unfairness.
        diffs = (raw - fair) / fair
        if sort_by_unfairness:
            # Sort based on unfairness.
            diffs, indices = torch.sort(diffs)
            preds = preds[indices]
            labels = labels[indices]
        # Bucketize and compute bucket accuracies.
        num_samples = preds.size()[0]
        num_buckets = min(20 * (1 if sort_by_unfairness else 4), num_samples)
        num_per_bucket = math.floor(num_samples / num_buckets)
        assert num_per_bucket > 0, \
            ("There must be at least one sample per bucket, but there are "
             f"{num_samples} samples and only {num_buckets} buckets!")
        # The resulting buckets are tuples of three values:
        #   (x-axis value for bucket, number predicted correctly, total)
        buckets = [
            (x,
             self.check_output(preds_, labels_),
             preds_.size()[0])
            for x, preds_, labels_ in [
                # Each bucket is defined by a tuple of three values:
                #   (x-axis value for bucket, predictions, ground truth labels).
                # The x-axis is the mean relative difference for this
                # bucket. A few values at the end may be discarded.
                (torch.mean(diffs[i:i + num_per_bucket]),
                 preds[i:i + num_per_bucket],
                 labels[i:i + num_per_bucket])
                for i in range(0, num_samples, num_per_bucket)]]
        if self.graph:
            # Plot each bucket's accuracy.
            pyplot.plot(
                ([x for x, _, _ in buckets]
                 if sort_by_unfairness else list(range(len(buckets)))),
                [c / t for _, c, t in buckets], "bo-")
            pyplot.ylim((-0.1, 1.1))
            if x_lim is not None:
                pyplot.xlim((x_lim[0] * 1.1, x_lim[1] * 1.1))
            pyplot.xlabel("Unfairness (fraction of fair)" if sort_by_unfairness else "Time")
            pyplot.ylabel("Classification accuracy")
            pyplot.tight_layout()
            pyplot.savefig(flp)
            pyplot.close()
        # Compute the overall accuracy.
        _, corrects, totals = zip(*buckets)
        acc = sum(corrects) / sum(totals)
        print(f"    Test accuracy: {acc * 100:.2f}%")
        return acc

    def test(self, fets, dat_in, dat_out_classes, dat_out_raw, dat_out_oracle,
             num_flws, arr_times=None, graph_prms=None):
        """
        Tests this model on the provided dataset and returns the test accuracy
        (higher is better).
        """
        # Compute the fair share. Convert from int to float to avoid
        # all values being rounded to 0.
        fair = torch.reciprocal(num_flws.type(torch.float))

        if self.graph:
            assert graph_prms is not None, "graph_prms is None!"
            out_dir = path.join(graph_prms["out_dir"], self.name)
            sort_by_unfairness = graph_prms["sort_by_unfairness"]
            # Set the x limits.Determine the maximum unfairness.
            x_lim = (
                # Compute the maximum unfairness.
                (-1, ((dat_out_raw - fair) / fair).max())
                if sort_by_unfairness else (0, graph_prms["dur_s"]))
            # Create the output directory.
            if not path.exists(out_dir):
                os.makedirs(out_dir)

            # Analyze feature importance.
            coefs = None
            try:
                coefs = self.net.coef_
            except AttributeError:
                # Coefficients are only; available with a linear kernel.
                print("Warning: Unable to extract coefficients!")
            if coefs is not None:
                imps, names = zip(*reversed(sorted(
                    [(imp, fets[idx]) for idx, imp in enumerate(coefs[0])],
                    key=lambda t: t[0])))
                # Pick the 20 features whose coefficients have the largest magnitude.
                imps, names = zip(*reversed(sorted(
                    [(imp, name) for _, imp, name in sorted(
                        [(abs(imp), imp, name) for imp, name in zip(imps, names)],
                        key=lambda t: t[0])[len(names) - 20:]],
                    key=lambda t: t[0])))
                num_fets = len(names)
                y_vals = list(range(num_fets))
                pyplot.figure(figsize=(7, 0.2 * num_fets))
                pyplot.barh(y_vals, imps, align="center")
                pyplot.yticks(y_vals, names)
                pyplot.ylim((-1, num_fets))
                pyplot.xlabel("Feature coefficient")
                pyplot.ylabel("Feature name")
                pyplot.tight_layout()
                pyplot.savefig(path.join(out_dir, f"features_{self.name}.pdf"))
                pyplot.close()

            # Analyze accuracy vs. unfairness for all flows and all
            # degrees of unfairness, for the Mathis Model oracle.
            print("Evaluting Mathis Model oracle:")
            self.__evaluate(
                dat_out_oracle, dat_out_classes, dat_out_raw, fair,
                path.join(
                    out_dir, f"accuracy_vs_unfairness_mathis_{self.name}.pdf"),
                x_lim, sort_by_unfairness)

            # Analyze, for each number of flows, accuracy vs. unfairness.
            flws_accs = []
            nums_flws = list(set(num_flws.tolist()))
            for num_flws_selected in nums_flws:
                print(f"Evaluating model for {num_flws_selected} flows:")
                valid = torch.where(num_flws == num_flws_selected)
                flws_accs.append(self.__evaluate(
                    torch.tensor(self.net.predict(dat_in[valid])),
                    dat_out_classes[valid], dat_out_raw[valid], fair[valid],
                    path.join(
                        out_dir,
                        (f"accuracy_vs_unfairness_{num_flws_selected}flows_"
                         f"{self.name}.pdf")),
                    x_lim, sort_by_unfairness))

            # Analyze accuracy vs. number of flows.
            x_vals = list(range(len(flws_accs)))
            pyplot.bar(x_vals, flws_accs, align="center")
            pyplot.xticks(x_vals, nums_flws)
            pyplot.ylim((0, 1.1))
            pyplot.xlabel("Total flows (1 unfair)")
            pyplot.ylabel("Classification accuracy")
            pyplot.tight_layout()
            pyplot.savefig(path.join(out_dir, f"accuracy_vs_num-flows_{self.name}.pdf"))
            pyplot.close()
        else:
            out_dir = "."
            sort_by_unfairness = False
            x_lim = None

        # Analyze accuracy vs unfairness for all flows and all degrees
        # of unfairness, for the model itself.
        print("Evaluating model:")
        return self.__evaluate(
            torch.tensor(self.net.predict(dat_in)), dat_out_classes,
            dat_out_raw, fair,
            path.join(out_dir, f"accuracy_vs_unfairness_{self.name}.pdf"),
            x_lim, sort_by_unfairness)


class LrSklearnWrapper(SvmSklearnWrapper):
    """ Wraps ab sklearn Logistic Regression model. """

    name = "LrSklearn"
    params = ["max_iter", "graph"]

    def new(self, **kwargs):
        self.graph = kwargs["graph"]
        # Use L1 regularization. Since the number of samples is
        # greater than the number of features, solve the primal
        # optimization problem instead of its dual. Automatically set
        # the class weights based on the class popularity in the
        # training data.Change the maximum number of iterations.
        self.net = linear_model.LogisticRegression(
            penalty="l1", dual=False, class_weight="balanced",
            solver="liblinear", max_iter=kwargs["max_iter"], verbose=1)
        return self.net


class LstmWrapper(PytorchModelWrapper):
    """ Wraps Lstm. """

    name = "Lstm"
    # For now, we do not use RTT ratio because our method of estimating
    # it cannot be performed by a general receiver.
    # in_spc = ["inter-arrival time", "RTT ratio", "loss rate"]
    in_spc = ["inter-arrival time", "loss rate"]
    out_spc = ["queue occupancy"]
    num_clss = 5
    # Cross-entropy loss is designed for multi-class classification tasks.
    los_fnc = torch.nn.CrossEntropyLoss
    opt = torch.optim.Adam
    params = ["lr"]

    def __init__(self, hid_dim=32, num_lyrs=1, out_dim=5):
        super(LstmWrapper, self).__init__()
        self.in_dim = len(self.in_spc)
        self.hid_dim = hid_dim
        self.num_lyrs = num_lyrs
        self.out_dim = out_dim

    def new(self):
        self.net = Lstm(self.in_dim, self.hid_dim, self.num_lyrs, self.out_dim)
        return self.net

    def init_hidden(self, batch_size):
        weight = next(self.net.parameters()).data
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
        clss = np.vectorize(
            functools.partial(
                percent_to_class, fair=1. / (sim.unfair_flws + sim.other_flws)),
            otypes=[int])(dat_out)
        clss_str = np.empty((clss.shape[0],), dtype=[("class", "int")])
        clss_str["class"] = clss
        return clss_str


class Lstm(torch.nn.Module):
    """ An LSTM that classifies a flow into one of five fairness categories. """

    def __init__(self, in_dim, hid_dim, num_lyrs, out_dim):
        super(Lstm, self).__init__()
        self.hid_dim = hid_dim
        self.lstm = torch.nn.LSTM(in_dim, self.hid_dim)
        self.fc = torch.nn.Linear(self.hid_dim, out_dim)
        self.sg = torch.nn.Sigmoid()
        print(f"Lstm - in_dim: {in_dim}, hid_dim: {self.hid_dim}, "
              f"num_lyrs: {num_lyrs}, out_dim: {out_dim}")

    def forward(self, x, hidden):
        # The LSTM itself, which also takes as input the previous hidden state.
        out, hidden = self.lstm(x, hidden)
        # Select the last piece as the LSTM output.
        out = out.contiguous().view(-1, self.hid_dim)
        # Classify the LSTM output.
        out = self.fc(out)
        out = self.sg(out)
        return out, hidden


#######################################################################
#### Old models. Present for archival purposes only. These are not ####
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


MODELS = {mdl.name: mdl for mdl in [
    BinaryDnnWrapper,
    SvmWrapper,
    SvmSklearnWrapper,
    LrSklearnWrapper,
    LstmWrapper
]}
MODEL_NAMES = sorted(MODELS.keys())
