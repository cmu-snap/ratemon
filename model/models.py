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
        assert self.num_clss > 0, "Invalid number of output classes!"
        assert self.los_fnc is not None, "No loss function!"
        assert self.opt is not None, "No optimizer!"

    def init_hidden(self, batch_size):
        """
        If this model is an LSTM, then this method returns the initialized
        hidden state. Otherwise, return None.
        """
        return None

    @staticmethod
    def convert_to_class(dat_out_all):
        """
        Converts real-valued feature values into classes. dat_out_all is a
        list of pairs, one for each simulation, of the form:
            (output feature Numpy array, number of flows)
        """
        return dat_out_all

    def modify_data(self, dat_all):
        """
        Performs an arbitrary transformation on the data. data_all is a
        list of tuples of the form: (input tensor, output tensor).
        """
        return dat_all

    def forward(self, x, hidden):
        raise Exception(("Attempting to call \"forward()\" on the Model base "
                         "class itself."))


class BinaryDnn(Model):
    """ A simple binary classifier neural network. """

    in_spc = ["inter-arrival time", "loss rate"]
    out_spc = ["queue occupancy"]
    num_clss = 2
    nums_nodes = [16, 32, 16, 2]
    # los_fnc = torch.nn.BCELoss
    los_fnc = torch.nn.CrossEntropyLoss
    opt = torch.optim.SGD

    def __init__(self, win=500):
        super(BinaryDnn, self).__init__()
        self.check()

        self.win = win
        # We must store these as indivual class variables (instead of
        # just storing them in self.fcs) because PyTorch looks at the
        # class variables to determine the model's trainable weights.
        self.fc0 = torch.nn.Linear(len(BinaryDnn.in_spc) * self.win, 64)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 2)
        self.fcs = [self.fc0, self.fc1, self.fc2, self.fc3]
        self.sg = torch.nn.Sigmoid()
        print(f"BinaryDnn - win: {self.win}, fc layers: {len(self.fcs)}")

    def forward(self, x, hidden=None):
        for fc in self.fcs:
            x = torch.nn.functional.relu(fc(x))
        return self.sg(x), hidden

    @staticmethod
    def convert_to_class(dat_out_all):
        # Verify that all feature maps have exactly one column.
        for dat_out, _ in dat_out_all:
            assert len(dat_out.dtype.names) == 1, "Should be only one column."
        return [
            # Map a conversion function across all entries. Note that
            # here an entry is an entire row, since each row is a
            # single tuple value.
            np.vectorize(
                functools.partial(
                    # Compare each queue occupancy percent with the fair
                    # percent. prc[0] assumes a single column.
                    lambda prc, fair: prc[0] > fair, fair=1. / num_flws),
                # Convert to integers.
                otypes=[int])(dat_out)
            for dat_out, num_flws in dat_out_all]

    def modify_data(self, dat_all):
        # Extract from the set of simulations many separate intervals
        # of length self.win. Each interval becomes a training
        # example.
        total_packets = sum([dat_in.size()[0] for dat_in, _ in dat_all])
        # Set the number of windows to be the number of adjacent
        # windows that fit in the set of simulations. However, when
        # actually selecting the windows, they may overlap.
        num_wins = float(total_packets) / self.win
        dat_all_new = []
        # Loop over all of the simulations.
        for dat_in, dat_out in dat_all:
            # The number of packets in this simulation.
            sim_packets = dat_in.size()[0]
            # Select random intervals from this simulation.
            for idx in random.sample(
                    # Do not pick indices between 0 and (self.win - 1) to make
                    # sure that all windows of length self.win ending on the
                    # chosen index fit within the simulation.
                    range(self.win, sim_packets),
                    # Calculate the fraction of windows that should come from
                    # this simulation.
                    math.ceil(sim_packets / total_packets * num_wins)):
                assert idx >= self.win, \
                    f"Improperly formed index ({idx}) for window ({self.win})!"
                dat_all_new.append(
                    (dat_in[idx - self.win:idx].flatten(),
                     # As an output feature, select only the final
                     # ground truth value. I.e., the final ground
                     # truth value for this window becomes the ground
                     # truth for the entire window.
                     dat_out[idx - self.win:idx].flatten()[-1]))
        # Verify that we selected at least as many windows as we intended to.
        num_selected_wins = len(dat_all_new)
        assert num_selected_wins >= num_wins, \
            "Insufficient windows: {num_selected_wins} < {num_wins}"
        return dat_all_new


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

    def __init__(self, hid_dim=32, num_lyrs=1, out_dim=5):
        super(Lstm, self).__init__()
        self.check()

        self.in_dim = len(self.in_spc)
        self.hid_dim = hid_dim
        self.num_lyrs = num_lyrs
        self.out_dim = out_dim
        self.lstm = torch.nn.LSTM(self.in_dim, self.hid_dim)
        self.fc = torch.nn.Linear(self.hid_dim, self.out_dim)
        self.sg = torch.nn.Sigmoid()
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
    def convert_to_class(dat_out_all):
        for dat_out, _ in dat_out_all:
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

        return [
            # Map a conversion function across all entries. Note that
            # here an entry is an entire row, since each row is a
            # single tuple value.
            np.vectorize(
                functools.partial(percent_to_class, fair=1. / num_flws),
                otypes=[int])(dat_out)
            for dat_out, num_flws in dat_out_all]


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
    "SimpleOne": SimpleOne,
    "FcOne": FcOne,
    "FcTwo": FcTwo,
    "FcThree": FcThree,
    "FcFour": FcFour
}
