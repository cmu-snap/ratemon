#!/usr/bin/env python3
"""
Models of the form:
  (current throughput, current RTT, target throughput) -> (ACK pacing interval)
"""

import torch


class LstmSimple(torch.nn.Module):
    # The specification of the input tensor format.
    # in_spc = ["inter-arrival time", "RTT ratio", "loss rate"]
    in_spc = ["inter-arrival time", "loss rate"]
    # The specification of the output tensor format.
    out_spc = ["queue occupancy"]

    def __init__(self, hid_dim=32, num_lyrs=1, out_dim=5):
        super(LstmSimple, self).__init__()
        self.in_dim = len(self.in_spc)
        self.hid_dim = hid_dim
        self.num_lyrs = num_lyrs
        self.out_dim = out_dim

        print(f"LstmSimple - in_dim: {self.in_dim}, hid_dim: {self.hid_dim}, "
              f"num_lyrs: {self.num_lyrs}, out_dim: {self.out_dim}")

        # Each input entry will correspond to a packet and contain two elements:
        #   (inter-arrival time, loss rate)
        # Guess: Set the hidden dimension to 32.
        self.lstm = torch.nn.LSTM(self.in_dim, self.hid_dim)
        # The input dimension (hidden dimension) is 32. We are
        # predicting 5 classes, so the output dimension is also 5.
        self.fc = torch.nn.Linear(self.hid_dim, self.out_dim)
        self.sg = torch.nn.Sigmoid()


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


class SimpleOne(torch.nn.Module):
    # The specification of the input tensor format.
    in_spc = ["average_throughput_after_bps"]
    # The specification of the output tensor format.
    out_spc = ["ack_period_us"]

    def __init__(self):
        super(SimpleOne, self).__init__()
        assert len(SimpleOne.in_spc) == 1
        assert len(SimpleOne.out_spc) == 1
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


class FcOne(torch.nn.Module):
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


# TODO: Do we need to manually initialize the weights in some way?
@torch.no_grad()
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.fill_(1.0)


MODELS = {
    "LstmSimple": LstmSimple,
    "SimpleOne": SimpleOne,
    "FcOne": FcOne,
    "FcTwo": FcTwo,
    "FcThree": FcThree,
    "FcFour": FcFour
}
