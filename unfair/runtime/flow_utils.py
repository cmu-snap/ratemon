"""Defines useful classes for representing flows."""

import ctypes
import sys
import threading
import time

from unfair.model import defaults, loss_event_rate, utils


class FlowKey(ctypes.Structure):
    """A struct to use as the key in maps in the corresponding eBPF program.

    Represents a flow fourtuple.
    """

    _fields_ = [
        ("local_addr", ctypes.c_uint),
        ("remote_addr", ctypes.c_uint),
        ("local_port", ctypes.c_ushort),
        ("remote_port", ctypes.c_ushort),
    ]

    def __init__(self, local_addr, remote_addr, local_port, remote_port):
        """Record the flow fourtuple."""
        super().__init__()
        self.local_addr = local_addr
        self.remote_addr = remote_addr
        self.local_port = local_port
        self.remote_port = remote_port

    def __str__(self):
        """Create a string representation of a flow fourtuple."""
        return (
            f"(R) {utils.int_to_ip_str(self.remote_addr)}:{self.remote_port} <-> "
            f"{utils.int_to_ip_str(self.local_addr)}:{self.local_port} (L)"
        )

    def __hash__(self):
        """Hash this just like a fourtuple."""
        return hash(
            (self.local_addr, self.remote_addr, self.local_port, self.remote_port)
        )

    def __eq__(self, other):
        """Compare this just like a fourtuple."""
        return (
            self.local_addr == other.local_addr
            and self.remote_addr == other.remote_addr
            and self.local_port == other.local_port
            and self.remote_port == other.remote_port
        )


class Flow:
    """Represents a single flow, identified by a four-tuple.

    A Flow object is created when we first receive a packet for the flow. A Flow object
    is deleted when it has been five minutes since we have checked this flow's
    fairness.

    Must acquire self.lock before accessing members.
    """

    def __init__(self, fourtuple, loss_event_windows):
        """Set up data structures for a flow."""
        self.ingress_lock = threading.RLock()
        # self.inference_flag = multiprocessing.Value(typecode_or_type="i", lock=True)
        self.fourtuple = fourtuple
        self.flowkey = FlowKey(*fourtuple)
        self.incoming_packets = []
        self.sent_tsvals = {}
        # Smallest RTT ever observed for this flow (microseconds). Used to calculate
        # the BDP. Updated whenever we compute features for this flow.
        self.min_rtt_us = sys.maxsize
        # The timestamp of the last packet on which we have run inference.
        self.latest_time_sec = time.time()
        self.label = defaults.Class.APPROX_FAIR
        self.decision = (defaults.Decision.NOT_PACED, None)
        self.loss_tracker = loss_event_rate.LossTracker(loss_event_windows)

    def __str__(self):
        """Create a string representation of this flow."""
        return str(self.flowkey)
