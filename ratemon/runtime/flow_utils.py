"""Defines useful classes for representing flows."""

import ctypes
import logging
import sys
import threading
import time

from ratemon.model import defaults, loss_event_rate, utils


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

    def __init__(self, fourtuple, loss_event_windows, start_time_us):
        """Set up data structures for a flow."""
        self.ingress_lock = threading.RLock()
        # self.inference_flag = multiprocessing.Value(typecode_or_type="i", lock=True)
        self.fourtuple = fourtuple
        self.flowkey = FlowKey(*fourtuple)
        self.incoming_packets = []
        self.sent_tsvals = {}
        # The time at which this flow started. Used to determine the relative packet
        # arrival time.
        self.start_time_us = start_time_us
        # The approximate delta between packet timestamps from libpcap and the
        # system clock.
        self.approx_pcap_system_delta_us = time.time() * 1e6 - start_time_us
        # Smallest RTT ever observed for this flow (microseconds). Used to calculate
        # the BDP. Updated whenever we compute features for this flow.
        self.min_rtt_us = sys.maxsize
        # The timestamp of the last packet on which we have run inference.
        self.latest_time_sec = time.time()
        self.label = defaults.Class.NEAR_TARGET
        self.decision = (defaults.Decision.NOT_PACED, None)
        self.loss_tracker = loss_event_rate.LossTracker(self, loss_event_windows)

    def __str__(self):
        """Create a string representation of this flow."""
        return str(self.flowkey)

    def is_interesting(self, timeout_us=5e6):
        """Whether the flow has seen any data in the last few seconds."""
        with self.ingress_lock:
            return self.incoming_packets and (
                time.time() * 1e6
                - self.approx_pcap_system_delta_us
                - self.incoming_packets[-1][4]
                < timeout_us
            )

    def is_ready(self, smoothing_window, longest_window):
        with self.ingress_lock:
            # This flow is ready for interence if...
            return (
                # ...we have at least as many packets as the smoothing window...
                len(self.incoming_packets) >= smoothing_window
                and (
                    # ...the time span covered by the packets is at least that which is
                    # required for the longest windowed input feature.
                    self.incoming_packets[-smoothing_window][4]
                    - self.incoming_packets[0][4]
                    >= self.min_rtt_us * longest_window
                )
            )


class FlowDB(dict):
    def __init__(self):
        super().__init__()
        # Maps each sender IP address to a map of flows.
        self._senders = {}
        # Acquire this lock when adding, removing, or iterating over flows; no
        # need to acquire this lock updating a flow object.
        self.lock = threading.RLock()

    def __setitem__(self, key, value):
        """key is a fourtuple and value is a Flow object."""
        with self.lock:
            sender_ip = value.flowkey.remote_addr
            # If this is the first flow from this sender, add the sender.
            if sender_ip not in self._senders:
                self._senders[sender_ip] = set()
            # Add the flow to the sender.
            self._senders[sender_ip].add(key)
            return super().__setitem__(key, value)

    def __delitem__(self, key):
        with self.lock:
            if key in self:
                sender_ip = self[key].flowkey.remote_addr
                if sender_ip in self._senders:
                    # Remove the flow from the sender.
                    if key in self._senders[sender_ip]:
                        self._senders[sender_ip].remove(key)
                    # If this was the last flow for this sender, remove the sender.
                    if len(self._senders[sender_ip]) == 0:
                        del self._senders[sender_ip]
            return super().__delitem__(key)

    def get_flows_from_sender(self, sender_ip, ignore_uninteresting=True):
        with self.lock:
            return {
                fourtuple
                for fourtuple in self._senders.get(sender_ip, set())
                if not ignore_uninteresting or self[fourtuple].is_interesting()
            }

    def sender_okay(
        self, sender_ip, smoothing_window, longest_window, ignore_uninteresting=True
    ):
        with self.lock:
            for fourtuple in self.get_flows_from_sender(sender_ip):
                if fourtuple not in self:
                    logging.warning("Flow %s not in flow DB", fourtuple)
                    continue
                # If the flow is both interesting and not ready, then the sender
                # as a whole is not ready...
                if (
                    not ignore_uninteresting or self[fourtuple].is_interesting()
                ) and not self[fourtuple].is_ready(smoothing_window, longest_window):
                    return False
            return True
