"""Calculate loss event rate and manage associated metadata."""

import collections
import logging

from ratemon.model import features


class LossTracker:
    def __init__(self, flow, window_sizes=None):
        if window_sizes is None:
            window_sizes = [8]
        assert window_sizes, "Must specify window_sizes."
        self.flow = flow
        self.window_sizes = window_sizes
        self.largest_window = max(self.window_sizes)
        # Track which packets are definitely retransmissions. Ignore these
        # packets when estimating the RTT. Note that because we are doing
        # receiver-side retransmission tracking, it is possible that there are
        # other retransmissions that we cannot detect.
        self.retrans_pkts = set()
        # All sequence numbers that have been received.
        self.unique_pkts = set()
        self.packet_seq = False  # cca in {"copa", "vivace"}
        self.highest_seq = 0
        self.weights = {
            max_loss_events: LossTracker.make_interval_weights(max_loss_events)
            for max_loss_events in self.window_sizes
        }
        self.loss_events = collections.deque()
        self.loss_events.appendleft(LossTracker.new_loss_event())

    @staticmethod
    def new_loss_event():
        return {
            "start_time_us": 0,
            # "total_losses": 0,
            "total_packets": 0,
        }

    @staticmethod
    def make_interval_weights(num_intervals):
        """Use to calculate loss event rate."""
        return [1 - i * 1 / num_intervals for i in range(num_intervals)]

    def update_for_new_packet(self, prev_pkt, cur_pkt, packets_lost):
        if packets_lost > 0:
            prev_arrival_time_us = prev_pkt[4]
            cur_arrival_time_us = cur_pkt[4]
            interloss_time_us = (
                cur_arrival_time_us - prev_arrival_time_us
            ) / packets_lost
            cur_rtt_us = cur_pkt[1]

            for pkt in range(packets_lost):
                loss_time_us = prev_arrival_time_us + (pkt + 1) * interloss_time_us
                if loss_time_us <= self.loss_events[0]["start_time_us"] + cur_rtt_us:
                    # This loss is within one RTT of the start of the current loss
                    # event, so it is still in the same loss event.
                    pass
                else:
                    # This loss is more than one RTT later than the start of the current
                    # loss event, so it starts a new loss event.
                    self.loss_events.appendleft(self.new_loss_event())
                    self.loss_events[0]["start_time_us"] = loss_time_us
                    if len(self.loss_events) > self.largest_window + 1:
                        # Keep around the data for one more loss event than the max
                        # because the current loss event might not be included in the
                        # loss event rate calculation.
                        self.loss_events.pop()
                # Account for the packet that was lost.
                # loss_events[0]["total_losses"] += 1
                self.loss_events[0]["total_packets"] += 1
        # Account for the packet that was actually received.
        self.loss_events[0]["total_packets"] += 1

    def get_packets_lost(self, prev_pkt, cur_pkt):
        # Note that Copa and Vivace use packet-level sequence numbers
        # instead of TCP's byte-level sequence numbers.
        current_seq = cur_pkt[0]
        prev_seq = prev_pkt[0]
        self.highest_seq = max(self.highest_seq, prev_seq)
        payload_bytes = cur_pkt[3]
        prev_payload_bytes = prev_pkt[3]
        if payload_bytes == 0 or prev_payload_bytes == 0:
            return 0

        # Detect sequence number wraparound and adjust the previous sequence number
        # accordingly.
        # TODO: Right now we just skip this packet.
        estimated_cur_seq = prev_seq + (1 if self.packet_seq else prev_payload_bytes)
        if estimated_cur_seq > 2**32:
            logging.warning("Warning: Sequence number wraparound detected")
            # old_prev_seq = prev_seq
            # # This might be negative, but that should be okay.
            # prev_seq = (estimated_cur_seq % 2**32) - payload_bytes
            self.highest_seq = 0
            # Just skip this packet...assume no loss.
            return 0

        retrans = (
            current_seq in self.unique_pkts
            or (prev_seq + (1 if self.packet_seq else prev_payload_bytes)) > current_seq
        )
        if retrans:
            # If this packet is a multiple retransmission, then this line
            # has no effect.
            self.retrans_pkts.add(current_seq)
        # If this packet has already been seen, then this line has no
        # effect.
        self.unique_pkts.add(current_seq)

        # Receiver-side loss rate estimation. Estimate the number of lost
        # packets since the last packet. Do not try anything complex or
        # prone to edge cases. Consider only the simple case where the last
        # packet and current packet are in order and not retransmissions.
        pkt_loss_cur_estimate = (
            0
            if (
                # The last packet was a retransmission.
                self.highest_seq != prev_seq
                or
                # The current packet is a retransmission.
                retrans
            )
            else round(
                (
                    current_seq
                    - (1 if self.packet_seq else prev_payload_bytes)
                    - prev_seq
                )
                / (1 if self.packet_seq else payload_bytes)
            )
        )
        assert pkt_loss_cur_estimate >= 0
        if pkt_loss_cur_estimate > 1000:
            logging.debug(
                "Warning: High packet loss estimate: %f", pkt_loss_cur_estimate
            )
        return pkt_loss_cur_estimate

    def calculate_loss_event_rate(self, weights):
        """Use to calculate loss event rate."""
        logging.info(
            "Flow %s len(self.loss_events): %s", self.flow, len(self.loss_events)
        )
        logging.info("Flow %s len(weights): %s", self.flow, len(weights))
        logging.info("Flow %s weights: %s", self.flow, weights)
        for i in range(min(len(self.loss_events), len(weights))):
            logging.info(
                "Flow %s self.loss_events[%s]: %s", self.flow, i, self.loss_events[i]
            )
            logging.info("Flow %s weights[%s]: %s", self.flow, i, weights[i])

        interval_total_0 = 0
        interval_total_1 = 0
        weights_total = 0
        for i in range(min(len(self.loss_events), len(weights))):
            interval_total_0 += self.loss_events[i]["total_packets"] * weights[i]
            weights_total += weights[i]
        for i in range(1, min(len(self.loss_events), len(weights) + 1)):
            interval_total_1 += self.loss_events[i]["total_packets"] * weights[i - 1]
        interval_total = max(interval_total_0, interval_total_1)
        interval_mean = interval_total / weights_total
        return 1 / interval_mean

    def loss_event_rate(self, pkts, all_pkts=False):
        """
        Each packet is a tuple of the form specified by features.PARSE_PACKETS_FETS:
            (
                seq,
                rtt_us,
                total_bytes,
                payload bytes,
                time_us,
            )
        """
        # Check if pkts is not a list of tuples, in which case it is probably a numpy
        # array, so convert it to a list of lists.
        if not (
            isinstance(pkts, list) and (len(pkts) == 0 or isinstance(pkts[0], tuple))
        ):
            pkts = pkts[list(features.get_names(features.PARSE_PACKETS_FETS))].tolist()

        num_pkts = len(pkts)
        # Since we need a previous packet, we cannot calculate anything for the first
        # packet. Assume no loss.
        packets_lost = [0]

        per_packet_loss_event_rate = {win: [0] for win in self.window_sizes}
        for idx in range(1, num_pkts):
            cur_packets_lost = self.get_packets_lost(pkts[idx - 1], pkts[idx])
            packets_lost.append(cur_packets_lost)
            if self.largest_window is not None:
                self.update_for_new_packet(pkts[idx - 1], pkts[idx], cur_packets_lost)

            if all_pkts:
                for max_loss_events, weights in self.weights.items():
                    per_packet_loss_event_rate[max_loss_events].append(
                        self.calculate_loss_event_rate(weights)
                    )

        if all_pkts:
            return packets_lost, per_packet_loss_event_rate

        return packets_lost, {
            max_loss_events: self.calculate_loss_event_rate(weights)
            for max_loss_events, weights in self.weights.items()
        }
