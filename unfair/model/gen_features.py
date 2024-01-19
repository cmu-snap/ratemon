#! /usr/bin/env python3
"""Parses the output of CloudLab experiments."""

import argparse
from contextlib import contextmanager
from io import UnsupportedOperation
import itertools
import logging
import multiprocessing
import subprocess
import os
from os import path
import random
import shutil
import sys
import time
import traceback

import json
import numpy as np

from unfair.model import cl_args, defaults, features, loss_event_rate, utils


def get_time_bounds(pkts, direction="data"):
    """Find when flows start and end.

    Returns the earliest and latest times in a particular direction of each flow
    in a trace. pkts is in the format produced by utils.parse_packets().

    Returns a list of tuples of the form:
        ( time of first packet, time of last packet )
    """
    dir_idx = 1 if direction == "ack" else 0
    # [[0, -1]] Selects the first and last arrival times.
    return [
        tuple(pkts[flw][dir_idx][features.ARRIVAL_TIME_FET][[0, -1]].tolist())
        for flw in pkts.keys()
    ]


@contextmanager
def open_exp(exp, exp_flp, untar_dir, out_dir, out_flp, always_reparse):
    """Untar an experiment.

    Lock the experiment to prevent two processes from opening it at once.
    Cleans up the lock and untarred files automatically.
    """
    lock_flp = path.join(out_dir, f"{exp.name}.lock")
    exp_dir = path.join(untar_dir, exp.name)
    # Keep track of what we do.
    locked = False
    untarred = False
    try:
        # Check the lock file for this experiment.
        if path.exists(lock_flp):
            print(f"Parsing already in progress: {exp_flp}")
            yield False, None
        # If the output file exists, then we do not need to parse this file.
        elif not always_reparse and path.exists(out_flp):
            print(f"Already parsed: {exp_flp}")
            yield False, None
        else:
            locked = True
            with open(lock_flp, "w", encoding="utf-8"):
                pass

            # Create a temporary folder to untar experiments.
            if not path.exists(untar_dir):
                os.mkdir(untar_dir)
            # If this experiment has already been untarred, then delete the old
            # files.
            if path.exists(exp_dir):
                shutil.rmtree(exp_dir)
            untarred = True
            subprocess.check_call(["tar", "-xf", exp_flp, "-C", untar_dir])
            yield True, exp_dir
    finally:
        # Remove an entity only if we created it.
        #
        # Remove untarred folder
        if locked and path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        # Remove lock file.
        if untarred and path.exists(lock_flp):
            os.remove(lock_flp)


def parse_opened_exp(
    exp,
    exp_flp,
    exp_dir,
    out_flp,
    skip_smoothed,
    receiver_ip,
    select_tail_percent=None,
    sender_fairness=False,
):
    """Parse an experiment. Return the smallest safe window size."""
    print(f"Parsing: {exp_flp}")
    if exp.name.startswith("FAILED"):
        print(f"Error: Experimant failed: {exp_flp}")
        return -1
    if exp.tot_flws == 0:
        print(f"Error: No flows to analyze in: {exp_flp}")
        return -1

    # Determine flow src and dst ports.
    params_flp = path.join(exp_dir, f"{exp.name}.json")
    if not path.exists(params_flp):
        print(f"Error: Cannot find params file ({params_flp}) in: {exp_flp}")
        return -1
    with open(params_flp, "r", encoding="utf-8") as fil:
        params = json.load(fil)
    # Dictionary mapping a flow to its flow's CCA. Each flow is a tuple of the
    # form: (sender port, receiver port)
    #
    # { (sender port, receiver port): CCA }
    flw_to_cca = {
        (sender_port, flw[6]): flw[2]
        for flw in params["flowsets"]
        for sender_port in flw[5]
    }
    flws = list(flw_to_cca.keys())

    # NOTE: Se no longer use the sender pcap.
    #
    # sender_pcap = path.join(exp_dir, f"sender-tcpdump-{exp.name}.pcap")
    # Look up the name of the receiver host.
    if "receiver" in params:
        receiver_pcap = path.join(
            exp_dir, f"{params['receiver'][0]}-tcpdump-{exp.name}.pcap"
        )
    else:
        receiver_pcap = path.join(exp_dir, f"server-tcpdump-{exp.name}.pcap")

    # if not (path.exists(sender_pcap) and path.exists(receiver_pcap)):
    if not path.exists(receiver_pcap):
        print(f"Warning: Missing pcap file in: {exp_flp}")
        return -1
    # NOTE: Disabled because not used.
    #
    # flw_to_pkts_sender = utils.parse_packets(sender_pcap, flw_to_cca)
    flw_to_pkts_receiver = utils.parse_packets(
        receiver_pcap, flw_to_cca, receiver_ip, select_tail_percent
    )

    # Log and drop flows with no data packets.
    flws_to_remove = []
    for flw, pkts in flw_to_pkts_receiver.items():
        if len(pkts[0]) == 0:
            print(f"\tWarning: No data packets for flow {flw} in: {exp_flp}")
            flws_to_remove.append(flw)
        if len(pkts[0]) == 1:
            print(f"\tWarning: No ACK packets for flow {flw} in: {exp_flp}")
            flws_to_remove.append(flw)
    flw_to_pkts_receiver = {
        flw: pkts
        for flw, pkts in flw_to_pkts_receiver.items()
        if flw not in flws_to_remove
    }
    flw_to_cca = {
        flw: cca for flw, cca in flw_to_cca.items() if flw not in flws_to_remove
    }
    flws = [flw for flw in flws if flw not in flws_to_remove]
    if not flws:
        print(f"Warning: No flows with packets in: {exp_flp}")
        return -1

    flw_to_pkts_receiver = utils.drop_packets_after_first_flow_finishes(
        flw_to_pkts_receiver, includes_acks=True
    )

    # NOTE: Disabled because not used.
    #
    # # Determine the path to the bottleneck queue log file.
    # toks = exp.name.split("-")
    # q_log_flp = path.join(
    #     exp_dir, "-".join(toks[:-1]) + "-forward-bottleneckqueue-" + toks[-1] + ".log"
    # )
    # q_log = None
    # if path.exists(q_log_flp):
    #     q_log = list(enumerate(utils.parse_queue_log(q_log_flp)))

    # Transform absolute times into relative times to make life easier.
    #
    # Determine the absolute earliest time observed in the experiment.
    earliest_time_us = min(
        first_time_us
        for bounds in [
            # NOTE: Disabled because not used.
            #
            # get_time_bounds(flw_to_pkts_sender, direction="data"),
            # get_time_bounds(flw_to_pkts_sender, direction="ack"),
            get_time_bounds(flw_to_pkts_receiver, direction="data"),
            get_time_bounds(flw_to_pkts_receiver, direction="ack"),
        ]
        for first_time_us, _ in bounds
    )
    # Subtract the earliest time from all times.
    for flw in flws:
        # NOTE: Disabled because not used.
        #
        # flw_to_pkts_sender[flw][0][features.ARRIVAL_TIME_FET] -= earliest_time_us
        # flw_to_pkts_sender[flw][1][features.ARRIVAL_TIME_FET] -= earliest_time_us
        flw_to_pkts_receiver[flw][0][features.ARRIVAL_TIME_FET] -= earliest_time_us
        flw_to_pkts_receiver[flw][1][features.ARRIVAL_TIME_FET] -= earliest_time_us

        # NOTE: Disabled because not used.
        #
        # assert (flw_to_pkts_sender[flw][0][features.ARRIVAL_TIME_FET] >= 0).all()
        # assert (flw_to_pkts_sender[flw][1][features.ARRIVAL_TIME_FET] >= 0).all()
        assert (flw_to_pkts_receiver[flw][0][features.ARRIVAL_TIME_FET] >= 0).all()
        assert (flw_to_pkts_receiver[flw][1][features.ARRIVAL_TIME_FET] >= 0).all()

    flws_time_bounds = get_time_bounds(flw_to_pkts_receiver, direction="data")

    # Process PCAP files from senders and receivers.
    # The final output, with one entry per flow.
    flw_results = {}

    # Keep track of the number of erroneous throughputs (i.e., higher than the
    # experiment bandwidth) for each window size. Each value is a pair where the first
    # entry is the number of erroneous throughputs originating from a single flow and
    # the second entry is the number of erroneous throughputs originating from the
    # combined throughput of all flows.
    win_to_errors = {win: [0, 0] for win in features.WINDOWS}

    # Create the (super-complicated) dtype. The dtype combines each metric at
    # multiple granularities.
    dtype = features.REGULAR_FETS + (
        [] if skip_smoothed else features.make_smoothed_features()
    )

    for flw_idx, flw in enumerate(flws):
        cca = flw_to_cca[flw]
        # Copa and PCC Vivace use packet-based sequence numbers as opposed to
        # TCP's byte-based sequence numbers.
        packet_seq = cca in {"copa", "vivace"}
        # NOTE: Disabled because not used.
        #
        # snd_data_pkts, snd_ack_pkts = flw_to_pkts_sender[flw]
        recv_data_pkts, recv_ack_pkts = flw_to_pkts_receiver[flw]

        first_data_time_us = recv_data_pkts[0][features.ARRIVAL_TIME_FET]

        # The final output. -1 implies that a value could not be calculated.
        output = np.full(len(recv_data_pkts), -1, dtype=dtype)

        # If this flow does not have any packets, then skip it.
        skip = False

        # NOTE: Disabled because not used.
        #
        # if snd_data_pkts.shape[0] == 0:
        #     skip = True
        #     print(f"Warning: No data packets sent for flow {flw_idx} in: " f"{exp_flp}")
        if recv_data_pkts.shape[0] == 0:
            skip = True
            print(
                f"Warning: No data packets received for flow {flw_idx} in: "
                f"{exp_flp}"
            )
        if recv_ack_pkts.shape[0] == 0:
            skip = True
            print(f"Warning: No ACK packets sent for flow {flw_idx} in: " f"{exp_flp}")
        if skip:
            flw_results[flw] = output
            continue

        # State that the windowed metrics need to track across packets.
        win_to_start_idx = {win: 0 for win in features.WINDOWS}
        # Total number of packet losses up to the current received
        # packet.
        pkt_loss_total_estimate = 0
        # Loss rate estimation.
        prev_seq = None
        prev_payload_bytes = None
        highest_seq = None
        # NOTE: Disabled because not used.
        #
        # # Use for Copa RTT estimation.
        # snd_ack_idx = 0
        # snd_data_idx = 0
        # Use for TCP and PCC Vivace RTT estimation.
        recv_ack_idx = 0

        # Track which packets are definitely retransmissions. Ignore these
        # packets when estimating the RTT. Note that because we are doing
        # receiver-side retransmission tracking, it is possible that there are
        # other retransmissions that we cannot detect.
        #
        # All sequence numbers that have been received.
        unique_pkts = set()
        # Sequence numbers that have been received multiple times.
        retrans_pkts = set()

        for j, recv_pkt in enumerate(recv_data_pkts):
            if j % 1000 == 0:
                print(
                    f"\tFlow {flw_idx + 1}/{exp.tot_flws}: "
                    f"{j}/{len(recv_data_pkts)} packets"
                )
            # Whether this is the first packet.
            first = j == 0
            # Note that Copa and Vivace use packet-level sequence numbers
            # instead of TCP's byte-level sequence numbers.
            recv_seq = recv_pkt[features.SEQ_FET]
            output[j][features.SEQ_FET] = recv_seq
            retrans = recv_seq in unique_pkts or (
                prev_seq is not None
                and prev_payload_bytes is not None
                and (prev_seq + (1 if packet_seq else prev_payload_bytes)) > recv_seq
            )
            if retrans:
                # If this packet is a multiple retransmission, then this line
                # has no effect.
                retrans_pkts.add(recv_seq)
            # If this packet has already been seen, then this line has no
            # effect.
            unique_pkts.add(recv_seq)

            recv_time_cur_us = recv_pkt[features.ARRIVAL_TIME_FET]
            output[j][features.ARRIVAL_TIME_FET] = recv_time_cur_us

            payload_bytes = recv_pkt[features.PAYLOAD_FET]
            wirelen_bytes = recv_pkt[features.WIRELEN_FET]
            output[j][features.PAYLOAD_FET] = payload_bytes
            output[j][features.WIRELEN_FET] = wirelen_bytes
            output[j][features.TOTAL_SO_FAR_FET] = (
                0 if first else output[j - 1][features.TOTAL_SO_FAR_FET]
            ) + wirelen_bytes
            output[j][features.PAYLOAD_SO_FAR_FET] = (
                0 if first else output[j - 1][features.PAYLOAD_SO_FAR_FET]
            ) + payload_bytes

            # Count how many flows were active when this packet was captured.
            active_flws = sum(
                1
                for first_time_us, last_time_us in flws_time_bounds
                if first_time_us <= recv_time_cur_us <= last_time_us
            )
            assert active_flws > 0, (
                f"Error: No active flows detected for packet {j} of "
                f"flow {flw_idx} in: {exp_flp}"
            )

            output[j][features.ACTIVE_FLOWS_FET] = active_flws
            output[j][features.BW_FAIR_SHARE_FRAC_FET] = utils.safe_div(1, active_flws)
            output[j][features.BW_FAIR_SHARE_BPS_FET] = (
                utils.safe_div(exp.bw_bps, active_flws) if exp.use_bess else -1
            )

            # Calculate RTT-related metrics.
            rtt_us = -1
            if not first and recv_seq != -1 and not retrans:
                if cca == "copa":
                    raise UnsupportedOperation("Support for Copa has been removed!")

                    # NOTE: Disabled because not used.
                    #
                    # # In a Copa ACK, the sender timestamp is the time at which
                    # # the corresponding data packet was sent. The receiver
                    # # timestamp is the time that the data packet was received
                    # # and the ACK was sent. This enables sender-side RTT
                    # # estimation. However, because the sender does not echo a
                    # # value back to the receiver, this cannot be used for
                    # # receiver-size RTT estimation.
                    # #
                    # # For now, we will just do sender-side RTT estimation. When
                    # # selecting which packets to use for the RTT estimate, we
                    # # will select the packet/ACK pair whose ACK arrived soonest
                    # # before packet j was sent. This means that the sender would
                    # # have been able to calculate this RTT estimate before
                    # # sending packet j, and could very well have included the
                    # # RTT estimate in packet j's header.
                    # #
                    # # First, find the index of the ACK that was received soonest
                    # # before packet j was sent.
                    # snd_ack_idx = utils.find_bound(
                    #     snd_ack_pkts[features.SEQ_FET],
                    #     recv_seq,
                    #     snd_ack_idx,
                    #     snd_ack_pkts.shape[0] - 1,
                    #     which="before",
                    # )
                    # snd_ack_seq = snd_ack_pkts[snd_ack_idx][features.SEQ_FET]
                    # # Then, find this ACK's data packet.
                    # snd_data_seq = snd_data_pkts[snd_data_idx][features.SEQ_FET]
                    # while snd_data_idx < snd_data_pkts.shape[0]:
                    #     snd_data_seq = snd_data_pkts[snd_data_idx][features.SEQ_FET]
                    #     if snd_data_seq == snd_ack_seq:
                    #         # Third, the RTT is the difference between the
                    #         # sending time of the data packet and the arrival
                    #         # time of its ACK.
                    #         rtt_us = (
                    #             snd_ack_pkts[snd_ack_idx][features.ARRIVAL_TIME_FET]
                    #             - snd_data_pkts[snd_data_idx][features.ARRIVAL_TIME_FET]
                    #         )
                    #         assert rtt_us >= 0, (
                    #             f"Error: Calculated negative RTT ({rtt_us} "
                    #             f"us) for packet {j} of flow {flw} in: "
                    #             f"{exp_flp}"
                    #         )
                    #         break
                    #     snd_data_idx += 1
                elif cca == "vivace":
                    # UDT ACKs may contain the RTT. Find the last ACK to be sent
                    # by the receiver before packet j was received.
                    recv_ack_idx = utils.find_bound(
                        recv_ack_pkts[features.ARRIVAL_TIME_FET],
                        recv_time_cur_us,
                        recv_ack_idx,
                        recv_ack_pkts.shape[0] - 1,
                        which="before",
                    )
                    udt_rtt_us = recv_ack_pkts[recv_ack_idx][features.TS_1_FET]
                    if udt_rtt_us > 0:
                        # The RTT is an optional field in UDT ACK packets. I
                        # assume that this means that if the RTT is not
                        # included, then the field will be 0.
                        rtt_us = udt_rtt_us
                else:
                    # This is a TCP flow. Do receiver-side RTT estimation using
                    # the TCP timestamp option. Attempt to find a new RTT
                    # estimate. Move recv_ack_idx to the first occurance of the
                    # timestamp option TSval corresponding to the current
                    # packet's TSecr.
                    recv_ack_idx_old = recv_ack_idx
                    tsval = recv_ack_pkts[recv_ack_idx][features.TS_1_FET]
                    tsecr = recv_pkt[features.TS_2_FET]
                    while recv_ack_idx < recv_ack_pkts.shape[0]:
                        tsval = recv_ack_pkts[recv_ack_idx][features.TS_1_FET]
                        if tsval == tsecr:
                            # If we found a timestamp option match, then update
                            # the RTT estimate.
                            rtt_us = (
                                recv_time_cur_us
                                - recv_ack_pkts[recv_ack_idx][features.ARRIVAL_TIME_FET]
                            )
                            break
                        recv_ack_idx += 1
                    else:
                        # If we never found a matching tsval, then use the
                        # previous RTT estimate and reset recv_ack_idx to search
                        # again on the next packet.
                        rtt_us = output[j - 1][features.RTT_FET]
                        recv_ack_idx = recv_ack_idx_old

            recv_time_prev_us = (
                -1 if first else output[j - 1][features.ARRIVAL_TIME_FET]
            )
            interarr_time_us = utils.safe_sub(recv_time_cur_us, recv_time_prev_us)
            output[j][features.INTERARR_TIME_FET] = interarr_time_us
            output[j][features.INV_INTERARR_TIME_FET] = utils.safe_mul(
                8 * 1e6 * wirelen_bytes, utils.safe_div(1, interarr_time_us)
            )

            output[j][features.RTT_FET] = rtt_us
            min_rtt_us = utils.safe_min(
                sys.maxsize if first else output[j - 1][features.MIN_RTT_FET], rtt_us
            )
            output[j][features.MIN_RTT_FET] = min_rtt_us
            rtt_estimate_ratio = utils.safe_div(rtt_us, min_rtt_us)
            output[j][features.RTT_RATIO_FET] = rtt_estimate_ratio

            # Receiver-side loss rate estimation. Estimate the number of lost
            # packets since the last packet. Do not try anything complex or
            # prone to edge cases. Consider only the simple case where the last
            # packet and current packet are in order and not retransmissions.
            pkt_loss_cur_estimate = (
                -1
                if (
                    recv_seq == -1
                    or prev_seq is None
                    or prev_seq == -1
                    or prev_payload_bytes is None
                    or prev_payload_bytes <= 0
                    or payload_bytes <= 0
                    or highest_seq is None
                    or
                    # The last packet was a retransmission.
                    highest_seq != prev_seq
                    or
                    # The current packet is a retransmission.
                    retrans
                )
                else round(
                    (recv_seq - (1 if packet_seq else prev_payload_bytes) - prev_seq)
                    / (1 if packet_seq else payload_bytes)
                )
            )

            if pkt_loss_cur_estimate != -1:
                pkt_loss_total_estimate += pkt_loss_cur_estimate
            loss_rate_cur = utils.safe_div(
                pkt_loss_cur_estimate, utils.safe_add(pkt_loss_cur_estimate, 1)
            )

            output[j][features.PACKETS_LOST_FET] = pkt_loss_cur_estimate
            prev_packets_lost_total = (
                0 if first else output[j - 1][features.PACKETS_LOST_TOTAL_FET]
            )
            # This feature is different, because we want to skip the current value if
            # it's -1.
            output[j][features.PACKETS_LOST_TOTAL_FET] = (
                prev_packets_lost_total
                if pkt_loss_cur_estimate == -1
                else utils.safe_add(prev_packets_lost_total, pkt_loss_cur_estimate)
            )
            output[j][features.LOSS_RATE_FET] = loss_rate_cur
            sqrt_loss_rate_cur = utils.safe_div(1, utils.safe_sqrt(loss_rate_cur))
            output[j][features.SQRT_LOSS_RATE_FET] = sqrt_loss_rate_cur
            mathis_tput_raw = utils.safe_mathis_tput_bps(
                output[j][features.PAYLOAD_FET],
                output[j][features.RTT_FET],
                output[j][features.LOSS_RATE_FET],
            )
            output[j][features.MATHIS_TPUT_LOSS_RATE_FET] = mathis_tput_raw

            # EWMA metrics.
            for (metric, _), alpha in itertools.product(
                features.EWMA_FETS, features.ALPHAS
            ):
                if skip_smoothed:
                    continue

                metric = features.make_ewma_metric(metric, alpha)
                if metric.startswith(features.INTERARR_TIME_FET):
                    new = interarr_time_us
                elif metric.startswith(features.INV_INTERARR_TIME_FET):
                    # Do not use the interarrival time EWMA to calculate the
                    # inverse interarrival time. Instead, use the true inverse
                    # interarrival time so that the value used to update the
                    # inverse interarrival time EWMA is not "EWMA-ified" twice.
                    new = output[j][features.INV_INTERARR_TIME_FET]
                elif metric.startswith(features.RTT_FET):
                    new = rtt_us
                elif metric.startswith(features.RTT_RATIO_FET):
                    new = rtt_estimate_ratio
                elif metric.startswith(features.LOSS_RATE_FET):
                    new = loss_rate_cur
                elif metric.startswith(features.SQRT_LOSS_RATE_FET):
                    new = (
                        1 if sqrt_loss_rate_cur in utils.UNSAFE else sqrt_loss_rate_cur
                    )
                elif metric.startswith(features.MATHIS_TPUT_LOSS_RATE_FET):
                    new = mathis_tput_raw
                else:
                    raise Exception(f"Unknown EWMA metric: {metric}")
                # Update the EWMA. If this is the first value, then use 0 are
                # the old value.
                output[j][metric] = utils.safe_update_ewma(
                    -1 if first else output[j - 1][metric], new, alpha
                )

            # If we cannot estimate the min RTT, then we cannot compute any
            # windowed metrics.
            if min_rtt_us != -1:
                # Move the window start indices later in time. The min RTT
                # estimate will never increase, so we do not need to investigate
                # whether the start of the window moved earlier in time.
                for win in features.WINDOWS:
                    win_to_start_idx[win] = utils.find_bound(
                        output[features.ARRIVAL_TIME_FET],
                        target=recv_time_cur_us - (win * min_rtt_us),
                        min_idx=win_to_start_idx[win],
                        max_idx=j,
                        which="after",
                    )

            # Windowed metrics.
            for (metric, _), win in itertools.product(
                features.WINDOWED_FETS, features.WINDOWS
            ):
                # If we cannot estimate the min RTT, then we cannot compute any
                # windowed metrics.
                if skip_smoothed or min_rtt_us == -1:
                    continue

                # Calculate windowed metrics only if an entire window has
                # elapsed since the start of the flow.
                win_size_us = win * min_rtt_us
                if recv_time_cur_us - first_data_time_us < win_size_us:
                    continue

                # A window requires at least two packets. Note that this means
                # the the first packet will always be skipped.
                win_start_idx = win_to_start_idx[win]
                if win_start_idx == j:
                    continue

                metric = features.make_win_metric(metric, win)
                if metric.startswith(features.INTERARR_TIME_FET):
                    new = utils.safe_div(
                        utils.safe_sub(
                            recv_time_cur_us,
                            output[win_start_idx][features.ARRIVAL_TIME_FET],
                        ),
                        j - win_start_idx,
                    )
                elif metric.startswith(features.INV_INTERARR_TIME_FET):
                    new = utils.safe_mul(
                        8 * 1e6 * wirelen_bytes,
                        utils.safe_div(
                            1,
                            output[j][
                                features.make_win_metric(
                                    features.INTERARR_TIME_FET, win
                                )
                            ],
                        ),
                    )
                elif metric.startswith(features.TPUT_FET):
                    tput_bps = utils.safe_tput_bps(output, win_start_idx, j)
                    # If the throughput exceeds the bandwidth, then print a
                    # warning and do not record this throughput.
                    if tput_bps != -1 and exp.use_bess and tput_bps > exp.bw_bps:
                        win_to_errors[win][0] += 1
                        continue
                    new = tput_bps
                elif metric.startswith(features.TPUT_SHARE_FRAC_FET):
                    # This is calculated at the end.
                    continue
                elif metric.startswith(features.TOTAL_TPUT_FET):
                    # This is calcualted at the end.
                    continue
                elif metric.startswith(features.TPUT_FAIR_SHARE_BPS_FET):
                    # This is calculated at the end.
                    continue
                elif metric.startswith(features.TPUT_TO_FAIR_SHARE_RATIO_FET):
                    # This is calculated at the end.
                    continue
                elif metric.startswith(features.RTT_FET):
                    new = utils.safe_mean(output[features.RTT_FET], win_start_idx, j)
                elif metric.startswith(features.RTT_RATIO_FET):
                    new = utils.safe_mean(
                        output[features.RTT_RATIO_FET], win_start_idx, j
                    )
                elif metric.startswith(features.LOSS_RATE_FET):
                    win_losses = utils.safe_sum(
                        output[features.PACKETS_LOST_FET], win_start_idx + 1, j
                    )
                    new = utils.safe_div(win_losses, win_losses + (j - win_start_idx))
                elif metric.startswith(features.SQRT_LOSS_RATE_FET):
                    new = utils.safe_div(
                        1,
                        utils.safe_sqrt(
                            output[j][
                                features.make_win_metric(features.LOSS_RATE_FET, win)
                            ]
                        ),
                    )
                elif metric.startswith(features.LOSS_EVENT_RATE_FET):
                    # Populated below.
                    continue
                elif metric.startswith(features.SQRT_LOSS_EVENT_RATE_FET):
                    # Populated below.
                    continue
                elif metric.startswith(features.MATHIS_TPUT_LOSS_RATE_FET):
                    new = utils.safe_mathis_tput_bps(
                        output[j][features.PAYLOAD_FET],
                        output[j][features.RTT_FET],
                        output[j][features.LOSS_RATE_FET],
                    )
                elif metric.startswith(features.MATHIS_TPUT_LOSS_EVENT_RATE_FET):
                    # Populated below.
                    continue
                else:
                    raise Exception(f"Unknown windowed metric: {metric}")
                output[j][metric] = new

            prev_seq = recv_seq
            prev_payload_bytes = payload_bytes
            highest_seq = (
                prev_seq if highest_seq is None else max(highest_seq, prev_seq)
            )
            # In the event of sequence number wraparound, reset the sequence
            # number tracking.
            #
            # TODO: Test sequence number wraparound logic.
            if (
                recv_seq != -1
                and recv_seq + (1 if packet_seq else payload_bytes) > 2**32
            ):
                print(
                    "Warning: Sequence number wraparound detected for packet "
                    f"{j} of flow {flw} in: {exp_flp}"
                )
                highest_seq = None
                prev_seq = None

        # Fill in loss event rate--related metrics: LOSS_EVENT_RATE_FET,
        # SQRT_LOSS_EVENT_RATE, and MATHIS_TPUT_LOSS_EVENT_RATE_FET.
        for win, loss_event_rates in (
            loss_event_rate.LossTracker(features.WINDOWS)
            .loss_event_rate(output, all_pkts=True)[1]
            .items()
        ):
            if len(loss_event_rates) != len(output):
                print(
                    f"Error: Number of loss event rates ({len(loss_event_rates)}) "
                    f"does not equal number of packets ({len(output)}) "
                    f"for window size {win} in flow {flw} in: {exp_flp}"
                )
                continue
            # Fill in loss event rate.
            output[
                features.make_win_metric(features.LOSS_EVENT_RATE_FET, win)
            ] = loss_event_rates

            # Fill in the sqrt of the loss event rate and the mathis tput based on loss
            # event rate.
            sqrt_loss_event_rate_metric = features.make_win_metric(
                features.SQRT_LOSS_EVENT_RATE_FET, win
            )
            mathis_tput_loss_event_rate_metric = features.make_win_metric(
                features.MATHIS_TPUT_LOSS_EVENT_RATE_FET, win
            )
            for j in range(len(output)):
                # Use the loss event rate to compute 1 / sqrt(loss event rate).
                output[j][sqrt_loss_event_rate_metric] = utils.safe_div(
                    1, utils.safe_sqrt(loss_event_rates[j])
                )
                output[j][
                    mathis_tput_loss_event_rate_metric
                ] = utils.safe_mathis_tput_bps(
                    output[j][features.PAYLOAD_FET],
                    output[j][features.RTT_FET],
                    loss_event_rates[j],
                )

        # NOTE: Disabled because not used.
        #
        # # Get the sequence number of the last received packet.
        # last_seq = output[-1][features.SEQ_FET]
        # if last_seq == -1:
        #     print(
        #         "Warning: Unable to calculate retransmission or bottleneck "
        #         "queue drop rates due to unknown last sequence number for "
        #         f"(UDP?) flow {flw_idx} in: {exp_flp}"
        #     )
        # else:
        #     # Calculate the true number of retransmissions using the sender
        #     # traces.
        #     #
        #     # Truncate the sent packets at the last occurence of the last packet to
        #     # be received.
        #     #
        #     # Find when the last received packet was sent. Assume that if this
        #     # packet was retransmitted, then the last retransmission is the one
        #     # that arrived at the receiver (which may be an incorrect
        #     # assumption).
        #     snd_idx = len(snd_data_pkts) - 1
        #     while snd_idx >= 0:
        #         if snd_data_pkts[snd_idx][features.SEQ_FET] == last_seq:
        #             # unique_snd_pkts, counts = np.unique(
        #             #     snd_data_pkts[:snd_idx + 1][features.SEQ_FET],
        #             #     return_counts=True)
        #             # unique_snd_pkts = unique_snd_pkts.tolist()
        #             # counts = counts.tolist()
        #             # all_retrans = [
        #             #     (seq, counts)
        #             #     for seq, counts in zip(unique_snd_pkts, counts)
        #             #     if counts > 1]
        #
        #             # tot_pkts = snd_idx + 1
        #
        #             # The retransmission rate is:
        #             #     1 - unique packets / total packets.
        #             output[-1][features.RETRANS_RATE_FET] = (
        #                 1
        #                 -
        #                 # Find the number of unique sequence numbers, from the
        #                 # beginning up until when the last received packet was
        #                 # sent.
        #                 np.unique(snd_data_pkts[: snd_idx + 1][features.SEQ_FET]).shape[
        #                     0
        #                 ]
        #                 /
        #                 # Convert from index to packet count.
        #                 (snd_idx + 1)
        #             )
        #             break
        #         snd_idx -= 1
        #     else:
        #         print(
        #             "Warning: Did not find when the last received packet "
        #             f"(seq: {last_seq}) was sent for flow {flw_idx} in: "
        #             f"{exp_flp}"
        #         )
        #
        #     # Calculate the true drop rate at the bottleneck queue using the
        #     # bottleneck queue logs.
        #     sender_port = flw[0]
        #     deq_idx = None
        #     drop_rate = None
        #     if q_log is None:
        #         print(f"Warning: Unable to find bottleneck queue log: {q_log_flp}")
        #     else:
        #         # Find the dequeue log corresponding to the last packet that was
        #         # received.
        #         for record_idx, record in reversed(q_log):
        #             if (
        #                 record[0] == "deq"
        #                 and record[2] == sender_port
        #                 and record[3] == last_seq
        #             ):
        #                 deq_idx = record_idx
        #                 break
        #     if deq_idx is None:
        #         print(
        #             "Warning: Did not find when the last received packet "
        #             f"(seq: {last_seq}) was dequeued for flow {flw_idx} in: "
        #             f"{exp_flp}"
        #         )
        #     else:
        #         # Find the most recent stats log before the last received
        #         # packet was dequeued.
        #         for _, record in reversed(q_log[:deq_idx]):
        #             if record[0] == "stats" and record[1] == sender_port:
        #                 drop_rate = record[4] / (record[2] + record[4])
        #                 break
        #     if drop_rate is None:
        #         print(
        #             "Warning: Did not calculate the drop rate at the bottleneck "
        #             f"queue for flow {flw_idx} in: {exp_flp}"
        #         )
        #     else:
        #         output[-1][features.DROP_RATE_FET] = drop_rate

        # Make sure that all output rows were used.
        used_rows = np.sum(output[features.ARRIVAL_TIME_FET] != -1)
        total_rows = output.shape[0]
        assert used_rows == total_rows, (
            f"Error: Used only {used_rows} of {total_rows} rows for flow "
            f"{flw_idx} in: {exp_flp}"
        )

        flw_results[flw] = output

    # Save memory by explicitly deleting the sent and received packets
    # after they have been parsed. This happens outside of the above
    # for-loop because only the last iteration's packets are not
    # automatically cleaned up by now (they go out of scope when the
    # *_pkts variables are overwritten by the next loop).

    # NOTE: Disabled because not used.
    #
    # del snd_data_pkts
    del recv_data_pkts
    del recv_ack_pkts

    if not skip_smoothed:
        # Maps window the index of the packet at the start of that window.
        win_to_start_idx = {win: 0 for win in features.WINDOWS}

        # Merge the flow data into a unified timeline.
        combined = []
        for flw in flws:
            num_pkts = flw_results[flw].shape[0]
            merged = np.empty(
                (num_pkts,),
                dtype=[
                    (features.WIRELEN_FET, "float64"),
                    (features.MIN_RTT_FET, "float64"),
                    ("sender port", "float64"),
                    ("receiver port", "float64"),
                    ("index", "float64"),
                ],
            )
            merged[features.WIRELEN_FET] = flw_results[flw][features.WIRELEN_FET]
            merged[features.MIN_RTT_FET] = flw_results[flw][features.MIN_RTT_FET]
            merged["sender port"].fill(flw[0])
            merged["receiver port"].fill(flw[1])
            merged["index"] = np.arange(num_pkts)
            combined.append(merged)
        zipped_arr_times, zipped_dat = utils.zip_timeseries(
            [flw_results[flw][features.ARRIVAL_TIME_FET] for flw in flws], combined
        )

        for j in range(zipped_arr_times.shape[0]):
            min_rtt_us = zipped_dat[j][features.MIN_RTT_FET]
            if min_rtt_us == -1:
                continue

            for win in features.WINDOWS:
                # The bounds should never go backwards, so start the
                # search at the current bound.
                win_to_start_idx[win] = utils.find_bound(
                    zipped_arr_times,
                    target=(
                        zipped_arr_times[j]
                        - (win * zipped_dat[j][features.MIN_RTT_FET])
                    ),
                    min_idx=win_to_start_idx[win],
                    max_idx=j,
                    which="after",
                )
                # If the window's trailing edge caught up with its
                # leading edge, then skip this flow.
                if win_to_start_idx[win] >= j:
                    continue

                total_tput_bps = utils.safe_div(
                    utils.safe_mul(
                        # Accumulate the bytes received by this flow during this
                        # window. When calculating the average throughput, we
                        # must exclude the first packet in the window.
                        utils.safe_sum(
                            zipped_dat[features.WIRELEN_FET],
                            start_idx=win_to_start_idx[win] + 1,
                            end_idx=j,
                        ),
                        8 * 1e6,
                    ),
                    utils.safe_sub(
                        zipped_arr_times[j], zipped_arr_times[win_to_start_idx[win]]
                    ),
                )
                # Check if the total throughput is erroneous, and if so, then do not
                # fill in features related to the total throughput.
                if exp.use_bess and total_tput_bps > exp.bw_bps:
                    win_to_errors[win][1] += 1
                    continue

                # Extract the flow to which this packet belongs, as well as
                # its index in its flow.
                flw = tuple(zipped_dat[j][["sender port", "receiver port"]].tolist())
                index = int(zipped_dat[j]["index"])
                flw_results[flw][index][
                    features.make_win_metric(features.TOTAL_TPUT_FET, win)
                ] = total_tput_bps
                # Use the total throughput and the number of active flows to
                # calculate the throughput fair share.
                flw_results[flw][index][
                    features.make_win_metric(features.TPUT_FAIR_SHARE_BPS_FET, win)
                ] = utils.safe_div(
                    total_tput_bps,
                    flw_results[flw][index][features.ACTIVE_FLOWS_FET],
                )
                # Divide the flow's throughput by the total throughput.
                tput_share = utils.safe_div(
                    flw_results[flw][index][
                        features.make_win_metric(features.TPUT_FET, win)
                    ],
                    total_tput_bps,
                )
                flw_results[flw][index][
                    features.make_win_metric(features.TPUT_SHARE_FRAC_FET, win)
                ] = tput_share
                # Calculate the ratio of tput share to bandwidth fair share.
                flw_results[flw][index][
                    features.make_win_metric(features.TPUT_TO_FAIR_SHARE_RATIO_FET, win)
                ] = utils.safe_div(
                    tput_share,
                    flw_results[flw][index][features.BW_FAIR_SHARE_FRAC_FET],
                )

    print(f"\tFinal window durations in: {exp_flp}:")
    for win in features.WINDOWS:
        print(
            f"\t\t{win}:",
            ", ".join(
                f"{dur_us} us" if dur_us > 0 else "unknown"
                for dur_us in (
                    win
                    * np.asarray(
                        [res[-1][features.MIN_RTT_FET] for res in flw_results.values()]
                    )
                ).tolist()
            ),
        )
    print(f"\tWindow errors in: {exp_flp}")
    for win in features.WINDOWS:
        print(
            f"\t\t{win}: per-flow: {win_to_errors[win][0]}, overall: {win_to_errors[win][1]}"
        )
    smallest_safe_win = 0
    for win in sorted(features.WINDOWS):
        if sum(win_to_errors[win]) == 0:
            print(f"\tSmallest safe window size is {win} in: {exp_flp}")
            smallest_safe_win = win
            break
    else:
        print(f"Warning: No safe window sizes in: {exp_flp}")

    # Determine if there are any NaNs or Infs in the results. For the results
    # for each flow, look through all features (columns) and make a note of the
    # features that bad values. Flatten these lists of feature names, using a
    # set comprehension to remove duplicates.
    bad_fets = {
        fet
        for flw_dat in flw_results.values()
        for fet in flw_dat.dtype.names
        if not np.isfinite(flw_dat[fet]).all()
    }
    if bad_fets:
        print(
            f"Warning: Experiment {exp_flp} has NaNs of Infs in features: "
            f"{bad_fets}"
        )

    # Save the results.
    if path.exists(out_flp):
        print(f"\tOutput already exists: {out_flp}")
    else:
        print(f"\tSaving: {out_flp}")
        np.savez_compressed(
            out_flp,
            **{str(k + 1): v for k, v in enumerate(flw_results[flw] for flw in flws)},
        )

    return smallest_safe_win


def parse_received_packets(
    flw,
    start_time_us,
    min_rtt_us,
    fets,
    previous_fets=None,
    win_metrics_start_idx=0,
):
    """Generate features for the inference runtime.

    Requires the absolute start time of the flow (microseconds measured against the
    same epoch as the packet timestamps in ARRIVAL_TIME_FET) and the existing minimum
    RTT (microseconds).

    In contrast to parse_opened_exp(), this function only has access to receiver
    information and only processes a single flow. Furthermore, windowed metrics are
    only populated from index win_metrics_start_idx through the last packet. Features
    that cannot be or are willfully not calculated are set to -1. The results may
    contain NaN or Inf values.

    Returns a tuple containing a structured numpy array with the resulting features,
    along with the updated minimum RTT (microseconds).

    TODO: We actually do not update the minimum RTT. Should we, or is that handled
          elsewhere?
    """
    num_pkts = len(fets)
    assert num_pkts, f"No packets provided for flow: {flw}"
    assert 1 <= win_metrics_start_idx <= num_pkts, (
        "Invalid start index for windowed features: "
        f"1 <= {win_metrics_start_idx} <= {num_pkts - 1}"
    )
    # Verify that the packets have been received in order (i.e., their
    # arrival times are monotonically increasing). Calculate the time
    # difference between subsequent packets and make sure that it is never
    # negative.
    assert (
        (fets[features.ARRIVAL_TIME_FET][1:] - fets[features.ARRIVAL_TIME_FET][:-1])
        >= 0
    ).all(), f"Packet arrival times are not monotonically increasing for flow: {flw}"
    # Transform absolute times into relative times from the start of the flow.
    fets[features.ARRIVAL_TIME_FET] -= start_time_us
    assert (fets[features.ARRIVAL_TIME_FET] >= 0).all(), "Negative arrival times!"

    if features.INTERARR_TIME_FET in fets.dtype.names:
        fets[features.INTERARR_TIME_FET][1:] = (
            fets[features.ARRIVAL_TIME_FET][1:] - fets[features.ARRIVAL_TIME_FET][:-1]
        )
        # Change any interarrival times of 0 to 1. This is technically incorrect. An
        # interarrival time of 0 us just means that the timestamping is not fine-grained
        # enough. It means that the instantaneous throughput is greater than ~12 Gbps.
        # But that's basically infinite in terms of our context of ~100 Mbps, so just
        # limiting it to 12 Gbps (by setting interarrival time to 1 us) seems fine.
        fets[features.INTERARR_TIME_FET][fets[features.INTERARR_TIME_FET] == 0] = 1
    if features.INV_INTERARR_TIME_FET in fets.dtype.names:
        fets[features.INV_INTERARR_TIME_FET][1:] = np.divide(
            8 * 1e6 * fets[features.WIRELEN_FET][1:],
            fets[features.INTERARR_TIME_FET][1:],
        )
    if features.RTT_RATIO_FET in fets.dtype.names:
        fets[features.RTT_RATIO_FET] = np.divide(fets[features.RTT_FET], min_rtt_us)
    if features.LOSS_RATE_FET in fets.dtype.names:
        fets[features.LOSS_RATE_FET] = fets[features.PACKETS_LOST_FET] / (
            fets[features.PACKETS_LOST_FET] + 1
        )
    if features.MIN_RTT_FET in fets.dtype.names:
        fets[features.MIN_RTT_FET] = min_rtt_us
    if features.SQRT_LOSS_RATE_FET in fets.dtype.names:
        np.reciprocal(
            np.sqrt(fets[features.LOSS_RATE_FET]), out=fets[features.SQRT_LOSS_RATE_FET]
        )

    # EWMA metrics.
    for (metric, _), alpha in itertools.product(features.EWMA_FETS, features.ALPHAS):
        metric = features.make_ewma_metric(metric, alpha)
        # If this is not a desired feature, then skip it.
        if metric not in fets.dtype.names:
            continue

        # For all metrics other than the Mathis model, fill in the 0th entry.
        if not metric.startswith(features.MATHIS_TPUT_LOSS_RATE_FET):
            fets[0][metric] = utils.safe_update_ewma(
                -1 if previous_fets is None else previous_fets[metric],
                fets[0][features.parse_ewma_metric(metric)[0]],
                alpha,
            )

        if metric.startswith(features.INTERARR_TIME_FET):
            # Improve cache locality and minimize branch checks by pushing this look
            # inwards.
            for j in range(1, num_pkts):
                fets[j][metric] = utils.safe_update_ewma(
                    fets[j - 1][metric], fets[j][features.INTERARR_TIME_FET], alpha
                )
        elif metric.startswith(features.INV_INTERARR_TIME_FET):
            for j in range(1, num_pkts):
                fets[j][metric] = utils.safe_update_ewma(
                    fets[j - 1][metric], fets[j][features.INV_INTERARR_TIME_FET], alpha
                )
        elif metric.startswith(features.RTT_FET):
            for j in range(1, num_pkts):
                fets[j][metric] = utils.safe_update_ewma(
                    fets[j - 1][metric], fets[j][features.RTT_FET], alpha
                )
        elif metric.startswith(features.RTT_RATIO_FET):
            for j in range(1, num_pkts):
                fets[j][metric] = utils.safe_update_ewma(
                    fets[j - 1][metric], fets[j][features.RTT_RATIO_FET], alpha
                )
        elif metric.startswith(features.LOSS_RATE_FET):
            for j in range(1, num_pkts):
                fets[j][metric] = utils.safe_update_ewma(
                    fets[j - 1][metric], fets[j][features.LOSS_RATE_FET], alpha
                )
        elif metric.startswith(features.SQRT_LOSS_RATE_FET):
            for j in range(1, num_pkts):
                new = fets[j][features.SQRT_LOSS_RATE_FET]
                fets[j][metric] = utils.safe_update_ewma(
                    fets[j - 1][metric], 1 if new in utils.UNSAFE else new, alpha
                )
        elif metric.startswith(features.MATHIS_TPUT_LOSS_RATE_FET):
            fets[0][metric] = utils.safe_update_ewma(
                -1 if previous_fets is None else previous_fets[metric],
                utils.safe_mathis_tput_bps(
                    fets[0][features.PAYLOAD_FET],
                    fets[0][features.RTT_FET],
                    fets[0][features.LOSS_RATE_FET],
                ),
                alpha,
            )
            for j in range(1, num_pkts):
                # tput = (MSS / RTT) * (C / sqrt(p))
                fets[j][metric] = utils.safe_update_ewma(
                    fets[j - 1][metric],
                    utils.safe_mathis_tput_bps(
                        fets[j][features.PAYLOAD_FET],
                        fets[j][features.RTT_FET],
                        fets[j][features.LOSS_RATE_FET],
                    ),
                    alpha,
                )
        else:
            raise Exception(f"Unknown EWMA metric: {metric}")

    # The index at which this window starts. Uninitialized for now.
    win_to_start_idx = {win: 0 for win in features.WINDOWS}

    # Windowed metrics.
    for j in range(win_metrics_start_idx, num_pkts):
        # Find the start indices for all windowed such that they end on the current
        # packet.
        win_to_start_idx = {
            win: utils.find_bound(
                fets[features.ARRIVAL_TIME_FET],
                target=fets[j][features.ARRIVAL_TIME_FET] - (win * min_rtt_us),
                min_idx=old_start_idx,
                max_idx=j - 1,
                which="after",
            )
            for win, old_start_idx in win_to_start_idx.items()
        }

        available_time_us = (
            fets[j][features.ARRIVAL_TIME_FET] - fets[0][features.ARRIVAL_TIME_FET]
        )

        for (metric, _), win in itertools.product(
            features.WINDOWED_FETS, features.WINDOWS
        ):
            metric = features.make_win_metric(metric, win)
            # If this is not a desired feature, then skip it.
            if metric not in fets.dtype.names:
                continue

            # Calculate windowed metrics only if an entire window has
            # elapsed since the start of the flow. Recall that the timestamps
            # have been adjusted to be relative to the start of the flow.
            win_size_us = win * min_rtt_us
            if available_time_us < win_size_us:
                logging.warning(
                    (
                        "Warning: Skipping windowed metric %s for packet %d"
                        "because we lack a full window (%d < %d)"
                    ),
                    metric,
                    j,
                    available_time_us,
                    win_size_us,
                )
                continue

            # A window requires at least two packets. Note that this means
            # that the first packet will always be skipped.
            win_start_idx = win_to_start_idx[win]
            assert win_start_idx < j, "Window does not contain at least two packtes."

            if metric.startswith(features.INTERARR_TIME_FET):
                new = (
                    fets[j][features.ARRIVAL_TIME_FET]
                    - fets[win_start_idx][features.ARRIVAL_TIME_FET]
                ) / (j - win_start_idx)
            elif metric.startswith(features.INV_INTERARR_TIME_FET):
                new = (
                    8
                    * 1e6
                    * fets[j][features.WIRELEN_FET]
                    / fets[j][features.make_win_metric(features.INTERARR_TIME_FET, win)]
                )
            elif metric.startswith(features.TPUT_FET):
                new = utils.safe_tput_bps(fets, win_start_idx, j)
            elif metric.startswith(features.RTT_FET):
                new = utils.safe_mean(fets[features.RTT_FET], win_start_idx, j)
            elif metric.startswith(features.RTT_RATIO_FET):
                new = utils.safe_mean(fets[features.RTT_RATIO_FET], win_start_idx, j)
            elif metric.startswith(features.LOSS_EVENT_RATE_FET):
                # Filled in already.
                continue
            elif metric.startswith(features.SQRT_LOSS_EVENT_RATE_FET):
                # 1 / sqrt(loss event rate).
                new = utils.safe_div(
                    1,
                    utils.safe_sqrt(
                        fets[j][
                            features.make_win_metric(features.LOSS_EVENT_RATE_FET, win)
                        ]
                    ),
                )
            elif metric.startswith(features.LOSS_RATE_FET):
                win_losses = utils.safe_sum(
                    fets[features.PACKETS_LOST_FET], win_start_idx + 1, j
                )
                new = utils.safe_div(win_losses, win_losses + (j - win_start_idx))
            elif metric.startswith(features.SQRT_LOSS_RATE_FET):
                new = utils.safe_div(
                    1,
                    utils.safe_sqrt(
                        fets[j][features.make_win_metric(features.LOSS_RATE_FET, win)]
                    ),
                )
            elif metric.startswith(features.MATHIS_TPUT_LOSS_RATE_FET):
                new = utils.safe_mathis_tput_bps(
                    fets[j][features.PAYLOAD_FET],
                    fets[j][features.RTT_FET],
                    fets[j][features.make_win_metric(features.LOSS_RATE_FET, win)],
                )
            elif metric.startswith(features.MATHIS_TPUT_LOSS_EVENT_RATE_FET):
                new = utils.safe_mathis_tput_bps(
                    fets[j][features.PAYLOAD_FET],
                    fets[j][features.RTT_FET],
                    fets[j][
                        features.make_win_metric(features.LOSS_EVENT_RATE_FET, win)
                    ],
                )
            else:
                raise Exception(f"Unknown windowed metric: {metric}")
            fets[j][metric] = new

    # Make sure that all fets rows were used.
    used_rows = np.sum(fets[features.ARRIVAL_TIME_FET] != -1)
    total_rows = fets.shape[0]
    assert (
        used_rows == total_rows
    ), f"Error: Used only {used_rows} of {total_rows} rows for flow {flw}"

    # Determine if there are any NaNs/Infs in the results. Just check the packets for
    # which we calculated windowed features as well.
    bad_fets = {
        fet
        for fet in fets.dtype.names
        if not np.isfinite(fets[win_metrics_start_idx:][fet]).any()
    }
    if bad_fets:
        logging.warning(
            "Warning: Flow %s has NaNs of Infs in features: %s", flw, bad_fets
        )

    # for fet in list(bad_fets) + [features.LOSS_RATE_FET]:
    #     if fet in fets.dtype.names:
    #         logging.info(
    #             "Bad fet: %s:\n\t%s",
    #             fet,
    #             "\n\t".join(str(x) for x in fets[win_metrics_start_idx:][fet]),
    #         )


def parse_exp(
    exp_flp,
    untar_dir,
    out_dir,
    skip_smoothed,
    receiver_ip,
    select_tail_percent,
    sender_fairness,
    always_reparse=False,
    parse_func=parse_opened_exp,
):
    """Lock, untar, and parse an experiment."""
    exp = utils.Exp(exp_flp)
    # Create output directory if it does not already exist.
    os.makedirs(out_dir, exist_ok=True)
    out_flp = path.join(out_dir, f"{exp.name}.npz")
    with open_exp(exp, exp_flp, untar_dir, out_dir, out_flp, always_reparse) as (
        locked,
        exp_dir,
    ):
        if locked and exp_dir is not None:
            try:
                return parse_func(
                    exp,
                    exp_flp,
                    exp_dir,
                    out_flp,
                    skip_smoothed,
                    receiver_ip,
                    select_tail_percent,
                    sender_fairness,
                )
            except AssertionError:
                traceback.print_exc()
                return -1
            except:
                traceback.print_exc()
                raise

    return -1


def _main():
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        description="Parses the output of CloudLab experiments."
    )
    psr.add_argument(
        "--exp-dir",
        help="The directory in which the experiment results are stored.",
        required=True,
        type=str,
    )
    psr.add_argument(
        "--untar-dir",
        help=(
            "The directory in which the untarred experiment intermediate "
            "files are stored (required)."
        ),
        required=True,
        type=str,
    )
    psr.add_argument(
        "--random-order",
        action="store_true",
        help="Parse experiments in a random order.",
    )
    psr.add_argument(
        "--skip-smoothed-features",
        action="store_true",
        help="Do not calculate EWMA and windowed features.",
    )
    psr.add_argument(
        "--parallel",
        default=multiprocessing.cpu_count(),
        help="The number of files to parse in parallel.",
        type=int,
    )
    psr.add_argument(
        "--num-exps", default=None, help="The number of experiments to parse.", type=int
    )
    psr.add_argument(
        "--select-tail-percent",
        help="The percentage (by time) of the tail of the PCAPs to select.",
        required=False,
        type=float,
    )
    psr.add_argument(
        "--receiver-ip",
        help=(
            "The IPv4 address of the receiver interface on which the "
            "PCAPs were captured."
        ),
        required=True,
        type=str,
    )
    psr, psr_verify = cl_args.add_out(psr)
    args = psr_verify(psr.parse_args())
    exp_dir = args.exp_dir
    untar_dir = args.untar_dir
    out_dir = args.out_dir
    skip_smoothed = args.skip_smoothed_features
    num_exps = args.num_exps

    # Find all experiments.
    pcaps = [
        (
            path.join(exp_dir, exp),
            untar_dir,
            out_dir,
            skip_smoothed,
            args.receiver_ip,
            args.select_tail_percent,
            False,
        )
        for exp in sorted(os.listdir(exp_dir))
        if exp.endswith(".tar.gz")
    ]
    if args.random_order:
        random.shuffle(pcaps)

    if num_exps is not None:
        assert num_exps > 0, f'"--num-exps" must be greater than 0, but is: {num_exps}'
        if len(pcaps) < num_exps:
            print(
                f"Warning: Only found {len(pcaps)} experiments, "
                f'but "--num-exps" is: {num_exps}'
            )
        num_exps = min(num_exps, len(pcaps))
        print(f"Selecting {num_exps}/{len(pcaps)} experiments.")
        pcaps = pcaps[:num_exps]

    print(f"Num files: {len(pcaps)}")
    tim_srt_s = time.time()
    if defaults.SYNC:
        smallest_safe_wins = {parse_exp(*pcap) for pcap in pcaps}
    else:
        with multiprocessing.Pool(processes=args.parallel) as pol:
            smallest_safe_wins = set(pol.starmap(parse_exp, pcaps))
    print(f"Done parsing - time: {time.time() - tim_srt_s:.2f} seconds")

    # Remove return values from experiments that were not parsed.
    smallest_safe_wins = [win for win in smallest_safe_wins if win != -1]
    if 0 in smallest_safe_wins:
        print("Some experiments had no safe window sizes.")
    print(
        "Smallest globally-safe window size:",
        max(smallest_safe_wins) if smallest_safe_wins else "No experiments parsed!",
    )


if __name__ == "__main__":
    _main()
