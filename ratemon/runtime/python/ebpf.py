import logging
import os
import time
from os import path

import numpy as np
from bcc import BPF, BPFAttachType, BPFProgType
from pyroute2 import IPRoute, protocols
from pyroute2.netlink.exceptions import NetlinkError


def load_ebpf():
    """Load the corresponding eBPF program."""
    # Load BPF text.
    bpf_flp = path.join(
        path.abspath(path.dirname(__file__)),
        "ratemon_runtime.c",
    )
    if not path.isfile(bpf_flp):
        logging.error("Could not find BPF program: %s", bpf_flp)
        return 1
    logging.info("Loading BPF program: %s", bpf_flp)
    with open(bpf_flp, "r", encoding="utf-8") as fil:
        bpf_text = fil.read()
    # Load BPF program.
    return BPF(text=bpf_text)


def configure_ebpf(args):
    """Set up eBPF hooks."""
    if min(args.listen_ports) >= 50000:
        # Use the listen ports to determine the wait time, so that multiple
        # instances of this program do not try to configure themselves at the same
        # time.
        rand_sleep = min(args.listen_ports) - 50000
        logging.info(
            "Waiting %f seconds to prevent race conditions...", rand_sleep)
        time.sleep(rand_sleep)

    try:
        bpf = load_ebpf()
    except:
        logging.exception("Error loading BPF program!")
        return None, None
    flow_to_rwnd = bpf["flow_to_rwnd"]

    # Set up a TC egress qdisc, specify a filter the accepts all packets, and attach
    # our egress function as the action on that filter.
    ipr = IPRoute()
    ifindex = ipr.link_lookup(ifname=args.interface)
    assert (
        len(ifindex) == 1
    ), f'Trouble looking up index for interface "{args.interface}": {ifindex}'
    ifindex = ifindex[0]

    logging.info("Attempting to create central qdisc")
    handle = 0x10000
    default = 0x200000
    responsible_for_central_tc = False
    try:
        ipr.tc("add", "htb", ifindex, handle, default=default)
    except NetlinkError:
        logging.warning(
            "Unable to create central qdisc. It probably already exists.")
    else:
        logging.info("Responsible for central TC")
        responsible_for_central_tc = True

    if not responsible_for_central_tc:
        # If someone else is responsible for the egress action, then we will just let
        # them do the work.
        logging.warning("Not configuring TC")
        return flow_to_rwnd, None

    # Read the TCP window scale on outgoing SYN-ACK packets.
    func_sock_ops = bpf.load_func("read_win_scale", bpf.SOCK_OPS)  # sock_stuff
    filedesc = os.open(args.cgroup, os.O_RDONLY)
    bpf.attach_func(func_sock_ops, filedesc, BPFAttachType.CGROUP_SOCK_OPS)

    # Overwrite advertised window size in outgoing packets.
    egress_fn = bpf.load_func("do_rwnd_at_egress", BPF.SCHED_ACT)
    action = dict(kind="bpf", fd=egress_fn.fd,
                  name=egress_fn.name, action="ok")

    # int prog_fd;

    prod_fd = bpf.load_func("bpf_iter__task", BPFProgtype.TRACING)
    int link_fd = bcc_iter_attach(prog_fd, NULL, 0)
    if (link_fd < 0) {
        std: : cerr << "bcc_iter_attach failed: " << link_fd << std: : endl
        return 1
    }

    int iter_fd = bcc_iter_create(link_fd)
    if (iter_fd < 0) {
        std: : cerr << "bcc_iter_create failed: " << iter_fd << std: : endl
        close(link_fd)
        return 1
    }

    try:
        # Add the action to a u32 match-all filter
        ipr.tc(
            "add-filter",
            "u32",
            ifindex,
            parent=handle,
            prio=10,
            protocol=protocols.ETH_P_ALL,  # Every packet
            target=0x10020,
            keys=["0x0/0x0+0"],
            action=action,
        )
    except:
        logging.exception("Error: Unable to configure TC.")
        return None, None

    def ebpf_cleanup():
        """Clean attached eBPF programs."""
        logging.info("Detaching sock_ops hook...")
        bpf.detach_func(func_sock_ops, filedesc, BPFAttachType.CGROUP_SOCK_OPS)
        logging.info("Removing egress TC...")
        ipr.tc("del", "htb", ifindex, handle, default=default)

    logging.info("Configured TC and BPF!")
    return flow_to_rwnd, ebpf_cleanup
