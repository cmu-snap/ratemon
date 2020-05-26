#! /usr/bin/env python3

import argparse
import multiprocessing as mp
import os
from os import path
import re
import subprocess as sp
import time

import matplotlib.pyplot as plt
import numpy as np


ACK_PERIOD_MIN = 0
ACK_PERIOD_MAX = 15_000
ACK_PERIOD_DELTA = 1000

DUR = 5

PCAP = True

# Warning: If you move this file from the "unfair/model" directory, then you
#          must update this variable.
NS3_DIR = path.join(path.dirname(path.realpath(__file__)), "..", "ns-3-unfair")

OUT_DIR = "."


def run(delay):
    out = sp.check_output(
        args=[path.join(NS3_DIR, "build", "scratch", "ai"), f"--dur={DUR}",
              f"--ackPeriod={delay}", f"--out-dir={OUT_DIR}",
              "--pcap={}".format("true" if PCAP else "false")],
        stderr=sp.STDOUT, env=os.environ).decode().split("\n")
    # out = sp.check_output(["./waf", f"--run=ai --ackPeriod={delay}"],
    #                       stderr=sp.STDOUT, env=os.environ).decode().split("\n")
    # print(out)
    for line in out:
        if "Throughput" in line:
            match = re.search(r"Throughput: ([\d.]+) Mb/s", line)
            assert match is not None, f"Improperly formed output line: {line}"
            return float(match.group(1))
    assert False, "No output lines contain \"Throughput\""


def main():
    global OUT_DIR
    print(f"ns-3 dir: {NS3_DIR}")

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--out-dir", type=str, default=".")
    OUT_DIR = parser.parse_args().out_dir

    # Compile ns-3.
    sp.check_call([path.join(NS3_DIR, "waf")])
    # Since we are running the application binary directly, we need to make sure
    # the ns-3 library can be found.
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/lib/gcc/x86_64-linux-gnu/7:{}".format(
            os.path.join(NS3_DIR, "build")))

    # Generate the configurations and simulate them.
    delays = np.array(list(
        range(ACK_PERIOD_MIN, ACK_PERIOD_MAX + 1, ACK_PERIOD_DELTA)))
    print("Num simulations: {}".format(len(delays)))
    start_time = time.time()
    with mp.Pool() as pool:
        data = np.array(pool.map(run, delays))
    print("Done with simulation")
    print("Real simulation time (s): {:.2f}".format(time.time() - start_time))

    print(data)
    # Graph results.
    plt.plot(delays / 1000, data)
    plt.xlabel("Ack delay (ms)")
    plt.ylabel("Average throughput (Mb/s)")
    plt.ylim(bottom=0, top=10)
    plt.savefig(path.join(OUT_DIR, "plot.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
