#!/usr/bin/env python3
"""
Parses bpftrace.log files generated by cctestbedv2.py's receiver metrics logging.
"""

import argparse
import sys

import numpy as np

PERCENTILES = [10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9, 99.99, 100.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate statistics about SKB length"
    )
    parser.add_argument("--in-file", type=str, help="Input file.", required=True)
    parser.add_argument("--out-file", type=str, help="Output file.", required=True)
    return parser.parse_args()


def main(args):
    lens = []
    with open(args.in_file, "r", encoding="utf-8") as fil:
        for line in fil:
            if "len," in line:
                lens.append(int(line.strip().split(",")[2]))
    print(f"Found {len(lens)} SKB lengths.")
    lens = np.asarray(lens)
    metrics = dict(zip(PERCENTILES, np.percentile(lens, PERCENTILES)))
    metrics["avg"] = np.mean(lens)
    msg = "Percentiles:\n\t" + "\n\t".join(
        f"{a}: {b:.2f}"
        for a, b in sorted(
            metrics.items(), key=(lambda x: x[0] if isinstance(x[0], float) else -1)
        )
    )
    print(msg)
    with open(args.out_file, "w", encoding="utf-8") as fil:
        fil.write(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
