#! /usr/bin/env python3

import json, argparse
import matplotlib
import matplotlib.pyplot as plt 
from typing import Iterable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Data json file")
    parser.add_argument("--save", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    matplotlib.rcParams.update({'font.size': 16})

    def clamp(x, lower):
        return x if x >= lower else lower

    def adjust(x):
        if isinstance(x, Iterable):
            return (adjust(val) for val in x)
        else:
            return clamp(x-2, 0)

    # TODO: save these in the data file
    DUR_s = 300
    USE_RENO = ["true", "false"]
    WARMUP_s = [100, DUR_s]

    for reno in USE_RENO: 
        _, ax = plt.subplots()
        for warm in WARMUP_s:
            this = ((conf,v) for conf, v in data if conf['warmup_s'] == warm and conf['use_reno'] == reno)
            xs, ys = zip(*((conf['num_other_flows'], v) for conf, v in this))
            pacing = "Without" # ("With" if warm < DUR_s else "Without") + " ACK pacing"
            marker = "+"
            if warm < DUR_s:
                pacing = "With"
                marker = "o"
            ax.plot(list(adjust(xs)), list(ys), label=f"{pacing} ACK Pacing", marker=marker)
        flow = 'Reno' if reno == 'true' else 'Cubic'
        ax.set_xlabel(f"Number of {flow} Flows")
        ax.set_ylabel("Jain's Fairness Index")
        ax.set_ylim(0, 1.1)
        ax.legend()
        plt.tight_layout()
        if args.save:
            plt.savefig(f"{flow}.pdf".lower())

    if not args.save:
        plt.show()
