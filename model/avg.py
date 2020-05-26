#! /usr/bin/env python3

import json, argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Data json file")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    DUR_s = 300
    pacing = [y for x, y in data if x['warmup_s'] < DUR_s]
    print(f"With pacing: {sum(pacing)/len(pacing)}")

    no_pacing = [y for x, y in data if x['warmup_s'] >= DUR_s]
    print(f"Without pacing: {sum(no_pacing)/len(no_pacing)}")



