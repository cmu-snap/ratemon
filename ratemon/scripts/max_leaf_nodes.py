#!/usr/bin/env python3

import argparse
import os
import sys
from os import path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the number of leaf nodes to use in a decision tree."
    )
    parser.add_argument(
        "--in-dir",
        help=(
            "Contains a directory for each number of leaf nodes. "
            'E.g., "100", "500", "200000", etc. '
            '"-1" Means no limit.'
        ),
        required=True,
        type=str,
    )
    args = parser.parse_args()
    assert path.isdir(args.in_dir)
    return args


def parse_results(max_leaf_nodes_dir):
    acc = parse_val_from_log(
        max_leaf_nodes_dir, lambda val: val.strip("%"), "Test accuracy:"
    )
    iters = parse_val_from_log(
        max_leaf_nodes_dir, lambda val: val, "Model fit in iterations:"
    )
    train_time_s = parse_val_from_log(
        max_leaf_nodes_dir,
        lambda val: val.strip(" seconds"),
        "Finished training - time:",
    )
    test_time_s = parse_val_from_log(
        max_leaf_nodes_dir,
        lambda val: val.strip(" seconds"),
        "Finished testing - time:",
    )

    models = [fln for fln in os.listdir(max_leaf_nodes_dir) if fln.startswith("model")]
    if len(models) != 1:
        return 1
    model_size_bytes = path.getsize(path.join(max_leaf_nodes_dir, models[0]))

    return acc, iters, train_time_s, test_time_s, model_size_bytes


def parse_val_from_log(max_leaf_nodes_dir, final_parsing, what):
    log_flp = path.join(max_leaf_nodes_dir, "output.log")
    with open(log_flp, "r", encoding="utf-8") as fil:
        lines = [line for line in fil if what in line]
        if not lines:
            raise RuntimeError(f'Error parsing "{what}" from: {log_flp}')
        val = final_parsing(lines[-1].split(what)[1].strip())
        try:
            val = float(val)
        except ValueError as exp:
            raise RuntimeError(f'Error parsing "{what}" from: {log_flp}') from exp
    return val


def plot_line_graph(x_data, y_data, x_label, y_label, title, filename):
    # sns.set(style="whitegrid")
    # fig, ax = plt.subplots()
    corrected_xs = [(max(x_data) * 10 if x == -1 else x) for x in x_data]
    plt.plot(corrected_xs, y_data, "o-")
    plt.xlabel(x_label)
    plt.xscale("log")
    plt.ylabel(y_label)
    plt.ylim(bottom=0, top=max(y_data) * 1.1)
    plt.xticks(
        corrected_xs,
        [
            ("No limit" if x == -1 else ("31 (default)" if x == 31 else x))
            for x in x_data
        ],
        rotation=45,
    )
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def graph(results, out_dir):
    xs, vals = zip(*sorted(results.items()))
    xs = list(xs)
    vals = list(vals)
    if xs[0] == -1:
        xs = xs[1:] + [-1]
        vals = vals[1:] + [vals[0]]
    accs, iterss, train_times, test_times, model_sizes = zip(*vals)

    plot_line_graph(
        xs,
        accs,
        "Max leaf nodes",
        "Accuracy",
        "Accuracy vs. max leaf nodes",
        path.join(out_dir, "accuracy.pdf"),
    )
    plot_line_graph(
        xs,
        iterss,
        "Max leaf nodes",
        "Iterations",
        "Iterations vs. max leaf nodes",
        path.join(out_dir, "iterations.pdf"),
    )
    plot_line_graph(
        xs,
        [x / 60 for x in train_times],
        "Max leaf nodes",
        "Training time (minutes)",
        "Training time vs. max leaf nodes",
        path.join(out_dir, "training_time.pdf"),
    )
    plot_line_graph(
        xs,
        [x / 60 for x in test_times],
        "Max leaf nodes",
        "Testing time (minutes)",
        "Testing time vs. max leaf nodes",
        path.join(out_dir, "testing_time.pdf"),
    )
    plot_line_graph(
        xs,
        [x / 1e9 for x in model_sizes],
        "Max leaf nodes",
        "Model size (GB)",
        "Model size vs. max leaf nodes",
        path.join(out_dir, "model_size.pdf"),
    )


def _main(args):
    results = {}
    for max_leaf_nodes in os.listdir(args.in_dir):
        try:
            max_leaf_nodes = int(max_leaf_nodes)
        except ValueError:
            continue
        results[max_leaf_nodes] = parse_results(
            path.join(args.in_dir, str(max_leaf_nodes))
        )
    graph(results, args.in_dir)


if __name__ == "__main__":
    sys.exit(_main(parse_args()))
