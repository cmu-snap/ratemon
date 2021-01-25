#! /usr/bin/env python3
""" General unit tests. """

import shutil
import subprocess
import shlex
import unittest
import os
from os import path
# Add the program directory to the system path, in case this script is
# being executed from a different directory using "python -m unittest ...".
import sys
sys.path.append(path.dirname(path.realpath(__file__)))


TEST_DATA_DIR = "model/test_data/"
LR_MODEL = (
    TEST_DATA_DIR +
    "1-3-False-100-2-False-5.0-linear-0.001-10-1-LrSklearn-0.09-False-0-2-l1" +
    "-False-None-False-False-9223372036854775807-0-9223372036854775807-0.1" +
    "-10-0.pickle")
SCALE_PARAM = TEST_DATA_DIR + "scale_params.json"
SIMULATIONS = TEST_DATA_DIR + "simulations/"
EXPERIMENTS = TEST_DATA_DIR + "experiments/"
PARSED_EXPERIMENTS = TEST_DATA_DIR + "parsed_experiments/"
UNTAR_DIR = TEST_DATA_DIR + "untar/"
TEST_OUTPUT_DIR = "model/test_output/"


class TestGeneral(unittest.TestCase):
    """ General unit tests. """

    def test_deps(self):
        """
        Tests that the Travis CI environment is configured with all necessary
        dependencies.

        Implicitly tests that all modules are free of syntax errors.
        """
        import check_mathis_accuracy
        import cl_args
        import correlation
        import defaults
        import fet_hists
        import gen_training_data
        import graph_one
        import hyper
        import models
        import parse_cloudlab
        import prepare_data
        import sim
        import test
        import train
        import training_param_sweep
        import utils

    def test_parsing(self):
        """
        Tests the 'parse_cloudlab' script, which parses cloudlab experiments
        and produces parsed npz file for training.
        """
        command_line_args = (
            "./model/parse_cloudlab.py "
            f"--exp-dir {EXPERIMENTS} "
            f"--untar-dir {UNTAR_DIR} "
            f"--out-dir {PARSED_EXPERIMENTS}")
        split_args = shlex.split(command_line_args)
        p = subprocess.Popen(split_args)
        p.wait()
        assert(p.returncode == 0)

        suffix = ".tar.gz"
        exp_name = [
            d for d in os.listdir(EXPERIMENTS) if d.endswith(suffix)][0][
            : -len(suffix)]

        # Check if output files are in parsed_experiments
        assert(os.path.exists(PARSED_EXPERIMENTS + f"{exp_name}.npz"))

    def test_training(self):
        command_line_args = (
            f"./model/train.py --data-dir {SIMULATIONS} "
            f"--model=LrSklearn --out-dir {TEST_OUTPUT_DIR} "
            "--num-sims=2 --max-iter=1 --keep-percent=5")

        split_args = shlex.split(command_line_args)
        p = subprocess.Popen(split_args)
        p.wait()
        assert(p.returncode == 0)

        # Check if output files are in test_output
        model_file = False
        scale_file = False
        for fname in os.listdir(TEST_OUTPUT_DIR):
            if fname.endswith('.pickle'):
                model_file = True
            if fname == "scale_params.json":
                scale_file = True
        assert(model_file and scale_file)

        # Remove files
        shutil.rmtree(TEST_OUTPUT_DIR)

    def test_evaluation(self):
        """
        Tests the 'test.py' script, which processes the simulation,
        evalute model performance, and produce various graphs.

        The test should also remove all the files generated from the script.
        """
        command_line_args = (f"./model/test.py --model {LR_MODEL} "
                             f"--scale-params {SCALE_PARAM} "
                             f"--standardize --simulation {SIMULATIONS} "
                             f"--out-dir {TEST_OUTPUT_DIR}")
        split_args = shlex.split(command_line_args)
        p = subprocess.Popen(split_args)
        p.wait()
        assert(p.returncode == 0)

        # Check if output files are in test_output
        assert(os.path.exists(TEST_OUTPUT_DIR + "results.txt"))
        assert(os.path.exists(TEST_OUTPUT_DIR + "queue_vs_accuracy.pdf"))
        assert(os.path.exists(TEST_OUTPUT_DIR + "rtt_vs_accuracy.pdf"))
        assert(os.path.exists(TEST_OUTPUT_DIR + "bandwidth_vs_accuracy.pdf"))
        # TODO(Ron): Add more file checks here when the other PR is merged in

        # Remove files
        shutil.rmtree(TEST_OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main()
