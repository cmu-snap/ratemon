#! /usr/bin/env python3
""" General unit tests. """

from os import path
# Add the program directory to the system path, in case this script is
# being executed from a different directory using "python -m unittest ...".
import sys
sys.path.append(path.dirname(path.realpath(__file__)))

import os
import unittest
import shlex
import subprocess
import shutil

TEST_DATA_DIR = "test_data/"
LR_MODEL = (TEST_DATA_DIR + "1-3-False-100-2-False-5.0-linear-0.001-10-20"
                            "-LrSklearn-0.09-False-0-1000-l1-False-None-True"
                            "-False-9223372036854775807-0-9223372036854775807"
                            "-0.1-10-0.pickle")
SCALE_PARAM = TEST_DATA_DIR + "scale_params.json"
SIMULATIONS = TEST_DATA_DIR + "simulations"
TEST_OUTPUT_DIR = "test_output/"

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
        import parse_dumbbell
        import prepare_data
        import sim
        import test
        import train
        import training_param_sweep
        import utils

    def test_evaluation(self):
        """
        Tests the 'test.py' script, which processes the simulation, 
        evalute model performance, and produce various graphs.

        The test should also remove all the files generated from the script.
        """
        command_line_args = (f"./test.py --model {LR_MODEL} --scale-params " 
                             f"{SCALE_PARAM} --standardize --simulation "
                             f"{SIMULATIONS} --out-dir {TEST_OUTPUT_DIR}")
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
