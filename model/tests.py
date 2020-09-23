#! /usr/bin/env python3
""" General unit tests. """

from os import path
# Add the program directory to the system path, in case this script is
# being executed from a different directory using "python -m unittest ...".
import sys
sys.path.append(path.dirname(path.realpath(__file__)))

import unittest


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


if __name__ == "__main__":
    unittest.main()
