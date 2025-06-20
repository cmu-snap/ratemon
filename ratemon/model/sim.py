"""Runs simulations."""

import json
import logging
import logging.handlers
import multiprocessing
import os
import subprocess
import time
import traceback
from os import path

# Path to the ns-3 top-level directory.
# Warning: If you move this file from the directory "ratemon/model", then you
#          must update this variable.
NS3_DIR = path.join(path.dirname(path.realpath(__file__)), "..", "ns-3-unfair")
# The name of the ns-3 application to run.
APP = "dumbbell"
# Used to synchronize writing to the error log.
LOCK = multiprocessing.Lock()
# Name of the error log file.
ERR_FLN = "failed.json"
# Name of the logger for this module.
LOGGER = path.basename(__file__).split(".")[0]


def check_output(cnf, logger, msg):
    """Runs a configuration and returns its output."""
    args = [
        path.join(NS3_DIR, "build", "scratch", APP),
    ] + [f"--{arg}={val}" for arg, val in cnf.items()]
    cmd = f"LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']} {' '.join(args)}"
    log = logging.getLogger(logger)
    log.info("Running%s: %s", f" ({msg})" if msg is not None else "", cmd)
    try:
        res = (
            subprocess.check_output(args, stderr=subprocess.STDOUT, env=os.environ)
            .decode()
            .split("\n")
        )
    except subprocess.CalledProcessError:
        traceback.print_exc()
        log.exception("Exception while running:\n%s\n\n", cmd)
        # Only one worker should access the error log at a time.
        LOCK.acquire()
        err_flp = path.join(cnf["out_dir"], ERR_FLN)
        # If the error log already exists, then read the existing log and append
        # to it. Otherwise, start a new log.
        if path.exists(err_flp):
            with open(err_flp, "r") as fil:
                err_log = json.load(fil)
        else:
            err_log = []
        # Record the full command (including LD_LIBRARY_PATH) in the error log
        # for ease of debugging. Do not do "cnf['cmd'] = ..." to maintain the
        # invariant that cnf contains only command line arguments.
        err_log.append(dict(cnf, cmd=cmd))
        with open(err_flp, "w") as fil:
            json.dump(err_log, fil, indent=4)
        LOCK.release()
        # The output is empty.
        res = []
    return res


def run(cnf, res_fnc, logger, idx, total):
    """
    Runs a configuration. If res_fnc is not None, then returns the result of
    parsing the configuration's output using res_fnc, otherwise returns None.
    """
    # Build the arguments array, run the simulation, and iterate over each line
    # in its output.
    out = check_output(cnf, logger, msg=f"{idx + 1:{f'0{len(str(total))}'}}/{total}")
    if res_fnc is None:
        return None
    return res_fnc(out)


def sim(
    eid,
    cnfs,
    out_dir,
    res_fnc=None,
    log_par=None,
    log_dst=None,
    dry_run=False,
    sync=False,
):
    """
    Simulates a set of configurations. Returns a list of pairs of the form:
        (configuration, result)
    """
    # Set up logging.
    logger = LOGGER if log_par is None else f"{log_par}.{LOGGER}"
    log = logging.getLogger(logger)
    if log_dst is not None:
        # Create email logger.
        # TODO: Create an email logger only if an SMTP server is running on the
        #       local machine.
        hdl = logging.handlers.SMTPHandler(
            mailhost="localhost",
            fromaddr=f"{os.getlogin()}@maas.cmcl.cs.cmu.edu",
            toaddrs=log_dst,
            subject=f"[{logger}] {eid}",
        )
        hdl.setLevel("ERROR")
        log.addHandler(hdl)

    # Compile ns-3.
    log.info("ns-3 dir: %s", NS3_DIR)
    subprocess.check_call(["./waf"], cwd=NS3_DIR)
    # Since we are running the application binary directly, we need to make sure
    # the ns-3 library can be found.
    os.environ["LD_LIBRARY_PATH"] = (
        f"/usr/lib/gcc/x86_64-linux-gnu/7:{path.join(NS3_DIR, 'build', 'lib')}:"
        "/opt/libtorch/lib"
    )

    # Record the configurations.
    with open(path.join(out_dir, "configurations.json"), "w") as fil:
        json.dump(cnfs, fil, indent=4)

    # Simulate the configurations.
    num_cnfs = len(cnfs)
    log.info("Num simulations: %s", num_cnfs)
    if dry_run:
        return []
    tim_srt_s = time.time()
    if sync:
        data = [
            run(cnf, res_fnc, logger, idx, num_cnfs) for idx, cnf in enumerate(cnfs)
        ]
    else:
        with multiprocessing.Pool() as pool:
            data = pool.starmap(
                run,
                ((cnf, res_fnc, logger, idx, num_cnfs) for idx, cnf in enumerate(cnfs)),
            )
    log.critical("Done with simulations - time: %.2f seconds", time.time() - tim_srt_s)
    return list(zip(cnfs, data))
