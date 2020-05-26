"""Runs simulations. """

import json
import logging
import logging.handlers
import multiprocessing
import os
from os import path
import subprocess
import time
import traceback


# Path to the ns-3 top-level directory.
# Warning: If you move this file from the directory "unfair/model", then you
#          must update this variable.
NS3_DIR = path.join(path.dirname(path.realpath(__file__)), "..", "ns-3-unfair")
# Used to synchronize writing to the error log.
LOCK = multiprocessing.Lock()
# Name of the error log file.
ERR_FLN = "failed.json"
# Name of the logger for this module.
LOGGER = path.basename(__file__).split(".")[0]

def check_output(cnf, logger):
    args = ([path.join(NS3_DIR, "build", "scratch", "ai-multi-flow"),] +
            [f"--{arg}={val}" for arg, val in cnf.items()])
    cmd = f"LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']} {' '.join(args)}"
    log = logging.getLogger(logger)
    log.info(f"Running: {cmd}")
    try:
        res = subprocess.check_output(
            args, stderr=subprocess.STDOUT, env=os.environ).decode().split("\n")
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


def run(cnf, res_fnc, logger):
    # Build the arguments array, run the simulation, and iterate over each line
    # in its output.
    out = check_output(cnf, logger)
    if res_fnc is None:
        return None
    return res_fnc(out)


def sim(eid, cnfs, out_dir, res_fnc=None, log_par=None, log_dst=None, dry_run=False, sync=False):
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
            subject=f"[{logger}] {eid}")
        hdl.setLevel("ERROR")
        log.addHandler(hdl)

    # Compile ns-3.
    log.info(f"ns-3 dir: {NS3_DIR}")
    subprocess.check_call(["./waf"], cwd=NS3_DIR)
    # Since we are running the application binary directly, we need to make sure
    # the ns-3 library can be found.
    os.environ["LD_LIBRARY_PATH"] = (
        f"/usr/lib/gcc/x86_64-linux-gnu/7:{path.join(NS3_DIR, 'build', 'lib')}:"
        "/opt/libtorch/lib")

    # Record the configurations.
    with open(path.join(out_dir, "configurations.json"), "w") as fil:
        json.dump(cnfs, fil, indent=4)

    # Simulate the configurations.
    log.info(f"Num simulations: {len(cnfs)}")
    if dry_run:
        return []
    tim_srt_s = time.time()
    if sync:
        data = [run(cnf, res_fnc, logger) for cnf in cnfs]
    else:
        with multiprocessing.Pool() as pool:
            data = pool.starmap(run, ((cnf, res_fnc, logger) for cnf in cnfs))
    log.critical(f"Done with simulations - time: {time.time() - tim_srt_s:.2f} seconds")
    return list(zip(cnfs, data))
