#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Compare two reward graphs based on the logging file.
# =============================================================================
import argparse
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import parse_logging_file, progress_bar

def main():
    parser = argparse.ArgumentParser(description="yumi_push_qmap")
    parser.add_argument("--outs", type=str, default="",
                        help="list of outs directory that should be evaluated \
                        format is out1§out2§...")
    parser.add_argument("--descs", type=str, default="",
                        help="list of description (e.g. legend in plots), \
                        format is desc1§desc2§...")
    parser.add_argument("--max_iteration", type=int, default=-1)
    args = parser.parse_args()

    # Seperate out directories and descriptions from input arguments.
    outs  = args.outs.split("§")
    descs = args.descs.split("§")
    max_iter = 150000 if args.max_iteration < 0 else args.max_iteration
    assert len(outs) == len(descs)
    # Load and check logging files from out directories.
    paths = [os.path.join(os.environ["YUMI_PUSH_OUTS"], x) for x in outs]
    for x in paths:
        if not os.path.isdir(x): raise IOError("Directory %s not existing !" % x)
    loggings = [parse_logging_file(os.path.join(x, "logging.txt"),
                                   max_episode=max_iter) for x in paths]

    # Plot rewards in same plot.
    fig = plt.figure()
    plt.plot(loggings[0]["reward_mean_1000"], '-', label=descs[0])
    for log, desc in zip(loggings[1:], descs[1:]):
        plt.plot(log["reward_mean_1000"], '--', label=desc)
    plt.legend()
    plt.xlabel("EPISODES")
    plt.ylabel("REWARD_MEAN")
    plt.savefig("rewards.png")
    plt.close()

if __name__ == "__main__":
    main()
