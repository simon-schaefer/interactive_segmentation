#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Deep Q learning hyperparameter plotting.
# =============================================================================
import argparse
import csv
import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="yumi_push_hypers")
    parser.add_argument("--out", type=str, default="",
                        help="outs directory that should be evaluated")
    parser.add_argument("--policy", type=str, default="",
                        help="policy directory that should be evaluated")
    args = parser.parse_args()

    # Load and check directory.
    directory, path = parse_path(args)
    if not os.path.isfile(os.path.join(path, "progress.csv")):
        raise IOError("No config found in {} !".format(path))
    # Read progress csv file and store as dictionary.
    hypers = pd.read_csv(os.path.join(path, "progress.csv"))
    print(hypers.head())

    # Plot parameters.
    plt.plot(hypers["steps"], hypers["% time spent exploring"])
    plt.grid(True)
    plt.title("exploration curve")
    plt.xlabel("STEPS"); plt.ylabel("EXPLORATION [%]")
    plt.savefig(os.path.join(path, "exploration_rate.png"))
    plt.close()

if __name__ == "__main__":
    main()
