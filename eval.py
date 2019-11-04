#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Model evaluation (simulation).
# =============================================================================
import argparse
import os

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yumi_push.common.checkpoint import Checkpoint
from yumi_push.common.io_utils import load_yaml
from yumi_push.common.openai_utils import *
from yumi_push.tasks import init_task

def main():
    parser = argparse.ArgumentParser(description="yumi_push_eval")
    parser.add_argument("--policy", type=str, default="",
                        help="policies directory that should be evaluated")
    parser.add_argument("--out", type=str, default="",
                        help="outs directory that should be evaluated")
    parser.add_argument("--visualize", action="store_true",
                        help="enable visualization (default=False)")
    parser.add_argument("--debug", action="store_true",
                        help="enable debug mode (default=False)")
    parser.add_argument("--activations", action="store_true",
                        help="enable cnn layer visualization (default=False)")
    parser.add_argument("--eval_steps", type=int, default=10,
                        help="number of evaluation steps")
    args = parser.parse_args()
    # Load and check directory.
    if not ((args.policy == "" and args.out != "") \
    or (args.policy != "" and args.out == "")):
        raise IOError("Usage: python3 eval.py --policy=... or --out=...")
    directory = args.policy if args.policy != "" else args.out
    parent_dir = os.environ["YUMI_PUSH_POLICIES"] if args.policy != "" \
                 else os.environ["YUMI_PUSH_OUTS"]
    path = os.path.join(parent_dir, directory)
    if not os.path.isdir(path):
        raise IOError("Directory {} not existing !".format(path))
    if not os.path.isdir(os.path.join(path, "checkpoints")):
        raise IOError("No model to load in {} !".format(path))
    if not os.path.isfile(os.path.join(path, "config.txt")):
        raise IOError("No config found in {} !".format(path))
    config = load_yaml(os.path.join(path, "config.txt"))
    config["visualize"]  = args.visualize
    config["debug"]      = args.debug
    config["working_dir"] = path
    config["eval_steps"] = config["seg_rew_list_len"] = args.eval_steps
    cam_config = load_yaml(os.environ["YUMI_PUSH_CONFIG"]+"/"+config['camera_config'])
    ckp = Checkpoint(config, tag=directory+"_eval")

    # Load task and components.
    task, _, _ = init_task(config, cam_config, ckp)
    learn_algo = config.get("train_algo", "ppo2")
    if learn_algo == "ppo2": ppo2(task, config, eval=True)
    elif learn_algo == "deepq": deepq(task, config, eval=True)
    else: raise ValueError("Invalid training algorithm {}!".format(learn_algo))
    task.close()

if __name__ == "__main__":
    main()
