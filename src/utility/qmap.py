#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Q values mapping.
# =============================================================================
import argparse
import os
import numpy as np

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from yumi_push.common.checkpoint import Checkpoint
from yumi_push.common.io_utils import load_yaml
from yumi_push.common.openai_utils import deepq
from yumi_push.tasks import init_task
from utils import *

def main():
    parser = argparse.ArgumentParser(description="yumi_push_qmap")
    parser.add_argument("--out", type=str, default="",
                        help="outs directory that should be evaluated")
    parser.add_argument("--policy", type=str, default="",
                        help="policy directory that should be evaluated")
    args = parser.parse_args()

    # Load and check directory.
    directory, path = parse_path(args)
    if not os.path.isdir(os.path.join(path, "checkpoints")):
        raise IOError("No model to load in {} !".format(path))
    if not os.path.isfile(os.path.join(path, "config.txt")):
        raise IOError("No config found in {} !".format(path))
    config = load_yaml(os.path.join(path, "config.txt"))
    config["visualize"] = False
    config["verbose"] = True
    config["debug"] = False
    config["scene"] = "cubes_fixed"
    config["rew_schedule"] = False
    config["train_steps"] = config["eval_steps"] = 1
    config["working_dir"] = path
    cam_config = load_yaml(os.environ["YUMI_PUSH_CONFIG"]+"/"+config['camera_config'])
    ckp = Checkpoint(config, tag=directory+"_qmap")

    # Task definition.
    task, sensor, actuator = init_task(config, cam_config, ckp)

    # Get Q values from agent.
    act = deepq(task, config, eval=True)
    obs, Qvalues, states = task.reset(), [], []
    for _ in range(config["task_max_trials_eps"]):
        sample_probs = task.get_action_probabilities()
        act_out = act(obs,sample_probs)
        Qvalues.append(act_out[1][0])
        _, segmented, _ = sensor.get_state()
        states.append(segmented)
        obs = task.step(act_out[0][0])[0]

    print("Start creating Q-maps ...")
    # Plot Q map by iterating over the whole array of actions, determine
    # the pixel by "unnormalizing" and assign the Q value.
    for iq, q_values in enumerate(Qvalues):
        n_actions    = task.action_space.n
        directions   = actuator.directions()
        n_directions = len(directions)
        n_steps      = int(np.sqrt(task.action_space.n/n_directions))
        q_map = np.ones((n_steps, n_steps, n_directions))*(-1000)
        for x in range(n_actions):
            action = actuator.undiscretize(np.asarray(x, dtype=int))
            q_map[action[0],action[1],action[2]] = q_values[x]
            progress_bar(x, n_actions)
        # Plotting output graphs.
        draw_graphs(q_map,states[iq],directions,ckp,name="{}_qvalues".format(iq))

    # Clearning up directory.
    os.remove(os.path.join(ckp.dir, "logging.txt"))
    os.remove(os.path.join(ckp.dir, "log.txt"))
    os.remove(os.path.join(ckp.dir, "progress.csv"))
    print("... finished creating Q-maps.")

if __name__ == "__main__":
    main()
