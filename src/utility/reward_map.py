#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Reward mapping.
# =============================================================================
import argparse
import os
import numpy as np

from yumi_push.common.checkpoint import Checkpoint
from yumi_push.common.io_utils import load_yaml
from yumi_push.common.openai_utils import deepq
from yumi_push.tasks import init_task
from utils import *

def main():
    parser = argparse.ArgumentParser(description="yumi_push_reward_map")
    parser.add_argument("--out", type=str, default="",
                        help="outs directory that should be evaluated")
    parser.add_argument("--policy", type=str, default="",
                        help="policy directory that should be evaluated")
    parser.add_argument("--config", type=str, default="",
                        help="config file that should be evaluated")
    args = parser.parse_args()
    # Load and check directory.
    if (int(args.policy == "")+int(args.out == "")+int(args.config == ""))!=2:
        raise IOError("Usage: python3 reward_map.py --policy=... or --out=... or --config=...")
    if args.policy != "":
        tag = args.policy
        file = os.path.join(args.policy, "config.txt")
        file = os.path.join(os.environ["YUMI_PUSH_POLICIES"], file)
    elif args.out != "":
        tag = args.out
        file = os.path.join(args.out, "config.txt")
        file = os.path.join(os.environ["YUMI_PUSH_OUTS"], file)
    else:
        tag = args.config.replace(".yaml", "")
        file = os.path.join(os.environ["YUMI_PUSH_CONFIG"], args.config)
    if not os.path.isfile(file):
        raise IOError("File {} not existing !".format(file))
    # Load configuration and create output file (checkpoint).
    config = load_yaml(file)
    config["visualize"] = False
    config["verbose"] = True
    config["debug"] = False
    config["scene"] = "cubes_fixed"
    config["rew_schedule"] = False
    config["train_algo"] = ""
    cam_config = load_yaml(os.environ["YUMI_PUSH_CONFIG"]+"/"+config['camera_config'])
    ckp = Checkpoint(config, tag=tag+"_reward_map")
    # Task definition.
    task, sensor, actuator = init_task(config, cam_config, ckp)
    # Get number of pushing actions (not implemented for all actuators, so will
    # throw error for "too continuous" actuators).
    directions = actuator.directions()
    n_directions = len(directions)
    # Iterate over all actions and determine reward
    # (assume start_point are 1&2 action and 3 pushing direction).
    print("Start creating reward map ...")
    rewards = None
    # For discrete action space the actions simply are a integer number.
    if config["act_discrete"]:
        n_actions = task.action_space.n
        n_action_steps = int(np.sqrt(task.action_space.n/n_directions))
        rewards = np.ones((n_action_steps,n_action_steps,n_directions))*(-1000)
        for x in range(n_actions):
            task.reset()
            action = actuator.undiscretize(np.asarray(x, dtype=int))
            _, reward, done, _ = task.step(np.asarray(x, dtype=int))
            rewards[action[0],action[1],action[2]] = reward
            progress_bar(x, n_actions)
    # For continuous action space discretize the actions (assumes action to be
    # [starting point, pushing_direction]).
    else:
        res_cont = 0.1 #[m]
        n_action_steps = int(2.0/res_cont)
        n_actions = n_directions*(n_action_steps**2)
        rewards = np.ones((n_action_steps,n_action_steps,n_directions))*(-1000)
        i = 0
        for ix in range(n_action_steps):
            for iy in range(n_action_steps):
                for id in range(n_directions):
                    x,y = -1.0+ix*res_cont,-1.0+iy*res_cont,
                    d   = -0.9+id*(2.0/n_directions)
                    task.reset()
                    _, reward, done, _ = task.step(np.asarray([x,y,d]))
                    rewards[ix,iy,id] = reward
                    i = i + 1
                    progress_bar(i, n_actions)
    # Plotting output graphs.
    _, segmented, _ = sensor.get_state()
    draw_graphs(rewards, segmented, directions, ckp, name="rewards")
    # Clearning up directory.
    os.remove(os.path.join(ckp.dir, "log_num_steps.pdf"))
    os.remove(os.path.join(ckp.dir, "log_reward.pdf"))
    os.remove(os.path.join(ckp.dir, "log_success_rate.pdf"))
    os.remove(os.path.join(ckp.dir, "logging.txt"))
    print("... finished creating reward map.")

if __name__ == "__main__":
    main()
