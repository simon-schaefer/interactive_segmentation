#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Testing multiple step episodes.
# =============================================================================
import argparse
import os
import numpy as np

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from yumi_push.networks.networks import *
from yumi_push.simulation import camera as sim_camera
from yumi_push.simulation import yumi as sim_robot
from yumi_push.simulation import world as sim_world
from yumi_push.tasks import sensors as task_sensors
from yumi_push.tasks import actuators as task_actuators
from yumi_push.tasks import task as task_task
from yumi_push.tasks import rewards as task_rewards

from yumi_push.common.checkpoint import Checkpoint
from yumi_push.common.io_utils import load_yaml
from yumi_push.common.openai_utils import *
from yumi_push.tasks import task
from yumi_push.tasks.misc import Status

# Redefine the task class so that the scene is not reseted, even if it
# is solved properly, by not resetting the scenario in reset and hardcoding
# the state as running.
class NewTask(task_task.Task):

    def __init__(self, sensor, actuator, reward_fn, environment,
                 config, ckp):
        super(NewTask, self).__init__(
            sensor, actuator, reward_fn, environment, config, ckp
        )

    def step(self, action):
        obs, reward, done, flags = super(NewTask, self).step(action)
        if self.status == Status.SUCCESS:
            self._ckp.can_reset = True
            self.reset()
        done = False
        return obs, reward, done, flags

def new_init_task(config, cam_config, ckp):
    world = sim_world.World(config, ckp)
    camera = sim_camera.RGBDCamera(world.physics_client, cam_config)
    robot  = sim_robot.Robot2DHand(config, ckp, world)
    actuator = task_actuators.actuator_factory(config, ckp, robot)
    sensor = task_sensors.Sensor(config, ckp, camera=camera)
    reward = task_rewards.RewardFn(config,ckp,camera)
    task = NewTask(sensor,actuator,reward,world,config,ckp)
    return task, sensor, actuator

# Get configuration from directory in out or policy folder, set config
# "must" parameters, initialize task and let it run.
def main():
    parser = argparse.ArgumentParser(description="yumi_push_qmap")
    parser.add_argument("--out", type=str, default="",
                        help="outs directory that should be evaluated")
    parser.add_argument("--policy", type=str, default="",
                        help="policy directory that should be evaluated")
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
    config["visualize"] = config["verbose"] = True
    config["debug"] = False
    config["rew_schedule"] = False
    config["working_dir"] = path
    cam_config = load_yaml(os.environ["YUMI_PUSH_CONFIG"]+"/"+config['camera_config'])
    ckp = Checkpoint(config, tag=directory+"_multistep")

    # Task definition.
    task, sensor, actuator = new_init_task(config, cam_config, ckp)
    learn_algo = config.get("train_algo", "ppo2")
    if learn_algo == "ppo2": ppo2(task, config, eval=True)
    elif learn_algo == "deepq": deepq(task, config, eval=True)
    else: raise ValueError("Invalid training algorithm {}!".format(learn_algo))
    task.close()



if __name__ == "__main__":
    main()
