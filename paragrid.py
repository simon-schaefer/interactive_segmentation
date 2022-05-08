#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Parameter grid search for training.
# =============================================================================
import itertools
from multiprocessing import Pool
import os

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from yumi_push.networks.networks import *
from yumi_push.common.checkpoint import Checkpoint
from yumi_push.common.io_utils import load_yaml
from yumi_push.common.openai_utils import *
from yumi_push.simulation import camera as sim_camera
from yumi_push.simulation import yumi as sim_robot
from yumi_push.simulation import world as sim_world
from yumi_push.tasks import sensors as task_sensors
from yumi_push.tasks import actuators as task_actuators
from yumi_push.tasks import task as task_task
from yumi_push.tasks import rewards as task_rewards

# Build a list of configuration parameters creating the parameter search
# grid space. The format is "param_name#value#type" (bool=[0,1]!).
grid = [
    ["train_algo#ppo2#str"],
    ["act_discrete#0#bool"],
    ["scene#cubes_smart#str"],
    ["sensor_type#distances#str"],
    ["train_minibatches_update#256#int"],
    ["rew_act_start_dis_succeed#1#bool"],
    ["model_network#fc_small#str", "model_network#fc_large#str"],
]
grid = list(itertools.product(*grid))

def train(paramset):
    config = load_yaml(os.environ["YUMI_PUSH_CONFIG"]+"/simulation.yaml")
    cam_config = load_yaml(os.environ["YUMI_PUSH_CONFIG"]+"/"+config['camera_config'])
    config["visualize"] = config["debug"] = config["verbose"] = False
    config["train_steps"] = 180000
    tag = "!"
    for param in paramset:
        assert len(param.split("#")) == 3
        p,v,t = param.split("#")
        if t == "str": config[p] = v
        elif t == "int": config[p] = int(v)
        elif t == "float": config[p] = float(v)
        elif t == "bool": config[p] = bool(int(v))
        else: raise ValueError("unknown type")
        tag = tag + param + "!"
    ckp = Checkpoint(config, tag=tag)
    config["working_dir"] = ckp.get_path("checkpoints/")
    # Execute training in given configuration.
    world = sim_world.World(config, ckp)
    camera = sim_camera.RGBDCamera(world.physics_client, cam_config)
    robot  = sim_robot.Robot2DHand(config, ckp, world)
    actuator = task_actuators.actuator_factory(config, ckp, robot)
    sensor = task_sensors.Sensor(config, ckp, camera=camera)
    reward = task_rewards.RewardFn(config,ckp,camera)
    task = task_task.Task(sensor,actuator,reward,world,config,ckp)
    learn_algo = config.get("train_algo", "ppo2")
    if learn_algo == "ppo2": ppo2(task, config)
    elif learn_algo == "deepq": deepq(task, config)
    else: raise ValueError("Invalid training algorithm {}!".format(learn_algo))
    task.close()

def main():
    pool = Pool(processes=len(grid))
    pool.map(train, grid)

if __name__ == "__main__":
    main()
