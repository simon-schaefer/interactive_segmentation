#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Convolution network visualization.
# =============================================================================
import argparse
import os
import sys
import numpy as np
import cv2
import math

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def plotNNFilter(units,image):
    matplotlib.use('TkAgg')
    filters = units.shape[3]
    image = np.reshape(image,[64,64])
    n_columns = 4
    n_rows = math.ceil(filters / n_columns) + 1
    plt.figure(1, figsize=(12,12))
    plt.subplot(n_rows+1, n_columns*2, 1)
    plt.title('Input Image')
    plt.imshow(image,cmap="gray")
    plt.axis('off')
    for i in range(filters):
        plt.subplot(n_rows, n_columns*2, i*2+1+n_columns*2)
        plt.axis('on')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.show()

def getActivations(session,layer,placeholder,image):
    #x_feed = tf.placeholder(tf.uint8, [1,64,64,1],name='ppo2_model/Ob')
    feed_data = np.reshape(image,[1,64,64,1])
    feed_data = feed_data.astype(np.float32)
    units = session.run(layer,feed_dict={placeholder:feed_data})
    plotNNFilter(units,image)

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
    args = parser.parse_args()
    # Load and check directory.
    directory, path = parse_path(args)
    if not os.path.isdir(os.path.join(path, "checkpoints")):
        raise IOError("No model to load in {} !".format(path))
    if not os.path.isfile(os.path.join(path, "config.txt")):
        raise IOError("No config found in {} !".format(path))
    config = load_yaml(os.path.join(path, "config.txt"))
    config["visualize"]  = args.visualize
    config["debug"]      = args.debug
    config["working_dir"] = path
    cam_config = load_yaml(os.environ["YUMI_PUSH_CONFIG"]+"/"+config['camera_config'])
    ckp = Checkpoint(config, tag=directory+"_eval")

    # Load task and components.
    world = sim_world.World(config, ckp)
    camera = sim_camera.RGBDCamera(world.physics_client, cam_config)
    robot  = sim_robot.Robot2DHand(config, ckp, world)
    actuator = task_actuators.actuator_factory(config, ckp, robot)
    sensor = task_sensors.Sensor(config, ckp, camera=camera)
    reward = task_rewards.RewardFn(config,ckp,camera)

    task = task_task.Task(sensor,actuator,reward,world,config,ckp)
    learn_algo = config.get("train_algo", "ppo2")
    if learn_algo == "ppo2": ppo2(task, config, eval=True)
    elif learn_algo == "deepq": deepq(task, config, eval=True)
    else: raise ValueError("Invalid training algorithm {}!".format(learn_algo))

    # Plot activations (if activated).
    if args.activations:
        fig = plt.figure()  # an empty figure with no axes
        fig.suptitle('No axes on this figure')  # Add a title so we know which it is
        fig, ax_lst = plt.subplots(2, 2)
        # Access layer activation of last evaluation by accessing the
        # tensorflow namespace, in which the network and all its parameters
        # are saved.
        session = model.sess
        imageToUse = sensor.get_state()
        graph = session.graph
        # First convolution.
        layer = graph.get_tensor_by_name("ppo2_model/pi/c1/Conv2D:0")
        tf_placeholder = session.graph.get_tensor_by_name("ppo2_model/Ob:0")
        getActivations(session,layer,tf_placeholder,imageToUse)
        # Second convolution.
        layer = graph.get_tensor_by_name("ppo2_model/pi/c2/Conv2D:0")
        tf_placeholder = session.graph.get_tensor_by_name("ppo2_model/Ob:0")
        getActivations(session,layer,tf_placeholder,imageToUse)

    task.close()

if __name__ == "__main__":
    main()
