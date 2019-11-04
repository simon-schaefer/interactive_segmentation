#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Testing utility functions.
# =============================================================================
import argparse
import cv2
import os
import glob
import numpy as np
import pandas as pd
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

def images_to_video(path:str, tag: str, video_name:str):
    imgs = glob.glob(path + "/debug/*.png")
    imgs = [x.split("/")[-1] for x in imgs if x.count(tag) > 0]
    img_nums = sorted(np.unique([int(x.split("_")[0]) for x in imgs]))
    fname = "{}_{}.png".format(str(img_nums[0]), tag)
    frame = cv2.imread(os.path.join(path, "debug", fname))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 10, (width,height))
    for fi,fnum in enumerate(img_nums):
        fname = "{}_{}.png".format(str(fnum), tag)
        video.write(cv2.imread(os.path.join(path, "debug", fname)))
        progress_bar(fi, len(img_nums))
    cv2.destroyAllWindows()
    video.release()

def parse_path(args: argparse.Namespace):
    if not ((args.policy == "" and args.out != "") \
    or (args.policy != "" and args.out == "")):
        raise IOError("Usage: python3 eval.py --policy=... or --out=...")
    directory = args.policy if args.policy != "" else args.out
    parent_dir = os.environ["YUMI_PUSH_POLICIES"] if args.policy != "" \
                 else os.environ["YUMI_PUSH_OUTS"]
    path = os.path.join(parent_dir, directory)
    if not os.path.isdir(path):
        raise IOError("Directory {} not existing !".format(path))
    return directory, path

def parse_logging_file(filename, max_episode=150000):
    """ Parse standartized logging file in order to get the sequence of
    rewards, the action/paction and status by step. """
    assert os.path.isfile(filename)
    num_lines  = sum(1 for line in open(filename, 'r'))
    logging_list = []
    episode_dict = {"episode":0, "rewards":[], "reward_sum":0, "status_final":None}
    print("\nRead logging file {} ... ".format(filename))
    with open(filename, "r") as file:
        for il, line in enumerate(file):
            if line.count("status") > 0 and line.count("reward") > 0:
                m = re.match(r'reward = (.*), status = Status.(.*)', line)
                reward, status = float(m.group(1)), m.group(2)
                episode_dict["rewards"].append(reward)
                if status != "RUNNING":
                     status = line.split("Status.")[-1].replace("\n","")
                     episode_dict["status_final"] = status
                     episode_dict["reward_sum"]   = sum(episode_dict["rewards"])
                     logging_list.append(episode_dict)
                     episode_dict = {"episode": episode_dict["episode"]+1,
                                      "rewards": [], "status_final": None}
            if episode_dict["episode"] > max_episode: break
            progress_bar(il, num_lines)
    episode_df = pd.DataFrame(logging_list)
    print("\nDetermine 1000-mean-reward ... ")
    rewards_1000_mean = np.zeros((len(episode_df["episode"]),1))
    last_update = -1
    for episode in episode_df["episode"]:
        if episode < 1000: rewards_1000_mean[episode] = 0.0
        elif last_update > 0 and episode - last_update < 100:
            rewards_1000_mean[episode] = rewards_1000_mean[episode-1]
        else:
            reward_mean = np.mean(episode_df["reward_sum"][episode-1000:episode])
            rewards_1000_mean[episode], last_update = reward_mean, episode
        progress_bar(episode, len(episode_df["episode"]))
    episode_df["reward_mean_1000"] = rewards_1000_mean
    return episode_df

def progress_bar(iteration: int, num_steps: int, bar_length: int=50) -> int:
    """ Draws progress bar showing the number of executed
    iterations over the overall number of iterations.
    Increments the iteration and returns it. """
    status = ""
    progress = float(iteration) / float(num_steps)
    if progress >= 1.0:
        progress, status = 1.0, "\r\n"
    block = int(round(bar_length * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (bar_length - block), round(progress*100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()
    return iteration + 1

def draw_graphs(values, segmented, directions, ckp, name=""):
    """ Draws heatmap and 3D isohypsen plot of values array (3D array) and
    save it in ckp path. Also draws segmented image plot as scene visualization. """
    segmented = ckp.segmented_to_image(segmented)
    plt.imshow(segmented)
    plt.savefig(os.path.join(ckp.dir, name+"_obs.png"))
    plt.close()
    max_value = 2000 #np.amax(values)
    for i, direction in enumerate(directions):
        X = np.linspace(0,values.shape[0],values.shape[0])
        Y = np.linspace(0,values.shape[1],values.shape[1])
        Z = values[:,:,i]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, Z, 100, cmap='viridis')
        ax.view_init(40,35)
        ax.set_xlabel("X [steps]")
        ax.set_ylabel("Y [steps]")
        ax.set_zlabel(name)
        plt.savefig(os.path.join(ckp.dir, "{}_3d_{}.png".format(name,direction)))
        plt.close()
        sns.heatmap(Z, vmax=max_value)
        plt.savefig(os.path.join(ckp.dir, "{}_2d_{}.png".format(name,direction)))
        plt.close()
