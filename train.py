#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Model training (simulation).
# =============================================================================
import os

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from yumi_push.common.checkpoint import Checkpoint
from yumi_push.common.io_utils import load_yaml
from yumi_push.common.openai_utils import *
from yumi_push.tasks import init_task

def main():
    config = load_yaml(os.environ["YUMI_PUSH_CONFIG"]+"/simulation.yaml")
    cam_config = load_yaml(os.environ["YUMI_PUSH_CONFIG"]+"/"+config['camera_config'])
    ckp = Checkpoint(config)
    config["working_dir"] = ckp.get_path("checkpoints/")
    task, _, _ = init_task(config, cam_config, ckp)
    learn_algo = config.get("train_algo", "ppo2")
    if learn_algo == "ppo2": ppo2(task, config)
    elif learn_algo == "deepq": deepq(task, config)
    else: raise ValueError("Invalid training algorithm {}!".format(learn_algo))

    task.close()

if __name__ == "__main__":
    main()
