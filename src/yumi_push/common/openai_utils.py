#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : OpenAI baselines wrapping utilities.
# =============================================================================
import os

from baselines.common.vec_env import DummyVecEnv
from baselines.deepq import learn as deepq_learn
from baselines.ppo2.ppo2 import learn as ppo2_learn

# =============================================================================
# Proximal Policy Optimization (ppo2).
# =============================================================================
def ppo2(task, config, eval=False):
    assert config["act_discrete"] == False
    dummy_env = DummyVecEnv([lambda: task])
    learn_args = {"env": dummy_env, "network": config["model_network"],
                  "total_timesteps": config["train_steps"],
                  "nsteps": config["train_steps_update"],
                  "nminibatches": config["train_minibatches_update"],
                  "noptepochs": config["train_epoch_update"],
                  "vf_coef": config["train_value_fn_coeff"],
                  "ent_coef": config["train_entropy_coeff"],
                  "gamma": config["train_gamma"], "lam": config["train_lambda"],
                  "lr": float(config["train_learning_rate"]),
                  "save_interval": config["ckp_model_save_interval"],
                  # Network parameters.
                  "nactions": task.action_space.shape[0]
                  }
    if eval:
        learn_args["total_timesteps"] = config["eval_steps"]
        learn_args["nsteps"] = config["eval_steps"]
        learn_args["nminibatches"] = config["eval_steps"]
        learn_args["lr"] = 0.0
        models_path = config["working_dir"] + "/checkpoints/"
        models_path = models_path + str(max([f for f in os.listdir(models_path)]))
        learn_args["load_path"] = models_path
    return ppo2_learn(**learn_args)

# =============================================================================
# Deep Q-Learning.
# =============================================================================
def deepq(task, config, eval=False):
    assert config["act_discrete"] == True
    dummy_env = DummyVecEnv([lambda: task])
    learn_args = {"env": dummy_env, "network": config["model_network"],
                  "total_timesteps": config["train_steps"],
                  "gamma": config["train_gamma"],
                  "buffer_size": config["train_buffer_size"],
                  "exploration_fraction": config["train_exploration_fraction"],
                  "exploration_final_eps": config["train_exploration_final_eps"],
                  "prioritized_replay": config["train_prioritized_replay"],
                  "lr": float(config["train_learning_rate"]),
                  "print_freq": config["ckp_model_save_interval"],
                  "checkpoint_freq": config["ckp_model_save_interval"],
                  "checkpoint_path": config["working_dir"],
                  "learning_starts": -1,
                  # Network parameters.
                  "nactions": 1
                  }
    if eval:
        learn_args["total_timesteps"] = config["eval_steps"]
        learn_args["learning_starts"] = config["eval_steps"]
        learn_args["exploration_fraction"] = 1.0/config["eval_steps"]
        learn_args["exploration_final_eps"] = 0.0
        learn_args["lr"] = 0.0
        models_path = config["working_dir"] + "/checkpoints/model"
        learn_args["load_path"] = models_path
    return deepq_learn(**learn_args)
