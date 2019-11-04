#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michel Breyer, Simon Schaefer, Patrick Pfreundschuh
# Description : OpenAI gym compatible base class for robotic learning tasks.
#               Tasks follow the standard episodic reinforcement learning model.
#               At each control step, the task assembles an observation from the
#               current state of its sensors. It then executes the command chosen
#               by an external agent, determines the next state of the system and
#               reports the reward associated with the transition.
#               Different tasks can be defined by providing implementations of
#               sensors, actuators and reward functions, as well as attaching
#               custom code to different events, such as the beginning and end of
#               episodes, through a callback system.
# =============================================================================
import gym
import numpy as np
import time

from yumi_push.tasks.misc import Status

class Task(gym.Env):
    """OpenAI gym compatible base class for robot learning tasks. """

    def __init__(self, sensor, actuator, reward_fn, environment,
                 config, ckp):
        """Initialize the task."""
        self._sensor = sensor
        self._actuator = actuator
        self._reward_fn = reward_fn
        self._environment = environment
        self._ckp = ckp
        self._debug_mode = config.get("debug", False)
        self._actuator_type = config.get("actuator")
        self._type = config.get("train_algo", "deepq")
        self.action_space = self._actuator.action_space
        self.observation_space = self._sensor.state_space
        self.num_envs = 1
        if self._type=="deepq": self.action_to_px=self.calculate_action_to_px()

    def reset(self):
        """Reset task and return observation of the initial state. """
        if self._ckp.can_reset:
            self._ckp.reset()
            self._environment.reset(test_scene=self._ckp.is_testing_step)
        obs, segmented, distances = self._sensor.get_state()
        self._actuator.reset()
        self._reward_fn.reset(segmented=segmented, depth=self._sensor.depth)
        self._ckp.update_last_state(obs, segmented, distances)
        self.status = Status.RUNNING
        return obs

    def step(self, action):
        """Advance the task by one step inputting action and outputting
        the tuple (obs, reward, done, info), where done is a boolean flag
        indicating whether the current episode finished. """
        if self._type == "deepq": action, qvalues = action[0][0], action[1]
        self._ckp.step()
        act = self._actuator.unnormalize(action)
        start_pos = np.asarray([act[0],act[1]])
        self._sensor.get_state()
        start_in_object = self._sensor.is_in_object(start_pos)
        start_closest_pos = self._sensor.get_closest_point(start_pos)
        # Execute action and transform the executed action back to image
        # domain for further usage (pact = action as starting and target
        # point in pixel domain).
        act = self._actuator.execute_action(action=act)
        start_pos  = np.asarray([act[0],act[1]])
        target_pos = np.asarray([act[2],act[3]])
        pact = (self._sensor.camera.m_to_px(start_pos),
                self._sensor.camera.m_to_px(target_pos))
        self._ckp.write_log("paction = {}".format(pact))
        # Observe new state and determine reward & status.
        obs, segmented, distances = self._sensor.get_state()
        reward, self.status, num_steps = self._reward_fn(
            segmented=segmented,
            depth=self._sensor.depth,
            closest_m=start_closest_pos,
            act_start_m=start_pos,
            act_target_m=target_pos,
            act_start_in_obj=start_in_object,
            num_objects=self._environment.get_num_objects()
        )
        done = self.status != Status.RUNNING
        # Logging step and draw final segmentation and action
        # in plot (only debug).
        self._ckp.add_log(reward, "reward", typ="add")
        self._ckp.add_log(num_steps, "num_steps", typ="set")
        self._ckp.add_log(
            int(self.status==Status.SUCCESS), "success_rate", typ="add"
        )
        self._ckp.save_action_as_plot(
            pact[0], pact[1], segmented.copy(), name="action.png"
        )
        self._ckp.save_seg_rew_map(name="seg_rew_map.png")
        if self._ckp.is_testing_step:
            self._ckp.save_action_dist(name="start_point_map.png")
        if self._type == "deepq" and self._ckp.is_testing_step:
            self._ckp.save_qmaps_as_plot(*self.convert_qvalues_to_qmaps(qvalues))
        #if self._ckp.is_testing_step:
        #    self._ckp.save_network_activation_as_plot(obs)
        self._ckp.update_last_state(obs, segmented, distances)
        self._ckp.update_start_pos(pact[0])
        return obs, reward, done, {}

    def close(self):
        self._ckp.write_log("\nCLOSING task")
        self._environment.close()

    def calculate_action_to_px(self):
        all_actions = np.arange(self._actuator.action_space.n)
        actions_px = np.zeros((self._actuator.action_space.n,2))
        for i in all_actions:
            action = self._actuator.unnormalize(i)
            actions_px[i,:] = self._sensor.camera.m_to_px(
                np.array([action[0],action[1]]))
        return actions_px.astype(int)

    def convert_qvalues_to_qmaps(self, q_values):
        """ Convert vector of qvalues in single map for every possible
        pushing direction for the chosen actuator. """
        n_actions    = self.action_space.n
        n_directions = len(self._actuator.directions())
        n_steps      = int(np.sqrt(n_actions/n_directions))
        q_maps       = np.ones((n_steps, n_steps, n_directions))*(-1000)
        q_values     = np.reshape(q_values, (q_values.size))
        for x in range(n_actions):
            action = self._actuator.undiscretize(np.asarray(x, dtype=int))
            action.extend([0] * (3 - len(action)))
            q_maps[action[0],action[1],action[2]] = q_values[x]
        return q_maps, self._actuator.directions()

    def get_action_probabilities(self):
        heatmap = self._sensor.get_sample_heatmap()
        probabilities = heatmap[self.action_to_px[:,0],self.action_to_px[:,1]]
        prob_norm = np.linalg.norm(probabilities)
        probabilities = probabilities/prob_norm

        return probabilities
