#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer, Patrick Pfreundschuh
# Description : Implementation of rewards (as described below).
# =============================================================================
import itertools
import numpy as np
import os
import time

import pybullet
import cv2 as cv
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours

from yumi_push.tasks.misc import Status

class RewardFn(object):

    def __init__(self, config, ckp, camera):
        # Store _configuration and get _config.
        self._camera = camera
        self._config = config
        self._ckp = ckp
        self._include = {}
        self._include["diff_dis"] = config.get("rew_inc_diff_dis", True)
        self._include["time_pen"] = config.get("rew_inc_time_penalty", True)
        self._include["act_start_dis"] = config.get("rew_inc_act_start_dis", True)
        self._include["pushing_dis"] = config.get("rew_inc_pushing_dis", True)
        # Build reward scheduler.
        self._scheduler = self._build_scheduler()
        # Auxiliary internal variables.
        self._max_inter_dist = config.get("rew_max_inter_dis", 0.1)
        self._last_obj_dist = None
        self._overall_obj_dist = None
        self._last_seg = None
        self._num_step = 0
        self.status = None
        self.reward = 0.0

    def __call__(self,**kwargs):
        """ Compute the reward of the last state transition. """
        self.reward = 0.0
        self._num_step = self._num_step + 1
        success = {}
        # Fail directly if start position in object.
        assert "act_start_in_obj" in kwargs.keys()
        if kwargs["act_start_in_obj"]:
            self.reward = -self._config.get("rew_in_obj_penalty", 99.9)
            self.status = Status.FAIL
            # Logging sum of rewards and status.
            self._ckp.write_log("reward = {}, status = {}".format(
                self.reward, self.status
            ))
            return (self.reward,self.status,self._num_step)
        # Difference distance reward.
        # Determines the difference in distance between the objects
        # (each other) from the last to the current observation.
        con_diff_dis = self._include["diff_dis"] and self._schedule("diff_dis")
        if con_diff_dis:
            reward = 0
            assert "segmented" in kwargs.keys() and "depth" in kwargs.keys()
            assert kwargs["segmented"].shape == kwargs["depth"].shape
            assert "num_objects" in kwargs.keys()
            new_obj_dist = self._object_dist(
                kwargs["segmented"], kwargs["depth"],
                max_rewarded_dist=self._max_inter_dist
            )
            diff_obj_dist = new_obj_dist - self._last_obj_dist
            self._last_obj_dist = new_obj_dist
            self._overall_obj_dist += diff_obj_dist
            reward = diff_obj_dist/self._max_inter_dist
            reward*= self._config.get("rew_diff_dis_max_reward", 10.0)
            if not (diff_obj_dist>0): reward *= 0.3
            self.reward += reward
            goal  = self._config.get("rew_diff_dis_goal_per_obj", 0.05)
            combs = list(itertools.combinations(range(kwargs["num_objects"]),2))
            goal  = goal*len(combs)
            success["diff_dis"] = self._overall_obj_dist >= goal
            self._ckp.write_log(
                "reward-diff-dis: {} (current = {:.3f}, total = {:.3f}, goal = {:.3f})".format(
                reward, diff_obj_dist, self._overall_obj_dist, goal
            ))
        # Time penalty.
        # Constant time penalty penalizing each further step needed
        # to reach the goal.
        con_time_pen = self._include["time_pen"]
        if con_time_pen:
            reward = self._config.get("rew_time_penalty", -10.0)
            self._ckp.write_log("reward-time-penalty: %f" % reward)
            self.reward += reward
        # Action starting point penalty.
        con_act_start_dis = self._include["act_start_dis"] \
                            and self._schedule("act_start_dis")
        if con_act_start_dis:
            reward = 0
            assert "closest_m" in kwargs.keys()
            assert kwargs["closest_m"].size == 2
            assert "act_start_m" in kwargs.keys()
            assert kwargs["act_start_m"].size == 2
            ptarget, pinit = kwargs["closest_m"], kwargs["act_start_m"]
            distance = np.linalg.norm(ptarget-pinit)
            goal = self._config.get("rew_act_start_dis_goal", 0.0)
            max_reward = self._config.get("rew_act_start_dis_max_reward", +10.0)
            min_reward = self._config.get("rew_act_start_dis_min_reward", -10.0)
            max_dist = self._config.get("rew_act_start_dis_max_dis", 2.0)
            do_success = self._config.get("rew_act_start_dis_succeed", False)
            # For distance < goal a streched sigmoid function is used which is
            # constraint to be the minimal reward at d = 0 and the maximal
            # reward at d = goal_distance.
            # Approach: s(x) = K1*sigmoid(d) + K2
            # s(0) = K1*sig(0) + K2 = rmin
            # s(g) = K1*sig(g) + K2 = rmax
            if distance < goal:
                t = 1 + np.exp(-goal)
                k2 = (min_reward - max_reward/2.0*t)/(1.0 - t/2)
                k1 = (max_reward - k2)*t
                reward += k1/(1 + np.exp(-distance)) + k2
            # Otherwise use right half of 1/x-function.
            # Approach: q(x) = K1/(d) + K2
            # q(g) = K1/g + K2 = rmax
            # q(md) = K1/md + K2 = rmax - (rmax - rmin)*0.9
            else:
                k1 = 0.9*(min_reward - max_reward)/(1/max_dist - 1/goal)
                k2 = max_reward - k1/goal
                reward += k1/distance + k2
            self.reward += reward
            success["act_start_dis"] = do_success and reward > 0.9*max_reward
        # Action pushing distance penalty.
        con_pushing_dis = self._include["pushing_dis"]
        if con_pushing_dis:
            assert "act_start_m" in kwargs.keys()
            assert kwargs["act_start_m"].size == 2
            assert "act_target_m" in kwargs.keys()
            assert kwargs["act_target_m"].size == 2
            ptarget, pinit = kwargs["act_target_m"], kwargs["act_start_m"]
            self.reward += -np.linalg.norm(ptarget-pinit)
        # Determine status, success if goal distance (sum of all
        # interobject distances) is reached, fail if maximal steps
        # are reached, otherwise runnning.
        self.status = Status.RUNNING
        if con_diff_dis and success["diff_dis"]:
            self.status = Status.SUCCESS
        elif con_act_start_dis and success["act_start_dis"]:
            self.status = Status.SUCCESS
        elif(self._num_step>= self._config.get("task_max_trials_eps", 10)):
            self.status = Status.TIME_LIMIT
        # Final (success) reward.
        if self.status == Status.SUCCESS:
            self.reward += self._config.get("rew_final", 10.0)
        # Logging sum of rewards and status.
        self._ckp.write_log("reward = {}, status = {}".format(
            self.reward, self.status
        ))
        return (self.reward,self.status,self._num_step)

    def reset(self,**kwargs):
        assert "segmented" in kwargs.keys() and "depth" in kwargs.keys()
        self._last_obj_dist = self._object_dist(
            kwargs["segmented"], kwargs["depth"],
            max_rewarded_dist=self._max_inter_dist
        )
        self._overall_obj_dist = self._last_obj_dist
        self._num_step = 0
        self.reward = 0.0

    # ==========================================================================
    # Scheduler.
    # ==========================================================================
    def _build_scheduler(self):
        if not self._config.get("rew_schedule",False): return None
        scheduler = {}
        for x in self._config.keys():
            if not x.count("rew_schedule_") > 0: continue
            k = x.replace("rew_schedule_","")
            v = self._config[x]
            assert len(v) == 2
            if v[1] == -1: v[1] = int(1e9)
            scheduler[k] = [v[0],v[1]]
        return scheduler

    def _schedule(self, key):
        if self._scheduler is None: return True
        iter = self._ckp.iteration
        return self._scheduler[key][0] <= iter <= self._scheduler[key][1]

    # ==========================================================================
    # Helper functions.
    # ==========================================================================
    def _min_object_dist(self,depth_img,edge_1_idx,edge_2_idx):
        min_dist = np.inf
        edge_1_idx = np.array(edge_1_idx,dtype=np.int16)
        edge_2_idx = np.array(edge_2_idx,dtype=np.int16)
        e1 = np.linspace(0,len(edge_1_idx)-1,len(edge_1_idx)).tolist()
        e2 = np.linspace(0,len(edge_2_idx)-1,len(edge_2_idx)).tolist()
        #get all different pixel combinations
        combinations = list(itertools.product(e1,e2))
        combinations = np.asarray(combinations,dtype=np.int16)
        p1 = edge_1_idx[combinations[:,0]]
        p2 = edge_2_idx[combinations[:,1]]
        Z1 = depth_img[p1[:,0],p1[:,1]]
        Z2 = depth_img[p2[:,0],p2[:,1]]
        m1 = self._camera.px_to_m(p1,Z1)
        m2 = self._camera.px_to_m(p2,Z2)
        #calculate squared distance
        pd = m1-m2
        pd = np.square(pd)
        pd = np.sqrt(pd[:,0]+pd[:,1])
        #get minimum
        min_dist = np.min(pd)
        min_arg = np.argmin(pd)
        return min_dist

    def _get_segments(self,seg_img):
        contours = find_contours(seg_img, 0.01)
        return contours

    def _object_dist(self,seg_img,depth,max_rewarded_dist):
        max_dist = max_rewarded_dist
        segments = self._get_segments(seg_img)
        #create all combinations of inter object comparisons
        segment_array = np.linspace(0,len(segments)-1,len(segments),dtype=int)
        combinations = list(itertools.combinations(segment_array,2))
        #add all minimal inter-object distances
        total_dist = 0
        truncated_dist = 0
        for j in range(0,len(combinations)):
            min_dist_obj = self._min_object_dist(depth,
                segments[combinations[j][0]],
                segments[combinations[j][1]]
            )
            truncated_dist = truncated_dist + np.min([max_dist,min_dist_obj])
            total_dist = total_dist + min_dist_obj
        return truncated_dist
