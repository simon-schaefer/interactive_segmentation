#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michel Breyer, Simon Schaefer
# Description : Implementation of sensors.
# =============================================================================
import gym
import gym.spaces
import numpy as np
import time

import cv2

from yumi_push.perception.distances import *
from yumi_push.perception.prefilter import *
from yumi_push.perception.segmentation import *

# =============================================================================
# General implementation of a sensor supporting the following devices ...
# - camera: segmentation of objects with information about how many
#           times they have been pushed already (~= probability of
#           right segmentation).
# =============================================================================
class Sensor(object):

    def __init__(self, config, ckp, **kwargs):
        super(Sensor, self).__init__()
        # Extract devices from kwargs argument.
        self.camera = kwargs.get("camera", None)
        self.robot  = kwargs.get("robot", None)
        # Visual perception algorithms.
        assert not self.camera is None
        self.rgb, self.depth = None, None
        self._prefilter = DepthPrefilter(config, ckp)
        seg_method = config.get("seg_method", "boolean")
        self._segmentor = segmentation_factory(seg_method, config, ckp)
        self._distancer = DistanceMap(config, ckp)
        # Initialize state space as image with the number of channels
        # depending on which devices and perception algorithms are used.
        self._state_type = config.get("sensor_type", "distances")
        self._state_discrete = config.get("sensor_discrete", False)
        down = config.get("model_obs_downscaling", 1)
        self._obs_sz = (int(self.camera.info.height/down),
                        int(self.camera.info.width/down))
        self.state_space = gym.spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32,
            shape=(self._obs_sz[0], self._obs_sz[1], 1),
        )
        # Auxialiary variables.
        self._ckp = ckp
        self.padding_dist = config.get("in_object_padding_dis", 0.0)
        self._uniform_sampling = config.get("uniform_sampling", True)

    def get_state(self):
        # Render RGBD image from internal camera sensor.
        self.rgb, self.depth, _ = self.camera.render_images()
        self._ckp.save_array_as_plot(self.rgb, "rgb.png")
        rgbd = np.stack((self.rgb[:,:,0], self.rgb[:,:,1],
                         self.rgb[:,:,2], self.depth), axis=2)
        # Visual perception operations.
        img_wo_bg = self._prefilter(rgbd)
        segmented = self._segmentor(img_wo_bg)
        distances = self._distancer(segmented)
        # Compute observation for network input based on state type.
        state = None
        if self._state_type == "segmented": state = segmented.copy()
        elif self._state_type == "distances": state = distances.copy()
        else:
            raise ValueError("Invalid sensor type {}!".format(self._state_type))
        state = cv2.resize(state,
            (self._obs_sz[1],self._obs_sz[0]), interpolation=cv2.INTER_NEAREST
        )
        if self._state_type == "segmented":
            self._ckp.save_array_as_plot(state, "observation.png")
        elif self._state_type == "distances":
            self._ckp.save_distances_as_plot(state, "observation.png")
        state = np.reshape(state,(state.shape[0],state.shape[1],1))
        return state, segmented, distances

    def is_in_object(self, point_m):
        assert point_m.size == 2
        start_closest_pos_m = self.get_closest_point(point_m)
        dist = np.linalg.norm(point_m-start_closest_pos_m)
        return dist < self.padding_dist

    def get_closest_point(self, point_m):
        assert point_m.size == 2
        point_p = self.camera.m_to_px(point_m)
        closest_p = self._distancer.get_closest_point(point_p)
        # Transform back to cartesian domain.
        z = self.depth[point_p[0], point_p[1]]
        return self.camera.px_to_m(closest_p, z)[0]

    def get_raw_rgbd(self):
        return (self.rgb, self.depth)

    def get_sample_heatmap(self):
        raw_dists = self._distancer._distances_raw
        if(self._uniform_sampling):
            return np.ones(np.shape(raw_dists))

        goal_val = np.shape(raw_dists)[0]/10
        dists_from_goal = np.absolute(raw_dists - goal_val)
        sorround_idx = np.where(dists_from_goal < 2 )
        obj_idx = np.where(raw_dists <= 2)
        sourround_ones = np.zeros(np.shape(raw_dists))*0.5
        sourround_ones[sorround_idx] = 1
        kernel_size = np.shape(raw_dists)[0]/5
        if(np.floor(kernel_size)%2==1):
            kernel_size = np.floor(kernel_size).astype(dtype=int)
        else:
            kernel_size = np.ceil(kernel_size).astype(dtype=int)

        smoothed = cv2.GaussianBlur(sourround_ones,(kernel_size,kernel_size),0)
        smoothed = smoothed + 0.2
        smoothed[obj_idx]=0
        self._ckp.save_distances_as_plot(smoothed, "sample_prob.png")
        return smoothed
