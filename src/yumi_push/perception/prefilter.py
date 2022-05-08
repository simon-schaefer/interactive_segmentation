#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Segmentation prefilter for background removal using RGBD.
# =============================================================================
import numpy as np

class Prefilter(object):

    def __init__(self, config, ckp):
        self._ckp = ckp

    def __call__(self):
        raise NotImplementedError

# =============================================================================
# DepthPrefilter
# The DepthPrefilter uses the RGBD depth information and a histogram analysis
# in order to detect and remove the background, assuming that most of the
# image is background. In this case the most occuring depth will be the
# background depth. Also the background is assumed to be more distant than the
# regarded objects which is the case for the robot pushing task.
# =============================================================================
class DepthPrefilter(Prefilter):

    def __init__(self, config, ckp):
        super(DepthPrefilter, self).__init__(config, ckp)
        self._cut = config.get("prefilt_cutting_part", 20.0)

    def __call__(self, image):
        assert len(image.shape) == 3 and image.shape[2] == 4
        unique, counts = np.unique(image[:,:,3], return_counts=True)
        max_count = counts[-1]
        max_distance = unique[-1]
        for i in range(len(counts)):
            if counts[i] == max_count:
                continue
            if counts[i] > max_count/self._cut:
                max_distance = unique[i]
        rgb_filtered = image[:,:,:3]
        rgb_filtered[image[:,:,3]>=max_distance-1e-2] = 0
        self._ckp.save_array_as_plot(rgb_filtered, "prefilter.png")
        return rgb_filtered
