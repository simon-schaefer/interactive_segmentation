#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Segmentation prefilter for background removal using RGBD.
# =============================================================================
import numpy as np

from scipy.ndimage import distance_transform_edt

class Distances(object):

    def __init__(self, config, ckp):
        self._ckp = ckp
        self._distances, self._distances_raw, self._indices = None, None, None

    def __call__(self, segmented):
        raise NotImplementedError

    def is_in_object(self, point_p):
        assert point_p.size == 2
        return self._distances_raw[point_p[0],point_p[1]] < 1

    def get_closest_point(self, point_p):
        assert point_p.size == 2
        closest_p = np.reshape(self._indices[:, point_p[0], point_p[1]], (1,2))
        return closest_p

# =============================================================================
# DistanceMap
# Creates map of distances from objects based on scipy.ndimage library, i.e.
# by converting segmentation to boolean mask and finding the closest euclidean
# distance to opposite boolean value for every pixel.
# Return distance map as well as stores distances map and array of indices of
# closest point to every pixel.
# =============================================================================
class DistanceMap(Distances):

    def __init__(self, config, ckp):
        super(DistanceMap, self).__init__(config, ckp)

    def __call__(self, segmented):
        assert len(segmented.shape) == 2
        # Determine maximal distance.
        w,h = segmented.shape
        # Apply scipy's distance transform.
        logical_mask = segmented <= 0
        self._distances_raw, self._indices = distance_transform_edt(
            logical_mask, return_indices=True, return_distances=True
        )
        max_distance = np.sqrt(w*w + h*h)
        obj_idxs = self._distances_raw < 1e-3
        self._distances = -self._distances_raw/max_distance
        self._distances[obj_idxs] = 1.0
        self._ckp.save_distances_as_plot(self._distances, "distance_map.png")
        return self._distances
