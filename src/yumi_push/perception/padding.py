#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Builds variations of segmented input image, each variation
#               is centered around one object. Therefore in order to have
#               a "constant" image shape, the shape is padded.
# =============================================================================
import numpy as np

from skimage.color import rgb2gray
from skimage.measure import find_contours

class Padding(object):

    def __init__(self, config, ckp):
        self._ckp = ckp
        self._obj_index_dict = None

    def __call__(self):
        raise NotImplementedError

    def get_object_from_index(self, index):
        if index in self._obj_index_dict.keys():
            return self._obj_index_dict[index]
        else:
            return None

    def get_num_channels(self):
        raise NotImplementedError

# =============================================================================
# ContourPadding
# In the segmented image objects are distinguished by finding contours using
# skimage find_contours function. The center of each contour is the center of
# each object (center of resulting image).
# =============================================================================
class ContourPadding(Padding):

    def __init__(self, config, ckp):
        super(ContourPadding, self).__init__(config, ckp)
        self._num_channels = 10

    def __call__(self, segmented):
        w,h = segmented.shape
        gray = rgb2gray(segmented)
        contours = find_contours(gray, 0.01)
        n_objects = len(contours)
        images = np.zeros((2*w,2*h,self._num_channels), dtype=np.uint8)
        self._obj_index_dict = {}
        for k,c in enumerate(contours):
            cx = int(np.mean([z[0] for z in c]))
            cy = int(np.mean([z[1] for z in c]))
            self._obj_index_dict[k] = (cx,cy)
            images[w-cx:2*w-cx,h-cy:2*h-cy,k] = segmented
        self._ckp.save_padding_as_plot(images, "pad_{}.png".format(n_objects))
        return images

    def get_num_channels(self):
        return self._num_channels
