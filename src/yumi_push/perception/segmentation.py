#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Segmentation algorithms converting RGB(D) image to segmented
#               classes masks.
# =============================================================================
import numpy as np

class Segmentation(object):

    def __init__(self, config, ckp):
        self._ckp = ckp
        self._labels = None

    def __call__(self):
        raise NotImplementedError

    def is_in_object(self, point_p):
        assert point_p.size == 2
        return self._labels[point_p[0],point_p[1]] > 0

def segmentation_factory(name, config, ckp):
    if name == "boolean":
        return BooleanSegmenation(config, ckp)
    else:
        raise ValueError("Invalid segmentation method {}".format(name))

# =============================================================================
# BooleanSegmenation
# Based on (prefiltered) RGB image create boolean mask, i.e. differ between
# objects and background only.
# =============================================================================
class BooleanSegmenation(Segmentation):

    def __init__(self, config, ckp):
        super(BooleanSegmenation, self).__init__(config, ckp)

    def __call__(self, image):
        assert len(image.shape) == 3
        w,h,_  = image.shape
        self._labels = np.zeros((w,h), dtype=np.int8)
        self._labels[np.max(image, axis=2) > 0] = 1
        self._ckp.save_segmentation_as_plot(self._labels, "segmentation.png")
        return self._labels

# =============================================================================
# ContourSamplingSegmentation
# This segmentation method takes both the shape/position of the object and its
# color into account. To do so after removing the backgroung using the depth
# information, a contour detection is applied. The detected contours are then
# compared to the previously known contours. In case the position did not change
# (same contour) the previous object index is assigned. Otherwise the mean
# color of the object is compared to the database of already existing objects
# and the closest color match is assigned. Thereby merely unique matches
# are allowed. In case more contours have been detected a new contour will be
# added to the database.
#
# Starting process (database build up):
# When a contour is added to the database it has to be checked whether the
# detected contour consists of several objects. In order to distinguish the
# objects some random colors are sampled in the contour area (throwing away
# background samples).  If the samples are all in a certain color range, the
# contour is considered as a single object, otherwise it is divided into
# several objects (using numpy simple color masking based on the samples).
#
# object_database: obj_idx --> (feature_vec, mean_color, mask)
# =============================================================================
# class ContourSamplingSegmentation(Segmentation):
#
#     def __init__(self, config, ckp):
#         super(ContourSamplingSegmentation, self).__init__(config, ckp)
#         self._object_database = {}
#         self._max_color_distance = config.get("seg_max_color_dis", 10)/255.0
#
#     def __call__(self, image):
#         assert len(image.shape) == 3 and image.shape[2] == 3
#         w,h,_ = image.shape
#         gray = rgb2gray(image)
#         contours = find_contours(gray, 0.01)
#         hsv = rgb2hsv(image)[:,:,:2]
#         if not len(self._object_database.keys()) == len(contours):
#             self._ckp.write_log(
#                 "segmentation - Reinit as len(db)={} and {} found".format(
#                 len(self._object_database.keys()), len(contours)
#                 ))
#             return self._initialize(contours, hsv)
#         labels   = np.zeros((w,h), dtype=np.int8)
#         matched  = {x: False for x in self._object_database.keys()}
#         for contour in contours:
#             # Compare detected contours to existing database.
#             match_idx = None
#             dims = self._extract_dimensions(contour)
#             for db_idx, db_feat in self._object_database.items():
#                 if dims == db_feat[0]:
#                     match_idx = db_idx
#                     matched[db_idx] = True
#                     self._ckp.write_log(
#                         "segmentation - Recognized object {}".format(
#                         db_idx
#                     ))
#                     break
#             # If contour have not been found but no new contours
#             # are added (same length as db), match by closest mean
#             # color and update database.
#             if match_idx is None:
#                 color = self._extract_color(dims, hsv)
#                 closest_dis, cl_idx = np.inf, 0
#                 for db_idx, db_feat in self._object_database.items():
#                     if matched[db_idx]: continue
#                     distance = np.linalg.norm(color-db_feat[1])
#                     if distance <= closest_dis:
#                         closest_dis = distance
#                         cl_idx = db_idx
#                 matched[cl_idx] = True
#                 match_idx = cl_idx
#                 mask = self._extract_mask(dims, image)
#                 self._ckp.write_log(
#                     "segmentation - Tracked object {} from {} to {}".format(
#                     cl_idx, self._object_database[cl_idx][0], dims
#                 ))
#                 self._object_database[cl_idx] = (dims, color, mask)
#             # Add (updated) mask to labels.
#             labels += self._object_database[match_idx][2]*match_idx
#         self._ckp.save_segmentation_as_plot(labels, "segmentation.png")
#         return labels
#
#     def _initialize(self, contours, image, N=20):
#         self._object_database = {}
#         w,h,_  = image.shape
#         objidx = 1
#         labels = np.zeros((w,h), dtype=np.int8)
#         for contour in contours:
#             dims = self._extract_dimensions(contour)
#             x_min,x_max,y_min,y_max = dims
#             samples = np.zeros((N,2)) #hue,saturation(w/o value)
#             i = 0
#             while i < N:
#                 x = np.random.randint(x_min, x_max)
#                 y = np.random.randint(y_min, y_max)
#                 samples[i,:] = image[x,y,:2]
#                 if np.sum(samples[i,:]) > 1e-5: # background = 0
#                     i = i + 1
#             # If all samples in the same color region treat the contour
#             # as single object and create database entry.
#             if np.sum(np.var(samples, axis=0)) <= self._max_color_distance:
#                 mean_color = self._extract_color(dims, image)
#                 mask = self._extract_mask(dims, image)
#                 self._object_database[objidx] = (dims, mean_color, mask)
#                 labels += mask*objidx
#                 self._ckp.write_log(
#                     "segmentation - Found single object at ({},{}) -> {}".format(
#                     x_min, y_min, objidx
#                 ))
#                 objidx = objidx + 1
#             # Otherwise try to split the contour using color information.
#             # Therefore find unique colors in the samples (clustering would
#             # be to comp. expensive, espc. as the number of clusters is
#             # unknown). For each color create a mask, dimension and finally
#             # a database entry.
#             else:
#                 colors = []
#                 maxdix = self._max_color_distance
#                 for i in range(N):
#                     y = samples[i,:]
#                     found = False
#                     for x in colors:
#                         if np.linalg.norm(y-x) < maxdix:
#                             found = True
#                             break
#                     if not found: colors.append(y)
#                 for color in colors:
#                     patch = image[x_min:x_max,y_min:y_max,:2].copy()
#                     mask = np.zeros((w,h), dtype=bool)
#                     patch = np.sum(np.abs(patch-color), axis=2) <= maxdix
#                     mask[x_min:x_max,y_min:y_max] = patch
#                     true_xys = np.nonzero(mask)
#                     pxmin,pxmax = np.amin(true_xys[0]),np.amax(true_xys[0])
#                     pymin,pymax = np.amin(true_xys[1]),np.amax(true_xys[1])
#                     self._object_database[objidx] = (dims, color, mask)
#                     labels += mask*objidx
#                     objidx = objidx + 1
#                 self._ckp.write_log(
#                     "segmentation - Found {} objects at ({},{}) -> {}-{}".format(
#                     len(colors), x_min, y_min, objidx-len(colors), objidx-1
#                 ))
#         self._ckp.save_segmentation_as_plot(labels, "segmentation.png")
#         return labels
#
#     def _extract_color(self, dims, image, N=5):
#         x_min,x_max,y_min,y_max = dims
#         samples = np.zeros((N,2))
#         i = 0
#         while i < N:
#             x = np.random.randint(x_min, x_max)
#             y = np.random.randint(y_min, y_max)
#             samples[i,:] = image[x,y,:2]
#             if np.sum(samples[i,:]) > 1e-5: # background = 0
#                 i = i + 1
#         color = np.mean(samples, axis=0)
#         return color
#
#     def _extract_dimensions(self, contour):
#         x_min, x_max = int(np.amin(contour[:,0])),int(np.amax(contour[:,0]))
#         y_min, y_max = int(np.amin(contour[:,1])),int(np.amax(contour[:,1]))
#         return (x_min,x_max,y_min,y_max)
#
#     def _extract_mask(self, dims, image):
#         x_min,x_max,y_min,y_max = dims
#         mask = np.zeros(image.shape[:2], dtype=bool)
#         patch = np.min(image[x_min:x_max,y_min:y_max,:],axis=2)
#         mask[x_min:x_max,y_min:y_max] = patch > 0
#         return mask

# =============================================================================
# KMeansSegmentation
# KMeans clustering algorithm taking both the color distance and the euclidean
# pixel distance into account (both normalized).
# Problem: unknown a priori number of objects in the scene and if known
#          (estimated) number of objects using find_contours, it is often
#          underestimted.
# =============================================================================
# class KMeansSegmentation(Segmentation):

#     def __init__(self, config, ckp):
#         super(KMeansSegmentation, self).__init__(config, ckp)

#     def __call__(self, image):
#         assert len(image.shape) == 3
#         w,h,_ = image.shape
#         # Find number of clusters using contour algorithm.
#         gray = rgb2gray(image)
#         contours = find_contours(gray, 0.01)
#         num_objects = len(contours)
#         # Clustering filtered image.
#         edges = canny(rgb2gray(image), sigma=5)
#         distances = distance_transform_edt(edges)
#         image_distance_array = np.zeros((w,h,4))
#         image_distance_array[:,:,:3] = image.astype(np.float32)/255
#         image_distance_array[:,:,3] = distances/np.sqrt(w**2+h**2)
#         image_distance_array = np.reshape(image_distance_array, (w*h,4))
#         model = KMeans(n_clusters=num_objects*2+1)
#         model.fit(image_distance_array)
#         labels = model.predict(image_distance_array)
#         labels = np.reshape(labels, (w,h))
#         labels[image < 0] = 0
#         self._ckp.save_segmentation_as_plot(labels, "segmentation.png")
#         return labels

# =============================================================================
# WatershedSegmentation
# Purely euclidean distance based segmentation method based on looking for
# local maxima of distance from edges.
# Problem: It works for sure for cicular shapes having clear local maxima
#          but not for quadratic shapes.
# =============================================================================
# class WatershedSegmentation(Segmentation):

#     def __init__(self, config, ckp):
#         super(WatershedSegmentation, self).__init__(config, ckp)

#     def __call__(self, image):
#         assert 2 <= len(image.shape) >= 3
#         if 3 == len(image.shape):
#             image = rgb2gray(image[:,:,:3])
#             image = np.reshape(image, (image.shape[0], image.shape[1]))
#         edges = canny(image, sigma=5)
#         distances = distance_transform_edt(edges)
#         local_maxs = peak_local_max(distances, indices=False, min_distance=0)
#         markers = label(local_maxs)
#         labels = watershed(-distances, markers, mask=image)
#         labels[image < 0] = 0
#         self._ckp.save_segmentation_as_plot(labels, "segmentation.png")
#         return labels

#     def get_num_channels(self):
#         return 1

# =============================================================================
# LevelSegmentation
# Finds edeges by looking for specific iso-valued contours in color shape
# using linear interpolation for continuity. For each contour a boolean
# mask is created using the fast marching algorithm (boolean), multiplied
# by the label and added to the zero labels array.
# Problem: Inconstant object index assignment due to different orders of
#          detected contours
# =============================================================================
# class LevelSegmentation(Segmentation):

#     def __init__(self, config, ckp):
#         super(LevelSegmentation, self).__init__(config, ckp)

#     def __call__(self, image):
#         assert len(image.shape) == 3 and image.shape[2] == 3
#         w,h,_ = image.shape
#         gray = rgb2gray(image)
#         contours = find_contours(gray, 0.01)
#         labels = np.zeros((w,h), dtype=np.int8)
#         background_mask = np.sum(image, axis=2) < 0
#         labels[background_mask] = 0
#         for ic, contour in enumerate(contours):
#             mask = self._contour2mask(contour, labels.shape)
#             labels[mask] = (ic+1)
#         self._ckp.save_segmentation_as_plot(labels, "segmentation.png")
#         return labels

#     def _contour2mask(self, contour, img_shape):
#         mask = np.zeros(img_shape, dtype=np.bool)
#         contour = np.round(contour).astype(np.int)
#         mask[contour[:,0],contour[:,1]] = True
#         x_min, y_min = np.amin(contour[:,0]), np.amin(contour[:,1])
#         x_max, y_max = np.amax(contour[:,0]), np.amax(contour[:,1])
#         x_start = int((x_min + x_max)/2) # assuming approximatly convexity
#         y_start = int((y_min + y_max)/2) # assuming approximatly convexity
#         to_explore = [np.asarray([x_start, y_start])]
#         cardinal = np.asarray([[1,0],[-1,0],[0,1],[0,-1]])
#         i = 0
#         while len(to_explore) > 0:
#             e = to_explore.pop()
#             if not (0<=e[0]<img_shape[0] and 0<=e[1]<img_shape[1]): continue
#             if mask[e[0],e[1]]: continue
#             new_elements = e + cardinal
#             for ne in new_elements:
#                 if not (0<=ne[0]<img_shape[0] and 0<=ne[1]<img_shape[1]):
#                     continue
#                 if not mask[ne[0],ne[1]]: to_explore.append(ne)
#             mask[e[0],e[1]] = True
#             i = i+1
#         return mask

#     def get_num_channels(self):
#         return 1

# =============================================================================
# LevelGraySegmentation
# Based on the depth image this segmentation method distinguishes objects
# by contour detection and shape filling. In opposite to the LevelSegmentation
# it does not assign classes by enumeration but by taking the mean gray
# value of the filled shape, to avoid switching of classes of the same object
# in subsequent images due to a different order of returned contours.
# Problem: Changing mean gray value due to changing size and color (while
#          shifting around)
# =============================================================================
# class LevelGraySegmentation(Segmentation):

#     def __init__(self, config, ckp):
#         super(LevelGraySegmentation, self).__init__(config, ckp)

#     def __call__(self, image):
#         assert len(image.shape) == 3 and image.shape[2] == 3
#         w,h,_ = image.shape
#         gray = rgb2gray(image)
#         contours = find_contours(gray, 0.01)
#         labels = np.zeros((w,h), dtype=np.int8)
#         background_mask = np.sum(image, axis=2) < 0
#         labels[background_mask] = 0
#         for contour in contours:
#             mask = self._contour2mask(contour, labels.shape)
#             labels[mask] = int(np.mean(gray[mask]))
#         self._ckp.save_segmentation_as_plot(labels, "segmentation.png")
#         return labels

#     def _contour2mask(self, contour, img_shape):
#         mask = np.zeros(img_shape, dtype=np.bool)
#         contour = np.round(contour).astype(np.int)
#         mask[contour[:,0],contour[:,1]] = True
#         x_min, y_min = np.amin(contour[:,0]), np.amin(contour[:,1])
#         x_max, y_max = np.amax(contour[:,0]), np.amax(contour[:,1])
#         x_start = int((x_min + x_max)/2) # assuming approximatly convexity
#         y_start = int((y_min + y_max)/2) # assuming approximatly convexity
#         to_explore = [np.asarray([x_start, y_start])]
#         cardinal = np.asarray([[1,0],[-1,0],[0,1],[0,-1]])
#         i = 0
#         while len(to_explore) > 0:
#             e = to_explore.pop()
#             if not (0<=e[0]<img_shape[0] and 0<=e[1]<img_shape[1]): continue
#             if mask[e[0],e[1]]: continue
#             new_elements = e + cardinal
#             for ne in new_elements:
#                 if not (0<=ne[0]<img_shape[0] and 0<=ne[1]<img_shape[1]):
#                     continue
#                 if not mask[ne[0],ne[1]]: to_explore.append(ne)
#             mask[e[0],e[1]] = True
#             i = i+1
#         return mask

#     def get_num_channels(self):
#         return 1

# =============================================================================
# UniqueHueSegmentation
# Based on the background cleaned RGB image it is assumed that colors are
# more or less unique.  Therefore the unique elements of the image are
# filtered while dropping close colors. To further distinguish objects while
# keeping the output space comparably small the hue value the hue value
# (HSV color space) is used in order to further describe the objects, after
# applying nearest neighbour suppression so that similar hue values are
# detected as one class.
# Problem: No distance measure and ambiguous hue value (e.g. black, white, red)
# =============================================================================
# class UniqueHueSegmentation(Segmentation):

#     def __init__(self, config, ckp):
#         super(UniqueHueSegmentation, self).__init__(config, ckp)
#         self._max_color_distance = config.get("seg_max_color_dis", 10)/255.0

#     def __call__(self, image):
#         assert len(image.shape) == 3 and image.shape[2] == 3
#         w,h,_ = image.shape
#         hue = rgb2hsv(image)[:,:,0]
#         unique, counts = np.unique(hue, return_counts=True)
#         #unique, counts = np.unique(
#         #    np.reshape(image,(-1, 3)), axis=0, return_counts=True
#         #)   # very slow (maybe convert to HSV in before)
#         labels = np.zeros((w,h), dtype=np.int8)
#         for iu, u in enumerate(unique):
#             if np.sum(u) < 1e-3: continue
#             mask = np.zeros((w,h), dtype=np.int8)
#             #dist = np.sum(np.abs(image - u), axis=2)
#             dist = np.abs(hue - u)
#             mask[dist < self._max_color_distance] = iu+1
#             labels += mask
#         labels[np.mean(image, axis=2) < 0] = 0
#         self._ckp.save_segmentation_as_plot(labels, "segmentation.png")
#         return labels

#     def get_num_channels(self):
#         return 1
