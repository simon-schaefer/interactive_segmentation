#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michel Breyer, Simon Schaefer
# Description : Various geometry primitives and transformation utilities.
#               This file mainly extends the existing utilies from the [tf]
#               (https://github.com/ros/geometry) package with conversions
#               between different formats.
# =============================================================================
import logging
import numpy as np

from yumi_push.common.transformations import *
from yumi_push.common import io_utils

# =============================================================================
# Tuple primitives (position, quat vector).
# =============================================================================
def tuple_addition(x,y):
    assert len(x) == len(y)
    z = [x[i]+y[i] for i in range(len(x))]
    return tuple(z)

def tuple_substraction(x,y):
    assert len(x) == len(y)
    z = [x[i]-y[i] for i in range(len(x))]
    return tuple(z)

def tuple_mult_scalar(x,a):
    z = [x[i]*a for i in range(len(x))]
    return tuple(z)

# =============================================================================
# Geometry primitives.
# =============================================================================
def from_pose(translation, rotation):
    """Create a transform from a translation vector and quaternion.

    Args:
        translation: A translation vector.
        rotation: A quaternion in the form [x, y, z, w].

    Returns:
        A 4x4 homogeneous transformation matrix.
    """
    transform = quaternion_matrix(rotation)
    transform[:3, 3] = translation
    return transform


def to_pose(transform):
    """Extract the translation vector and quaternion from the given transform.

    Args:
        transform: A 4x4 homogeneous transformation matrix.

    Returns:
        A translation vector and quaternion in the form [x, y, z, w].

    """
    translation = transform[:3, 3]
    rotation = quaternion_from_matrix(transform)
    return translation, rotation


def from_dict(serialized_transform):
    """Deserialize a transform from a Python dict.

    Args:
        serialized_transform (dict): A dict storing translation and rotation.

    Returns:
        A 4x4 homogeneous transformation matrix.

    Examples:
        >>> transform = {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]}
        >>> transform_utils.from_dict(transform)
        array([[1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])
    """
    translation = serialized_transform["translation"]
    rotation = serialized_transform["rotation"]
    return from_pose(translation, rotation)


def to_dict(transform):
    """Write a transform to a dict.

    Args:
        transform: A 4x4 homogeneous transformation matrix.

    Returns:
        A dict storing the transform.
    """
    translation, rotation = to_pose(transform)
    return {"translation": translation, "rotation": rotation}


def from_yaml(file_path):
    """Read a transform from a yaml file.

    Example of the content of such a file:

        transform:
            translation: [1., 2., 3.]
            rotation: [0., 0., 0., 1.]

    Args:
        file_path: The path to the YAML file.

    Returns:
        A 4x4 homogeneous transformation matrix.
    """
    cfg = io_utils.load_yaml(file_path)
    return from_dict(cfg["transform"])


def random_unit_vector(rand=None):
    if rand is None:
        rand = np.random.uniform(-1.0, 1.0, 3)
    v = rand / np.linalg.norm(rand)
    assert np.isclose(np.linalg.norm(v), 1., atol=1e-8)
    return v

# =============================================================================
# Plotting utilities.
# =============================================================================
def transparent_cmap(cmap, N=255):
    """ Create transparent color map from given colormap. """
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 1.0, num=N+4)
    return mycmap
