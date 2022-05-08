#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Definition of simulation constants.
# =============================================================================

# Mapping object names (by definition in urdf file) to their
# dimensions in (x,y,z). It is thereby assumed that the object
# has similar dimensions visually as well as inertially.
object_sizes = {
    "cube" : (0.2, 0.2),
    "cube_large": (0.4, 0.4)
}

# Workspace (plane) size and origin.
workspace_origin = {
    "small": (-0.5, -0.5, 0.0),
    "large": (-1.0, -1.0, 0.0),
}
workspace_size = {
    "small": (1.0, 1.0, 0.0),
    "large": (2.0, 2.0, 0.0),
}
