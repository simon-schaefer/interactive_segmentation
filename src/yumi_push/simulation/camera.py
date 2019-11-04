# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michel Breyer, Simon Schaefer
# Description : Camera simulation using pybullet's renderer.
# =============================================================================
import numpy as np
import time

import pybullet

from yumi_push.simulation import constants as sim_const

class RGBDCamera(object):
    """OpenCV compliant camera model using PyBullet's built-in renderer.

    Attributes:
        info (CameraInfo): The intrinsics of this camera.
    """

    def __init__(self, physics_client, cam_config):
        self._physics_client = physics_client
        self.info = CameraInfo.from_dict(cam_config)
        self._near = cam_config['near']
        self._far = cam_config['far']
        self._camera_pos = cam_config['camera_position']
        self._target_pos = cam_config['target_position']
        self._up_direction = cam_config['up_direction']
        self.viewMatrix = pybullet.computeViewMatrix(
            self._camera_pos, self._target_pos, self._up_direction
        )
        self.projection_matrix = _build_projection_matrix(
            self.info.height, self.info.width,
            self.info.K, self._near, self._far
        )
        # World to camera transformation.
        R = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        t = self._camera_pos
        self.T_WC = np.zeros([4,4])
        self.T_WC[0:3,0:3] = R
        self.T_WC[0:3,3] = np.transpose(t)
        self.T_WC[3,3] = 1

    def render_images(self):
        """Render synthetic RGB and depth images.

        Args:
            view_matrix: The transform from world to camera frame.

        Returns:
            A tuple of RGB (height x width x 3 of uint8) and depth (heightxwidth
            of float32) images as well as a segmentation mask as np.array.
        """
        gl_view_matrix = self.viewMatrix
        gl_projection_matrix = self.projection_matrix.flatten(order='F')
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)
        result = self._physics_client.getCameraImage(
            width=self.info.width,
            height=self.info.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_projection_matrix,
            #shadow=0,
            #lightDirection=[1.0,0.0,2.0],
            #flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            #renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
            renderer=pybullet.ER_TINY_RENDERER)
        # Extract RGB image
        rgb = np.asarray(result[2], dtype=np.uint8)
        rgb = np.reshape(rgb, (self.info.height, self.info.width, 4))[:, :, :3]
        # Extract depth image
        near, far = self._near, self._far
        depth_buffer = np.asarray(result[3], np.float32).reshape(
            (self.info.height, self.info.width)
        )
        depth = 1.0 * far * near / (far - (far - near) * depth_buffer)
        # Extract segmentation mask
        mask = np.asarray(result[4], dtype=np.uint8)
        mask = np.reshape(mask, (self.info.height, self.info.width))
        # Reshape images to useful format.
        return rgb, depth, mask

    def px_to_m(self,px,Z):

        #input : px[y,x], Z
        #output: m[x,y] in world/actuator coordinate system
        K = self.info.K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        m = np.zeros((4,px.shape[0]),dtype=np.float32)
        x, y = px[:,1], px[:,0]
        m[1,:] = np.multiply((y-cy)/fy,Z)
        m[0,:] = np.multiply((x-cx)/fx,Z)
        m[3,:] = 1
        m = np.matmul(np.linalg.inv(self.T_WC), m)
        m =  m[0:2,:]
        m = np.transpose(m)
        return m
        # return np.reshape(m[0:2,:], (px.shape[0],2))

    def m_to_px(self,world_point):
        # input : m[x,y] in world/actuator coordinate system
        # output: px[y,x]
        assert len(world_point) == 2
        P = np.append(world_point,[0,1])
        K = self.info.K
        p = np.matmul(K,np.matmul(self.T_WC,P)[:3])
        p = np.divide(p, p[2])
        x, y = p[1], p[0]
        return np.array(np.round([x,y]),dtype=int)


def _gl_ortho(left, right, bottom, top, near, far):
    """Implementation of OpenGL's glOrtho subroutine."""
    ortho = np.diag([2./(right-left), 2./(top-bottom), - 2./(far-near), 1.])
    ortho[0, 3] = - (right + left) / (right - left)
    ortho[1, 3] = - (top + bottom) / (top - bottom)
    ortho[2, 3] = - (far + near) / (far - near)
    return ortho

def _build_projection_matrix(height, width, K, near, far):

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]


    # C = np.array([[fx, 0., cx, 0.],
    #             [0., fy, cy, 0.],
    #             [0., 0., 1, 0],
    #              [0., 0., 0, 1.]])
    # T = np.array([[1, 0, 0, 0],
    #               [0, 1, 0, 0],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]])
    #
    # # C = np.array([[K, np.transpose([0,0,0]),
    # #              [0, 0, 0,1]]])
    #
    #
    # perspective = np.matmul(C,T)
    perspective = np.array([[fx, 0., -cx, 0.],
                            [0., fy, -cy, 0.],
                            [0., 0., near + far, near * far],
                            [0., 0., -1., 0.]])

    ortho = _gl_ortho(0.0, width, 0.0, height, near, far)
    return np.matmul(ortho, perspective)

# =============================================================================
# Camera Info convenience class.
# =============================================================================
class CameraInfo(object):
    """Camera information similar to ROS sensor_msgs/CameraInfo.

    Attributes:
        height (int): The camera image height.
        width (int): The camera image width.
        K (np.ndarray): The 3x3 intrinsic camera matrix.
    """

    def __init__(self, height, width, K):
        """Initialize a camera info object."""
        self.height = height
        self.width = width
        self.K = K

    @classmethod
    def from_dict(cls, camera_info):
        """Construct a CameraInfo object from a dict.

        Args:
            camera_info (dict): A dict containing the height, width and
                intrinsics of a camera. For example:

                {'height': 480,
                 'width': 640,
                 'K': [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]}
        """
        height = camera_info['height']
        width = camera_info['width']
        K = np.reshape(camera_info['K'], (3, 3))
        return cls(height, width, K)

    def to_dict(self):
        """Store a camera info object to a dict.

        Returns:
            A dict containing the height, width and intrinsics. For example:

            {'height': 480,
             'width': 640,
             'K': [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]}
        """
        return {'height': self.height, 'width': self.width, 'K': self.K}
