#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michel Breyer, Simon Schaefer
# Description : Implementation of simulated yumi.
# =============================================================================
import numpy as np
import os
import pybullet

from yumi_push.common import misc, transformations
from yumi_push.simulation import constants as sim_consts
from yumi_push.tasks.actuators import Robot

# =============================================================================
# Robot hand implementation (not connected to origin i.e. can reach every
# destination in action space directly without joints).
# =============================================================================
class Robot2DHand(Robot):

    def __init__(self, config, ckp, world):
        super(Robot2DHand, self).__init__()
        self._world = world
        self._config = config
        self._ckp = ckp
        self.reset()

    def reset(self):
        """Load an URDF model of our robot and add it to the virtual world."""
        urdf=os.path.join(os.environ["YUMI_PUSH_MODELS"],"robot_hand.urdf")
        self._model = self._world.add_model(
            model_path=urdf,
            position=[-10.0, -10.0, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0],
            is_robot=True)
        self._model.set_dynamics(mass=self._config.get("act_mass", 10.0),
        lateralFriction=0,spinningFriction=10,rollingFriction=10,
        linearDamping=0,angularDamping=0)

    def get_pose(self):
        """Returns the current position of the robot tool."""
        return self._model.get_pose()

    def get_state(self):
        """Returns the current position of the robot tool."""
        return self.get_pose()

    def get_workspace(self):
        """ Return robot's set of possible locations (i.e. workspace)
        in 2D as (x,y,z)-origin and (x,y,z)-space. """
        wid = self._config["workspace"]
        return sim_consts.workspace_origin[wid], sim_consts.workspace_size[wid]

    def goto_pose(self, position, orientation, log=True, discrete=True):
        assert len(position) == 3 and len(orientation) == 4
        if log: self._movement_last = (self.get_pose()[0], position)
        self._world.physics_client.resetBasePositionAndOrientation(
        self._model.uid, posObj=position, ornObj=orientation)
        self._world.run(0.2)


    def goto_pose_delta(self, translation, yaw_rotation, log=True):
        assert len(translation) == 3
        assert type(yaw_rotation) == float
        position, orientation = self._model.get_pose()
        _, _, yaw = transformations.euler_from_quaternion(orientation)
        # Compute the new target pose of the gripper in world frame
        v_goal = 0.5
        d_trans = np.linalg.norm(translation)
        v_time = d_trans/v_goal
        trans_unit  =  translation/d_trans
        v_move = trans_unit * v_goal
        self._world.physics_client.resetBaseVelocity(self._model.uid,
        linearVelocity=v_move,angularVelocity=[0.000,0,0])
        self._world.run(v_time)
        # position, orientation = self._model.get_pose()
        # print("real pos: ", position)
