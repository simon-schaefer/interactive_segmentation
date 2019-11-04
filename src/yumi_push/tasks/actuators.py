#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michel Breyer, Simon Schaefer
# Description : Implementation of actuator (generalized) for defined task.
#               Depending on whether a simulation or the real-world Yumi
#               is used, the robot class can be switched. Both implementations
#               for the robot should follow the abstract "Robot" class to
#               ensure compability.
# =============================================================================
from enum import Enum
import gym
import gym.spaces
import numpy as np
from scipy.ndimage import distance_transform_edt

from yumi_push.common import transformations

# =============================================================================
# General implementation.
# =============================================================================
class Robot(object):

    def __init__(self):
        super(Robot, self).__init__()
        self._movement_last = ((0,0,0), (0,0,0))

    def goto_pose(self, position, orientation, log=True):
        """ Go directly to 3D position and orientation.
        For pure position control set orietation to None.
        The log flag means whether the last_movement variable
        should be updated by this movement or not. """
        raise NotImplementedError

    def goto_pose_delta(self, translation, yaw_rotation, log=True):
        """ Go from current position to next position by
        translating and rotation around z-axis.
        For pure position control set orietation to None.
        The log flag means whether the last_movement variable
        should be updated by this movement or not. """
        raise NotImplementedError

    def reset(self, position, orientation):
        """Reset robot joints to initial state. """
        raise NotImplementedError

    def get_pose(self):
        """Return pose of end-effector as tuple of tuples, i.e. (pose,
        orientation) with pose = (x,y,z) and orientation = (qx,qy,qz,qw). """
        raise NotImplementedError

    def get_last_movement(self):
        """Return the starting and target positions of the last
        logged movement of the robot. """
        return self._movement_last

    def get_state(self):
        """Return robot's state description (e.g. joint states). """
        raise NotImplementedError

    def get_workspace(self):
        """ Return robot's set of possible locations (i.e. workspace)
        in 2D as (x,y,z)-origin and (x,y,z)-space. """
        raise NotImplementedError

class Actuator(object):

    def __init__(self, robot, config, ckp):
        self.robot = robot
        self._ckp = ckp
        self.is_discrete = config.get("act_discrete", False)
        self.do_clipping = config.get("act_clipping", True)
        self._res = config.get("act_discrete_steps", 20)

    def reset(self):
        self.robot.reset()

    def execute_action(self, action, **kwargs):
        raise NotImplementedError

    def build_action_space(self, num_vars, reses):
        self._num_vars = num_vars
        if self.is_discrete:
            assert len(reses) == num_vars
            self._reses = reses
            space = 1
            for x in self._reses: space = space*x
            return gym.spaces.Discrete(space)
        else:
            return gym.spaces.Box(
                low=-1.0, high=1.0, shape=(num_vars,), dtype=np.float32
            )

    def unnormalize(self, action):
        if not self.is_discrete:
            assert action.size == self.action_space.shape[0]
            if self.do_clipping:
                action = np.clip(action, a_min=-1.0, a_max=1.0)
            else:
                action = np.tanh(action)
            assert (action >= -1.0).all() and (action <= 1.0).all()
        else:
            unnorm = self.undiscretize(action)
            for i in range(len(unnorm)):
                unnorm[i] = (unnorm[i]/self._reses[i]*2 - 1.0)
            action = np.asarray(unnorm)
        if len(action) == 3:
            action = np.asarray([action[1],-action[0],action[2]])
        elif len(action) == 2:
            action = np.asarray([action[1],-action[0]])
        elif len(action) == 4:
            action = np.asarray([action[1],-action[0],action[3],-action[2]])
        else:
            raise NotImplementedError("No action conversion known !")
        return action

    def undiscretize(self, action):
        assert action.size == 1
        assert action.dtype == int
        unnorm_action = []
        action_cpy = action.copy()
        for i in range(self._num_vars):
            divisor = int(np.prod(self._reses[i+1:]))
            x = action_cpy // divisor
            action_cpy -= x*divisor
            unnorm_action.append(x)
        return unnorm_action

# =============================================================================
# StartTargetActuator
# Robot hand following the action: start pos, target_pos.
# Action is assumed to be given normalized (i.e. in [0.0, 1.0]), conversion
# to real coordinates is done by the actuator. Action vector:
# 1) x-starting position
# 2) y-starting position
# 3) x-target position
# 4) y-target position
# =============================================================================
class StartTargetActuator(Actuator):

    def __init__(self, config, ckp, robot):
        super(StartTargetActuator, self).__init__(robot, config, ckp)
        self.action_space = self.build_action_space(4,reses=[self._res]*4)

    def execute_action(self, action, **kwargs):
        assert action.size == 4
        pinit = np.asarray([action[0], action[1], 0.0])
        ptarget = np.asarray([action[2], action[3], 0.0])
        self.robot.goto_pose(pinit, (0,0,0,1))
        self.robot.goto_pose_delta(ptarget - pinit, 0.0)
        self.robot.goto_pose(np.asarray([ptarget[0],ptarget[1],-10]),
                             np.asarray([0,0,0,1]))
        self._ckp.write_log(
            "action = (sx={:.3f}, sy={:.3f}, tx={:.3f}, ty={:.3f})".format(
            action[0], action[1], action[2], action[3]
        ))
        return action

    def unnormalize(self, action):
        action = super(StartTargetActuator, self).unnormalize(action)
        wo, ws = self.robot.get_workspace()
        sx = (action[0]+1)/2.0*ws[0] + wo[0]
        sy = (action[1]+1)/2.0*ws[1] + wo[1]
        tx = (action[2]+1)/2.0*ws[0] + wo[0]
        ty = (action[3]+1)/2.0*ws[1] + wo[1]
        return np.asarray([sx,sy,tx,ty])

# =============================================================================
# StartCardinalHalfActuator
# Robot hand following the action: start pos, "boolean" pushing direction.
# Action is assumed to be given normalized (i.e. in [0.0, 1.0]), conversion
# to real coordinates is done by the actuator. Action vector:
# 1) x-starting position
# 2) y-starting position
# 3) Pushing direction (up > 0, right < 0)
# =============================================================================
class StartCardinalHalfActuator(Actuator):

    def __init__(self, config, ckp, robot):
        super(StartCardinalHalfActuator, self).__init__(robot, config, ckp)
        self.action_space = self.build_action_space(3,[self._res,self._res,2])
        self._pushing_dis = config.get("act_pushing_dis", 0.5)

    def execute_action(self, action, **kwargs):
        assert action.size == 3
        pinit = np.asarray([action[0], action[1], 0.0])
        if action[2]: xt, yt = pinit[0], pinit[1] + self._pushing_dis
        else: xt, yt = pinit[0] + self._pushing_dis, pinit[1]
        ptarget = np.asarray([xt, yt, 0.0])
        self.robot.goto_pose(pinit, (0,0,0,1))
        self.robot.goto_pose_delta(ptarget - pinit, 0.0)
        self.robot.goto_pose(np.asarray([ptarget[0],ptarget[1],-10]),
                             np.asarray([0,0,0,1]))
        action = np.asarray([pinit[0],pinit[1],ptarget[0],ptarget[1]])
        self._ckp.write_log(
            "action = (sx={:.3f}, sy={:.3f}, tx={:.3f}, ty={:.3f})".format(
            action[0], action[1], action[2], action[3]
        ))
        return action

    def unnormalize(self, action):
        action = super(StartCardinalHalfActuator, self).unnormalize(action)
        wo, ws = self.robot.get_workspace()
        sx = (action[0]+1)/2.0*ws[0] + wo[0]
        sy = (action[1]+1)/2.0*ws[1] + wo[1]
        up = action[2] > 0
        return np.asarray([sx,sy,up])

    def directions(self):
        return ["right", "up"]

# =============================================================================
# StartCardinalActuator
# Robot hand following the action: start pos, "boolean" pushing direction.
# Action is assumed to be given normalized (i.e. in [0.0, 1.0]), conversion
# to real coordinates is done by the actuator. Action vector:
# 1) x-starting position
# 2) y-starting position
# 3) Pushing direction (-1 < left < -0.5, -0.5 < up < 0,
#                       0 < right < 0.5, 0.5 < down < 1.0)
# =============================================================================
class StartCardinalActuator(Actuator):

    def __init__(self, config, ckp, robot):
        super(StartCardinalActuator, self).__init__(robot, config, ckp)
        self.action_space = self.build_action_space(3,[self._res,self._res,4])
        self._pushing_dis = config.get("act_pushing_dis", 0.5)

    def execute_action(self, action, **kwargs):
        assert action.size == 3
        assert action[2] in [0,1,2,3]
        pinit = np.asarray([action[0], action[1], 0.0])
        if action[2] == 0: xt, yt = pinit[0] - self._pushing_dis, pinit[1]
        elif action[2] == 1: xt, yt = pinit[0], pinit[1] + self._pushing_dis
        elif action[2] == 2: xt, yt = pinit[0] + self._pushing_dis, pinit[1]
        elif action[2] == 3: xt, yt = pinit[0], pinit[1] - self._pushing_dis
        ptarget = np.asarray([xt, yt, 0.0])
        self.robot.goto_pose(pinit, (0,0,0,1))
        self.robot.goto_pose_delta(ptarget - pinit, 0.0)
        self.robot.goto_pose(np.asarray([ptarget[0],ptarget[1],-10]),
                             np.asarray([0,0,0,1]))
        action = np.asarray([pinit[0],pinit[1],ptarget[0],ptarget[1]])
        self._ckp.write_log(
            "action = (sx={:.3f}, sy={:.3f}, tx={:.3f}, ty={:.3f})".format(
            action[0], action[1], action[2], action[3]
        ))
        return action

    def unnormalize(self, action):
        action = super(StartCardinalActuator, self).unnormalize(action)
        wo, ws = self.robot.get_workspace()
        sx = (action[0]+1)/2.0*ws[0] + wo[0]
        sy = (action[1]+1)/2.0*ws[1] + wo[1]
        direction = 0
        if -1.0 <= action[2] < -0.5: direction = 0 #left
        if -0.5 <= action[2] < +0.0: direction = 1 #up
        if +0.0 <= action[2] < +0.5: direction = 2 #right
        if +0.5 <= action[2] <= 1.0: direction = 3 #down
        return np.asarray([sx,sy,direction])

    def directions(self):
        return ["left", "up", "right", "down"]

# =============================================================================
# StartOnlyActuator
# Robot hand following the action: start pos.
# Action is assumed to be given normalized (i.e. in [0.0, 1.0]), conversion
# to real coordinates is done by the actuator. Action vector:
# 1) x-starting position
# 2) y-starting position
# =============================================================================
class StartOnlyActuator(Actuator):

    def __init__(self, config, ckp, robot):
        super(StartOnlyActuator, self).__init__(robot, config, ckp)
        self.action_space = self.build_action_space(2,[self._res,self._res])

    def execute_action(self, action, **kwargs):
        assert action.size == 2
        pinit = np.asarray([action[0], action[1], 0.0])
        ptarget = pinit
        self.robot.goto_pose(pinit, (0,0,0,1))
        self.robot.goto_pose_delta(ptarget - pinit, 0.0)
        self.robot.goto_pose(np.asarray([ptarget[0],ptarget[1],-10]),
                             np.asarray([0,0,0,1]))
        action = np.asarray([pinit[0],pinit[1],ptarget[0],ptarget[1]])
        self._ckp.write_log(
            "action = (sx={:.3f}, sy={:.3f}, tx={:.3f}, ty={:.3f})".format(
            action[0], action[1], action[2], action[3]
        ))
        return action

    def unnormalize(self, action):
        action = super(StartOnlyActuator, self).unnormalize(action)
        wo, ws = self.robot.get_workspace()
        sx = (action[0]+1)/2.0*ws[0] + wo[0]
        sy = (action[1]+1)/2.0*ws[1] + wo[1]
        return np.asarray([sx,sy])

    def directions(self):
        return ["point"]

# =============================================================================
# StartOnlyActuator
# Simplified version of the start-target actuator above, only actuating the
# starting position of the pushing action, while the orientation and the
# pushing distance are defined by taking the closest object and pushing in
# its direction. Action vector:
# 1) x-starting position
# 2) y-starting position
# =============================================================================
# class StartOnlyActuator(Actuator):
#
#     def __init__(self, config, ckp, robot):
#         super(StartOnlyActuator, self).__init__(robot, config, ckp)
#         self.action_space = self.build_action_space(2)
#         self._dl = config.get("act_addtional_dis", 0.3)
#
#     def execute_action(self, action, **kwargs):
#         assert action.size == 2
#         assert "closest_object" in kwargs.keys()
#         assert kwargs["closest_object"].size == 2
#         pinit = np.asarray([action[0], action[1], 0.0])
#         ptarget = kwargs["closest_object"]
#         self.robot.goto_pose(pinit, (0,0,0,1))
#         t = np.asarray([ptarget[0]-pinit[0],ptarget[1]-pinit[1],0.0])
#         l = np.linalg.norm(t)
#         t = t/l*(l + self._dl) if l > 1e-2 else np.asarray([0,0,0])
#         ptarget = pinit + t
#         if(np.abs(ptarget[0])>0.8):
#             ptarget[0] = np.sign(ptarget[0])*0.8
#         if(np.abs(ptarget[1])>0.8):
#             ptarget[1] = np.sign(ptarget[0])*0.8
#         t = ptarget-pinit
#         self.robot.goto_pose_delta(t, 0.0)
#         self.robot.goto_pose(np.asarray([ptarget[0],ptarget[1],-10]),
#         np.asarray([0,0,0,1]))
#         self._ckp.write_log(
#             "action = (sx={:.3f}, sy={:.3f}, tx={:.3f}, ty={:.3f})".format(
#             pinit[0], pinit[1], ptarget[0], ptarget[1]
#         ))
#         action = (pinit[0], pinit[1], ptarget[0], ptarget[1])
#         return action
#
#     def unnormalize(self, action):
#         action = super(StartTargetActuator, self).unnormalize(action)
#         wo, ws = self.robot.get_workspace()
#         sx = (action[0]+1)/2.0*ws[0]*0.8 + wo[0]
#         sy = (action[1]+1)/2.0*ws[1]*0.8 + wo[1]
#         return np.asarray([sx,sy])

# =============================================================================
# Selector
# =============================================================================
def actuator_factory(config, ckp, robot):
    name = config.get("actuator", "start_target")
    if name == "start_target":
        return StartTargetActuator(config, ckp, robot)
    elif name == "start_cardinal_half":
        return StartCardinalHalfActuator(config, ckp, robot)
    elif name == "start_cardinal":
        return StartCardinalActuator(config, ckp, robot)
    elif name == "start_only":
        return StartOnlyActuator(config, ckp, robot)
    else:
        raise ValueError("Invalid actuator {}".format(name))


# =============================================================================
# StartDistActuator
# Simplified version of the start-target actuator above, only actuating the
# starting position of the pushing action, while the orientation is
# defined by taking the closest object and pushing in its direction until it
# reaches the object and pushes the additional distance beyond
# its direction. Action vector:
# 1) x-starting position
# 2) y-starting position
# 3) additional distance
# =============================================================================
# class StartDistActuator(Actuator):
#
#     def __init__(self, config, ckp, robot):
#         super(StartDistActuator, self).__init__(robot, config, ckp)
#         self._dl = config.get("act_addtional_dis", 0.3)
#
#     def reset(self):
#         self.robot.reset()
#
#     def execute_action(self, action, **kwargs):
#         assert action.size == 3
#         assert "closest_object" in kwargs.keys()
#         assert kwargs["closest_object"].size == 2
#         pinit = np.asarray([action[0], action[1], 0.0])
#         ptarget = kwargs["closest_object"]
#         self.robot.goto_pose(pinit, (0,0,0,1))
#         t = np.asarray([ptarget[0]-pinit[0],ptarget[1]-pinit[1],0.0])
#         l = np.linalg.norm(t)
#         t = t*(l + action[2])/l if l > 1e-2 else np.asarray([0,0,0])
#         ptarget = pinit + t
#         if(np.abs(ptarget[0])>0.8):
#             ptarget[0] = np.sign(ptarget[0])*0.8
#         if(np.abs(ptarget[1])>0.8):
#             ptarget[1] = np.sign(ptarget[0])*0.8
#         t = ptarget-pinit
#         self.robot.goto_pose_delta(t, 0.0)
#         self.robot.goto_pose(np.asarray([ptarget[0],ptarget[1],-10]),
#         np.asarray([0,0,0,1]))
#         self._ckp.write_log(
#             "action = (sx={:.3f}, sy={:.3f}, tx={:.3f}, ty={:.3f})".format(
#             pinit[0], pinit[1], ptarget[0], ptarget[1]
#         ))
#         action = (pinit[0], pinit[1], ptarget[0], ptarget[1])
#         return action
#
#     def unnormalize(self, action):
#         assert action.size == 3
#         assert (action >= -1.0).all() and (action <= 1.0).all()
#         print(action)
#         wo, ws = self.robot.get_workspace()
#         x = action[0]*0.8
#         y = action[1]*0.8
#         d = (action[2]+1)/2*0.3 + 0.1
#         return np.asarray([x,y,d])
