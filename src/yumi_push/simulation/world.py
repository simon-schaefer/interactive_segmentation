#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michel Breyer, Simon Schaefer
# Description : Simulation world class (convenience wrapper for pybullet).
# =============================================================================
from enum import Enum
import os
import numpy as np
import random
import sys
import time

import pybullet
from pybullet_utils import bullet_client

import yumi_push.simulation.constants as constants

class World(object):
    """Convenience class for managing a simulated world.

    Attributes:
        physics_client: Client connected to a physics server.
        models: List of models registered to this world.
        sim_time: Simulated time since last World reset.
    """

    class Events(Enum):
        RESET = 0
        STEP = 1

    def __init__(self, config, ckp):
        # Create simulation from pybullet package and configure it
        # (visualizer depending on configuration).
        self._visualize = config.get("visualize", True)
        self.physics_client = bullet_client.BulletClient(
            pybullet.GUI if self._visualize else pybullet.DIRECT
        )
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=2.0, cameraYaw=0.0, cameraPitch=-89.5,
            cameraTargetPosition=(0.0,0.0,0.0)
        )
        self.physics_client.configureDebugVisualizer(
            pybullet.COV_ENABLE_SHADOWS,0)
        self.physics_client.configureDebugVisualizer(
            pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        self.physics_client.configureDebugVisualizer(
            pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
        self.physics_client.configureDebugVisualizer(
            pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
        self.physics_client.configureDebugVisualizer(
            pybullet.COV_ENABLE_GUI,0)
        # Crete auxialiary variables.
        self._models = []
        self._plane = None
        self._plane_id = config["workspace"]
        self._robot_uids = []
        self.sim_time = 0.0
        self._time_step = config.get("sim_dt", 1./240.)
        self._solver_iters = config.get("sim_solver_iters", 150)
        self._real_time = config.get('visualize', True)
        # Define scenes from arguments.
        self._scene      = scene_factory(config, ckp, self)
        self._scene_test = _Cubes2DFixed(config, ckp, self)
        self._ckp = ckp
        self.reset()

    def step(self):
        """Advance the simulation by one step."""
        if self._ckp.iteration <= 1 and self._visualize: return
        self.physics_client.stepSimulation()
        self.sim_time += self._time_step
        if self._real_time:
            time.sleep(max(0., self.sim_time - time.time() + self._real_start_time))

    def reset(self, test_scene=False):
        """Reset the world. This removes all models."""
        self.physics_client.resetSimulation()
        self.physics_client.setPhysicsEngineParameter(
            fixedTimeStep=self._time_step,
            numSolverIterations=self._solver_iters,
            enableConeFriction=1)
        self.physics_client.setGravity(0.0, 0.0, -9.81)
        self._models = []
        self._robot_uids = []
        self.sim_time = 0.0
        self._add_plane()
        self._real_start_time = time.time()
        if test_scene: self._scene_test.reset()
        else:          self._scene.reset()
        self.run(0.5)

    def run(self, duration):
        """ Run the simulation for the given duration. """
        for _ in range(int(duration / self._time_step)):
            self.step()

    def is_valid(self, safety_distance=0.1):
        """ Checks whether all models state are still valid, i.e. all
        objects still are in the workspace. """
        wo = constants.workspace_origin[self._plane_id]
        ws = constants.workspace_size[self._plane_id]
        sd = safety_distance
        for model in self.get_models():
            position, _ = model.get_pose()
            x, y, z = position
            is_valid = (wo[0]+sd < x < wo[0]+ws[0]-sd) \
                   and (wo[1]+sd < y < wo[1]+ws[1]-sd)
            if not is_valid: return False
        return True

    def add_model(self, model_path, position, orientation,
                  is_robot=False, is_plane=False, scaling=1.0, static=False):
        assert model_path.endswith(".urdf")
        model = _Model(self.physics_client)
        model.load(
            model_path, position, orientation, scaling,
            static, rand_color=not (is_robot or is_plane)
        )
        self._models.append(model)
        if is_robot: self._robot_uids.append(model.uid)
        return model

    def add_block(self, model_path, position, orientation,
                  is_robot=False, is_plane=False, scaling=1.0, static=False):
        assert model_path.endswith(".urdf")
        model = _Model(self.physics_client)
        model.load(
            model_path, position, orientation, scaling,
            static, rand_color=not (is_robot or is_plane)
        )
        model.set_dynamics(mass=0.01)
        self._models.append(model)
        if is_robot: self._robot_uids.append(model.uid)
        return model

    def _add_plane(self):
        urdf = "plane_{}.urdf".format(self._plane_id)
        urdf_path = os.path.join(os.environ["YUMI_PUSH_MODELS"], urdf)
        model = self.add_model(
            urdf_path, [0.0,0.0,-0.01],[0.0,0.0,0.0,1.0], is_plane=True
        )
        assert model.name == "plane"
        self._plane = model

    def remove_model(self, model):
        assert not model.uid == self._plane.uid
        assert not model.uid in self._robot_uids
        self.physics_client.removeBody(model.uid)
        self._models.remove(model)

    def get_contacts(self, model_A, model_B):
        """Return all contact points between the two given models."""
        result = self.physics_client.getContactPoints(model_A.uid, model_B.uid)
        contacts = []
        for contact in result:
            contacts.append({"position": contact[5],
                             "normal": contact[7],
                             "depth": contact[8],
                             "force": contact[9]})
        return contacts

    def get_objects(self):
        """ Return information about all non-plane/non-robot objects. """
        objects = []
        for m in self._models:
            if m.uid == self._plane.uid or m.uid in self._robot_uids:
                continue
            pose = m.get_pose()
            dims = tuple(m.scale*x for x in constants.object_sizes[m.name])
            objects.append((pose, dims))
        return objects

    def get_num_objects(self):
        return len(self.get_objects())

    def get_models(self):
        """ Return list of models, without the plane model. """
        return [x for x in self._models if not x.uid == self._plane.uid \
                                       and not x.uid in self._robot_uids]

    def get_plane_contacts(self):
        """ Return contact points of models with plane. """
        models = self.get_models()
        contacts = []
        for model in models:
            contacts.append(self.get_contacts(self._plane, model))
        return contacts

    def close(self):
        self.physics_client.disconnect()

# =========================================================================
# Manipulation robot wrapper for Pybullet.
# =========================================================================
class _Model(object):
    """Wrapper for manipulating robot models in Pybullet.

    Models consist of base body and chain of links connected through joints.

    Attributes:
        joints: A dict mapping joint names to Joint objects.
        links: A dict mapping link names to Link objects.
        name: The name of the model.
        uid: The unique id of the model within the physics server.
    """

    def __init__(self, physics_client):
        self._physics_client = physics_client

    def load(self, model_path, position, orientation,
             scaling=1.0, static=False, rand_color=True):
        """Load a robot model from its URDF file."""
        model_path = os.path.expanduser(model_path)
        self.uid = self._physics_client.loadURDF(
            model_path, position, orientation,
            globalScaling=scaling, useFixedBase=static)
        # if rand_color: self._physics_client.changeVisualShape(
        #    self.uid, -1, rgbaColor=self._random_color()
        # )
        if rand_color: self._physics_client.changeVisualShape(
            self.uid, -1, rgbaColor=(0.286,0.4,0.64,1.0)
        )
        self.name = self._physics_client.getBodyInfo(self.uid)[1].decode("utf-8")
        self.scale = scaling
        joints, links = {}, {}
        # Add every joint and link occuring in a link.
        for i in range(self._physics_client.getNumJoints(self.uid)):
            joint_info = self._physics_client.getJointInfo(self.uid, i)
            joint_name = joint_info[1].decode("utf8")
            joint_limits = {"lower": joint_info[8], "upper": joint_info[9],
                            "force": joint_info[10]}
            joints[joint_name] = _Joint(
                self._physics_client, self.uid, i, joint_limits)
            link_name = joint_info[12].decode("utf8")
            links[link_name] = _Link(self._physics_client, self.uid, i)
        self.joints, self.links = joints, links

    def _random_color(self):
        c = np.random.rand(1,3)
        if np.amax(c) < 0.5: c += 0.5
        return [*(c.tolist()[0]), 1.0]

    def get_pose(self):
        """Return the pose of the model base."""
        pos, quat = self._physics_client.getBasePositionAndOrientation(self.uid)
        return np.asarray(pos), np.asarray(quat)

    def set_pose(self, position, orientation):
        """Set the pose (position, orientation) of the model base."""
        self._physics_client.resetBasePositionAndOrientation(
            self.uid, position, orientation
        )

    def get_velocity(self):
        """Return the linear and angular velocity of model base. """
        linear, angular = self._physics_client.getBaseVelocity(self.uid)
        return np.asarray(linear), np.asarray(angular)

    def set_velocity(self, linear, angular):
        """Set linear and angular velocity of model base. """
        self._physics_client.resetBaseVelocity(
            self.uid, linear, angular
        )

    def apply_force(self, force_vec, force_base):
        """Apply force to model base which is fixed at given position. """
        self._physics_client.applyExternalForce(
            self.uid, -1, force_vec, force_base, pybullet.WORLD_FRAME
        )

    def get_dynamics(self):
        """Return information about internal dynamics such as name, index,
        physical properties such as friction coefficient, etc. """
        return self._physics_client.getDynamicsInfo(self.uid, -1)

    def set_dynamics(self, **kwargs):
        """Set internal dynamics parameters such as mass, friction, etc. """
        self._physics_client.changeDynamics(self.uid, -1, **kwargs)

class _Link(object):

    def __init__(self, physics_client, model_id, link_id):
        self._physics_client = physics_client
        self.model_id = model_id
        self.uid = link_id

    def get_pose(self):
        link_state = self._physics_client.getLinkState(self.model_id, self.uid)
        position, orientation = link_state[0], link_state[1]
        return np.asarray(position), np.asarray(orientation)

class _Joint(object):

    def __init__(self, physics_client, model_id, joint_id, limits):
        self._physics_client = physics_client
        self.model_id = model_id
        self.uid = joint_id
        self.limits = limits

    def get_position(self):
        joint_state = self._physics_client.getJointState(self.model_id, self.uid)
        return joint_state[0]

    def set_position(self, position):
        self._physics_client.setJointMotorControl2(
            self.model_id, self.uid,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.limits["force"])

# =========================================================================
# Implementation of different scenes.
# =========================================================================
class _Scenes(object):

    def __init__(self, config, ckp, world):
        self._world = world
        self._config = config
        self._ckp = ckp
        # Keep track of which positions are already occupied.
        # Has to filled and cleared in child classes (!).
        self._pos_occupied = []
        # Border limit i.e. Manhatten distance from plane border to
        # last permitted position.
        self._border_limit = self._config.get("scene_border_limit", 0.2)
        # Workspace constants.
        plane_id = self._config["workspace"]
        self._ws_o = constants.workspace_origin[plane_id]
        self._ws_s = constants.workspace_size[plane_id]

    def reset(self):
        raise NotImplementedError

    def cubes_random_smart_set(self, cube, min_objects, max_objects,
                               prob_neigh, cube_size_random=False):
        """ Randomly generate positions of [min-max] number of objects
        with probability prob_neigh to be next to each other. Thereby,
        keep attention to not place one cube at the same position twice.
        Return cube poses as list. """
        # Sample random number of objects and spawn them.
        n_objects  = np.random.randint(min_objects, max_objects + 1)
        ws_o, ws_s = self._ws_o, self._ws_s
        # For each object with certain probability either spawn the next
        # cube next to the current one or randomly somewhere (unoccupied).
        p             = prob_neigh
        next_is_neigh = False
        pos_last      = None
        poses         = []
        model_names   = [random.choice(list(constants.object_sizes.keys())) \
            for i in range(n_objects)] if cube_size_random else [cube]*n_objects
        models_used   = []
        for i in range(n_objects):
            pos = None
            trials, failed = 0, False
            cs = constants.object_sizes[model_names[i]][0] + 0.01
            while not self._is_valid_position(pos, cs) and not failed:
                if next_is_neigh:
                    rand_idx = np.random.binomial(1, 0.5)
                    rand_sgn = 1 if np.random.binomial(1, 0.5) else -1
                    pos = pos_last.copy()
                    pos[rand_idx] = pos[rand_idx] + rand_sgn*(cs_last + cs)/2.0
                else:
                    pos = np.r_[
                        int(np.random.uniform(ws_o[0],ws_o[0]+ws_s[0])/cs)*cs,
                        int(np.random.uniform(ws_o[1],ws_o[1]+ws_s[1])/cs)*cs,
                        0.1
                    ]
                trials = trials + 1
                failed = trials > 10
            if failed: continue
            orientation = np.r_[0, 0, 0, 1.0]
            poses.append((pos, orientation))
            models_used.append(model_names[i])
            self._pos_occupied.append((pos[0]+cs/2,pos[0]-cs/2,
                                       pos[1]+cs/2,pos[1]-cs/2))
            pos_last, cs_last = pos, cs
            next_is_neigh = (np.random.binomial(1, p) == 1)
        # Clean up class variables.
        self._pos_occupied = []
        return poses, models_used

    def _is_valid_position(self, position, cs):
        """ Checks whether position is valid to add on plane, including
        checking whether the position is occupied already,
        checking whether the position is within the plane,
        checking whether the position is None. """
        plane_id   = self._config["workspace"]
        ws_o, ws_s = self._ws_o, self._ws_s
        bl         = self._border_limit
        if position is None: return False
        for pos_occ in self._pos_occupied:
            valid = position[0] - cs/2 >= pos_occ[0]
            valid = valid or position[0] + cs/2 <= pos_occ[1]
            valid = valid or position[1] - cs/2 >= pos_occ[2]
            valid = valid or position[1] + cs/2 <= pos_occ[3]
            if not valid: return False
        if not (ws_o[0]+bl < position[0] < ws_o[0]+ws_s[0]-bl): return False
        if not (ws_o[1]+bl < position[1] < ws_o[1]+ws_s[1]-bl): return False
        return True

    def get_model_path(self, *subdir):
        return os.path.join(os.environ["YUMI_PUSH_MODELS"], *subdir)

class _EmptySurface(_Scenes):
    """ Flat surface with no objects. """

    def __init__(self, config, ckp, world):
        super(_EmptySurface, self).__init__(config, ckp, world)

    def reset(self):
        self._ckp.write_log("scene = empty")

class _Cubes2DFixed(_Scenes):
    """ Identical cubes being at fixed position (midth). Number is cubes is
    maximal number in configuration (scene_max_objects), the probability of cubes
    spawning next to each other is determined by configuration
    (scene_prob_neighbor). Cube configuration is determined once in the beginning
    and not changed afterwards. """

    def __init__(self, config, ckp, world):
        super(_Cubes2DFixed, self).__init__(config, ckp, world)
        np.random.seed(1)
        cube_model  = config.get("scene_cubes", "cube")
        num_objects = config.get("scene_max_objects", 6)
        self._poses, self._model_names = self.cubes_random_smart_set(
            cube_model, num_objects, num_objects,
            prob_neigh=config.get("scene_prob_neighbor", 0.5),
            cube_size_random=config.get("scene_cube_random_choice", False)
        )

    def reset(self):
        positions = []
        for i, pose in enumerate(self._poses):
            model = self.get_model_path(self._model_names[i] + ".urdf")
            self._world.add_model(model, pose[0], pose[1], scaling=1.0)
            positions.append((round(pose[0][0],2), round(pose[0][1],2)))
        self._world.run(self._config.get("scene_resting_time", 0.1))
        self._ckp.write_log("scene = cubes at positions {}".format(positions))

class _Cubes2DPreset(_Scenes):
    """ Identical cubes being at fixed position (midth). Number is cubes is
    maximal number in configuration (scene_max_objects), the probability of cubes
    spawning next to each other is determined by configuration
    (scene_prob_neighbor). Cube configuration is determined once in the beginning
    and not changed afterwards. """

    def __init__(self, config, ckp, world):
        super(_Cubes2DPreset, self).__init__(config, ckp, world)
        cube_model = config.get("scene_cubes", "cube")
        self._model = self.get_model_path(cube_model + ".urdf")
        num_objects = config.get("scene_max_objects", 6)
        prob_spawn_neighbor = config.get("scene_prob_neighbor", 0.5)

        pose_1 = np.array([-0.6,-0.6,0.1])
        pose_2 = np.array([0.6,0.6,0.1])
        orientation_1 = np.array([0, 0, 0, 1])
        orientation_2 = np.array([0, 0, 0, 1])
        poses = []
        poses.append([pose_1, orientation_1])
        poses.append([pose_2, orientation_2])
        self._poses = poses

    def reset(self):
        positions = []
        for pose in self._poses:
            self._world.add_model(self._model, pose[0], pose[1], scaling=1.0)
            positions.append((round(pose[0][0],2), round(pose[0][1],2)))
        self._world.run(self._config.get("scene_resting_time", 0.1))
        self._ckp.write_log("scene = cubes at positions {}".format(positions))

class _Cubes2DSmart(_Scenes):
    """ Randomly but smartly distributed identical cubes. Intelligent
    distribution of cubes includes to ensure that cubes are not in the
    same spot and to higher the probability of cubes next to each other. """

    def __init__(self, config, ckp, world):
        super(_Cubes2DSmart, self).__init__(config, ckp, world)
        self._cube = config.get("scene_cubes", "cube")
        self._max_objects = config.get("scene_max_objects", 6)
        self._min_objects = config.get("scene_min_objects", 1)
        self._prob_spawn_neighbor = config.get("scene_prob_neighbor", 0.5)
        self._cube_choice = config.get("scene_cube_random_choice", False)

    def reset(self):
        poses, model_names = self.cubes_random_smart_set(
            self._cube, self._min_objects, self._max_objects,
            self._prob_spawn_neighbor, self._cube_choice
        )
        positions = []
        for i, pose in enumerate(poses):
            model = self.get_model_path(model_names[i] + ".urdf")
            self._world.add_model(model, pose[0], pose[1], scaling=1.0)
            positions.append((round(pose[0][0],2), round(pose[0][1],2)))
        self._world.run(self._config.get("scene_resting_time", 0.1))
        self._ckp.write_log("scene = cubes at positions {}".format(positions))

def scene_factory(config, ckp, world):
    name = config.get("scene", "empty")
    if name == "empty":
        return _EmptySurface(config, ckp, world)
    elif name == "cubes_fixed":
        return _Cubes2DFixed(config, ckp, world)
    elif name == "cubes_smart":
        return _Cubes2DSmart(config, ckp, world)
    elif name == "cubes_preset":
        return _Cubes2DPreset(config, ckp, world)
    else:
        raise ValueError("Invalid scene {}".format(name))
