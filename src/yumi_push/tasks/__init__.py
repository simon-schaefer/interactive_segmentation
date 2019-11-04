from yumi_push.networks.networks import *
from yumi_push.simulation import camera as sim_camera
from yumi_push.simulation import yumi as sim_robot
from yumi_push.simulation import world as sim_world
from yumi_push.tasks import sensors as task_sensors
from yumi_push.tasks import actuators as task_actuators
from yumi_push.tasks import task as task_task
from yumi_push.tasks import rewards as task_rewards


def init_task(config, cam_config, ckp):
    world = sim_world.World(config, ckp)
    camera = sim_camera.RGBDCamera(world.physics_client, cam_config)
    robot  = sim_robot.Robot2DHand(config, ckp, world)
    actuator = task_actuators.actuator_factory(config, ckp, robot)
    sensor = task_sensors.Sensor(config, ckp, camera=camera)
    reward = task_rewards.RewardFn(config,ckp,camera)
    task = task_task.Task(sensor,actuator,reward,world,config,ckp)
    return task, sensor, actuator
