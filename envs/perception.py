import rldev
import carla_utils as cu
from carla_utils import carla

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import open3d
import cv2
from . import params
from .tools import get_vehicle_bbx_vec, save_bbx, save_pcd, Vis



class PerceptionSemanticLidar(object):
    dim_state = 1
    
    def __init__(self, config, **kwargs):
        self.config = config
        if config.render:
            self.vis = Vis()
        return


    def run_step(self, step_reset, timestamp, agent: cu.BaseAgent, vehicles: List[cu.BaseAgent]):
        self.step_reset, self.timestamp = step_reset, timestamp

        current_state = agent.get_state()
        obstacles = [o for o in vehicles if o.get_state().distance_xyz(current_state) < params.perception_range]

        ### bounding boxes
        obstacle_labels = []
        for o in obstacles:
            t = o.get_transform()
            t_local = cu.cvt.CarlaTransform.cua_state(o.get_state().world2local(current_state))
            t_local.rotation.roll = t.rotation.roll
            t_local.rotation.pitch = t.rotation.pitch
            bbx = o.vehicle.bounding_box

            bbx_vec = get_vehicle_bbx_vec(PseudoVehicle(o.vehicle, t_local, bbx))
            obstacle_labels.append(bbx_vec)
        obstacle_labels = np.stack(obstacle_labels, axis=0)

        ### point cloud
        sensor_lidar = agent.sensors_master['sensor.lidar.ray_cast_semantic', 'view']
        lidar_data: carla.SemanticLidarMeasurement = sensor_lidar.get_raw_data()
        lidar_data = np.fromstring(bytes(lidar_data.raw_data),
            dtype=np.dtype([
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('CosAngle', np.float32),
                ('ObjIdx', np.uint32),
                ('ObjTag', np.uint32),
            ])
        )
        # points = np.array([lidar_data['x'], -lidar_data['y'], lidar_data['z']]).T
        points = np.array([lidar_data['x'], lidar_data['y'], lidar_data['z']]).T

        image = agent.sensors_master.get_camera().data[...,:-1]
        # image = cv2.resize(image, self.dim_state.obs)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) /255
        plt.imshow(image)
        plt.pause(0.001)


        print('time step: ', step_reset, timestamp)

        if self.config.save:
            save_bbx(os.path.join(params.save_path, f'episode_{step_reset}/bbx_{timestamp}.npy'), obstacle_labels)
            save_pcd(os.path.join(params.save_path, f'episode_{step_reset}/pcd_{timestamp}.npy'), points)

        return rldev.Data(bbxs=obstacle_labels, points=points)


    def visualize(self, obstacle_bbx_vecs, points):
        self.vis.run_step(obstacle_bbx_vecs, points)


    def destroy(self):
        if self.config.render:
            self.vis.destroy()




class PseudoVehicle(object):
    def __init__(self, vehicle, t, bbx):
        self.attributes = vehicle.attributes
        self.type_id = vehicle.type_id
        self.transform = t
        self.bounding_box = bbx
    def get_transform(self):
        return self.transform

