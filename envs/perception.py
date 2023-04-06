from ast import Return
from json import tool
import re
from tkinter.messagebox import NO
from tkinter.tix import MAX
from matplotlib import colors
import rllib
import carla_utils as cu
from carla_utils import carla

import numpy as np
import cv2
import math
import open3d
from .params import MAP_RANGE, ACTION_DIM, FEASIBLE_REGION_RANGE, HISTORY_LENGTH, PIXEL_PER_METER, IF_VEHICLE_CHANNEL

from .tools import Transformation, down_sampling, viz_o3d, cv2_show_img, CoarseSimulator
    

class PerceptionSemanticLidar(object):
    
    if IF_VEHICLE_CHANNEL:
        dim_state = cu.basic.Data(
            ego=ACTION_DIM,
            route=40,
            region_map=(MAP_RANGE, MAP_RANGE), 
            obstacle_map=(MAP_RANGE, MAP_RANGE), 
            vehicle_map=(MAP_RANGE, MAP_RANGE)
            )
    else:
        dim_state = cu.basic.Data(
        ego=ACTION_DIM,
        route=40,
        region_map=(MAP_RANGE, MAP_RANGE), 
        obstacle_map=(MAP_RANGE, MAP_RANGE)
        )
    
    def __init__(self, config, **kwargs):

        self.perp_gt_route = cu.perception.GroundTruthRoute(config, self.dim_state.route, perception_range=30)
        
        self.all_transform_data = []
        self.all_obstacle_lidar_data = []
        self.all_vehicle_lidar_data = []
        
        self.simulator = CoarseSimulator(PIXEL_PER_METER)
        
        pass


    def run_step(self, step_reset, timestamp, agents):
        self.step_reset, self.timestamp = step_reset, timestamp

        agent = agents[0]
        current_transform = agent.get_transform()
        
        self.wheelbase = agent.wheelbase
        self.max_velocity = agent.max_velocity
        self.bounding_box = agent.bounding_box
        
        ego = self.get_state_ego(agent)
        route = self.perp_gt_route.run_step(agent)
        # left handed system => right handed system
        x, y, theta = current_transform.location.x, -current_transform.location.y, -np.deg2rad(current_transform.rotation.yaw)
        route[20:] *= -1

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
        
        points_road, points_obstacle, points_vehicle = self.lidar_preprocess(lidar_data)
        self.update_data_multiple_frames(x, y, theta, points_obstacle, points_vehicle)
        
        # Feasible Region
        region_map = self.points_to_region_map(points_road)
        
        # Obstacle
        obstacle_map = self.points_to_map(multi_frame=True, region_map=region_map)
                
        if IF_VEHICLE_CHANNEL:
            # Vehicle
            vehicle_map = self.points_to_map(multi_frame=True, is_vehicle=True)
            # self.cv2_save_img('vehicle_map', vehicle_map)
            cv2_show_img('vehicle_map', vehicle_map)
            return rllib.basic.Data(ego=ego, route=route, region_map=region_map, obstacle_map=obstacle_map, vehicle_map=vehicle_map)
        else:
            return rllib.basic.Data(ego=ego, route=route, region_map=region_map, obstacle_map=obstacle_map)


    def update_data_multiple_frames(self, x, y, theta, points_obstacle, points_vehicle):
        iter_num = HISTORY_LENGTH if self.timestamp == 1 else 1
        for _ in range(iter_num):
            if self.timestamp != 1:
                self.all_transform_data.pop(0)
                self.all_obstacle_lidar_data.pop(0)
                self.all_vehicle_lidar_data.pop(0)
            self.all_transform_data.append(np.array([x, y, theta]))
            self.all_obstacle_lidar_data.append(np.array(points_obstacle[:]))
            self.all_vehicle_lidar_data.append(np.array(points_vehicle[:]))
            

    def get_state_ego(self, agent):
        state = np.array([
                agent.get_state().v / agent.max_velocity,
                agent.steer_model.y
            ], dtype=np.float32)
        
        if ACTION_DIM == 1:
            return state[0]
        else:
            return state
            

    def lidar_preprocess(self, semantic_point_cloud):
        ''' PointCloud Preorocess'''     
        # print('semantic_point_cloud: ', semantic_point_cloud.shape)
        
        # Right-handed Coordinate System
        points = np.array([semantic_point_cloud['x'], -semantic_point_cloud['y'], semantic_point_cloud['z']]).T
        labels = np.array(semantic_point_cloud['ObjTag'])
        
        # Crop
        range_mask = np.logical_and(
            np.logical_and(
                np.abs(points[:, 0]) < FEASIBLE_REGION_RANGE, 
                np.abs(points[:, 1]) < FEASIBLE_REGION_RANGE
                ),
            np.abs(points[:, 2]) < 3 # -3 ~ 3
            )
        points = points[range_mask]
        labels = labels[range_mask]
        
        # DownSampling
        points, colors = down_sampling(points, labels)
                
        # Semantic Filter
        road_mask = np.all(colors == np.array([128, 64, 128]) / 255.0, 1)
        points_road = points[road_mask]
        colors_road = colors[road_mask]
        
        vehicle_mask = np.all(colors == np.array([0, 0, 142]) / 255.0, 1)
        points_vehicle = points[vehicle_mask]
        colors_vehicle = colors[vehicle_mask]
        
        obstacle_mask = ~road_mask
        points_obstacle = points[obstacle_mask]
        colors_obstacle = colors[obstacle_mask]
        
        # Visualization
        # viz_o3d(points_vehicle, colors_vehicle)
        
        return points_road, points_obstacle, points_vehicle
    
    
    def points_to_region_map(self, points):
        ''' PointCloud to Map'''
        map = np.zeros((MAP_RANGE, MAP_RANGE)).astype(np.float32)

        u = (- points[:, 0] * PIXEL_PER_METER + MAP_RANGE // 2).astype(int)
        v = (- points[:, 1] * PIXEL_PER_METER + MAP_RANGE // 2).astype(int)
    
        map[u, v] = 1
        kernel = np.ones((5,5), np.uint8)
        cv2.circle(map, (MAP_RANGE // 2, MAP_RANGE // 2), int(3.5 * PIXEL_PER_METER), 1, -1) # clear ego points
        map = cv2.dilate(map, kernel, iterations=3)

        return map


    def points_to_map(self, multi_frame=False, is_vehicle=False, region_map=None):
        
        map = np.ones((MAP_RANGE, MAP_RANGE)).astype(np.float32)
        
        if multi_frame == True:
            for i in range(HISTORY_LENGTH):
                if is_vehicle:
                    tmp_points = np.asarray(self.all_vehicle_lidar_data[i])
                else:
                    tmp_points = np.asarray(self.all_obstacle_lidar_data[i])
                
                points = Transformation(
                    tmp_points,
                    np.asarray(self.all_transform_data[i]),
                    np.asarray(self.all_transform_data[-1]),
                    FEASIBLE_REGION_RANGE
                    )
                u = (- points[:, 0] * PIXEL_PER_METER + MAP_RANGE // 2).astype(int)
                v = (- points[:, 1] * PIXEL_PER_METER + MAP_RANGE // 2).astype(int)
                map[u, v] = 1 - (i + 1) / HISTORY_LENGTH
        else:
            if is_vehicle:
                tmp_points = np.asarray(self.all_vehicle_lidar_data[-1])
            else:
                tmp_points = np.asarray(self.all_obstacle_lidar_data[-1])

            u = (- points[:, 0] * PIXEL_PER_METER + MAP_RANGE // 2).astype(int)
            v = (- points[:, 1] * PIXEL_PER_METER + MAP_RANGE // 2).astype(int)
            map[u, v] = 0
        
        if is_vehicle:
            kernel = np.ones((5,5), np.uint8)
        else:
            kernel = np.ones((3,3), np.uint8)
            
        map = cv2.erode(map, kernel, iterations=2)
        
        # Add Map
        if not is_vehicle:
            obs_mask = region_map < 1
            map[obs_mask] = 0
            
        return map


    def viz(self, data, if_show=False):
        
        region_map = data.region_map
        obstacle_map = data.obstacle_map
        
        # self.cv2_save_img('region_map', region_map)
        # self.cv2_save_img('obstacle_map', obstacle_map)
        
        if if_show:
            # Marked Image
            marked_map = obstacle_map[:]
            marked_map = cv2.cvtColor(marked_map, cv2.COLOR_GRAY2RGB)
            for i in range(20):
                x = data.route[i] * 30.
                y = data.route[i+20] * 30.
                # left handed system => right handed system
                cv2.circle(marked_map, (int(MAP_RANGE // 2 - y * PIXEL_PER_METER), int(MAP_RANGE // 2 - x * PIXEL_PER_METER)), 3, (0,0,255), -1)
            
            # Predicted Path
            # curvature = np.tan(steer) / wheelbase
            vel = data.ego[0] * self.max_velocity * PIXEL_PER_METER
            ang_vel = -data.ego[1] / self.wheelbase * vel
            path_x, path_y, path_theta = self.simulator.predict_state(vel, ang_vel, int(MAP_RANGE / 2), int(MAP_RANGE / 2), 0, 0.1, 10)
                
            for i in range(len(path_theta)):
                h = path_x[i].astype(np.int32)
                w = path_y[i].astype(np.int32)
                cv2.circle(marked_map, (w, h), 3, (255,0,0), -1)
                
            # ego
            cv2.rectangle(marked_map, 
                        (int(MAP_RANGE/2 - self.bounding_box.y * PIXEL_PER_METER), int(MAP_RANGE / 2 - self.bounding_box.x * PIXEL_PER_METER)),
                        (int(MAP_RANGE/2 + self.bounding_box.y * PIXEL_PER_METER), int(MAP_RANGE / 2 + self.bounding_box.x * PIXEL_PER_METER)),
                        (255, 255, 0),
                        -1
                        )
                
            scale = 2
            bar = cv2.cvtColor(np.ones((40, MAP_RANGE * scale)).astype(np.float32), cv2.COLOR_GRAY2RGB)
            concat_img = cv2.vconcat([
                cv2.resize(cv2.cvtColor(region_map, cv2.COLOR_GRAY2RGB), (MAP_RANGE * scale, MAP_RANGE * scale)),
                bar, 
                cv2.resize(marked_map, (MAP_RANGE * scale, MAP_RANGE * scale)), 
                bar, 
                cv2.resize(cv2.cvtColor(obstacle_map, cv2.COLOR_GRAY2RGB), (MAP_RANGE * scale, MAP_RANGE * scale) )
                ])
            
            cv2.putText(concat_img, f'feasible_region_map',
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(concat_img, f'marked_map',
                        (5, 55 + MAP_RANGE * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(concat_img, f'obstacle_map',
                        (5, 95 + MAP_RANGE * scale * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)    
            
            self.marked_map = marked_map
            cv2_show_img("concat_img", concat_img)
        
        
    def cv2_save_img(self, name, img, step_reset=True, timestep=True, iteration=-1):
        
        if step_reset == True and timestep == True and iteration == -1:
            img_name = '_'.join(['./results/tmp/'+ name, str(self.step_reset), str(self.timestamp), '.png'])
            cv2.imwrite(img_name, (img*255).astype(np.uint8))
        else:
            img_name = '_'.join(['./results/tmp/'+ name, str(self.step_reset), str(self.timestamp), iteration, '.png'])
            cv2.imwrite(img_name, (img*255).astype(np.uint8))
            

            

        
        