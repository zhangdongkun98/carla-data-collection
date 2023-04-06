import cv2
import utm
import math
import open3d
import numpy as np
from PIL import Image
from .params import MAP_RANGE, ACTION_DIM, FEASIBLE_REGION_RANGE_SCOUT


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
LABEL_COLORS = np.array([
    (255, 255, 255), # 0 None
    (70, 70, 70),    # 1 Building
    (100, 40, 40),   # 2 Fences
    (55, 90, 80),    # 3 Other
    (220, 20, 60),   # 4 Pedestrian
    (153, 153, 153), # 5 Pole
    (157, 234, 50),  # 6 RoadLines
    (128, 64, 128),  # 7 Road
    (244, 35, 232),  # 8 Sidewalk
    (107, 142, 35),  # 9 Vegetation
    (0, 0, 142),     # 10 Vehicle
    (102, 102, 156), # 11 Wall
    (220, 220, 0),   # 12 TrafficSign
    (70, 130, 180),  # 13 Sky
    (81, 0, 81),     # 14 Ground
    (150, 100, 100), # 15 Bridge
    (230, 150, 140), # 16 RailTrack
    (180, 165, 180), # 17 GuardRail
    (250, 170, 30),  # 18 TrafficLight
    (110, 190, 160), # 19 Static
    (170, 120, 50),  # 20 Dynamic
    (45, 60, 150),   # 21 Water
    (145, 170, 100), # 22 Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


def gps2xy(lat, lon):
    return utm.from_latlon(lat, lon)[:2]


def xy2gps(x, y):
    return utm.to_latlon(x, y, 51, "R")


def direction_estimate(balloon_position, scout_position):

    position_1 = [222845.3948031321, 3351756.9051995813]
    position_2 = [222912.4333699645, 3351729.3134000297]
    position_0 = [222879.0291507287, 3351743.1054393286]

    k = (position_2[1] - position_1[1]) / (position_2[0] - position_1[0])
    b_balloon = balloon_position[1] - k * balloon_position[0]
    b_scout = scout_position[1] - k * scout_position[0]

    return b_balloon - b_scout > 0

def calc_yaw(orientation):
    
    atan2_y = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
    atan2_x = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
    yaw = math.atan2(atan2_y , atan2_x)
    # print("yaw : ", np.rad2deg(yaw))
    
    return np.rad2deg(yaw)


def cv2_rotate(img, yaw):
    
    # matrix
    M = cv2.getRotationMatrix2D((MAP_RANGE/2, MAP_RANGE/2), -yaw, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # new bound
    nW = int((MAP_RANGE * sin) + (MAP_RANGE * cos))
    nH = int((MAP_RANGE * cos) + (MAP_RANGE * sin))
    
    # translate
    M[0, 2] += (nW / 2) - MAP_RANGE / 2
    M[1, 2] += (nH / 2) - MAP_RANGE / 2
    
    img = cv2.warpAffine(img, M, (nW, nH))

    # center crop
    x = img.shape[0] / 2 - MAP_RANGE / 2
    y = img.shape[1] / 2 - MAP_RANGE / 2
    
    return img[int(x):int(x+MAP_RANGE), int(y):int(y+MAP_RANGE)]

            
def cv2_show_img(name, img):
        
    cv2.imshow(name, img)
    cv2.waitKey(10)


def PIL_rotate(img, yaw):
    
    img = Image.fromarray(np.array(img))
    img = np.array(img.rotate(-yaw))
    
    return img


def down_sampling(points_cloud, labels, voxel_size=1.5):
    
    down_points_cloud = open3d.geometry.PointCloud()
    down_points_cloud.points = open3d.utility.Vector3dVector(points_cloud)
    if labels is not None:
        point_color = LABEL_COLORS[labels]
        down_points_cloud.colors = open3d.utility.Vector3dVector(point_color)
    down_points_cloud = down_points_cloud.voxel_down_sample(voxel_size=voxel_size)
    
    return np.asarray(down_points_cloud.points), np.asarray(down_points_cloud.colors)
      

def viz_o3d(points, colors=None):

    point_viz = open3d.geometry.PointCloud()
    point_viz.points = open3d.utility.Vector3dVector(points)
    
    if colors is not None:
        point_viz.colors = open3d.utility.Vector3dVector(colors)
    
    open3d.visualization.draw_geometries([point_viz])


def Transformation(lidar_current, current_transform, target_transform, feasible_range):
    
    '''
    right handed system
    input:
        numpy.ndarray
    output:
        numpy.ndarray
    '''
    
    lidar_coordinates_current = np.stack([lidar_current[:, 0], lidar_current[:, 1], np.ones(lidar_current.shape[0])], axis=0)

    cos_target = math.cos(target_transform[2])
    sin_target = math.sin(target_transform[2])
    T_target = np.array([
        [cos_target, -sin_target, target_transform[0]],
        [sin_target,  cos_target, target_transform[1]],
        [         0,           0,                  1]])

    cos_current = math.cos(current_transform[2])
    sin_current = math.sin(current_transform[2])
    T_current = np.array([
        [cos_current, -sin_current, current_transform[0]],
        [sin_current,  cos_current, current_transform[1]],
        [          0,            0,                   1]])

    T_target_inv = np.linalg.inv(T_target)
    T_current_target = T_target_inv.dot(T_current)

    lidar_coordinates_target = T_current_target.dot(lidar_coordinates_current)
    
    range_mask = np.logical_and(
        np.abs(lidar_coordinates_target[0]) < feasible_range,
        np.abs(lidar_coordinates_target[1]) < feasible_range
        )
    
    return (lidar_coordinates_target[0:2, :].T)[range_mask]


def Coordinate_Transformation(d_x, d_y, current_x, current_y, current_yaw, target_x, target_y, target_yaw):

    T_target = np.array([
        [math.cos(target_yaw), -math.sin(target_yaw), target_x],
        [math.sin(target_yaw),  math.cos(target_yaw), target_y],                  
        [                   0,                     0,        1]])

    T_current = np.array([
        [math.cos(current_yaw), -math.sin(current_yaw), current_x], 
        [math.sin(current_yaw),  math.cos(current_yaw), current_y],
        [                    0,                      0,         1]])

    T_target_inv = np.linalg.inv(T_target)
    T_current_to_target = T_target_inv.dot(T_current)

    coordinates_current = [d_x, d_y, 1]
    coordinates_target = T_current_to_target.dot(coordinates_current)

    return coordinates_target[0], coordinates_target[1]


def map2points(obs_map):

    position = np.where(obs_map > 0.8)
    arg_h = position[0]
    arg_w = position[1]
    p_x = MAP_RANGE / 2 - arg_h
    p_y = MAP_RANGE / 2 - arg_w
    
    p_x = p_x[np.where(p_x>0)]
    p_y = p_y[np.where(p_x>0)]

    # Nearest Points
    laser_len = 180
    laser = np.ones([3, laser_len]) * MAP_RANGE / 2
    laser[2, :] = 1e6
    
    for i in range(len(p_x)):
        tmp_angle = math.atan2(p_y[i], p_x[i]) # -pi /2 ~ pi / 2
        int_angle = math.floor((tmp_angle + math.pi / 2) / math.pi * laser_len)

        dis2 = p_x[i] ** 2 + p_y[i] ** 2
        if(dis2 < laser[2, int_angle]):
            laser[0, int_angle] = p_x[i]
            laser[1, int_angle] = p_y[i]
            laser[2, int_angle] = dis2
            
    points = (laser[0:2].T).astype(np.float32)
    
    # DBSCAN
    # points = np.vstack([p_x,p_y]).astype(np.float32)
    # points = points.T
    # C = DBSCAN.DBSCAN_points(points)
    
    return points


def marginal_laser(points):

    laser_len = 180
    marginal_laser = np.ones([3, laser_len]) * FEASIBLE_REGION_RANGE_SCOUT
    marginal_laser[2, :] = 1e6
    
    for i in range(len(points[:, 0])):
        tmp_angle = math.atan2(points[:, 1][i], points[:, 0][i]) # -pi /2 ~ pi / 2
        index_angle = math.floor((tmp_angle + math.pi / 2) / math.pi * laser_len) % 180

        dis2 = points[:, 0][i] ** 2 + points[:, 1][i] ** 2
        if(dis2 < marginal_laser[2, index_angle]):
            marginal_laser[0, index_angle] = points[:, 0][i]
            marginal_laser[1, index_angle] = points[:, 1][i]
            marginal_laser[2, index_angle] = dis2
    points = (marginal_laser[0:2].T).astype(np.float32)
    
    return points

class CoarseSimulator():
    def __init__(self, pixel_per_meter, max_acc=0, max_ang_acc=0):
        super(CoarseSimulator, self).__init__()
        self.max_acc = max_acc  # m/s^2
        self.max_ang_acc = max_ang_acc  # rad/s^2
        self.pixel_per_meter = pixel_per_meter
    
    def predict_state(self, vel, ang_vel, x, y, th, dt, pre_step):
        next_xs = []
        next_ys = []
        next_ths = []

        for _ in range(pre_step):
            x = - vel * np.cos(th) * dt  * self.pixel_per_meter + x
            y = - vel * np.sin(th) * dt  * self.pixel_per_meter + y
            th = ang_vel * dt + th

            next_xs.append(x)
            next_ys.append(y)
            next_ths.append(th)
            
        return next_xs, next_ys, next_ths
    
    
    