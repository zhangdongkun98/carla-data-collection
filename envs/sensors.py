import carla_utils as cu
from carla_utils import carla

import numpy as np

from .params import perception_range, lidar_z


def image_callback(weak_self, data):
    # data: carla.Image
    self = weak_self()
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    self.raw_data = data
    self.data = array



lidar_callback = cu.sensor.sensor_callback.DefaultCallback.sensor_lidar_ray_cast_semantic


sensors_param_list = [
    {
        'type_id': 'sensor.other.collision',
        'role_name': 'default',
        'transform': carla.Transform(carla.Location(x=2.5, z=0.7)),
    },

    # {
    #     'type_id': 'sensor.camera.rgb',
    #     'role_name': 'view',
    #     'image_size_x': 640,
    #     'image_size_y': 360,
    #     'fov': 120,
    #     'transform':carla.Transform(carla.Location(x=0, z=2.8), carla.Rotation(pitch=-5)),
    #     'callback': image_callback,
    # },

    {
        'type_id': 'sensor.lidar.ray_cast_semantic',
        'role_name': 'view',
        'channels': 32,
        'range': perception_range,
        'points_per_second': 100000,
        'rotation_frequency': 40.0,
        'upper_fov': 10.0,
        'lower_fov': -30.0,
        'horizontal_fov': 360.0,
        'transform': carla.Transform(carla.Location(x=0, z=lidar_z), carla.Rotation(pitch=0)),
        'callback': lidar_callback,
    },

]


