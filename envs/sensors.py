import carla_utils as cu
from carla_utils import carla

import numpy as np

from . import params


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

    {
        'type_id': 'sensor.camera.rgb',
        'role_name': 'view',
        'image_size_x': 640,
        'image_size_y': 360,
        'fov': 120,
        'transform':carla.Transform(carla.Location(x=0, z=22.8), carla.Rotation(pitch=-90, yaw=-90)),
        'callback': image_callback,
    },

    {
        'type_id': 'sensor.lidar.ray_cast_semantic',
        'role_name': 'view',
        'channels': params.channels,
        'range': params.perception_range,
        'points_per_second': params.points_per_second,
        'rotation_frequency': params.rotation_frequency,
        'upper_fov': 10.0,
        'lower_fov': -30.0,
        'horizontal_fov': 360.0,
        'transform': carla.Transform(carla.Location(x=0, z=params.lidar_z), carla.Rotation(pitch=0)),
        'callback': lidar_callback,
    },

]


