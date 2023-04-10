decision_frequency = 20
control_frequency = 20
perception_range = 80

num_steps = 1000
num_vehicles = 100

### lidar
lidar_z = 2.8  ### transform
channels = 64
rotation_frequency = 40
points_per_channel_each_step = 600
points_per_second = points_per_channel_each_step * rotation_frequency * channels


def generate_argparser():
    from carla_utils.utils import default_argparser
    argparser = default_argparser()

    argparser.add_argument('-d', dest='description', default='Nothing', help='[Method] description.')

    argparser.add_argument('--seed', default=0, type=int, help='seed.')
    argparser.add_argument('--num-episodes', default=10, type=int, help='number of episodes.')
    argparser.add_argument('--save', action='store_true', help='save_data')
    
    argparser.add_argument('--evaluate', action='store_true', help='evaluate models (default: False)')
    return argparser



save_path = './raw_data'
