decision_frequency = 20
control_frequency = 20
perception_range = 50

num_steps = 1000
num_vehicles = 100

### lidar transform
lidar_z = 2.8


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
