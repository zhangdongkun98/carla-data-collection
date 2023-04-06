import rldev
import carla_utils as cu
from carla_utils import carla

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt



def generate_argparser():
    from carla_utils.utils import default_argparser
    argparser = default_argparser()

    argparser.add_argument('-d', dest='description', default='Nothing', help='[Method] description.')

    argparser.add_argument('--seed', default=0, type=int, help='seed.')
    argparser.add_argument('--num-episodes', default=20000, type=int, help='number of episodes.')
    
    argparser.add_argument('--evaluate', action='store_true', help='evaluate models (default: False)')
    return argparser


def run_one_episode(env):
    env.reset()
    while True:
        action = env.action_space.sample()
        experience, epoch_done, info = env.step(action)
        if env.config.render:
            env.render()
        
        if epoch_done == True:
            break
    return


if __name__ == "__main__":
    config = rldev.YamlConfig()
    args = generate_argparser().parse_args()
    config.update(args)

    mode = 'train'
    if config.evaluate == True:
        mode = 'evaluate'
        config.seed += 1
    rldev.setup_seed(config.seed)

    ### tensorboard log
    writer = rldev.create_dir(config, model_name='Demo', mode=mode)

    ### env
    from envs.env_v0 import Env_v0 as Env
    env = Env(config, writer, mode=mode, env_index=-1)



    print('\n' + rldev.prefix(__name__) + 'env: ', rldev.get_class_name(env))
    print('\nconfig: \n', config)
    try:
        cu.destroy_all_actors(config.core)
        for _ in range(config.num_episodes):
            run_one_episode(env)
    finally:
        writer.close()
        env.destroy()
