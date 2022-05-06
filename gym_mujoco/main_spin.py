from asyncio.log import logger
import pickle
from random import random
import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad
import numpy as np
import gym

import argparse

from spinup import sac_pytorch as sac
from spinup import ppo_pytorch as ppo
from spinup import td3_pytorch as td3

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int)
    args = parser.parse_args()
    
    env = quad.QuadrotorPlusHoverEnv(randomize_reset=True, env_bounding_box=5)
    env_fn = lambda : env

    # logger_kwargs_sac = dict(output_dir='sac{}'.format(args.run), exp_name='quad_sac{}'.format(args.run))
    # logger_kwargs_ppo = dict(output_dir='ppo{}'.format(args.run), exp_name='quad_ppo{}'.format(args.run))
    logger_kwargs_td3 = dict(output_dir='rand_init_td3{}'.format(args.run), exp_name='rand_init_quad_td3{}'.format(args.run))

    # sac(env_fn=env_fn, logger_kwargs=logger_kwargs_sac, max_ep_len=5000, epochs=70)
    # ppo(env_fn=env_fn, logger_kwargs=logger_kwargs_ppo, max_ep_len=5000, epochs=70)
    td3(env_fn=env_fn, logger_kwargs=logger_kwargs_td3, max_ep_len=5000, epochs=75)