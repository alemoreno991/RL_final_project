from asyncio.log import logger
import pickle
from random import random
import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad
import numpy as np
import gym

import argparse

from spinup import td3_pytorch as td3

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int)
    parser.add_argument('--bbox', type=int, default=1.2)
    parser.add_argument('--epochs', type=int, default=22)
    args = parser.parse_args()
    
    env = quad.QuadrotorPlusHoverEnv(randomize_reset=True, env_bounding_box=args.bbox)
    env_fn = lambda : env

    bboxstr = ""
    if args.bbox != 1.2:
        bboxstr = "bbox{}".format(args.bbox)

    logger_kwargs_td3 = dict(output_dir='rand_init_td3{}{}'.format(args.run, bboxstr), exp_name='rand_init_quad_td3{}'.format(args.run))

    td3(env_fn=env_fn, logger_kwargs=logger_kwargs_td3, max_ep_len=5000, epochs=args.epochs)