import argparse
import numpy as np
from termcolor import colored

import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad

from spinup import sac_pytorch as sac
from spinup import ppo_pytorch as ppo
from spinup import td3_pytorch as td3
from spinup import ddpg_pytorch as ddpg
from spinup import vpg_pytorch as vpg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # CLI to configure output data path
    parser.add_argument('--filename', type=str)
    
    # CLI to configure the number of runs
    parser.add_argument('--runs', type=int, default=1)
    
    # CLI to configure the maximum number of episodes and epochs
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--episodes', type=int, default=3000)
    
    # CLI to tune the environment
    parser.add_argument('--randomize_reset', action='store', default=False, const=True, nargs="?")
    parser.add_argument('--disorient', action='store', default=False, const=True, nargs="?")
    parser.add_argument('--observation_noise_std', type=float, default=0)
    parser.add_argument('--env_bounding_box', type=float, default=1.2)
    parser.add_argument('--init_max_vel', type=float, default=0.5)
    parser.add_argument('--init_max_angular_vel', type=float, default=0.1*np.pi)
    parser.add_argument('--init_max_attitude', type=float, default= np.deg2rad(60))
    parser.add_argument('--bonus_to_reach_goal', type=float, default=15.0)
    parser.add_argument('--max_reward_for_velocity_towards_goal', type=float, default=2.0)
    parser.add_argument('--position_reward_constant', type=float, default=5.0)
    parser.add_argument('--orientation_reward_constant', type=float, default=0.02)
    parser.add_argument('--linear_velocity_reward_constant', type=float, default=0.01)
    parser.add_argument('--angular_velocity_reward_constant', type=float, default=0.001)
    parser.add_argument('--action_reward_constant', type=float, default=0.0025)
    parser.add_argument('--reward_for_staying_alive', type=float, default=5.0)
    
    # CLI to configure the algorithm we wanna train the agent with
    parser.add_argument('--SAC', action='store', default=False, const=True, nargs="?")
    parser.add_argument('--TD3', action='store', default=False, const=True, nargs="?")
    parser.add_argument('--DDPG', action='store', default=False, const=True, nargs="?")
    parser.add_argument('--PPO', action='store', default=False, const=True, nargs="?")
    parser.add_argument('--VPG', action='store', default=False, const=True, nargs="?")
    args = parser.parse_args()

    # Configure the environment as defined by the user CLI
    env = quad.QuadrotorPlusHoverEnv(
        randomize_reset=args.randomize_reset,
        disorient=args.disorient,
        observation_noise_std=args.observation_noise_std,
        env_bounding_box=args.env_bounding_box,
        init_max_vel=args.init_max_vel,
        init_max_angular_vel=args.init_max_angular_vel,
        init_max_attitude=args.init_max_attitude,
        bonus_to_reach_goal=args.bonus_to_reach_goal,
        max_reward_for_velocity_towards_goal=args.max_reward_for_velocity_towards_goal,
        position_reward_constant=args.position_reward_constant,
        orientation_reward_constant=args.orientation_reward_constant,
        linear_velocity_reward_constant=args.linear_velocity_reward_constant,
        angular_velocity_reward_constant=args.angular_velocity_reward_constant,
        action_reward_constant=args.action_reward_constant,
        reward_for_staying_alive=args.reward_for_staying_alive
    )
    env_fn = lambda : env


    for i in range(args.runs):
        logger_kwargs = dict(
            output_dir='{}_{}'.format(args.filename, i), 
            exp_name='quad_{}'.format(i)
        )
    
        if True == args.VPG:
            print(colored('VPG...', 'blue'))
            vpg(env_fn=env_fn, 
                logger_kwargs=logger_kwargs, 
                max_ep_len=args.episodes, 
                epochs=args.epochs
            )

        if True == args.PPO:
            print(colored('PPO...', 'blue'))
            ppo(env_fn=env_fn, 
                logger_kwargs=logger_kwargs, 
                max_ep_len=args.episodes, 
                epochs=args.epochs
            )

        if True == args.DDPG:
            print(colored('DDPG...', 'blue'))
            ddpg(env_fn=env_fn, 
                logger_kwargs=logger_kwargs, 
                max_ep_len=args.episodes, 
                epochs=args.epochs
            )

        if True == args.TD3:
            print(colored('TD3...', 'blue'))
            td3(env_fn=env_fn, 
                logger_kwargs=logger_kwargs, 
                max_ep_len=args.episodes, 
                epochs=args.epochs
            )

        if True == args.SAC:
            print(colored('SAC...', 'blue'))
            sac(env_fn=env_fn, 
                logger_kwargs=logger_kwargs, 
                max_ep_len=args.episodes, 
                epochs=args.epochs
            )
