################################################################################
# It is necessary to execute the following command before running this script
#
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
#
#
#   Reference: https://doi.org/10.48550/arXiv.1509.02971
#
################################################################################
import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad
from DDPG import DDPG
from plotter import live_plotter

import numpy as np
from datetime import datetime

env = quad.QuadrotorPlusHoverEnv()

params = {}
params["act_dim"] = 4
params["obs_dim"] = 18
params["lr_actor"]  = 1e-4
params["lr_critic"] = 1e-3
params["gamma"] = 0.995
params["tau"] = 0.01
params["action_noise_std"] = 0.4
params["num_episodes"] = 2000
params["batch_size"] = 128 


agent = DDPG(
    act_dim=params["act_dim"],
    obs_dim=params["obs_dim"],
    lr_actor=params["lr_actor"],
    lr_q_value=params["lr_critic"],
    gamma=params["gamma"],
    tau=params["tau"],
    action_noise_std=params["action_noise_std"]
)

nepisode = params["num_episodes"]
batch_size = params["batch_size"] 
iteration = 0
x = []
y = []
line1 = []
for i_episode in range(nepisode):
    done = False
    obs0 = env.reset()
    ep_rwd = 0

    while not done:
        #env.render()

        act = agent.step(obs0)

        obs1, rwd, done, _ = env.step(act)

        agent.memory.store_transition(obs0, act, rwd/10, obs1, done)

        obs0 = obs1
        ep_rwd += rwd

        if iteration >= batch_size * 3:
            agent.learn()

        iteration += 1

        
    print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)

    y.append(ep_rwd)
    x.append(i_episode)
    line1 = live_plotter(x,y,line1, identifier="DDPG", params=params)

now = datetime.now() # current date and time
date = now.strftime("%m-%d-%Y:%H:%M:%S")
live_plotter(x, y, line1, filename="img/DDPG_"+date+".png")

env.close()
