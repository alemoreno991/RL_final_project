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

env = quad.QuadrotorPlusHoverEnv()

agent = DDPG(
    act_dim=4, 
    obs_dim=18,
    lr_actor  =0.00003, 
    lr_q_value=0.0002, 
    gamma=0.8, 
    tau=0.01, 
    action_noise_std=0.3
)

nepisode = 1800
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

        agent.memory.store_transition(obs0, act, rwd, obs1, done)

        obs0 = obs1
        ep_rwd += rwd

        if iteration >= 128 * 3:
            agent.learn()

        iteration += 1

        
    print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)

    y.append(ep_rwd)
    x.append(i_episode)
    line1 = live_plotter(x,y,line1)

live_plotter(x, y, line1, filename="img/DDPG.png")
env.close()
