################################################################################
# It is necessary to execute the following command before running this script
#
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
#
################################################################################
import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad
from PPO import PPO

import numpy as np
from plotter import live_plotter

env = quad.QuadrotorPlusHoverEnv()

agent = PPO(
    act_dim=4, 
    obs_dim=18,
    lr_actor=0.00001, 
    lr_value=0.00025, 
    gamma=0.99, 
    clip_range=0.2
)

nepisode = 10000
x = []
y = []
line1 = []
for i_episode in range(nepisode):
    obs0 = env.reset()
    ep_rwd = 0
    done = False
    t = 0

    while not done:
        #env.render()
        act, _ = agent.step(obs0)
        obs1, rwd, done, _ = env.step(act)

        agent.memory.store_transition(obs0, act, rwd)

        obs0 = obs1
        ep_rwd += rwd

        if (t + 1) % 32 == 0 or True == done:
            _, last_value = agent.step(obs1)
            agent.learn(last_value, done)
        t = t + 1

    print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)

    y.append(ep_rwd)
    x.append(i_episode)
    line1 = live_plotter(x,y,line1)

env.close()


