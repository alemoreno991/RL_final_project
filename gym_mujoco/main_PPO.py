################################################################################
# It is necessary to execute the following command before running this script
#
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
#
#
#   Reference: https://doi.org/10.48550/arXiv.1707.06347
#
################################################################################
import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad
from PPO import PPO

import numpy as np
from datetime import datetime
from plotter import live_plotter


env = quad.QuadrotorPlusHoverEnv()

params = {}
params["act_dim"] = 4
params["obs_dim"] = 18
params["lr_actor"]  = 1e-4
params["lr_critic"] = 1e-3
params["gamma"] = 0.99
params["clip_range"] = 0.2
params["num_episodes"] = 6000
params["batch_size"] = 128

agent = PPO(
    act_dim=params["act_dim"],
    obs_dim=params["obs_dim"],
    lr_actor=params["lr_actor"],
    lr_value=params["lr_critic"],
    gamma=params["gamma"],
    clip_range=params["clip_range"]
)

nepisode = params["num_episodes"]
batch_size = params["batch_size"]
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

        if (t + 1) % batch_size == 0 or True == done:
            _, last_value = agent.step(obs1)
            agent.learn(last_value, done)
        t = t + 1

    print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)

    y.append(ep_rwd)
    x.append(i_episode)
    line1 = live_plotter(x,y,line1, identifier="PPO", params=params)

now = datetime.now() # current date and time
date = now.strftime("%m-%d-%Y:%H:%M:%S")
live_plotter(x, y, line1, filename="img/PPO_"+date+".png")
env.close()


