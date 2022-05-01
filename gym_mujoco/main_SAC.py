################################################################################
# It is necessary to execute the following command before running this script
#
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
#
################################################################################
import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad
from SAC import SAC

import numpy as np
import time 


env = quad.QuadrotorPlusHoverEnv()

agent = SAC(
    act_dim=4, 
    obs_dim=18,
    lr_actor=1e-3, 
    lr_value=1e-3, 
    gamma=0.99, 
    tau=0.995
)

nepisode = 1000
batch_size = 128
iteration = 0

for i_episode in range(nepisode):
    done = False
    obs0 = env.reset()
    ep_rwd = 0

    while not done:
        env.render()
        
        act = agent.step(obs0)

        obs1, rwd, done, _ = env.step(act)

        agent.replay_buffer.store_transition(obs0, act, rwd, obs1, done)

        obs0 = obs1
        ep_rwd += rwd

        if iteration >= batch_size * 3:
            agent.learn()

        iteration += 1

    print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)

env.close()
