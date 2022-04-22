################################################################################
# It is necessary to execute the following command before running this script
#
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
#
################################################################################
import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad

import numpy as np
import time 



env = quad.QuadrotorPlusHoverEnv()

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

for _ in range(20):
    done = False
    s = env.reset()
    while not done:
        env.render()
        a = 0.4905 * np.ones(4)
        s_tilde,r,done,_ = env.step(a) # take a random action
        s=s_tilde
    time.sleep(1)

env.close()