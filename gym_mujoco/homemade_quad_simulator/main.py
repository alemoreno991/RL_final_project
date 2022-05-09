################################################################################
#
#   Reference: https://doi.org/10.48550/arXiv.1801.01290
#
################################################################################
import sys 
sys.path.insert(0, './quad_sim')

from quad_sim.simulateQuadrotorDynamicsHF import SimulateQuadrotorDynamicsHF

import numpy as np
from datetime import datetime
from SAC import SAC
from plotter import live_plotter

params = {}
params["act_dim"] = 4
params["obs_dim"] = 12
params["lr_actor"]  = 1e-4
params["lr_critic"] = 1e-3 
params["gamma"] = 0.99
params["tau"] = 0.995
params["num_episodes"] = 2000
params["batch_size"] = 128 


Q = np.diag([1,1,1,1,1,1,5,5,5,5,5,5])
R = np.eye(4)

env = SimulateQuadrotorDynamicsHF(Q, R)

agent = SAC(
    act_dim=params["act_dim"],
    obs_dim=params["obs_dim"],
    lr_actor=params["lr_actor"],
    lr_value=params["lr_critic"],
    gamma=params["gamma"],
    tau=params["tau"]
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
        act = agent.step(obs0)

        obs1, rwd, done = env.step(act, np.zeros(3))

        agent.replay_buffer.store_transition(obs0, act, rwd, obs1, done)

        obs0 = obs1
        ep_rwd += rwd

        if iteration >= batch_size * 3:
            agent.learn()

        iteration += 1

    print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)
    
    y.append(ep_rwd)
    x.append(i_episode)
    line1 = live_plotter(x,y,line1, identifier="SAC", params=params)

now = datetime.now() # current date and time
date = now.strftime("%m-%d-%Y:%H:%M:%S")
live_plotter(x, y, line1, filename="img/SAC_"+date+".png")