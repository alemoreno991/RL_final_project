################################################################################
# It is necessary to execute the following command before running this script
#
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
#
#
#   Reference: https://doi.org/10.48550/arXiv.1801.01290
#
################################################################################
import pickle
import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad
import numpy as np
import argparse
from SAC import SAC
from datetime import datetime

def test(agent, options):
    env = quad.QuadrotorPlusHoverEnv()
    done = False
    obs0 = env.reset(options=options)
    action_state = {}
    action_state[0] = {'action':None, 'state': obs0}
    ct = 0
    while not done:
        # env.render()
        ct+=1
        
        act = agent.step(obs0)

        obs1, _, done, _ = env.step(act)
        action_state[ct] = {'action':act, 'state': obs1}
    
    return action_state

def train(params): 
    env = quad.QuadrotorPlusHoverEnv(randomize_reset=True)

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
    for i_episode in range(nepisode):
        done = False
        obs0 = env.reset()
        ep_rwd = 0

        while not done:
            # env.render()
            
            act = agent.step(obs0)

            obs1, rwd, done, _ = env.step(act)

            agent.replay_buffer.store_transition(obs0, act, rwd, obs1, done)

            obs0 = obs1
            ep_rwd += rwd

            if iteration >= batch_size * 3:
                agent.learn()

            iteration += 1

        # print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)
        
        y.append(ep_rwd)
        # x.append(i_episode)
        # line1 = live_plotter(x,y,line1, identifier="SAC", params=params)

    # now = datetime.now() # current date and time
    # date = now.strftime("%m-%d-%Y:%H:%M:%S")
    # live_plotter(x, y, line1, filename="img/SAC_"+date+".png")
    env.close()
    
    return agent, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int)
    args = parser.parse_args()

    i = args.run
    data = {}
    time = datetime.now().strftime("%m-%d-%Y:%H:%M:%S")
    params = {}
    params["act_dim"] = 4
    params["obs_dim"] = 18
    params["lr_actor"]  = 1e-4
    params["lr_critic"] = 1e-3 
    params["gamma"] = 0.99
    params["tau"] = 0.995
    params["num_episodes"] = 2
    params["batch_size"] = 128 

    agent, returns = train(params)

    data["returns"] = returns

    options = {}
    pos_0 = [1.,1.,0.]
    quat = [1., 0., 0., 0.]
    vel = [0.,0.,0.]
    ang_vel = [0.,0.,0.]
    
    options["custom"] = pos_0 + quat + vel + ang_vel
    data["test0"] = test(agent, options)

    pos_0 = [1.,1.,1.]
    quat = [1., 0., 0., 0.]
    vel = [0.,0.,0.]
    ang_vel = [0.,0.,0.]
    
    options["custom"] = pos_0 + quat + vel + ang_vel
    data["test1"] = test(agent, options)

    with open('data/SAC_run{}_{}.pkl'.format(i, time), 'wb') as f:
        pickle.dump(data, f)
