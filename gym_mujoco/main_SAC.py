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
from termcolor import colored

def test(agent, options):
    env = quad.QuadrotorPlusHoverEnv()
    done = False
    obs0 = env.reset(options=options)
    action = [] 
    state = [obs0]
    while not done:
        # env.render()
        
        act = agent.step(obs0)
        action.append(act)

        obs1, _, done, _ = env.step(act)
        state.append(obs1)
    
    return action, state

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
    y = []
    for _ in range(nepisode):
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

        y.append(ep_rwd)

    env.close()
    
    return agent, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int)
    parser.add_argument('--time', type=str)
    args = parser.parse_args()
    i = args.run
    time = args.time
    
    data = {}
    params = {}
    params["act_dim"] = 4
    params["obs_dim"] = 18
    params["lr_actor"]  = 1e-4
    params["lr_critic"] = 1e-3 
    params["gamma"] = 0.99
    params["tau"] = 0.995
    params["num_episodes"] = 4000
    params["batch_size"] = 128 

    print(colored('Training...', 'green'))
    agent, returns = train(params)

    data["returns"] = returns

    print(colored('Running Test 1...', 'green'))
    options = {}
    pos_0 = [1.,1.,0.]
    quat = [1., 0., 0., 0.]
    vel = [0.,0.,0.]
    ang_vel = [0.,0.,0.]
    
    options["custom"] = pos_0 + quat + vel + ang_vel
    data["test0"] = {}
    data["test0"]["actions"], data["test0"]["states"] = test(agent, options)

    print(colored('Running Test 2...', 'green'))
    pos_0 = [0.,1.,1.]
    quat = [1., 0., 0., 0.]
    vel = [0.,0.,-1.]
    ang_vel = [0.3,0.,0.]
    
    options["custom"] = pos_0 + quat + vel + ang_vel
    data["test1"] = {}
    data["test1"]["actions"], data["test1"]["states"] = test(agent, options)

    print(colored('Running Test 3...', 'green'))
    options = {}
    pos_0 = [1.,0.,0.5]
    quat = [1., -0.08, 0.05, 0.]
    vel = [0.,0.,0.]
    ang_vel = [0.,0.,0.]
    
    options["custom"] = pos_0 + quat + vel + ang_vel
    data["test2"] = {}
    data["test2"]["actions"], data["test2"]["states"] = test(agent, options)

    print(colored('Running Test 4...', 'green'))
    pos_0 = [-2.,-2.,0.]
    quat = [1., -0.48, 0.35, 0.15]
    vel = [-5.,2.,-3.]
    ang_vel = [-0.1,0.2,0.7]
    
    options["custom"] = pos_0 + quat + vel + ang_vel
    data["test3"] = {}
    data["test3"]["actions"], data["test3"]["states"] = test(agent, options)

    print(colored('Saving information...', 'green'))
    with open('data/SAC_run{}_{}.pkl'.format(i, time), 'wb') as f:
        pickle.dump(data, f)
