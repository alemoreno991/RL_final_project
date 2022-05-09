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

from plotter import live_plotter

def test(agent, options):
    env = quad.QuadrotorPlusHoverEnv()
    done = False
    obs0 = env.reset(options=options)
    action = [] 
    state = [obs0]
    while not done:
        # env.render()  # Uncomment to visualize the trainning process
        
        act = agent.step(obs0)
        action.append(act)

        obs1, _, done, _ = env.step(act)
        state.append(obs1)
    
    env.close()

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
    for i_episode in range(nepisode):
        done = False
        obs0 = env.reset()
        ep_rwd = 0

        while not done:
            # env.render()  # Uncomment to visualize the trainning process
            
            act = agent.step(obs0)

            obs1, rwd, done, _ = env.step(act)

            agent.replay_buffer.store_transition(obs0, act, rwd, obs1, done)

            obs0 = obs1
            ep_rwd += rwd

            if iteration >= batch_size * 3:
                agent.learn()

            iteration += 1

        y.append(ep_rwd)
        print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)

    env.close()
    
    return agent, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--episodes', type=int, default=3500)
    parser.add_argument('--lr_actor', type=float, default=1e-4)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    i = args.run
    suffix = args.suffix
    
    data = {}
    params = {}
    params["act_dim"] = 4
    params["obs_dim"] = 18
    params["lr_actor"]  = args.lr_actor
    params["lr_critic"] = args.lr_critic
    params["gamma"] = args.gamma
    params["tau"] = 0.995
    params["num_episodes"] = args.episodes
    params["batch_size"] = args.batch_size

    print(colored('Training...', 'green'))
    agent, returns = train(params)

    data["returns"] = returns

    print(colored('Running Test 1...', 'green'))
    options = {}
    pos_0 = [0., 0., 1]
    quat = [1., 0., 0., 0.]
    vel = [0.,0.,0.]
    ang_vel = [0.,0.,0.]
    
    options["custom"] = pos_0 + quat + vel + ang_vel
    data["test0"] = {}
    data["test0"]["actions"], data["test0"]["states"] = test(agent, options)

    print(colored('Running Test 2...', 'green'))
    options = {}
    pos_0 = [0., 0., 1.]
    quat = [1., 0.9, 0.08, 0.8]
    vel = [0.,0.,0.]
    ang_vel = [0.,0.,0.]
    options["custom"] = pos_0 + quat + vel + ang_vel
    data["test1"] = {}
    data["test1"]["actions"], data["test1"]["states"] = test(agent, options)

    print(colored('Running Test 3...', 'green'))
    options = {}
    pos_0 = [0., 0., 1.]
    quat = [1., 0.1, -0.06, 1.8]
    vel = [1.,0.7,-0.5]
    ang_vel = [0.,0.,0.]
    
    options["custom"] = pos_0 + quat + vel + ang_vel
    data["test2"] = {}
    data["test2"]["actions"], data["test2"]["states"] = test(agent, options)

    print(colored('Running Test 4...', 'green'))
    options = {}
    pos_0 = [0., 0., 1.]
    quat = [1., -0.2, 0.16, 1.8]
    vel = [-5.3,1.57,3.25]
    ang_vel = [1.3,-0.5,3.14]
    
    options["custom"] = pos_0 + quat + vel + ang_vel
    data["test3"] = {}
    data["test3"]["actions"], data["test3"]["states"] = test(agent, options)

    print(colored('Saving information...', 'green'))
    with open('data/SAC_run{}_{}.pkl'.format(i, suffix), 'wb') as f:
        pickle.dump(data, f)