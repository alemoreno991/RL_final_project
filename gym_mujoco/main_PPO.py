################################################################################
# It is necessary to execute the following command before running this script
#
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
#
#
#   Reference: https://doi.org/10.48550/arXiv.1707.06347
#
################################################################################
import pickle
import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad
import numpy as np
import argparse
from PPO import PPO
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
        
        act, _ = agent.step(obs0)
        action.append(act)

        obs1, _, done, _ = env.step(act)
        state.append(obs1)
    
    env.close()

    return action, state

def train(params): 
    env = quad.QuadrotorPlusHoverEnv(randomize_reset=True)

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
    y = []
    for i_episode in range(nepisode):
        done = False
        obs0 = env.reset()
        ep_rwd = 0
        t=0

        while not done:
            # env.render()
            
            act, _ = agent.step(obs0)

            obs1, rwd, done, _ = env.step(act)

            agent.memory.store_transition(obs0, act, rwd)

            obs0 = obs1
            ep_rwd += rwd

            if (t + 1) % batch_size == 0 or True == done:
                _, last_value = agent.step(obs1)
                agent.learn(last_value, done)
            t = t + 1

        y.append(ep_rwd)
        print('Ep: %i' % i_episode, "|Ep_r: %i" % ep_rwd)

    env.close()
    
    return agent, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=8000)
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
    params["clip_range"] = 0.2
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
    with open('data/PPO_run{}_{}.pkl'.format(i, suffix), 'wb') as f:
        pickle.dump(data, f)