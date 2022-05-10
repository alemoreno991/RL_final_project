from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.td3.core as core
from spinup.utils.logx import EpochLogger
import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad
import pickle

def test_td3(env_fn, ac, seed=0, max_ep_len=4000):

    torch.manual_seed(seed)
    np.random.seed(seed)

    test_env = env_fn()
    act_dim = test_env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = test_env.action_space.high[0]

    # Create actor-critic module and target networks
    # ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    testflag = False 

    # traj = [[0,0],[1, 0], [1, 1], [0,0]]
    # traj = [[0, 0], [-2, 0], [1, -1], [1, 1]]

    # traj = [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]
    # traj = [[0, 0], [-2, 0], [0, -2], [2, 0], [0, 2]]

    # traj = [[0, 0],[1, 0], [2, 0],[2, 1], [2, 2],[1, 2] [0, 2], [0, 1], [0, 0]]
    traj = [[0, 0], [-1, 0],[-1, 0], [0, -1],[0, -1], [1, 0],[1, 0], [0, 1],[0, 1]]


    offset_x = 0
    offset_y = 0

    options = {}
    pos_0 = [0,0.,3]
    quat = [1, 0,0,0]
    vel = [0.,0.,0.]
    ang_vel = [0.,0.,0.]
    options["custom"] = pos_0 + quat + vel + ang_vel

    o, d, ep_len = test_env.reset(options), False, 0

    actions = []
    states = [o]

    ct = 0

    wait = 0

    while not(d or (ep_len == max_ep_len)):
        # test_env.render()
        action = get_action(o, 0)
        o, _, d, _ = test_env.step(action)

        o[0] += offset_x
        o[1] += offset_y

        states.append(test_env.get_state())

        if wait < 0 and (abs(o[0]) < 0.02 and abs(o[1]) < 0.02 and abs(o[2]) < 0.02):
            wait += 1
        elif (abs(o[0]) < 0.02 and abs(o[1]) < 0.02 and abs(o[2]) < 0.02):
            wait = 0
            ct += 1
            if (ct > 8):
                continue
            offset_x += traj[ct][0]
            offset_y += traj[ct][1]

        actions.append(action)
        
        ep_len += 1
    
    return actions, states

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    args = parser.parse_args()

    env = quad.QuadrotorPlusHoverEnv(randomize_reset=True, env_bounding_box=100)

    loaded_ac = torch.load('./sq_td3_diffR1/pyt_save/model.pt')
    env.reset()
    data = {}
    data["test0"] = {}
    data["test0"]["actions"], data["test0"]["states"] = test_td3(lambda : env, loaded_ac)

    with open('data/square.pkl', 'wb') as f:
        pickle.dump(data, f)