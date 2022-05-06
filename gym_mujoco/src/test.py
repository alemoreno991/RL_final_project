import gym_multirotor.envs.mujoco.quadrotor_plus_hover as quad
import numpy as np
import torch
import json
import argparse
import pickle
from pathlib import Path

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def test(agent_model, initial_cond, seed=0, max_ep_len=400):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = quad.QuadrotorPlusHoverEnv()
    agent = torch.load(agent_model)

    print(initial_cond)
    state, done, ep_ret, ep_len = env.reset(initial_cond), False, 0, 0

    a = []
    s = [state]
    while not(done or (ep_len == max_ep_len)):
        # test_env.render()
        action = agent.act(torch.as_tensor(state, dtype=torch.float32))
        state, _, done, _ = env.step(action)
        ep_len += 1

        a.append(action)
        s.append(state)    
    
    env.close()

    return a, s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--agent', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    init_conditions = read_json(args.input)

    data = {}
    for i in range(len(init_conditions)):
        X0 = {}
        pos_0 = init_conditions["test{}".format(i)]["pos0"]
        quat0 = init_conditions["test{}".format(i)]["quat0"]
        vel0   = init_conditions["test{}".format(i)]["vel0"]
        omega0 = init_conditions["test{}".format(i)]["omega0"]
        X0["custom"] = pos_0 + quat0 + vel0 + omega0
        
        data["test{}".format(i)] = {}
        data["test{}".format(i)]["actions"], data["test{}".format(i)]["states"] = test( args.agent + '/pyt_save/model.pt', X0)

    with open(args.output, 'wb') as f:
        pickle.dump(data, f)