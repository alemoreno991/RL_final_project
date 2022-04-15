import time
import gym
from my_cartpole import MyCartPoleEnv

env = MyCartPoleEnv()
print(env.action_space)
print(env.observation_space)
input("Press Enter to continue...")

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()