import random
import sys
import time

sys.path.append(".")
sys.path.append("minigrid")
import numpy as np
import torch
import torch.nn as nn

from minigrid.wrappers import *
from drl_lib import dqn


def process_obs(obs):
    return torch.tensor(obs).float().permute(2, 0, 1)

env_name = "MiniGrid-DistShift1-v0"
max_timestep = 10

env = ObstacleImgObsWrapper(FullyObsWrapper(gym.make(env_name)),n_obstacle = 3)
env.reset()
step =0
done = False
while (step < max_timestep):
    step += 1
    action = 1
    # take action in env:
    next_state, reward, done, info, _ = env.step(env.used_actions[action])
    next_state = process_obs(next_state)
    state = next_state


env.close()

