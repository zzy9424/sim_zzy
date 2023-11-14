import random

import torch

from replays.default_replay import DefaultReplay

max_size = 1000
batch_size = 32

replay_buffer = DefaultReplay(max_size,batch_size)
size = torch.Size([3, 7, 7])

for step in range(10000):
    state = torch.randn(size)
    action = random.choice([1, 2, 3])
    reward =  random.random()
    next_state = torch.randn(size)
    done = random.choice([True, False])
    replay_buffer.add((state, action, reward, next_state, float(done)),priority=replay_buffer.max_priority(),age=step)

for step in range(10000):
    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, _, indices = replay_buffer.sample(step)
    print(indices)