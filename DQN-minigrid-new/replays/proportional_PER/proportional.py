import numpy
import random

import numpy as np
import torch

from . import sum_tree
from replays.base_replay import BaseReplay


class ProportionalPER(BaseReplay):

    def __init__(self,max_size,batch_size,alpha=0.7,beta = 0.7):
        super().__init__(max_size,batch_size)
        self.tree = sum_tree.SumTree(self.max_size)
        self.alpha = alpha
        self.beta = beta
        self.writer = None

        self.max_p = 1.0

    def get_cursor_idx(self):
        return self.tree.cursor

    def max_priority(self):
        return self.max_p

    def add(self, data, priority,age):
        self.tree.add(data, priority**self.alpha, age)
        self.size = self.tree.size

    def sample(self,timestep):
        if self.tree.filled_size() < self.batch_size:
            return None, None, None

        indices = []
        weights = []
        priorities = []
        state, action, reward, next_state, done = [], [], [], [], []
        for _ in range(self.batch_size):
            r = random.uniform(0, 1)
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append((1./self.max_size/priority)**self.beta if priority > 1e-16 else 0)
            indices.append(index)
            self.priority_update([index], [0]) # To avoid duplicating
            s, a, r, s_, d = data
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        self.priority_update(indices, priorities) # Revert priorities
        avg_age = self.tree.get_age(indices,timestep)
        self.writer.add_scalar("sample age", avg_age, global_step=timestep)
        # Normalize for stability
        max_weight = max(weights)
        if max_weight!=0:
            for i in range(len(weights)):
                weights[i] /= max_weight
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done), weights, indices

    def priority_update(self, indices, priorities):
        for i, p in zip(indices, priorities):
            if isinstance(p,torch.Tensor):
                p = abs(p[0].item())
            p = p ** self.alpha
            if p > self.max_p:
                self.max_p = p
            self.tree.val_update(i, p,min_p=0.01)

    # def reset_alpha(self, alpha):
    #     self.alpha, old_alpha = alpha, self.alpha
    #     priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
    #     self.priority_update(range(self.tree.filled_size()), priorities)
