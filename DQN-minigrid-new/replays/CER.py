import math

import numpy as np

from replays.base_replay import BaseReplay


class CER(BaseReplay):
    def __init__(self, max_size,batch_size):
        super().__init__(max_size,batch_size)
        self.buffer=[]
        self.age=[]
        self.max_p=0

    def get_cursor_idx(self):
        return self.size

    def sample_from_recent(self,size):
        indexes = np.random.randint(len(self.buffer)-size*10, len(self.buffer), size)
        state, action, reward, next_state, done = [], [], [], [], []

        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        return state,action,reward,next_state,done,indexes

    def max_priority(self):
        return 1

    def priority_update(self, indices, priorities):
        pass

    def add(self, data,priority,age):
        self.size += 1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(data)
        self.age.append(age)

    def sample(self,timestep):
        batch_size = self.batch_size

        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size / 5)]
            self.size = len(self.buffer)

        state, action, reward, next_state, done, recent_indices = self.sample_from_recent(math.floor(self.batch_size/10))
        # print("recent_sample:",math.floor(self.batch_size/10))
        indexes = np.random.randint(0, len(self.buffer), size=batch_size-math.floor(self.batch_size/10))
        # print("long_term_sample:",batch_size-math.floor(self.batch_size/10))
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        indices = list(indexes)
        indices.extend(list(recent_indices))
        avg_age = self.get_age(indices,timestep)
        self.writer.add_scalar("sample age", avg_age, global_step=timestep)
        weights = [1] * batch_size
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done),weights,indices

    def get_age(self,indices,timestep):
        res = 0
        for i in indices:
            res += (timestep-self.age[i])
        return res/len(indices)