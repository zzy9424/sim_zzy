import numpy as np

from replays.base_replay import BaseReplay


class OnlineReplay(BaseReplay):
    def __init__(self, max_size,batch_size):
        super().__init__(batch_size+1,batch_size)
        self.buffer=[]

    def get_cursor_idx(self):
        return self.size

    def max_priority(self):
        return 1

    def priority_update(self, indices, priorities):
        pass

    def add(self, data,priority,age):
        if self.size == self.max_size:
            del self.buffer[0]
        else:
            self.size += 1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(data)

    def sample(self,timestep):
        batch_size = self.batch_size

        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []

        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done),None,indexes