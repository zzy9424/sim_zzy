import numpy as np

from base_replay import BaseReplay
import torch
import torch.nn as nn

class AER_DQN(nn.Module):# with the same architecture as the deep Q-network
    def __init__(self,input_shape, output_shape):
        super(AER_DQN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (2, 2)),
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.get_conv_output_dim(), 16),#[16,16]
            nn.Tanh(),
            nn.Linear(16, output_shape)
        )
        #print(self.get_conv_output_dim())

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc_layer(x)
        return x

    def get_conv_output_dim(self):
        # x = torch.zeros(self.input_shape).unsqueeze(0)
        # x = self.conv_layer(x)
        # return int(torch.prod(torch.tensor(x.shape)))
        return self.conv_layer(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.nn.init.uniform_(layer.weight, -1, 1)
                torch.nn.init.uniform_(layer.bias, -1, 1)

class AER(BaseReplay):
    def __init__(self, max_size,batch_size,p_lambda = 2):#p_lambda推荐取2，2.5，3，4
        super().__init__(max_size,batch_size)
        self.buffer=[]
        self.age = []
        self.p_lambda = p_lambda

        self.embedded_state_network = AER_DQN(input_shape=(3, 7, 7), output_shape=18)
        self.embedded_state_network.init_weights()

    def get_cursor_idx(self):
        return self.size

    def max_priority(self):
        return 1

    def priority_update(self, indices, priorities):
        pass

    def add(self, data,priority,age):
        self.size += 1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(data)
        self.age.append(age)

    def sample(self, timestep,current_s):
        batch_size = self.batch_size
        p_lambda = self.p_lambda
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size / 5)]
            del self.age[0:int(self.size / 5)]
            self.size = len(self.buffer)

        pre_indices = np.random.randint(0, len(self.buffer), size=batch_size*p_lambda)
        pre_state, pre_action, pre_reward, pre_next_state, pre_done = [], [], [], [], [],
        priorities = []
        for i in pre_indices:
            s, a, r, s_, d = self.buffer[i]
            phi_sj = self.embedded_state_network.forward(s.unsqueeze(0)).squeeze(0)
            phi_st = self.embedded_state_network.forward(current_s.unsqueeze(0)).squeeze(0)
            phi_sj = np.array(phi_sj.detach().numpy())
            phi_st = np.array(phi_st.detach().numpy())
            priority = - np.linalg.norm(phi_sj - phi_st, ord=2)
            priorities.append(priority)
            pre_state.append(np.array(s, copy=False))
            pre_action.append(np.array(a, copy=False))
            pre_reward.append(np.array(r, copy=False))
            pre_next_state.append(np.array(s_, copy=False))
            pre_done.append(np.array(d, copy=False))

        top_n_priority_indices = np.argsort(-np.array(priorities))[:self.batch_size]  # n = batch_size
        # print(top_n_priority_indices)
        state, action, reward, next_state, done, indices = [], [], [], [], [], []
        for i in top_n_priority_indices:
            indices.append(pre_indices[i])
            state.append(pre_state[i])
            action.append(pre_action[i])
            reward.append(pre_reward[i])
            next_state.append(pre_next_state[i])
            done.append(pre_done[i])

        avg_age = self.get_age(indices,timestep)
        if self.writer is not None:
            self.writer.add_scalar("sample age", avg_age, global_step=timestep)

        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done),None,indices

    def get_age(self,indices,timestep):
        res = 0
        for i in indices:
            res += (timestep-self.age[i])
        return res/len(indices)