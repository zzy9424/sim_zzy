import math
from collections import deque
import sys
sys.path.append("..")
sys.path.append(".")
import numpy
import random
import time
import numpy as np
import torch

from replays.base_replay import BaseReplay
from replays.proportional_PER import sum_tree

class EPER(BaseReplay):

    def __init__(self,max_size,batch_size,alpha=0.7,beta = 0.7,lambda_init=0.5,verbose=False):
        super().__init__(max_size,batch_size)
        self.verbose = verbose
        self.stage = 1

        self.alpha = alpha
        self.beta = beta
        self.max_p = 1
        self.saved_net = None
        self.lamda_init = lambda_init
        self.lamda = self.lamda_init
        self.lamda_decay = lambda_init/(self.max_size/2)

        self.new_buffer = sum_tree.SumTree(self.max_size)
        self.old_buffer = sum_tree.SumTree(self.max_size)

        self.sample_from_new = self.actual_sample_from_new = self.batch_size
        self.sample_from_old = self.actual_sample_from_old = 0


        self.td_error_buffer_size = 100
        self.td_error_buffer = deque(maxlen=self.td_error_buffer_size)
    def update_saved_net(self,net):
        self.saved_net = net

    def check_stage_change(self,td_error_list):
        return
        c_td = sum(td_error_list) / len(td_error_list)
        average_td = sum(self.td_error_buffer) / len(self.td_error_buffer)
        if c_td > average_td*2:
            self.stage_1_to_2()
        else:
            self.td_error_buffer.append(c_td)

    def stage_1_to_2(self):
        self.old_buffer = self.new_buffer
        self.new_buffer = sum_tree.SumTree(self.max_size)
        self.lamda = self.lamda_init
        self.stage = 2
        self.max_p = 1
        self.td_error_buffer = deque(maxlen=self.td_error_buffer_size)
        for i in range(self.old_buffer.filled_size()):
            if self.old_buffer.data[i] is not None:
                self.priority_update(self.old_buffer, [i], [self.max_p])

    def stage_2_to_1(self):
        self.stage = 1
        self.max_p = 1
        self.sample_from_new = self.actual_sample_from_new = self.batch_size
        self.sample_from_old = self.actual_sample_from_old = 0
        for i in range(self.new_buffer.filled_size()):
            if self.new_buffer.data[i] is not None:
                self.priority_update(self.new_buffer, [i], [self.max_p])

    def recalculate_lamda(self):
        if self.stage == 1:
            return
        proportion = self.new_buffer.size / (self.old_buffer.size + self.new_buffer.size)
        # print("prop",proportion,"lamba",self.lamda)
        if self.lamda > proportion:
            # TODO decay lamda
            self.lamda -= self.lamda_decay
        else:
            self.lamda = proportion
        self.sample_from_new = math.ceil(self.lamda * self.batch_size)
        self.sample_from_old = self.batch_size - self.sample_from_new

    def add(self, data, priority=None,age=0):
        self.size = self.new_buffer.size + self.old_buffer.size
        if priority is None:
            priority = self.max_p
        self.new_buffer.add(data, priority ** self.alpha, age)
        if self.stage == 2:
            self.recalculate_lamda()
            if self.size >= self.max_size and self.old_buffer.size!=0:
                self.old_buffer.remove()


    def _sample_from(self,buffer,sample_size):
        indices = []
        weights = []
        priorities = []
        state, action, reward, next_state, done = [], [], [], [], []
        for _ in range(sample_size):
            data, priority, index = buffer.find(random.uniform(0, 1))
            weights.append((1. / self.max_size / priority) ** self.beta if priority > 1e-16 else 0)
            indices.append(index)
            priorities.append(priority)
            self.priority_update(buffer, [index], [0])  # To avoid duplicating

            s, a, r, s_, d = data
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        self.priority_update(buffer, indices, priorities)  # Revert priorities
        # print(indices)
        # print(priorities)
        # print("=" * 20)

        return state, action, reward, next_state, done,weights,indices

    def sample(self, timestep):
        # arr = self.new_buffer.leaf()
        # self.max_p = np.max(arr)
        #
        # arr = np.array(arr)
        # arr = arr[arr != 0]
        # mean, max_val, min_val, q3, median, q1 = np.mean(arr), np.max(arr), np.min(arr), np.percentile(arr,75), np.median(arr), np.percentile(arr, 25)
        # print("New Size: {}, Mean: {:.3f},  Max: {:.2f}, Min: {:.2f}, 75%: {:.2f}, Median: {:.2f}, 25%: {:.2f}".format(
        #         len(arr), mean, max_val, min_val, q3, median, q1),end="\t")
        # if self.stage == 2:
        #     arr = self.old_buffer.leaf()
        #     arr = np.array(arr)
        #     arr = arr[arr != 0]
        #     mean, max_val, min_val, q3, median, q1 = np.mean(arr), np.max(arr), np.min(arr), np.percentile(arr,75), np.median(arr), np.percentile(arr, 25)
        #     print("Old Size: {}, Mean: {:.3f}, Max: {:.2f}, Min: {:.2f}, 75%: {:.2f}, Median: {:.2f}, 25%: {:.2f}".format(len(arr), mean, max_val, min_val, q3, median, q1))
        # else:
        #     print()
        if self.old_buffer.size <= 1 and self.stage == 2:
            self.stage_2_to_1()
        # avg_age = 0
        self.actual_sample_from_new = self.sample_from_new
        self.actual_sample_from_old = self.sample_from_old
        if self.stage == 1:
            state, action, reward, next_state, done,weights,indices = self._sample_from(self.new_buffer,self.batch_size)
            # avg_age += self.new_buffer.get_age(indices,timestep)
        else:
            if self.new_buffer.size < self.sample_from_new:
                self.actual_sample_from_new = self.new_buffer.size
                self.actual_sample_from_old = self.batch_size - self.actual_sample_from_new
            sample = self._sample_from(self.new_buffer, self.actual_sample_from_new)
            state, action, reward, next_state, done, weights, indices = sample
            # avg_age += (self.new_buffer.get_age(indices,timestep))*len(indices)
            # new_age=self.new_buffer.get_age(indices,timestep)
            # if self.writer is not None:
            #     self.writer.add_scalar("new_age", new_age, global_step=timestep)
            # for i in indices:
            #     print("age:",self.new_buffer.get_age_by_idx(i,timestep),"\tpri",self.new_buffer.get_val(i))
            # print("-"*20)
            if self.actual_sample_from_old != 0:
                sample = self._sample_from(self.old_buffer, self.actual_sample_from_old)
                t_state, t_action, t_reward, t_next_state, t_done, t_weights, t_indices = sample
                state = state + t_state
                action = action + t_action
                reward = reward + t_reward
                next_state = next_state + t_next_state
                done = done + t_done
                weights = weights + t_weights
                indices = indices + t_indices
                # for i in t_indices:
                #     print("age:", self.old_buffer.get_age_by_idx(i, timestep), "\tpri", self.old_buffer.get_val(i))
                # print("=" * 20)
                # avg_age += self.old_buffer.get_age(t_indices,timestep)*len(t_indices)
                # old_age = self.old_buffer.get_age(t_indices, timestep)
                # self.writer.add_scalar("old age", old_age, global_step=timestep)
            # avg_age /= len(indices)
            # print()
        # if self.writer is not None and self.stage == 2:
        #     self.writer.add_scalar("sample:new/old", self.actual_sample_from_new / (self.actual_sample_from_old+self.actual_sample_from_new), global_step=timestep)
        #     self.writer.add_scalar("buffer:new/old", self.new_buffer.size / (self.old_buffer.size+self.new_buffer.size),
        #                            global_step=timestep)
        #     self.writer.add_scalar("lambda", self.lamda,
        #                            global_step=timestep)

        # self.writer.add_scalar("sample age", avg_age, global_step=timestep
        # Normalize for stability
        max_weight = max(weights)
        if max_weight!=0:
            for i in range(len(weights)):
                weights[i] /= max_weight
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(
            done), weights, indices

    def get_q(self, indices, Q, next_state, next_action,reward,done,gamma,writer,episode,td_target_Q):
        if self.stage == 1:
            target_Q = td_target_Q
        else:
            target_Q = self.saved_net(next_state).detach().max(1)[0].unsqueeze(1)
            target_Q = reward + ((1 - done) * gamma * target_Q).detach()
        ed_error = abs(target_Q - Q)
        ed_error_new = [abs(e.item()) for e in ed_error if e < self.actual_sample_from_new]
        ed_error_old = [abs(1 / e.item()) for e in ed_error if e >= self.actual_sample_from_new]
        self.priority_update(self.new_buffer, indices[:self.actual_sample_from_new], ed_error_new)
        if self.sample_from_old != 0:
            self.priority_update(self.old_buffer, indices[self.actual_sample_from_new+1:], ed_error_old)


    def priority_update(self, buffer, indices, priorities):
        for i, p in zip(indices, priorities):
            p = p ** self.alpha
            if p > self.max_p:
                self.max_p = p
            if buffer.data[i] is not None:
                buffer.val_update(i, p, min_p=buffer.min_p)

    def calculate_idx_diff(self,indices):
        avg_sample_index_delta = 0
        for batch_idx, buffer_idx in enumerate(indices):
            if batch_idx < self.sample_from_new:
                avg_sample_index_delta += self.new_buffer.size - buffer_idx
            else:
                avg_sample_index_delta += (self.old_buffer.size - buffer_idx) + self.new_buffer.size

        return avg_sample_index_delta / len(indices)

if __name__ == '__main__':
    er = EPER(100, 10, alpha=1, beta=1, verbose=True)
    for i in range(100):
        er.add(["a","a","a","a","a"])
        print("lamda",er.lamda,"new",er.sample_from_new,"old",er.sample_from_old)
        er.recalculate_lamda()
    er.stage_1_to_2()
    print("=========")
    for i in range(100):
        er.add(["b","b","b","b","b"])
        print("lamda",er.lamda,"new",er.sample_from_new,"old",er.sample_from_old)
        # print(er.sample()[0])
        er.recalculate_lamda()