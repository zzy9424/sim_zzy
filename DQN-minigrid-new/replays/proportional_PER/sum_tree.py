#! -*- coding:utf-8 -*-

import sys
import os
import math

class SumTree(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = math.ceil(math.log(max_size+1, 2))+1
        self.tree_size = 2**self.tree_level-1
        self.tree = [0 for i in range(self.tree_size)]
        self.data = [None for i in range(self.max_size)]
        self.age = [None for i in range(self.max_size)]
        self.size = 0
        self.cursor = 0
        self.min_p = 0.01



    def add(self, contents, value, age):
        index = self.cursor
        self.cursor = (self.cursor+1)%self.max_size
        self.size = min(self.size+1, self.max_size)

        self.data[index] = contents
        self.age[index] = age
        self.val_update(index, value, min_p=self.min_p)

        # if self.size % 1000 ==0:
        #     non_zero_count = sum(1 for item in self.leaf() if item != 0)
        #     print("Non-Zero count:", non_zero_count)
        #     print("Min count:", sum(1 for item in self.leaf() if item == self.min_p))
        #     average = sum(self.leaf()) / non_zero_count
        #     print("平均数为:", average)
    def remove(self):
        index = self.cursor
        # print("cursor_now:",self.cursor,"cursor_back:", self.cursor,"zero_index:", index,"size:",self.size)
        self.cursor = (self.cursor+1)%self.max_size
        self.size = max(self.size-1, 0)
        self.data[index] = None
        # print("None count:",sum(1 for item in self.data if item is None))
        # print("Non-Zero count:", sum(1 for item in self.leaf() if item != 0))
        # print("Size:",self.size)
        self.val_update(index, 0)

    def get_val(self, index):
        tree_index = 2**(self.tree_level-1)-1+index
        return self.tree[tree_index]

    def val_update(self, index, value, min_p=0.00):
        if value < min_p:
            value = min_p
        tree_index = 2**(self.tree_level-1)-1+index
        diff = value-self.tree[tree_index]

        self.reconstruct(tree_index, diff)


    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex-1)/2)
            self.reconstruct(tindex, diff)

    def find(self, value, norm=True):
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):
        if 2**(self.tree_level-1)-1 <= index:
            return self.data[index-(2**(self.tree_level-1)-1)], self.tree[index], index-(2**(self.tree_level-1)-1)

        left = self.tree[2*index+1]
        if value <= left:
            return self._find(value,2*index+1)
        else:
            return self._find(value - left, 2 * (index + 1))


    def print_tree(self):
        for k in range(1, self.tree_level+1):
            for j in range(2**(k-1)-1, 2**k-1):
                print(self.tree[j], end=' ')
            print()

    def filled_size(self):
        return self.max_size

    def leaf(self):
        leaf=[]
        for i in range(2**(self.tree_level-1)-1,self.tree_size):
            leaf.append(self.tree[i])
        return leaf

    # input: array of indices
    def get_age(self,indices,timestep_now):
        res = 0
        for i in indices:
            res += (timestep_now-self.age[i])
        return res/len(indices)

    def get_age_by_idx(self,idx,timestep_now):
        return timestep_now-self.age[idx]
if __name__ == '__main__':
    s = SumTree(10)
    for i in range(20):
        s.add(2**i, i)
    s.print_tree()
    print(s.find(0.2))
