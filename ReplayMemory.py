import random
from collections import namedtuple
import numpy as np
import torch


def get_experience_tensor(*args):
    transition_args = (torch.Tensor(args[0][0]), torch.Tensor(args[0][1]),
                       torch.Tensor(np.array(args[0][2:])).flatten(),
                       torch.Tensor(args[1]), torch.Tensor(args[2]),
                       torch.Tensor(args[3][0]), torch.Tensor(args[3][1]),
                       torch.Tensor(np.array(args[4]['dt'])), torch.Tensor(np.array(args[4]['rewarding'])))
    return transition_args


class ReplayMemory():
    def __init__(self, capacity, checkpoint_memory=None, memory_size=0):
        # We keep track of the number of experiences,
        # and we overwrite the early experiences by the later,
        # after episode_num reaches max capacity

        self.Transition = namedtuple('Transition',
                                     ('init_map', 'init_mental_state', 'states_params',
                                      'goal_map', 'reward',
                                      'next_map', 'next_mental_state',
                                      'dt', 'rewarding'))
        self.max_len = int(capacity)
        if checkpoint_memory is None:
            self.experience_index = 0
            self.memory = np.zeros((self.max_len,), dtype=object)

        else:
            self.memory = checkpoint_memory
            self.experience_index = memory_size

        print('memory size: ', self.experience_index)

    def push_experience(self, *args): # args: init_state, goal_map, reward, next_sate
        self.memory[self.experience_index % self.max_len] = get_experience_tensor(*args)
        self.experience_index += 1

    def weighted_sample_without_replacement(self, k):
        sample_indices = random.sample(range(0, min(self.experience_index, self.max_len)), k)
        sample = self.memory[sample_indices]

        return sample

    def sample(self, size):
        sample = self.weighted_sample_without_replacement(k=size)
        return sample

    def __len__(self):
        return self.experience_index % self.max_len
