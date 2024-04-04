import os
import random
import torch
import numpy as np
from torch.nn import init
from torch import nn, optim
from TransitionNet import TransitionNet, weights_init_orthogonal
from gymnasium import spaces
import math
from ReplayMemory import ReplayMemory


def get_random_action(state: list):
    env_map = np.array(state[0])
    goal_map = np.zeros_like(env_map[0, :, :])

    all_object_locations = np.stack(np.where(env_map), axis=1)
    goal_index = np.random.randint(low=0, high=all_object_locations.shape[0], size=())
    goal_location = all_object_locations[goal_index, 1:]

    goal_map[goal_location[0], goal_location[1]] = 1
    return goal_map


class Transition:
    def __init__(self, params, device='auto'):
        self.params = params
        self.device = 'cuda' if ((device == 'auto' or device == 'cuda') and torch.cuda.is_available()) else 'cpu'
        self.gamma = params.GAMMA
        self.transition_net = TransitionNet(params, self.device)
        self.transition_net.apply(weights_init_orthogonal)
        self.epsilon = .95
        self.epsilon_range = [.95, .05]
        self.batch_size = params.BATCH_SIZE
        self.optimizer = optim.Adam(self.transition_net.parameters(), lr=params.INIT_LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(capacity=self.params.MEMORY_CAPACITY)

    def _get_batch_tensor(self, batch):
        # 'init_map', 'init_mental_state', 'states_params',
        # 'goal_map', 'reward',
        # 'next_map', 'next_mental_state'
        assert type(batch) is self.memory.Transition, 'batch should be a memory Transition name tuple'
        init_map, init_mental_state, states_params, goal_map, reward, next_map, next_mental_state = [], [], [], [], [], [], []
        for i in range(len(batch.init_map)):
            init_map.append(batch.init_map[i])
            init_mental_state.append(batch.init_mental_state[i])
            states_params.append(batch.states_params[i])
            goal_map.append(batch.goal_map[i])
            reward.append(batch.reward[i])
            next_map.append(batch.next_map[i])
            next_mental_state.append(batch.next_mental_state[i])
        init_map = torch.stack(init_map, dim=0)
        init_mental_state = torch.stack(init_mental_state, dim=0)
        states_params = torch.stack(states_params, dim=0)
        goal_map = torch.stack(goal_map, dim=0)
        reward = torch.stack(reward, dim=0)
        next_map = torch.stack(next_map, dim=0)
        next_mental_state = torch.stack(next_mental_state, dim=0)
        return init_map, init_mental_state, states_params, goal_map, reward, next_map, next_mental_state

    def save_experience(self, *args):
        self.memory.push_experience(*args)

    def optimize(self):
        if len(self.memory) < 3 * self.batch_size:
            return 0.
        transition_sample = self.memory.sample(self.batch_size)
        sample = self.memory.Transition(*zip(*transition_sample))
        self.transition_net.train()
        # ('init_state', 'goal_map', 'reward', 'next_state')
        init_map, \
            init_mental_state, \
            states_params, \
            goal_map, \
            reward, \
            next_map, \
            next_mental_state = self._get_batch_tensor(sample)

        pred_reward_mental_state = self.transition_net(init_map, goal_map, init_mental_state, states_params).cpu()

        loss = self.criterion(pred_reward_mental_state,
                              torch.cat([reward.unsqueeze(1), next_mental_state], dim=1))
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.transition_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

    def save(self, path):
        torch.save(self.transition_net.state_dict(), os.path.join(path, 'model.pt'))
