import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init


# import Utilities


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class TransitionNet(nn.Module):
    def __init__(self, params, device):
        self.params = params
        self.device = device
        super(TransitionNet, self).__init__()

        env_layer_num = self.params.OBJECT_TYPE_NUM + 1 + 1  # +1 for agent layer + 1 for goal map layer

        kernel_size = 2
        self.conv = nn.Sequential(nn.Conv2d(in_channels=env_layer_num,
                                            out_channels=128,
                                            kernel_size=kernel_size), nn.ReLU(),
                                  nn.Conv2d(in_channels=128,
                                            out_channels=64,
                                            kernel_size=kernel_size + 1), nn.ReLU(),
                                  nn.Conv2d(in_channels=64,
                                            out_channels=64,
                                            kernel_size=kernel_size + 2), nn.ReLU()
                                  ).to(self.device)
        each_channel_size = params.WIDTH - kernel_size - (kernel_size + 1) - (kernel_size + 2) - (kernel_size + 3) + 4
        self.linear = nn.Linear(in_features=64 * each_channel_size ** 2,
                                out_features=128).to(self.device)
        self.fc = nn.Sequential(
            nn.Linear(in_features=128 + 6,  # +2 for needs, +4 for params
                      out_features=64), nn.ReLU(),
            nn.Linear(in_features=64,
                      out_features=32), nn.ReLU(),
            nn.Linear(in_features=32,
                      out_features=16), nn.ReLU()).to(self.device)

        self.reward_mental_states_dt_readout = nn.Linear(in_features=16, out_features=4)
        self.rewarding_readout = nn.Linear(in_features=16, out_features=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, env_map, goal_map, mental_states, states_params):
        x = torch.cat([env_map, goal_map.unsqueeze(1)], dim=1).to(self.device)
        mental_states = mental_states.to(self.device)
        states_params = states_params.to(self.device)
        # batch_size = x.shape[0]
        y = self.conv(x)
        y0 = y.flatten(start_dim=1, end_dim=-1)

        y = F.relu(self.linear(y0))
        x = torch.concat([y, mental_states, states_params], dim=1)
        y = self.fc(x)
        reward_mental_states_dt = self.reward_mental_states_dt_readout(y)
        rewarding = self.rewarding_readout(y)

        return reward_mental_states_dt, rewarding  # 0: reward 1,2: mental_state
