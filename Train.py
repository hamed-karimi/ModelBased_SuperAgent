import os.path
import shutil
from copy import deepcopy
from Transition import Transition, get_random_action
from torch.utils.tensorboard import SummaryWriter
from Environment import Environment
from ReplayMemory import ReplayMemory
import numpy as np

class Train:
    def __init__(self, utils):
        self.params = utils.params
        self.episode_num = int(self.params.EPISODE_NUM)
        self.batch_size = int(self.params.BATCH_SIZE)
        self.step_num = int(self.params.EPISODE_STEPS)
        self.device = self.params.DEVICE
        self.res_folder = utils.res_folder
        self.log_dir = os.path.join(self.res_folder, 'log')
        self.tensor_writer = SummaryWriter()
        # self.tensorboard_call_back = CallBack(res_dir=self.res_folder, log_freq=self.params.PRINT_REWARD_FREQ, )

    def train_policy(self):
        print('start')
        few_many = [np.random.choice(['few', 'many']) for _ in range(self.params.OBJECT_TYPE_NUM)]
        environment = Environment(params=self.params, few_many_objects=few_many)
        transition = Transition(params=self.params)
        for episode in range(self.episode_num):
            state, _ = environment.reset()
            episode_loss = 0
            for step in range(self.step_num):
                goal_map = get_random_action(state=state)
                new_state, reward, terminated, truncated, _ = environment.step(goal_map)

                # ('init_state', 'goal_map', 'reward', 'next_state')
                transition.save_experience(state, goal_map, reward, new_state)
                state = deepcopy(new_state)

            episode_loss += transition.optimize()

            self.tensor_writer.add_scalar("Loss", episode_loss / self.step_num, episode)
            self.tensor_writer.add_scalar("lr", transition.lr_scheduler.get_last_lr()[0], episode)

        transition.save(self.res_folder)
