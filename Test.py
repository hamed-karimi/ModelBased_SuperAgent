import itertools
import math
from copy import deepcopy
# from stable_baselines3 import PPO
import os
from sbx import PPO
import torch
import numpy as np
from ReplayMemory import get_experience_tensor

from Transition import Transition, get_random_action
from Environment import Environment
import matplotlib.pyplot as plt


def get_predefined_parameters(params, param_name):
    if param_name == 'all_mental_states':
        all_param = [[-10, -5, 0, 5, 10]] * params.OBJECT_TYPE_NUM
    elif param_name == 'all_object_rewards':
        # all_param = [[0, 4, 8, 12, 16, 20]] * num_object
        param_range = params.ENVIRONMENT_OBJECT_REWARD_RANGE
        all_param = np.expand_dims(np.linspace(param_range[0],
                                               param_range[1], num=min(param_range[1] - param_range[0] + 1, 4),
                                               dtype=int), axis=0).tolist() * params.OBJECT_TYPE_NUM
    elif param_name == 'all_mental_states_change':
        # all_param = [[0, 1, 2, 3, 4, 5]] * num_object
        param_range = params.MENTAL_STATES_SLOPE_RANGE
        all_param = np.expand_dims(np.linspace(param_range[0],
                                               param_range[1],
                                               num=min(param_range[1] - param_range[0] + 1, 4), dtype=int),
                                   axis=0).tolist() * params.OBJECT_TYPE_NUM
    else:
        print('no such parameters')
        return
    num_param = len(all_param[0]) ** params.OBJECT_TYPE_NUM
    param_batch = []
    for i, ns in enumerate(itertools.product(*all_param)):
        param_batch.append(list(ns))
    return param_batch


class Test:
    def __init__(self, utils):
        self.params = utils.params
        self.device = 'cuda' if ((
                                         self.params.DEVICE == 'auto' or self.params.DEVICE == 'cuda') and torch.cuda.is_available()) else 'cpu'
        self.res_folder = utils.res_folder
        self.transition = self.load_model(self.params)
        self.height = utils.params.HEIGHT
        self.width = utils.params.WIDTH
        self.object_type_num = utils.params.OBJECT_TYPE_NUM
        self.color_options = [[1, 0, .2], [0, .8, .2], [0, 0, 0]]
        self.goal_shape_options = ['*', 's', 'P', 'o', 'D', 'X']
        self.objects_color_name = ['r', 'g', 'b']  # 2: stay

        self.all_mental_states = get_predefined_parameters(self.params, 'all_mental_states')
        self.all_object_rewards = get_predefined_parameters(self.params, 'all_object_rewards')
        self.all_mental_states_change = get_predefined_parameters(self.params, 'all_mental_states_change')

    def get_top_figure_title(self, parameters):
        [init_mental_state,
         next_mental_state,
         reward,
         pred_mental_state,
         pred_reward,
         dt,
         rewarding,
         pred_dt,
         pred_rewarding] = parameters
        title = r''
        for i in range(len(init_mental_state)):
            title += r"$n^{0}_{1}:{2:.2f}$, ".format('{0}', '{' + self.objects_color_name[i] + '}',
                                                     init_mental_state[i])
        title += '\n'
        for i in range(len(next_mental_state)):
            title += r"$n^{0}_{1}:{2:.2f}$, ".format('{1}', '{' + self.objects_color_name[i] + '}',
                                                     next_mental_state[i])
        title += r"r:{0:.2f}, ".format(reward)
        title += r"t:{0:.2f}, is_r:{1}".format(dt, rewarding)
        title += "\n"
        for i in range(len(pred_mental_state)):
            title += r"${0}^{1}_{2}:{3:.2f}$, ".format('{pn}', '{1}', '{' + self.objects_color_name[i] + '}',
                                                       pred_mental_state[i])
        title += r"pr:{0:.2f}, ".format(pred_reward)
        title += r"pt:{0:.2f}, pis_r:{1:.2f}".format(pred_dt, pred_rewarding)
        return title

    def get_title(self, shape, next_mental_state, next_reward, pred_mental_state, pred_reward):
        title = ''
        for i in range(len(next_mental_state)):
            title += r"$   n_{0}:{1:.2f}$, ".format('{' + self.objects_color_name[i] + '}',
                                                    next_mental_state[i])
        title += r"r_{0}:{1:.2f}".format(shape, next_reward)
        title += '\n'
        title += 'p: '
        for i in range(len(pred_mental_state)):
            title += r"$n_{0}:{1:.2f}$, ".format('{' + self.objects_color_name[i] + '}',
                                                 pred_mental_state[i])
        title += r"r_{0}:{1:.2f}".format(shape, pred_reward)
        return title

    def get_right_figure_title(self, pred_rewards, pred_mental_states, next_rewards, next_mental_states,
                               each_type_object_num):
        title = ''  # 'ms_0, ms_1, p_ms_0, p_ms_1, rw, prw\n'
        ind = 0
        for obj_type in range(len(each_type_object_num)):
            for obj in range(each_type_object_num[obj_type]):
                title += self.get_title(self.goal_shape_options[obj],
                                        next_mental_states[ind], next_rewards[ind],
                                        pred_mental_states[ind], pred_rewards[ind])
                title += '\n'
                ind += 1
        title += self.get_title('.', next_mental_states[-1], next_rewards[-1],
                                pred_mental_states[-1], pred_rewards[-1])
        return title

    def get_object_shape_dictionary(self, object_locations, agent_location, each_type_object_num):
        shape_map = dict()
        for obj_type in range(self.object_type_num):
            at_type_object_locations = object_locations[object_locations[:, 0] == obj_type]
            for at_obj in range(each_type_object_num[obj_type]):
                key = tuple(at_type_object_locations[at_obj, 1:].tolist())
                shape_map[key] = self.goal_shape_options[at_obj]
        key = tuple(agent_location.tolist())
        shape_map[key] = '.'
        return shape_map

    def get_goal_location_from_goal_map(self, goal_map):
        goal_location = np.argwhere(goal_map)[0]
        return goal_location

    def next_environment(self):
        for i in range(25):
            few_many = [np.random.choice(['few', 'many']) for _ in range(self.params.OBJECT_TYPE_NUM)]
            environment = Environment(self.params, few_many_objects=few_many)
            state, _ = environment.reset()
            goal_map = get_random_action(state)
            object_locations = np.stack(np.where(state[0][1:, :, :]), axis=1)
            each_type_object_num = environment.each_type_object_num
            new_state, reward, terminated, truncated, info = environment.step(goal_map)

            # init_map, init_mental_state, states_params, goal_map, reward, next_map, next_mental_state
            tensors = get_experience_tensor(state, goal_map, reward, new_state, info)
            yield tensors, each_type_object_num, object_locations

    def test_random_goal_selection(self):
        row_num = 5
        col_num = 5
        for test_id in range(3):
            fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))
            for setting_id, outputs in enumerate(self.next_environment()):
                tensors, each_type_object_num, object_locations = outputs
                init_map, init_mental_state, states_params, goal_map, reward, next_map, next_mental_state, dt, rewarding = tensors
                agent_location = np.argwhere(init_map[0, :, :]).flatten()
                goal_location = np.argwhere(goal_map).flatten()
                selected_goal_type = np.argwhere(init_map[:, goal_location[0], goal_location[1]]).min().item()
                selected_goal_type = 2 if selected_goal_type == 0 else selected_goal_type - 1
                r = setting_id // col_num
                c = setting_id % col_num

                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                ax[r, c].invert_yaxis()

                shape_map = self.get_object_shape_dictionary(object_locations, agent_location, each_type_object_num)

                with torch.no_grad():
                    pred_reward_mental_state_dt, \
                        pred_rewarding = self.transition.transition_net(init_map.unsqueeze(0),
                                                                        goal_map.unsqueeze(0),
                                                                        init_mental_state.unsqueeze(0),
                                                                        states_params.unsqueeze(0))
                    pred_reward_mental_state_dt = pred_reward_mental_state_dt.cpu()
                    pred_rewarding = pred_rewarding.cpu().item()

                    pred_reward = pred_reward_mental_state_dt[0, 0]
                    pred_mental_state = pred_reward_mental_state_dt[0, 1:3]
                    pred_dt = pred_reward_mental_state_dt[0, 3].item()

                for i in range(self.height):
                    for j in range(self.width):
                        if (torch.tensor([i, j]) == agent_location).all():
                            spot_shape = shape_map[tuple(goal_location.tolist())]
                            goal_type = selected_goal_type
                            face_color = self.color_options[goal_type]
                            alpha = .4
                        elif tuple([i, j]) in shape_map.keys():
                            spot_shape = shape_map[tuple([i, j])]
                            ind = np.argwhere((object_locations[:, 1:] == [i, j]).all(axis=1)).min()
                            goal_type = object_locations[ind, 0]
                            face_color = 'none'
                            alpha = 1
                        else:
                            spot_shape = '_'
                            goal_type = 2
                            alpha = 1
                            face_color = self.color_options[goal_type]

                        # goal_type = 2 if goal_type == 0 else goal_type - 1
                        size = 10 if spot_shape == '.' else 100
                        ax[r, c].scatter(j, i,
                                         marker=spot_shape,
                                         s=size,
                                         alpha=alpha,
                                         edgecolor=self.color_options[goal_type],
                                         facecolor=face_color)

                ax[r, c].set_title(self.get_top_figure_title([init_mental_state,
                                                              next_mental_state,
                                                              reward,
                                                              pred_mental_state,
                                                              pred_reward,
                                                              dt,
                                                              rewarding,
                                                              pred_dt,
                                                              pred_rewarding]), fontsize=8)
                ax[r, c].set(adjustable='box', aspect='equal')

            plt.tight_layout(pad=0.1, w_pad=6, h_pad=1)
            fig.savefig('{0}/{1}_{2}.png'.format(self.res_folder, 'test', test_id))
            plt.close()

    def next_agent_on_environment(self):
        for object_reward in self.all_object_rewards:
            for mental_state_slope in self.all_mental_states_change:
                environment = Environment(self.params, ['few', 'many'])

                for subplot_id, mental_state in enumerate(self.all_mental_states):
                    for i in range(self.height):
                        for j in range(self.width):
                            environment.init_environment_for_test(
                                [i, j],
                                mental_state,
                                mental_state_slope,
                                object_reward)
                            yield environment, subplot_id

    def test_agents_at_all_locations(self):
        fig, ax = None, None
        row_num = 3
        col_num = 3
        plot_id = 0
        for setting_id, output in enumerate(self.next_agent_on_environment()):
            environment = output[0]
            # subplot_id = output[1]
            object_locations, agent_location = environment.get_possible_goal_locations()
            agent_location = agent_location.squeeze()
            state = environment.get_observation()
            each_type_object_num = environment.each_type_object_num
            env_map = torch.Tensor(state[0]).unsqueeze(0)
            mental_states = torch.Tensor(state[1]).unsqueeze(0)
            states_params = torch.Tensor(np.array(state[2:])).flatten().unsqueeze(0)

            if setting_id % (col_num * row_num) == 0:
                fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))
                plot_id = 0
            else:
                plot_id += 1

            r = plot_id // col_num
            c = plot_id % col_num

            shape_map = self.get_object_shape_dictionary(np.argwhere(state[0][1:, :, :]),
                                                         agent_location, each_type_object_num)

            # env_map, goal_map, mental_states, states_params
            pred_rewards, pred_mental_states, next_rewards, next_mental_states = self.get_all_predictions(environment)

            ax[r, c].scatter(agent_location[1],
                             agent_location[0],
                             marker='2',
                             s=200,
                             edgecolor='b')
            for i in range(self.height):
                for j in range(self.width):
                    if (np.array([i, j]) == object_locations).all(axis=1).any():
                        spot_shape = shape_map[tuple([i, j])]
                        ind = np.argwhere((object_locations == [i, j]).all(axis=1)).min()
                        goal_type = np.argwhere(state[0][1:, :, :])[ind, 0]
                        face_color = 'none'
                        alpha = 1
                    else:
                        spot_shape = '.'
                        goal_type = 2
                        alpha = 1
                        face_color = self.color_options[goal_type]

                    # goal_type = 2 if goal_type == 0 else goal_type - 1
                    size = 10 if spot_shape == '.' else 100
                    ax[r, c].scatter(j, i,
                                     marker=spot_shape,
                                     s=size,
                                     alpha=alpha,
                                     edgecolor=self.color_options[goal_type],
                                     facecolor=face_color)
            ax[r, c].set_title(
                self.get_right_figure_title(pred_rewards, pred_mental_states, next_rewards, next_mental_states,
                                            each_type_object_num),
                # loc='center', #ha='left', va='center',
                x=1, y=0, loc='left',
                fontsize=8)
            ax[r, c].tick_params(length=0)
            # ax[r, c].set(adjustable='box')
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            ax[r, c].invert_yaxis()
            if (setting_id + 1) % (row_num * col_num) == 0:
                plt.tight_layout(pad=15, w_pad=8, h_pad=1.5)

                fig.savefig('{0}/slope_{1}-{2}_or_{3}-{4}__{5}.png'.format(self.res_folder,
                                                                           states_params[0][0],
                                                                           states_params[0][1],
                                                                           states_params[0][2],
                                                                           states_params[0][3],
                                                                           setting_id + 1))
                plt.close()

    def load_model(self, params):
        if params.USE_PRETRAINED:
            model_path = os.path.join('./pretrained', 'model.pt')
        else:
            model_path = os.path.join(self.res_folder, 'model.pt')
        model_parameters = torch.load(model_path, map_location=self.device)
        transition = Transition(params)
        transition.transition_net.load_state_dict(model_parameters)
        return transition

    def get_all_predictions(self, environment: Environment):
        state = environment.get_observation()
        env_map = torch.Tensor(state[0]).unsqueeze(0)
        mental_states = torch.Tensor(state[1]).unsqueeze(0)
        states_params = torch.Tensor(np.array(state[2:])).flatten().unsqueeze(0)

        object_locations, agent_location = environment.get_possible_goal_locations()
        if (object_locations == agent_location).all(axis=1).any():
            all_goal_locations = deepcopy(object_locations)
        else:
            all_goal_locations = np.concatenate([object_locations, agent_location], axis=0)

        next_rewards, next_mental_states = [], []
        pred_rewards, pred_mental_states = [], []
        for goal_location in all_goal_locations:
            imagined_environment = deepcopy(environment)
            goal_map = torch.zeros_like(env_map[:, 0, :, :])
            goal_map[0, goal_location[0], goal_location[1]] = 1
            pred_reward_mental_state = self.transition.transition_net(env_map, goal_map, mental_states,
                                                                      states_params).cpu()
            pred_reward = pred_reward_mental_state[0, 0].detach().item()
            pred_mental_state = pred_reward_mental_state[0, 1:].detach()

            next_obs, next_reward, _, _, _ = imagined_environment.step(goal_map=goal_map.squeeze().numpy())

            next_rewards.append(next_reward)
            next_mental_states.append(next_obs[1])
            pred_rewards.append(pred_reward)
            pred_mental_states.append(pred_mental_state)

        return pred_rewards, pred_mental_states, next_rewards, next_mental_states
