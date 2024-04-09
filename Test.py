import math

# from stable_baselines3 import PPO
import os
from sbx import PPO
import torch
import numpy as np
from ReplayMemory import get_experience_tensor

from Transition import Transition, get_random_action
from Environment import Environment
import matplotlib.pyplot as plt


# def get_predefined_parameters(params, param_name):
#     if param_name == 'all_mental_states':
#         all_param = [[-10, -5, 0, 5, 10]] * params.OBJECT_TYPE_NUM
#     elif param_name == 'all_object_rewards':
#         # all_param = [[0, 4, 8, 12, 16, 20]] * num_object
#         param_range = params.ENVIRONMENT_OBJECT_REWARD_RANGE
#         all_param = np.expand_dims(np.linspace(param_range[0],
#                                                param_range[1], num=min(param_range[1] - param_range[0] + 1, 4),
#                                                dtype=int), axis=0).tolist() * params.OBJECT_TYPE_NUM
#     elif param_name == 'all_mental_states_change':
#         # all_param = [[0, 1, 2, 3, 4, 5]] * num_object
#         param_range = params.MENTAL_STATES_SLOPE_RANGE
#         all_param = np.expand_dims(np.linspace(param_range[0],
#                                                param_range[1],
#                                                num=min(param_range[1] - param_range[0] + 1, 4), dtype=int),
#                                    axis=0).tolist() * params.OBJECT_TYPE_NUM
#     else:
#         print('no such parameters')
#         return
#     num_param = len(all_param[0]) ** params.OBJECT_TYPE_NUM
#     param_batch = []
#     for i, ns in enumerate(itertools.product(*all_param)):
#         param_batch.append(list(ns))
#     return param_batch


class Test:
    def __init__(self, utils):
        self.params = utils.params
        self.device = 'cuda' if ((self.params.DEVICE == 'auto' or self.params.DEVICE == 'cuda') and torch.cuda.is_available()) else 'cpu'
        self.res_folder = utils.res_folder
        self.transition = self.load_model(self.params)
        self.height = utils.params.HEIGHT
        self.width = utils.params.WIDTH
        self.object_type_num = utils.params.OBJECT_TYPE_NUM
        self.color_options = [[1, 0, .2], [0, .8, .2], [0, 0, 0]]
        self.goal_shape_options = ['*', 's', 'P', 'o', 'D', 'X']
        self.objects_color_name = ['r', 'g', 'b']  # 2: stay

    def get_figure_title(self, parameters):
        [init_mental_state,
         next_mental_state,
         reward,
         pred_mental_state,
         pred_reward] = parameters
        title = r''
        # title = '$n_{0}: {1:.2f}'.format('{' + self.objects_color_name[0] + '}', init_mental_state[0])
        for i in range(len(init_mental_state)):
            title += r"$n^{0}_{1}:{2:.2f}$, ".format('{0}', '{' + self.objects_color_name[i] + '}', init_mental_state[i])
        title += '\n'
        for i in range(len(next_mental_state)):
            title += r"$n^{0}_{1}:{2:.2f}$, ".format('{1}', '{' + self.objects_color_name[i] + '}', next_mental_state[i])
        title += r"rw:{0:.2f}".format(reward)
        title += "\n"
        for i in range(len(pred_mental_state)):
            title += r"${0}^{1}_{2}:{3:.2f}$, ".format('{pn}', '{1}', '{' + self.objects_color_name[i] + '}', pred_mental_state[i])
        title += r"prw:{0:.2f}".format(pred_reward)

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

    def next_agent_and_environment(self):
        for i in range(25):
            few_many = [np.random.choice(['few', 'many']) for _ in range(self.params.OBJECT_TYPE_NUM)]
            environment = Environment(self.params, few_many_objects=few_many)
            state, _ = environment.reset()
            goal_map = get_random_action(state)
            object_locations = np.stack(np.where(state[0][1:, :, :]), axis=1)
            each_type_object_num = environment.each_type_object_num
            new_state, reward, terminated, truncated, _ = environment.step(goal_map)

            # init_map, init_mental_state, states_params, goal_map, reward, next_map, next_mental_state
            tensors = get_experience_tensor(state, goal_map, reward, new_state)
            yield tensors, each_type_object_num, object_locations

    def get_goal_directed_actions(self):
        row_num = 5
        col_num = 5
        for test_id in range(3):
            fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))
            for setting_id, outputs in enumerate(self.next_agent_and_environment()):
                tensors, each_type_object_num, object_locations = outputs
                init_map, init_mental_state, states_params, goal_map, reward, next_map, next_mental_state = tensors
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
                    pred_reward_mental_state = self.transition.transition_net(init_map.unsqueeze(0),
                                                                              goal_map.unsqueeze(0),
                                                                              init_mental_state.unsqueeze(0),
                                                                              states_params.unsqueeze(0)).cpu()
                    pred_reward = pred_reward_mental_state[0, 0]
                    pred_mental_state = pred_reward_mental_state[0, 1:]

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

                ax[r, c].set_title(self.get_figure_title([init_mental_state,
                                                          next_mental_state,
                                                          reward,
                                                          pred_mental_state,
                                                          pred_reward]), fontsize=8)
                ax[r, c].set(adjustable='box', aspect='equal')

            plt.tight_layout(pad=0.1, w_pad=6, h_pad=1)
            fig.savefig('{0}/{1}_{2}.png'.format(self.res_folder, 'test', test_id))
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
