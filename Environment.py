import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Environment(gym.Env):
    def __init__(self, params, few_many_objects):
        self.params = params
        self.height = params.HEIGHT
        self.width = params.WIDTH
        self.few_many_objects = few_many_objects
        self.metadata = {"render_modes": None}
        self.object_type_num = params.OBJECT_TYPE_NUM
        self._no_reward_threshold = -5
        self._goal_selection_step = 0
        self._env_map = np.zeros((1 + self.object_type_num, self.height, self.width), dtype=int)
        self._mental_states = np.empty((self.object_type_num,), dtype=np.float64)
        self._mental_states_slope = np.empty((self.object_type_num,), dtype=np.float64)
        self._environment_object_reward = np.empty((self.object_type_num,), dtype=np.float64)
        self._environment_states_parameters = [self._mental_states_slope, self._environment_object_reward]
        self._environment_states_parameters_range = [self.params.MENTAL_STATES_SLOPE_RANGE,
                                                     self.params.ENVIRONMENT_OBJECT_REWARD_RANGE]

        # self.controller = Controller(self.height, self.width)

        self.observation_space = spaces.Tuple(
            # Usually, it will not be possible to use elements of this space directly in learning code.
            # However, you can easily convert Dict observations to
            # flat arrays by using a gymnasium.wrappers.FlattenObservation wrapper
            (spaces.Box(0, 1, shape=(1 + self.object_type_num,
                                     self.height, self.width), dtype=int),  # 'env_map'
             spaces.Box(self.params.INITIAL_MENTAL_STATES_RANGE[0], 2 ** 63 - 2,
                        shape=(self.object_type_num,), dtype=float),  # 'mental_states'
             spaces.Box(self.params.MENTAL_STATES_SLOPE_RANGE[0], self.params.MENTAL_STATES_SLOPE_RANGE[1],
                        shape=(self.object_type_num,), dtype=float),  # 'mental_states_slope'
             spaces.Box(self.params.ENVIRONMENT_OBJECT_REWARD_RANGE[0], self.params.ENVIRONMENT_OBJECT_REWARD_RANGE[1],
                        shape=(self.object_type_num,), dtype=float))  # 'environment_object_reward'
        )
        # self.action_space = spaces.MultiDiscrete([self.height, self.width])
        self.action_space = spaces.Box(-2 ** 63, 2 ** 63 - 2,
                                       shape=(self.params.WIDTH * self.params.HEIGHT,), dtype=float)

    def sample(self):  # return size: ndarray (198, )
        self._env_map = np.zeros_like(self._env_map, dtype=int)
        self._init_random_map(reset=True)
        self._init_random_mental_states()
        self._init_random_parameters()
        # flat_observation = self._flatten_observation()
        observation = self.get_observation()
        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env_map = np.zeros_like(self._env_map, dtype=int)
        self._goal_selection_step = 0
        sample_observation = self.sample()
        return sample_observation, dict()

    def step(self, goal_map):
        self._goal_selection_step += 1
        goal_location = np.argwhere(goal_map).flatten()  # location of 1 in map
        diagonal_steps, straight_steps = self._get_agent_distance_to_location(goal_location[0], goal_location[1])
        step_length = math.sqrt(2) * diagonal_steps + straight_steps
        self._update_agent_locations(goal_location)
        dt = np.array(1) if step_length < 1.4 else step_length
        object_reward = self._env_map[1:,
                        self._agent_location[0],
                        self._agent_location[1]] * self._environment_object_reward
        self._update_object_locations()

        mental_states_cost = self._get_mental_states_cost(diagonal_steps, straight_steps)
        self._update_mental_state_after_step(dt=dt)

        positive_mental_states_before_reward = self._total_positive_mental_states()
        self._update_mental_states_after_object(u=object_reward)
        positive_mental_states_after_reward = self._total_positive_mental_states()
        mental_states_reward = np.maximum(0,
                                          positive_mental_states_before_reward - positive_mental_states_after_reward)
        reward = np.array(mental_states_reward - step_length - mental_states_cost)

        terminated = False
        truncated = False
        # be careful about this, we might need to try to have always (or after 5 goal selection step) terminated=False,
        # and just maximize the reward.
        # (observation, reward, terminated, truncated, info)
        return self.get_observation(), reward, terminated, truncated, dict()

    def render(self):
        return None

    def _get_mental_states_cost(self, diagonal_steps, straight_steps):
        if diagonal_steps == 0 and straight_steps == 0:  # stays
            cost = self._total_positive_mental_states()
            return cost
        else:
            cost = 0
            mental_states = self._mental_states.copy()

        for step in range(diagonal_steps):
            carried_need = np.maximum(0, mental_states).sum() * math.sqrt(2)
            cost += carried_need
            mental_states += math.sqrt(2) * self._mental_states_slope

        for step in range(straight_steps):
            carried_need = np.maximum(0, mental_states).sum()
            cost += carried_need
            mental_states += self._mental_states_slope

        return cost

    def get_observation(self):
        observation = [self._env_map.copy(), self._mental_states.copy()]
        for i in range(len(self._environment_states_parameters)):
            observation.append(self._environment_states_parameters[i].copy())
        return observation

    def _update_agent_locations(self, new_location):
        self._env_map[0, self._agent_location[0], self._agent_location[1]] = 0
        self._agent_location = new_location
        self._env_map[0, self._agent_location[0], self._agent_location[1]] = 1

    def _update_object_locations(self):
        if self._env_map[1:, self._agent_location[0], self._agent_location[1]].sum() == 0:  # not reached an object
            return
        reached_object_type = np.argwhere(self._env_map[1:, self._agent_location[0], self._agent_location[1]])[0, 0]
        self.each_type_object_num[reached_object_type] += 1
        self._init_random_map(object_num_on_map=self.each_type_object_num, reset=False)  # argument is kind of redundant
        self._env_map[reached_object_type + 1, self._agent_location[0], self._agent_location[1]] = 0
        self.each_type_object_num[reached_object_type] -= 1

    def _get_agent_distance_to_location(self, x2, y2):
        dx = abs(x2 - self._agent_location[0])
        dy = abs(y2 - self._agent_location[1])
        min_dis = min(dx, dy)
        max_dis = max(dx, dy)
        diagonal_steps = min_dis
        straight_steps = max_dis - min_dis
        # return math.sqrt(2) * diagonal_steps + straight_steps
        return diagonal_steps, straight_steps

    def _total_positive_mental_states(self):
        total_need = np.maximum(0, self._mental_states).sum()
        return total_need

    def _update_mental_state_after_step(self, dt):
        dz = (self._mental_states_slope * dt)
        self._mental_states += dz

    def _update_mental_states_after_object(self, u):  # u > 0
        mental_states_threshold = np.empty_like(self._mental_states)
        for i, state in enumerate(self._mental_states):
            if state < self._no_reward_threshold:
                mental_states_threshold[i] = state
            else:
                mental_states_threshold[i] = self._no_reward_threshold

        self._mental_states += -(1 * u)
        self._mental_states = np.maximum(self._mental_states, mental_states_threshold)

    def _init_object_num_on_map(self) -> np.array:
        # e.g., self.few_many_objects : ['few', 'many']
        few_range = np.array([1, 2, 3, 4])
        many_range = np.array([1, 2, 3, 4])
        ranges = {'few': few_range,
                  'many': many_range}
        each_type_object_num = np.zeros((self.object_type_num,), dtype=int)
        for i, item in enumerate(self.few_many_objects):
            at_type_obj_num = np.random.choice(ranges[item])
            each_type_object_num[i] = at_type_obj_num

        return each_type_object_num

    def _init_random_map(self, object_num_on_map=None, reset=False):  # add agent location
        if self._env_map[0, :, :].sum() == 0:  # no agent on map
            self._agent_location = np.random.randint(low=0, high=self.height, size=(2,))
            self._env_map[0, self._agent_location[0], self._agent_location[1]] = 1

        object_num_already_on_map = self._env_map[1:, :, :].sum(axis=(1, 2))
        if object_num_on_map is None:
            self.each_type_object_num = self._init_object_num_on_map()
        else:
            self.each_type_object_num = object_num_on_map

        ###  ERASE THESE
        # object_num_already_on_map = [1, 0]
        # self._env_map[1, self._agent_location[0], self._agent_location[1]] = 1
        ####
        object_num_to_init = self.each_type_object_num - object_num_already_on_map

        object_count = 0
        check_no_object_layer_start = 1 if reset else 0
        for obj_type in range(self.object_type_num):
            for at_obj in range(object_num_to_init[obj_type]):
                while True:
                    sample_location = np.random.randint(low=0, high=[self.height, self.width],
                                                        size=(self.object_type_num,))
                    if self._env_map[check_no_object_layer_start:, sample_location[0], sample_location[1]].sum() == 0:
                        self._env_map[1 + obj_type, sample_location[0], sample_location[1]] = 1
                        break

                object_count += 1

    def _init_random_mental_states(self):
        self._mental_states[:] = self._get_random_vector(attr_range=self.params.INITIAL_MENTAL_STATES_RANGE,
                                                         prob_equal=self.params.PROB_EQUAL_PARAMETERS)

    def _init_random_parameters(self):
        for i in range(len(self._environment_states_parameters_range)):
            self._environment_states_parameters[i][:] = self._get_random_vector(
                attr_range=self._environment_states_parameters_range[i],
                prob_equal=self.params.PROB_EQUAL_PARAMETERS,
                only_positive=True)

    def _get_random_vector(self, attr_range, prob_equal=0, only_positive=False):
        p = random.uniform(0, 1)
        if p <= prob_equal:
            size = 1
        else:
            size = self.object_type_num

        random_vec = np.random.uniform(low=attr_range[0],
                                       high=attr_range[1],
                                       size=(size,))
        if only_positive:
            random_vec = np.abs(random_vec)
        return random_vec

    def init_environment_for_test(self, agent_location, mental_states, mental_states_slope,
                                  object_reward):  # mental_states_parameters
        self._env_map[0, :, :] = 0  # np.zeros_like(self._env_map[0, :, :], dtype=int)
        self._agent_location = np.array(agent_location)
        self._env_map[0, agent_location[0], agent_location[1]] = 1
        each_type_object_num = None
        if hasattr(self, 'each_type_object_num'):
            each_type_object_num = self.each_type_object_num
        self._init_random_map(each_type_object_num)
        # object_locations = np.argwhere(self._env_map[1:, :, :])

        self._mental_states = np.array(mental_states, dtype=float)
        self._environment_states_parameters[0][:] = np.array(mental_states_slope, dtype=float)
        self._environment_states_parameters[1][:] = np.array(object_reward, dtype=float)

    def get_possible_goal_locations(self):
        object_locations = np.argwhere(self._env_map[1:, :, :])[:, 1:]
        agent_locations = np.argwhere(self._env_map[0, :, :])
        return object_locations, agent_locations