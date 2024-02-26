from MORL_stablebaselines3.envs.gridworlds import gridworld_base
from MORL_stablebaselines3.envs.gridworlds.gridworld_base import Gridworld, RandomPlayer, Viewport

import gym
from gym import spaces

import numpy as np
import copy
import time
from collections import deque

class MOEnvDummy():

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        obs = np.zeros(shape=self.observation_space.shape)
        return obs

class MOGridworld(Gridworld):
    """Base class for multi objective gridworld environments.

    """

    def __init__(self,
                 map=gridworld_base.TEST_MAP,
                 object_mapping=gridworld_base.TEST_MAPPING,
                 viewport=Viewport(),
                 from_pixels=True,
                 inflation=1,
                 random_items=[],
                 random_items_frame=0,
                 init_agents=[],
                 agent_start=[0, 0],
                 agent_color=(0.0, 0.0, 255.0),
                 encounter_other_agents=False,
                 max_steps=50,
                 preference=np.array([-1,-5,+2,-2,-1]),
                 include_agents=True,
                 agent_preferences=[]):
        self.preference = preference
        self.include_agents = include_agents
        self.agent_preferences = agent_preferences
        super(MOGridworld, self).__init__(map=map,
                                          object_mapping=object_mapping,
                                          viewport=viewport,
                                          from_pixels=from_pixels,
                                          inflation=inflation,
                                          random_items=random_items,
                                          random_items_frame=random_items_frame,
                                          init_agents=init_agents,
                                          agent_start=agent_start,
                                          agent_color=agent_color,
                                          encounter_other_agents=encounter_other_agents,
                                          max_steps=max_steps)

    def step(self, action):

        self.step_count += 1

        new_pos = self.get_new_pos_from_action(action, self.agent_pos)

        reward = np.zeros(shape=(self.max_idx + 1,), dtype='uint8')
        reward[0] = 1 # Added one time step
        if self.is_walkable(new_pos[0], new_pos[1]):
            self.agent_pos = new_pos
        else:
            reward[1] = 1 # Agent tried to walk in illegal direction

        # Update grid agents
        agent_rewards = 0
        for agent in self.agents:
            agent.step(self)
            agent_rewards += agent.reward
            reward[agent.idx] += agent.reward
            agent.reset_reward()

        if self.from_pixels:
            obs = self.create_image_observation()
        else:
            obs = self.create_discrete_observation()

        idxs = self.encounter_object_idx(self.agent_pos[0], self.agent_pos[1], self.encounter_other_agents)
        for idx in idxs:
             reward[idx] += 1

        done = self.is_done(self.include_agents, self.agent_preferences)

        return (obs, reward, done, agent_rewards)

    def encounter_object_idx(self, column, row, encounter_agents=True):
        idxs = []
        if self.within_bounds(column, row) and self.grid[row, column] is not None:
            idxs.append(self.grid[row, column].idx)
            if self.grid[row, column].is_consumed:
                self.grid[row, column] = None

        if encounter_agents:
            for agent in self.agents:
                if (agent.position[0]==column) and (agent.position[1]==row):
                    idxs.append(agent.idx)
                    if agent.is_consumed:
                        self.agents.remove(agent)

        return idxs

    def is_done(self, include_agents=True, agent_preferences=[]):
        if self.step_count > self.max_steps:
            return True

        for row in range(self.grid.shape[0]):
            for column in range(self.grid.shape[1]):
                if self.grid[row, column] is not None and self.preference[self.grid[row, column].idx] > 0:
                    return False
                if self.grid[row, column] is not None and self.grid[row, column].idx in agent_preferences:
                    return False

        if include_agents:
            for agent in self.agents:
                if self.preference[agent.idx] > 0:
                    return False

        return True

    def reset(self, preference=None):
        if preference is not None:
            self.preference = preference
        self.load_map(self.map, self.object_mapping)
        self.agent_pos = self.agent_start
        self.agents.clear()
        for agent in self.init_agents:
            self.agents.append(copy.deepcopy(agent))
        for agent in self.agents:
            agent.reset()
        self.step_count = 0
        if self.from_pixels:
            obs = self.create_image_observation()
        else:
            obs = self.create_discrete_observation()
        return obs

class MORandomPlayer(RandomPlayer):

    def __init__(self, env, num_episodes):
        super().__init__(env,num_episodes)

    def step_env(self):
        obs, reward, done, info = self.env.step(self.env.action_space.sample())
        return (reward, done)

if __name__=="__main__":
    my_grid = MOGridworld(gridworld_base.TEST_MAP, 
                          gridworld_base.TEST_MAPPING,
                          preference=np.array([-1,-5,+2,-2,-1, -1]))
    player = MORandomPlayer(my_grid, 2)
    player.run()
