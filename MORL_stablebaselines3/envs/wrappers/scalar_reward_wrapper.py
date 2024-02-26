import IPython
import numpy as np
import torch
from typing import Optional
from gym import Env
import gym
import gymnasium
from MORL_stablebaselines3.envs.utils import Array
import math
# from MORL_stablebaselines3.morl.utility_function_torch import Utility_Function
# def scalar_reward_wrapper(cls, weight):
#     """ Class decorator for sauteing an environment. """
#     class ScalarRewardEnv(cls):
#         def __init__(
#                 self,
#                 reward_weights=weight,
#                 render_mode: Optional[str] = None,
#                 **kwargs
#         ):
#
#             self.render_mode = render_mode
#             self.reward_weights = np.array(reward_weights)
#             super().__init__()
#             self.cur_timesteps = 0
#             self.returns = 0
#             if isinstance(self.action_space, gymnasium.spaces.Box):
#                 self.action_space = gym.spaces.Box(low=self.action_space.low, high=self.action_space.high,
#                                                shape=self.action_space.shape, dtype=self.action_space.dtype)
#             elif isinstance(self.action_space, gymnasium.spaces.Discrete):
#                 self.action_space = gym.spaces.Discrete(self.action_space.n)
#             # IPython.embed()
#             if isinstance(self.observation_space, gymnasium.spaces.Box):
#                 self.observation_space = gym.spaces.Box(low=self.observation_space.low, high=self.observation_space.high,
#                                                         shape=(self.observation_space.shape[0],),
#                                                         dtype=self.observation_space.dtype)
#             elif isinstance(self.observation_space, gymnasium.spaces.Discrete):
#                 self.observation_space = gym.spaces.Discrete(self.observation_space.n)
#
#         def reset(self) -> np.ndarray:
#             """Resets the environment."""
#             obs, info = super().reset()
#             self.cur_timesteps = 0
#             self.returns = 0
#             return obs
#
#         def step(self, action):
#             """ Step through the environment. """
#             next_obs, vector_reward, done, _, info = super().step(action)
#
#             scalar_reward = np.dot(vector_reward, self.reward_weights)
#             info['zt'] = np.zeros(2)
#             self.cur_timesteps += 1
#             self.returns += scalar_reward
#             if done:
#                 ep_rew = self.returns
#                 ep_len = self.cur_timesteps
#                 ep_info = {"r": ep_rew, "l": ep_len}
#                 info["episode"] = ep_info
#             return next_obs, scalar_reward, done, info
#
#     return ScalarRewardEnv

class ScalarRewardEnv(gym.Wrapper):
    def __init__(
            self,
            env,
            reward_weights,
            **kwargs
    ):
        self.reward_weights = np.array(reward_weights)
        super().__init__(env)
        self.cur_timesteps = 0
        self.returns = 0
        self.vec_returns = np.zeros(2)
        if isinstance(self.action_space, gymnasium.spaces.Box):
            self.action_space = gym.spaces.Box(low=self.action_space.low, high=self.action_space.high,
                                               shape=self.action_space.shape, dtype=self.action_space.dtype)
        elif isinstance(self.action_space, gymnasium.spaces.Discrete):
            self.action_space = gym.spaces.Discrete(self.action_space.n)
        # IPython.embed()
        if isinstance(self.observation_space, gymnasium.spaces.Box):
            self.observation_space = gym.spaces.Box(low=self.observation_space.low, high=self.observation_space.high,
                                                    shape=(self.observation_space.shape[0],),
                                                    dtype=self.observation_space.dtype)
        elif isinstance(self.observation_space, gymnasium.spaces.Discrete):
            self.observation_space = gym.spaces.Discrete(self.observation_space.n)

    def reset(self) -> np.ndarray:
        """Resets the environment."""
        obs, info = super().reset()
        self.cur_timesteps = 0
        self.returns = 0
        self.vec_returns = np.zeros(2)
        return obs

    def step(self, action):
        """ Step through the environment. """
        next_obs, vector_reward, done, _, info = super().step(action)

        scalar_reward = np.dot(vector_reward, self.reward_weights)
        self.vec_returns += vector_reward[: 2]
        self.cur_timesteps += 1
        self.returns += scalar_reward
        info['zt'] = np.zeros(2)
        if hasattr(self.env, "_max_episode_steps") and self.cur_timesteps >= self.env._max_episode_steps:
            done = True
            info['max_steps_reached'] = True
        if done:
            ep_rew = self.returns
            ep_len = self.cur_timesteps
            # TODO: add vec returns ep_info = {"r": ep_rew, "l": ep_len, "vec_r": }
            ep_info = {"r": ep_rew, "l": ep_len, "zt": self.vec_returns}
            info["episode"] = ep_info
        return next_obs, scalar_reward, done, info


if __name__ == "__main__":
    pass