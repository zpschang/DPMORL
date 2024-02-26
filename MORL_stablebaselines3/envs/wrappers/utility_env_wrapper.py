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
import sys
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

class ObsInfoWrapper(gym.Wrapper):
    def __init__(
            self,
            env, 
            reward_dim, 
            reward_dim_indices
    ):
        super().__init__(env)

        self.cur_timesteps = 0
        self.reward_dim = reward_dim
        self.reward_dim_indices = reward_dim_indices
        self.actual_reward_dim = self.env.reward_dim
        self.zt = np.zeros(self.actual_reward_dim)
        if isinstance(self.action_space, gymnasium.spaces.Box):
            self.action_space = gym.spaces.Box(low=self.action_space.low, high=self.action_space.high,
                                           shape=self.action_space.shape, dtype=self.action_space.dtype)
        elif isinstance(self.action_space, gymnasium.spaces.Discrete):
            self.action_space = gym.spaces.Discrete(self.action_space.n)
        if isinstance(self.observation_space, gymnasium.spaces.Box):
            self.obs_high = np.array(self.observation_space.high, dtype=np.float32)
            self.obs_low = np.array(self.observation_space.low, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=self.observation_space.low, high=self.observation_space.high,
                                                   shape=self.observation_space.shape,
                                                   dtype=self.observation_space.dtype)
        elif isinstance(self.observation_space, gymnasium.spaces.Discrete):
            # self.observation_space = gym.spaces.Discrete(self.observation_space.n + 2)
            self.observation_space = gym.spaces.Discrete(self.observation_space.n)

    def reset(self) -> np.ndarray:
        """Resets the environment."""
        obs, info = super().reset()
        self.zt = np.zeros(self.actual_reward_dim)
        self.cur_timesteps = 0
        return obs

    def _augment_state(self, state: np.ndarray, returns: np.ndarray):
        """Augmenting the state with the safety state, if needed"""
        augmented_state = np.hstack([state, returns])
        return augmented_state

    def step(self, action):
        """ Step through the environment. """
        next_obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        zt_next = self.zt + reward
        info['true_reward'] = reward
        info['zt'] = zt_next
        self.cur_timesteps += 1
        self.zt = zt_next
        if hasattr(self.env, "_max_episode_steps") and self.cur_timesteps >= self.env._max_episode_steps:
            done = True
            info['max_steps_reached'] = True
        if done:
            ep_rew = self.zt
            ep_len = self.cur_timesteps
            ep_info = {"r": ep_rew, "l": ep_len}
            info["episode"] = ep_info

        # augmented_state = self._augment_state(next_obs, self.zt)
        return next_obs, reward[self.reward_dim_indices], done, info

class MultiEnv_UtilityFunction(VecEnvWrapper):
    def __init__(
            self,
            venv: VecEnv,
            utility_function,
            discount_factor=0.99,
            reward_dim=2,
            augment_state=False,
            **kwargs
    ):
        super().__init__(venv)
        self.num_envs = venv.num_envs
        self.utility_function = utility_function
        self.reward_dim = reward_dim
        self.augment_state = augment_state

        self.zt = np.zeros([self.num_envs, self.reward_dim])
        self.gamma = discount_factor  # same to gamma for RL
        self.action_space = venv.action_space
        
        if self.augment_state:
            low = np.hstack([venv.observation_space.low, np.full((self.reward_dim, ), -np.inf)])
            high = np.hstack([venv.observation_space.high, np.full((self.reward_dim, ), np.inf)])
            
            self.observation_space = gym.spaces.Box(low=low, high=high,
                                                    shape=low.shape,
                                                    dtype=venv.observation_space.dtype)
            self.min_val = self.utility_function.min_val[np.newaxis, :]
            self.max_val = self.utility_function.max_val[np.newaxis, :]
        else:
            self.observation_space = venv.observation_space
        
            
    def step_wait(self):
        return self.venv.step_wait()

    def update_utility_function(self, func):
        self.utility_function = func
        self.utility_function.eval()

    def reset(self) -> np.ndarray:
        """Resets the environment."""
        obs = self.venv.reset()
        self.zt = np.zeros([self.num_envs, self.reward_dim])
        self.total_reward = np.zeros([self.num_envs])
        if self.augment_state:
            normalized_return = (self.zt - self.min_val) / (self.max_val - self.min_val)
            obs = self._augment_state(obs, normalized_return)
        return obs

    def _augment_state(self, state: np.ndarray, returns: np.ndarray):
        """Augmenting the state with the safety state, if needed"""
        augmented_state = np.hstack([state, returns])
        return augmented_state

    def step(self, action):
        """ Step through the environment. """
        next_obs, reward, done, info = super().step(action)

        zt_next = self.zt + reward
        with torch.no_grad():
            new_reward = self.utility_function(zt_next) - self.utility_function(self.zt)
        self.total_reward += new_reward
        self.zt = zt_next
        if done.any():
            if self.augment_state:
                normalized_return = (self.zt - self.min_val) / (self.max_val - self.min_val)
                for index, info_env in enumerate(info):
                    if 'terminal_observation' in info_env:
                        assert done[index]
                        info_env['terminal_observation'] = np.concatenate([info_env['terminal_observation'], normalized_return[index]], 0)
            
            self.total_reward[done] = 0.0
            self.zt[done] = 0.0
        # Augment state with reward gained
        if self.augment_state:
            normalized_return = (self.zt - self.min_val) / (self.max_val - self.min_val)
            next_obs = self._augment_state(next_obs, normalized_return)

        return next_obs, new_reward, done, info


if __name__ == "__main__":
    pass