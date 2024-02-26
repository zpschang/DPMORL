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
def morl_env_wrapper(cls):
    """ Class decorator for sauteing an environment. """
    class MORLEnv_UtilityFunction(cls):
        def __init__(
                self,
                discount_factor=0.99,
                render_mode: Optional[str] = None,
                **kwargs
        ):

            self.render_mode = render_mode
            super().__init__()

            self.utility_function = None

            self.cur_timesteps = 0
            # IPython.embed()
            # self.zt = np.zeros(self.reward_space.shape[0]) # For more-than-2 dim ones
            self.zt = np.zeros(2)
            self.gamma = discount_factor  # same to gamma for RL
            if isinstance(self.action_space, gymnasium.spaces.Box):
                self.action_space = gym.spaces.Box(low=self.action_space.low, high=self.action_space.high,
                                               shape=self.action_space.shape, dtype=self.action_space.dtype)
            elif isinstance(self.action_space, gymnasium.spaces.Discrete):
                self.action_space = gym.spaces.Discrete(self.action_space.n)
            # IPython.embed()
            if isinstance(self.observation_space, gymnasium.spaces.Box):
                self.obs_high = np.array(np.hstack([self.observation_space.high, np.full(self.zt.shape, np.inf)]),
                                         dtype=np.float32)
                self.obs_low = np.array(np.hstack([self.observation_space.low, np.full(self.zt.shape, -np.inf)]),
                                        dtype=np.float32)
                self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high,
                                                        shape=(self.observation_space.shape[0]+self.zt.shape[0],),
                                                        dtype=self.observation_space.dtype)
            elif isinstance(self.observation_space, gymnasium.spaces.Discrete):
                self.observation_space = gym.spaces.Discrete(self.observation_space.n+2)

        def update_utility_function(self, func):
            self.utility_function = func
            self.utility_function.eval()

        def reset(self) -> np.ndarray:
            """Resets the environment."""
            obs, info = super().reset()
            self.zt = np.zeros(2)
            augmented_state = self._augment_state(obs, self.zt)
            self.cur_timesteps = 0
            return augmented_state

        def _augment_state(self, state: np.ndarray, returns: np.ndarray):
            """Augmenting the state with the safety state, if needed"""
            augmented_state = np.hstack([state, returns])
            return augmented_state

        def step(self, action):
            """ Step through the environment. """
            next_obs, reward, done, _, info = super().step(action)

            # zt_next = self.zt + reward
            zt_next = self.zt + reward[: 2]
            # TODO: zt_next = self.zt + self._augment_state(reward, -info['cost']) # next_safety_state := budget - cost
            # TODO: remove safety state expanding
            # IPython.embed()
            info['true_reward'] = reward
            info['zt'] = zt_next
            self.cur_timesteps += 1
            reward = self.gamma * self.utility_function(zt_next[np.newaxis, :]) - \
                     self.utility_function(self.zt[np.newaxis, :])
            self.zt = zt_next
            if done:
                ep_rew = self.zt
                ep_len = self.cur_timesteps
                ep_info = {"r": ep_rew, "l": ep_len}
                info["episode"] = ep_info

            augmented_state = self._augment_state(next_obs, self.zt)
            return augmented_state, reward, done, info

        # def reward_fn(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        #     """ Compute rewards in a batch. """
        #     reward = self.wrap._reward_fn(states, actions, next_states, is_tensor=True)
        #     if self.use_state_augmentation:
        #         # shape reward for model-based predictions
        #         reward = self.utility_function.get_logits(self.zt)
        #     return reward
    return MORLEnv_UtilityFunction


if __name__ == "__main__":
    pass