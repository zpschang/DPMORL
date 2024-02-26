import IPython
import numpy as np
import torch
from gym import Env
from gym import spaces
from MORL_stablebaselines3.envs.utils import Array
from MORL_stablebaselines3.morl.utility_function import Utility_Function


def morl_env(cls):
    """ Class decorator for sauteing an environment. """

    class MORLEnv(Env, Utility_Function):
        def __init__(
                self,
                safety_budget: float = 1.0,
                saute_discount_factor: float = 0.99,
                max_ep_len: int = 200,
                min_rel_budget: float = 1.,  # minimum relative (with respect to safety_budget) budget
                max_rel_budget: float = 1.,  # maximum relative (with respect to safety_budget) budget
                test_rel_budget: float = 1.,  # test relative budget
                unsafe_reward: float = 0,
                use_reward_shaping: bool = True,  # ablation
                use_state_augmentation: bool = True,  # ablation
                discount_factor: float = 0.99,
                # utility_function: Utility_Function = None,
                **kwargs
        ):
            assert safety_budget > 0, "Please specify a positive safety budget"
            assert saute_discount_factor > 0 and saute_discount_factor <= 1, "Please specify a discount factor in (0, 1]"
            assert min_rel_budget <= max_rel_budget, "Minimum relative budget should be smaller or equal to maximum relative budget"
            assert max_ep_len > 0

            self.wrap = cls(**kwargs)
            self.use_reward_shaping = use_reward_shaping
            self.use_state_augmentation = use_state_augmentation
            self.max_ep_len = max_ep_len
            self.utility_function = None
            self.min_rel_budget = min_rel_budget
            self.max_rel_budget = max_rel_budget
            self.test_rel_budget = test_rel_budget

            self.reward_shape = 2  # Scalar reward, _safety_budget
            self.zt = np.zeros(self.reward_shape)
            self.gamma = discount_factor  # same to gamma for RL

            if saute_discount_factor < 1:
                safety_budget = safety_budget * (1 - saute_discount_factor ** self.max_ep_len) / (
                            1 - saute_discount_factor) / self.max_ep_len
            self._safety_budget = np.float32(safety_budget)

            # self._safety_state = 1.
            self._safety_state = 0. # Here we treat the cost as a dimension of reward, bigger better
            self._saute_discount_factor = saute_discount_factor
            self._unsafe_reward = unsafe_reward

            self.action_space = self.wrap.action_space
            self.obs_high = self.wrap.observation_space.high
            self.obs_low = self.wrap.observation_space.low
            if self.use_state_augmentation:
                self.obs_high = np.array(np.hstack([self.obs_high, np.inf, np.inf]), dtype=np.float32)
                self.obs_low = np.array(np.hstack([self.obs_low, -np.inf, -np.inf]), dtype=np.float32)
            self.observation_space = spaces.Box(high=self.obs_high, low=self.obs_low)

        @property
        def safety_budget(self):
            return self._safety_budget

        @property
        def saute_discount_factor(self):
            return self._saute_discount_factor

        @property
        def unsafe_reward(self):
            return self._unsafe_reward

        def update_utility_function(self, func):
            self.utility_function = func

        def reset(self) -> np.ndarray:
            """Resets the environment."""
            ob = self.wrap.reset()
            if self.wrap._mode == "train":
                self._safety_state = self.wrap.np_random.uniform(low=self.min_rel_budget, high=self.max_rel_budget)
            elif self.wrap._mode == "test" or self.wrap._mode == "deterministic":
                self._safety_state = self.test_rel_budget
            else:
                raise NotImplementedError("this error should not exist!")
            self.zt = np.zeros(self.reward_shape)
            augmented_state = self._augment_state(ob, self.zt)
            return augmented_state

        def _augment_state(self, state: np.ndarray, safety_state: np.ndarray):
            """Augmenting the state with the safety state, if needed"""
            augmented_state = np.hstack([state, safety_state]) if self.use_state_augmentation else state
            return augmented_state

        def safety_step(self, cost: np.ndarray) -> np.ndarray:
            """ Update the normalized safety state z' = (z - l / d) / gamma. """
            self._safety_state += cost / self.safety_budget # maximum the cost
            self._safety_state /= self.saute_discount_factor
            return self._safety_state

        def step(self, action):
            """ Step through the environment. """
            next_obs, reward, done, info = self.wrap.step(action)
            next_safety_state = self.safety_step(info['cost'])
            zt_next = self.zt + self._augment_state(reward, next_safety_state) # next_safety_state := budget - cost
            # TODO: zt_next = self.zt + self._augment_state(reward, -info['cost']) # next_safety_state := budget - cost
            # TODO: remove safety state expanding
            # IPython.embed()
            info['true_reward'] = reward
            info['next_safety_state'] = next_safety_state
            info['zt'] = self.zt
            reward = self.gamma * self.utility_function.get_logits(zt_next[np.newaxis, :]) - \
                     self.utility_function.get_logits(self.zt[np.newaxis, :])
            self.zt = zt_next
            augmented_state = self._augment_state(next_obs, self.zt)
            return augmented_state, reward, done, info

        def reshape_reward(self, reward: Array, next_safety_state: Array):
            """ Reshaping the reward. """
            if self.use_reward_shaping:
                reward = reward * (next_safety_state > 0) + self.unsafe_reward * (next_safety_state <= 0)
            return reward

        # def reward_fn(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        #     """ Compute rewards in a batch. """
        #     reward = self.wrap._reward_fn(states, actions, next_states, is_tensor=True)
        #     if self.use_state_augmentation:
        #         # shape reward for model-based predictions
        #         reward = self.utility_function.get_logits(self.zt)
        #     return reward

    return MORLEnv


if __name__ == "__main__":
    pass