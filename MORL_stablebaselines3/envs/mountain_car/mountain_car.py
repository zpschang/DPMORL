import numpy as np
import math
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from MORL_stablebaselines3.envs.wrappers.saute_env import saute_env
from MORL_stablebaselines3.envs.wrappers.safe_env import SafeEnv
from MORL_stablebaselines3.envs.wrappers.morl_env_torch import morl_env_torch

# from MORL_stablebaselines3.envs.wrappers.morl_env import morl_env

mcar_cfg = dict(
    action_dim=1,
    action_range=[-1, 1],
    unsafe_reward=0.,
    saute_discount_factor=1.0,
    max_ep_len=200,
    min_rel_budget=1.0,
    max_rel_budget=1.0,
    test_rel_budget=1.0,
    use_reward_shaping=True,
    use_state_augmentation=True
)


class OurMountainCarEnv(Continuous_MountainCarEnv):
    def __init__(self):
        # self.pre_actions = []
        self.pre_actions_norm_sum = 0
        self.safe_timesteps = 0
        super().__init__()

    def step(self, action):
        position = prev_position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        # velocity += force * self.power - 0.0075 * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Convert a possible numpy bool to a Python bool.
        arrive = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        done = arrive or bool(self.safe_timesteps >= mcar_cfg["max_ep_len"])

        reward = position - prev_position  # + 0.1 * (velocity - self.goal_velocity)
        if arrive:
            reward += 10.0

        # reward -= math.pow(action[0], 2) * 0.1 # remove penalty on action

        self.state = np.array([position, velocity], dtype=np.float32)
        return self.state, reward, done, {}


class SafeMountainCarEnv(SafeEnv, OurMountainCarEnv):
    """Safe Mountain Car Environment."""

    def __init__(self, mode: int = "train", **kwargs):
        self._mode = mode
        # self.pre_actions = []
        self.pre_actions_norm_sum = 0
        super().__init__(**kwargs)

    def _get_obs(self):
        return self.state

    def step(self, action):
        self.safe_timesteps += 1
        return super().step(action)

    def reset(self):
        if self._mode == "train":
            # making our lives easier with random starts 
            self.state = np.array([
                self.np_random.uniform(low=-0.6, high=0.4),
                self.np_random.uniform(low=-self.max_speed, high=self.max_speed)
            ])
        elif self._mode == "test":
            self.state = np.array([
                self.np_random.uniform(low=-0.6, high=-0.4),
                0
            ])
        self.pre_actions_norm_sum = 0
        self.safe_timesteps = 0
        return np.array(self.state)

    def _safety_cost_fn(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> np.ndarray:
        """Computes a fuel cost on the mountain car"""
        if self.safe_timesteps:
            cost = (self.pre_actions_norm_sum + np.linalg.norm(actions)) / (self.safe_timesteps + 1) - \
                   self.pre_actions_norm_sum / self.safe_timesteps
        else:
            cost = (self.pre_actions_norm_sum + np.linalg.norm(actions)) / (self.safe_timesteps + 1)
        self.pre_actions_norm_sum += np.linalg.norm(actions)
        return cost
        # return np.linalg.norm(actions)
        # TODO: Energy Efficiency
        # TODO: return max(1 - np.linalg.norm(actions) * self.factor, 0)


@saute_env
class SautedMountainCarEnv(SafeMountainCarEnv):
    """Sauted safe mountain car."""


@morl_env_torch
class MORLMountainCarEnv(SafeMountainCarEnv):
    """morl mountain car."""
