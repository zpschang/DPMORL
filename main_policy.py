import gym
import numpy as np

import matplotlib
matplotlib.rcParams.update({'font.size': 22})

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import matplotlib.pyplot as plt
import mo_gymnasium
import gymnasium
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import math
from os import path
from typing import Callable, List, Dict, Tuple
import torch
from gym import spaces
from gym.utils import seeding
import time
import os
import argparse
from datetime import date
import glob

from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from mo_gymnasium.envs.breakable_bottles.breakable_bottles import BreakableBottles
from mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure
from mo_gymnasium.envs.fishwood.fishwood import FishWood
from mo_gymnasium.envs.four_room.four_room import FourRoom
from mo_gymnasium.envs.fruit_tree.fruit_tree import FruitTreeEnv
from mo_gymnasium.envs.highway.highway import MOHighwayEnv
from mo_gymnasium.envs.lunar_lander.lunar_lander import MOLunarLander
from mo_gymnasium.envs.mario.mario import MOSuperMarioBros
from mo_gymnasium.envs.mujoco.half_cheetah import MOHalfCheehtahEnv
from mo_gymnasium.envs.mujoco.hopper import MOHopperEnv
from mo_gymnasium.envs.mujoco.reacher import MOReacherEnv
from mo_gymnasium.envs.reacher.reacher import ReacherBulletEnv
from mo_gymnasium.envs.minecart.minecart import Minecart
from mo_gymnasium.envs.resource_gathering.resource_gathering import ResourceGathering
from mo_gymnasium.envs.water_reservoir.dam_env import DamEnv
from mo_gymnasium.envs.continuous_mountain_car.continuous_mountain_car import MOContinuousMountainCar
from MORL_stablebaselines3.envs.wrappers.morl_env_wrapper import morl_env_wrapper
from MORL_stablebaselines3.envs.wrappers.utility_env_wrapper import MultiEnv_UtilityFunction, ObsInfoWrapper
from mo_gymnasium.utils import MONormalizeReward, MORecordEpisodeStatistics, MOSyncVectorEnv
from MORL_stablebaselines3.envs.wrappers.scalar_reward_wrapper import ScalarRewardEnv
from MORL_stablebaselines3.utility_function.utility_function_parameterized import Utility_Function_Parameterized
from MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Programmed
from MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Linear
from MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Diverse_Goal

from DIPG.diverse_goal_env import DiverseGoalEnv
import copy

def make_env(env_name, rank, utility_function, reward_dim, reward_dim_indices, seed=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        if type(env_name) == str:
            env = mo_gymnasium.make(env_name)
            env.name = env_name
        else:
            env = env_name()
        env = ObsInfoWrapper(env, reward_dim=reward_dim, reward_dim_indices=reward_dim_indices)
        return env
    set_random_seed(seed)
    return _init

def choose_gpu(args):
    import os, socket, pynvml
    pynvml.nvmlInit()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
    s.close()
    if args.gpu == "all":
        memory_gpu = []
        masks = np.ones(pynvml.nvmlDeviceGetCount())
        for gpu_id, mask in enumerate(masks):
            if mask == -1:
                continue
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_gpu.append(meminfo.free / 1024 / 1024)
        gpu1 = np.argmax(memory_gpu)
    else:
        gpu1 = args.gpu
    print("****************************Choosen GPU : {}****************************".format(gpu1))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu1)
    if args.total_timesteps == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def config_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description="Training a set of Policies with DPMORL")
    parser.add_argument(
        '--env',
        help="env name",
        type=str,
        choices=["MountainCar", "BreakableBottles", "DeepSeaTreasure",
                 "FishWood", "FourRoom", "FruitTree",
                 "Highway", "LunarLander", "SuperMarioBros",
                 "HalfCheetah", "Hopper", "Reacher",
                 "ReacherBullet", "ResourceGathering", "WaterReservoir", "Minecart", "DiverseGoal"],
        default="MountainCar",
    )
    parser.add_argument(
        '--exp_name', 
        type=str,
        default='dpmorl'
    )
    parser.add_argument(
        '--clip_loss',
        help="TODO: Add clip loss to maintain the Lipschitz continuity of the utility function",
        action='store_true',
    )
    parser.add_argument(
        '--utility_epochs',
        help="Epochs for training one utility function",
        type=int,
        default=200,
    )
    parser.add_argument(
        '--seed',
        help="random seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        '--lr',
        help="Learning rate for utility functions",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        '--lamda',
        help="The hyperparameter of using env reward",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        '--norm',
        help="Normalize the input of the utility function (return)",
         type=str2bool, nargs='?', const=True,
    )
    parser.add_argument(
        '--num_test_episodes',
        help="The number of episode for testing",
        type=int,
        default=100,
    )
    parser.add_argument("--keep_scale", type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument("--reward_two_dim", type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--reward_dim_indices", type=str, default='')
    parser.add_argument("--linear_utility", type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--augment_state", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--test_only", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument(
        '--num_envs',
        help="The number of envs for vecenv",
        type=int,
        default=20,
    )
    parser.add_argument(
        '--num_policies',
        help="The number of policies for MORL optimization",
        type=int,
        default=1,
    )
    parser.add_argument(
        '--max_num_policies',
        help="The number of policies for MORL optimization",
        type=int,
        default=20,
    )
    parser.add_argument(
        '--total_timesteps',
        help="The number of total timesteps for training and evaluating a policy",
        type=float,
        default=1e7,
    )
    parser.add_argument(
        '--iters',
        help="The number of iterations for policy evaluation / utility function training",
        type=int,
        default=50,
    )
    parser.add_argument(
        '--gpu',
        help="Choose the specific gpu for training",
        type=str,
        default='all',
    )
    return parser.parse_args()

def env_functions(env_name):
    if env_name == "MountainCar":
        return MOContinuousMountainCar
    elif env_name == "BreakableBottles":
        return BreakableBottles
    elif env_name == "DeepSeaTreasure":
        return DeepSeaTreasure
    elif env_name == "FishWood":
        return FishWood
    elif env_name == "FourRoom":
        return FourRoom
    elif env_name == "FruitTree":
        return FruitTreeEnv
    elif env_name == "Highway":
        return MOHighwayEnv
    elif env_name == "LunarLander":
        return MOLunarLander
    elif env_name == "SuperMarioBros":
        return MOSuperMarioBros
    elif env_name == "HalfCheetah":
        return MOHalfCheehtahEnv
    elif env_name == "Hopper":
        return MOHopperEnv
    elif env_name == "Reacher":
        return MOReacherEnv
    elif env_name == "ReacherBullet":
        return ReacherBulletEnv
    elif env_name == "ResourceGathering":
        return ResourceGathering
    elif env_name == "WaterReservoir":
        return DamEnv
    elif env_name == "Minecart":
        return Minecart
    elif env_name == "DiverseGoal":
        return DiverseGoalEnv
    else:
        raise NotImplementedError("Please write the right and implemented env name!")


def get_id_name(env_name):
    if env_name == "MountainCar":
        return "mo-mountaincarcontinuous-v0"
    elif env_name == "BreakableBottles":
        return "breakable-bottles-v0"
    elif env_name == "DeepSeaTreasure":
        return "deep-sea-treasure-v0"
    elif env_name == "FishWood":
        return "fishwood-v0"
    elif env_name == "FourRoom":
        return "four-room-v0"
    elif env_name == "FruitTree":
        return "fruit-tree-v0"
    elif env_name == "Highway":
        return "mo-highway-v0"
    elif env_name == "LunarLander":
        return "mo-lunar-lander-v2"
    elif env_name == "SuperMarioBros":
        return "mo-supermario-v0"
    elif env_name == "HalfCheetah":
        return "mo-halfcheetah-v4"
    elif env_name == "Hopper":
        return "mo-hopper-v4"
    elif env_name == "Reacher":
        return "mo-reacher-v4"
    elif env_name == "ReacherBullet":
        return "mo-reacher-v0"
    elif env_name == "ResourceGathering":
        return "resource-gathering-v0"
    elif env_name == "WaterReservoir":
        return "water-reservoir-v0"
    elif env_name == 'Minecart':
        return "minecart-v0"
    elif env_name == "DiverseGoal":
        return DiverseGoalEnv
    else:
        raise NotImplementedError("Please write the right and implemented env name!")

import os
import json
from stable_baselines3.common.callbacks import BaseCallback

class ReturnLogger(BaseCallback):
    def __init__(self, save_dir, env_name, algo_name, policy_id, iter, seed, verbose=0):
        super(ReturnLogger, self).__init__(verbose)
        self.episode_vec_returns = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.env_name = env_name
        self.algo_name = algo_name
        self.seed = seed
        self.iter = iter
        self.policy_id = policy_id

    def _on_step(self) -> bool:
        # vec env
        if isinstance(self.locals.get("infos"), tuple) or isinstance(self.locals.get("infos"), list):
            for info in self.locals.get("infos"):
                if "episode" in info:
                    self.episode_vec_returns.append(info["episode"]["r"])
                    if len(self.episode_vec_returns) % 100 == 0:
                        print('return', info["episode"]["r"])
        else:
            if self.locals.get("done") and "episode" in self.locals.get("infos"):
                self.episode_vec_returns.append(self.locals.get("infos")["episode"]["r"])
        return True

    def _on_training_end(self) -> None:
        file_name = f"MORL_{self.env_name}_{self.algo_name}_policy{self.policy_id}_seed{self.seed}_{self.iter}.npz"
        file_path = os.path.join(self.save_dir, file_name)

        np.savez_compressed(file_path, episode_vec_returns=self.episode_vec_returns)


def gymnasium2gym(env):
    if isinstance(env.action_space, gymnasium.spaces.Box):
        env.action_space = gym.spaces.Box(low=env.action_space.low, high=env.action_space.high,
                                           shape=env.action_space.shape, dtype=env.action_space.dtype)
    elif isinstance(env.action_space, gymnasium.spaces.Discrete):
        env.action_space = gym.spaces.Discrete(env.action_space.n)
    if isinstance(env.observation_space, gymnasium.spaces.Box):
        env.obs_high = np.array(np.hstack([env.observation_space.high, np.full((env.observation_space.high.shape[0], env.envs[0].zt.shape[0]), np.inf)]),
                                 dtype=np.float32)
        env.obs_low = np.array(np.hstack([env.observation_space.low, np.full((env.observation_space.high.shape[0], env.envs[0].zt.shape[0]), -np.inf)]),
                                dtype=np.float32)
        env.observation_space = gym.spaces.Box(low=env.obs_low, high=env.obs_high,
                                                shape=(env.observation_space.shape[0], env.observation_space.shape[1] + env.envs[0].zt.shape[0],),
                                                dtype=env.observation_space.dtype)
    elif isinstance(env.observation_space, gymnasium.spaces.Discrete):
        env.observation_space = gym.spaces.Discrete(env.observation_space.n + 2)
    return env

def evaluate_policy(model, env, num_test_episodes):
    episode_returns = []
    obs = env.reset()
    trajectories = []
    current_trajectory = []
    
    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, infos = env.step(action)
        for info in infos:
            if "episode" in info:
                episode_returns.append(info["episode"]["r"])
                if len(episode_returns) % 10 == 0:
                    print(f'progress: {len(episode_returns)}/{num_test_episodes}')
        if done[0]:
            current_trajectory.append(infos[0]['terminal_observation'].copy())
            trajectories.append(np.array(current_trajectory))
            current_trajectory = []
        current_trajectory.append(obs[0].copy())
        if len(episode_returns) >= num_test_episodes:
            break
    episode_returns = episode_returns[:num_test_episodes]
    return np.array(episode_returns), trajectories


if __name__ == "__main__":
    import pickle
    with open('normalization_data/data.pickle', 'rb') as file:
        normalization_data = pickle.load(file)
        
    args = config_args()

    choose_gpu(args)

    base_env_name = env_functions(args.env)

    alg_name = "PPO"

    test_env = base_env_name()

    print(f"env: {args.env}, env reward dim: {test_env.reward_dim}")

    gym_id_name = get_id_name(args.env)

    num_cpu = args.num_envs  # Number of processes to use
    epochs = args.utility_epochs
    learning_rate = args.lr
    
    utility_dir = 'experiments/' + args.exp_name
    os.makedirs(utility_dir, exist_ok=True)
    reward_shape = test_env.reward_dim
    if args.reward_two_dim:
        reward_shape = 2
    if args.reward_dim_indices == '':
        reward_dim_indices = list(range(reward_shape))
    else:
        reward_dim_indices = eval(args.reward_dim_indices)
        reward_shape = len(reward_dim_indices)
    print(f'{reward_dim_indices = }, {reward_shape = }')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.linear_utility:
        utility_class_programmed = Utility_Function_Linear
    else:
        utility_class_programmed = Utility_Function_Programmed
    
    norm = True
    if args.env == 'DiverseGoal':
        args.total_timesteps //= 20
        utility_class_programmed = Utility_Function_Diverse_Goal
        norm = False
        print('Using Diverse Goal Utility Function')
    
    utility_function = utility_class_programmed(reward_shape=reward_shape, norm=norm, lamda=args.lamda, function_choice=0, keep_scale=args.keep_scale)
    num_utility_programmed = len(utility_function.utility_functions)
    
    # Load pretrained utility functions
    assert os.path.isdir(f'utility-model-selected/dim-{reward_shape}'), 'There is no pretrained utility functions provided. '
    num_pretrained_utility = len(glob.glob(f'utility-model-selected/dim-{reward_shape}/*'))
    pretrained_utility_paths = [f'utility-model-selected/dim-{reward_shape}/utility-{i}.pt'
                                for i in range(num_pretrained_utility)]
    
    pretrained_utility_functions = []
    for path in pretrained_utility_paths:
        model = Utility_Function_Parameterized(reward_shape=reward_shape, norm=norm, lamda=args.lamda, max_weight=0.5, keep_scale=args.keep_scale, size_factor=1)
        model.load_state_dict(torch.load(path))
        model.eval()
        model = model.cuda()
        pretrained_utility_functions.append(model)
    num_utility_pretrained = len(pretrained_utility_functions)
    
    if args.linear_utility or args.env == 'DiverseGoal':
        num_utility_pretrained = 0

    policies = []
    utility_functions_optims = []
    
    total_steps = args.total_timesteps
    iterations = args.iters
    pre_zts = None
    task_name = f"DPMORL.{args.env}.{'no_norm.' if args.norm == False else ''}LossNormLamda_{args.lamda}"
    utility_dir = os.path.join(utility_dir, task_name)
    
    num_total_policies = min(num_utility_programmed + num_utility_pretrained, args.max_num_policies)
    print(f'{num_total_policies = }')
    
    def get_utility(policy_idx):
        if policy_idx < num_utility_programmed:
            utility_function = utility_class_programmed(reward_shape=reward_shape, norm=norm, lamda=args.lamda, function_choice=policy_idx, keep_scale=args.keep_scale)
        else:
            utility_function = pretrained_utility_functions[policy_idx - num_utility_programmed]
        return utility_function
    if not args.test_only:
        for policy_idx in range(num_total_policies):
            utility_function = get_utility(policy_idx)
            if args.env in normalization_data:
                utility_function.min_val = normalization_data[args.env]['min'][0][reward_dim_indices]
                utility_function.max_val = normalization_data[args.env]['max'][0][reward_dim_indices]
                print('normalization data:', normalization_data[args.env])
            else:
                print('normalization data: None')
            optim, optim_init_state = None, None
            utility_functions_optims.append([utility_function, optim, optim_init_state])
            env = DummyVecEnv(
                [make_env(gym_id_name, i, utility_function, reward_dim=reward_shape, reward_dim_indices=reward_dim_indices, seed=args.seed) for i in range(num_cpu)], 
                reward_dim=reward_shape
            )
            env = MultiEnv_UtilityFunction(env, utility_function, reward_dim=reward_shape, augment_state=args.augment_state)
            env.update_utility_function(utility_function)

            policy = PPO("MlpPolicy", env, verbose=1, device='cuda', n_epochs=5)

            if policy_idx < num_utility_programmed:
                policy_name = f'program-{policy_idx}'
            else:
                policy_name = f'pretrain-{policy_idx-num_utility_programmed}'
            print(f"Training policy {policy_idx + 1} with {total_steps} steps...")
            curtime = time.time()
            return_logger = ReturnLogger(utility_dir, args.env, alg_name, policy_name, 0, args.seed)
            policy.learn(total_timesteps=total_steps, callback=return_logger, progress_bar=True)
            print(f"Training one policy with one utility function using time {time.time() - curtime:.2f} seconds.")
            policy.save(f'{utility_dir}/policy-{policy_name}')
        
    if args.test_only:
        for policy_idx in range(0, num_total_policies):
            if policy_idx < num_utility_programmed:
                policy_name = f'program-{policy_idx}'
            else:
                policy_name = f'pretrain-{policy_idx-num_utility_programmed}'
            print(f"Evaluating policy {policy_idx+1} with {args.num_test_episodes} episodes...")
            if not os.path.exists(f'{utility_dir}/policy-{policy_name}.zip'):
                print(f'{policy_name} does not exist')
                continue
            policy = PPO.load(f'{utility_dir}/policy-{policy_name}')
            curtime = time.time()
            env = DummyVecEnv(
                [make_env(gym_id_name, i, utility_function, reward_dim=reward_shape, reward_dim_indices=reward_dim_indices, seed=args.seed) for i in range(10)], 
                reward_dim=reward_shape
            )
            env = MultiEnv_UtilityFunction(env, utility_function, reward_dim=reward_shape, augment_state=args.augment_state)
            test_returns, trajectories = evaluate_policy(policy, env, args.num_test_episodes)
            np.savez_compressed(os.path.join(utility_dir,
                                             f"test_returns_policy_{policy_name}.npz"),
                                test_returns=test_returns)
            if args.env == 'DiverseGoal':
                from DIPG.diverse_goal_env import plot_env
                plot_name = rf'Trajectory of Policy $\pi_{policy_idx}$'
                plot_env(test_env, trajectories[:20], plot_name, f'{utility_dir}/trajectory-{policy_name}.png')

