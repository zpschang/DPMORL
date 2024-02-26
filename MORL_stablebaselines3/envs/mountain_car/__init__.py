from MORL_stablebaselines3.envs.mountain_car.mountain_car import SafeMountainCarEnv, SautedMountainCarEnv, MORLMountainCarEnv, mcar_cfg
from MORL_stablebaselines3.envs.mountain_car.mo_mountain_car import MOMountainCarEnv
from gym.envs import register

print('LOADING SAFE ENVIROMENTS')

register(
    id='SafeMountainCar-v0',
    entry_point='MORL_stablebaselines3.envs.mountain_car:SafeMountainCarEnv',
    max_episode_steps=mcar_cfg['max_ep_len'],
)

register(
    id='SautedMountainCar-v0',
    entry_point='MORL_stablebaselines3.envs.mountain_car:SautedMountainCarEnv',
    max_episode_steps=mcar_cfg['max_ep_len'],
)

register(
    id='MORLMountainCar-v0',
    entry_point='MORL_stablebaselines3.envs.mountain_car:MORLMountainCarEnv',
    max_episode_steps=mcar_cfg['max_ep_len'],
)

register(
    id='OriginalMORLMountainCar-v0',
    entry_point='MORL_stablebaselines3.envs.mountain_car:MOMountainCarEnv',
    max_episode_steps=mcar_cfg['max_ep_len'],
)
