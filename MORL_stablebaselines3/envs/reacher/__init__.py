from MORL_stablebaselines3.envs.reacher.reacher import SafeReacherEnv, SautedReacherEnv, MORLReacherEnv, reacher_cfg
from gym.envs import register

print('LOADING SAFE ENVIROMENTS')

register(
    id='SafeReacher-v0',
    entry_point='envs.reacher:SafeReacherEnv',
    max_episode_steps=reacher_cfg['max_ep_len'],
)

register(
    id='SautedReacher-v0',
    entry_point='envs.reacher:SautedReacherEnv',
    max_episode_steps=reacher_cfg['max_ep_len'],
)

register(
    id='MORLReacher-v0',
    entry_point='envs.reacher:MORLReacherEnv',
    max_episode_steps=reacher_cfg['max_ep_len'],
)
