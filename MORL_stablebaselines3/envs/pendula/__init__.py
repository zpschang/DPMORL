from MORL_stablebaselines3.envs.pendula.single_pendulum import pendulum_cfg, SafePendulumEnv, SautedPendulumEnv, MORLPendulumEnv
from MORL_stablebaselines3.envs.pendula.double_pendulum import double_pendulum_cfg, SafeDoublePendulumEnv, \
    SautedDoublePendulumEnv, MORLDoublePendulumEnv
from gym.envs import register

print('LOADING SAFE ENVIROMENTS') 

register(
    id='SafePendulum-v0',
    entry_point='envs.pendula:SafePendulumEnv',
    max_episode_steps=pendulum_cfg['max_ep_len']
)

register(
    id='SautedPendulum-v0',
    entry_point='envs.pendula:SautedPendulumEnv',
    max_episode_steps=pendulum_cfg['max_ep_len']
)

register(
    id='SafeDoublePendulum-v0',
    entry_point='envs.pendula:SafeDoublePendulumEnv',
    max_episode_steps=double_pendulum_cfg['max_ep_len']
)

register(
    id='SautedDoublePendulum-v0',
    entry_point='envs.pendula:SautedDoublePendulumEnv',
    max_episode_steps=double_pendulum_cfg['max_ep_len']
)

register(
    id='MORLPendulum-v0',
    entry_point='envs.pendula:MORLPendulumEnv',
    max_episode_steps=pendulum_cfg['max_ep_len'],
)

register(
    id='MORLDoublePendulum-v0',
    entry_point='envs.pendula:MORLDoublePendulumEnv',
    max_episode_steps=pendulum_cfg['max_ep_len'],
)