import time

from MORL_stablebaselines3.envs.gridworlds import gridworld_base
from MORL_stablebaselines3.envs.gridworlds.mo_gridworld_base import MOGridworld
from MORL_stablebaselines3.envs.gridworlds.gridworld_base import GridObject, HunterAgent

import numpy as np

GATHERING_MAPPING = {
    '#': GridObject(True, False, 0, (255.0, 255.0, 255.0), 1),
    'o': GridObject(True, True, 0, (0.0, 255.0, 0.0), 2),
    'p': GridObject(True, True, 1, (255.0, 0.0, 0.0), 3),
    'q': GridObject(True, True, 0, (255.0, 255.0, 0.0), 4),
    ' ': None
}

GATHERING_MAP = [
'        ',
'        ',
'        ',
'        ',
'        ',
'        ',
'        ',
'        ',
]


class MOGatheringEnv(MOGridworld):

    def __init__(self,
                 from_pixels=True,
                 agent_start=[0,0],
                 agent_color=(0.0, 0.0, 255.0),
                 preference=np.array([-1,-5,+20,-20,-20,+0]),
                 random_items=['p','o','p','o','p','o','q','q'],
                 random_items_frame=2,
                 agents=[]):

        agent0 = HunterAgent(3, True, False, 0, (255.0, 0.0, 255.0), 5)
        agent0.set_position([7,7])

        GATHERING_AGENTS = [agent0]

        super(MOGatheringEnv, self).__init__(map=GATHERING_MAP,
                                             object_mapping=GATHERING_MAPPING,
                                             random_items=random_items,
                                             random_items_frame=random_items_frame,
                                             from_pixels=from_pixels,
                                             init_agents=GATHERING_AGENTS,
                                             agent_start=agent_start,
                                             agent_color=agent_color,
                                             preference=preference,
                                             max_steps=30, include_agents=False)


if __name__=="__main__":
    from gym_mo.envs.gridworlds.mo_gridworld_base import MORandomPlayer
    
    my_grid = MOGatheringEnv()
    player = MORandomPlayer(my_grid, 2)
    player.run()
    

