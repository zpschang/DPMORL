from gym import spaces

from MORL_stablebaselines3.envs.gridworlds.mo_gridworld_base import MOGridworld
from MORL_stablebaselines3.envs.gridworlds.gridworld_base import GridObject

import numpy as np
import time

SEA_MAPPING = {
    '#': GridObject(False, False, 0, (255.0, 255.0, 255.0), 1),
    'a': GridObject(True, True, 1, (0.0, 125.0, 0.0), 2),
    'b': GridObject(True, True, 2, (0.0, 135.0, 0.0), 3),
    'c': GridObject(True, True, 3, (0.0, 145.0, 0.0), 4),
    'd': GridObject(True, True, 5, (0.0, 155.0, 0.0), 5),
    'e': GridObject(True, True, 8, (0.0, 165.0, 0.0), 6),
    'f': GridObject(True, True, 16, (0.0, 175.0, 0.0), 7),
    'g': GridObject(True, True, 24, (0.0, 185.0, 0.0), 8),
    'h': GridObject(True, True, 50, (0.0, 195.0, 0.0), 9),
    'i': GridObject(True, True, 74, (0.0, 205.0, 0.0), 10),
    'j': GridObject(True, True, 124, (0.0, 215.0, 0.0), 11),
    ' ': None
}

SEA_MAP = [
'          ',
'a         ',
'#b        ',
'##c       ',
'###def    ',
'######    ',
'######    ',
'######gh  ',
'########  ',
'########i ',
'#########j',
]

class MODeepSeaTresureEnv(MOGridworld):

    def __init__(self,
                 from_pixels=True,
                 preference=np.array([-1,-5,+1,+2,+3,+5,+8,+16,+24,+50,+74,+124])):
        super(MODeepSeaTresureEnv, self).__init__(map=SEA_MAP,
                                                  object_mapping=SEA_MAPPING,
                                                  from_pixels=from_pixels,
                                                  agent_start=[0,10],
                                                  preference=preference)
        self.treasure_depths = np.array([9, 8, 7, 6, 6, 6, 3, 3, 1, 0])

        self.modified_reward_space = spaces.Box(self.reward_low[: 2], self.reward_high[: 2], dtype=np.float32)


    def is_done(self, include_agents=False, agent_preferences=[]):
        if self.step_count > self.max_steps:
            return True

        if self.treasure_depths[self.agent_pos[0]] == self.agent_pos[1]:
            return True

        return False


if __name__=="__main__":
    from MORL_stablebaselines3.envs.gridworlds.mo_gridworld_base import MORandomPlayer

    my_grid = MODeepSeaTresureEnv(from_pixels=True)
    player = MORandomPlayer(my_grid, 10)
    player.run()
