import time

from MORL_stablebaselines3.envs.gridworlds import gridworld_base
from MORL_stablebaselines3.envs.gridworlds.mo_gridworld_base import MOGridworld
from MORL_stablebaselines3.envs.gridworlds.gridworld_base import GridObject, ScrollerAgent

import numpy as np

TRAFFIC_MAPPING = {
    '#': GridObject(False, False, 0, (255.0, 255.0, 255.0), 1),
    'o': GridObject(True, True, 100, (0.0, 255.0, 0.0), 2),
    's': GridObject(True, False, -10, (255.0, 255.0, 0.0), 3),
    ' ': None
}

TRAFFIC_MAP = [
'osssssso',
' ssssss ',
' ##ssss ',
'  sssss ',
'  ##sss ',
'    sss ',
'  ##### ',
'        ',
]

agent0 = ScrollerAgent(False, ScrollerAgent.GOING_UP, 3, [2,3], [[1,1],[6,7]], True, False, 0, (255.0, 0.0, 0.0), 4)
agent0.set_position([1,6])
agent0.reset()
agent1 = ScrollerAgent(False, ScrollerAgent.GOING_DOWN, 3, [2,3], [[2,2],[6,7]],True, False, 0, (255.0, 0.0, 0.0), 4)
agent1.set_position([2,7])
agent1.reset()
agent2 = ScrollerAgent(False, ScrollerAgent.GOING_UP, 1, [1,3], [[3,3],[4,7]], True, False, 0, (255.0, 0.0, 0.0), 4)
agent2.set_position([3,4])
agent2.reset()
agent3 = ScrollerAgent(False, ScrollerAgent.GOING_DOWN, 2, [1,3], [[4,4],[2,7]], True, False, 0, (255.0, 0.0, 0.0), 4)
agent3.set_position([4,6])
agent3.reset()
agent4 = ScrollerAgent(False, ScrollerAgent.GOING_UP, 5, [4,6], [[5,5],[2,7]], True, False, 0, (255.0, 0.0, 0.0), 4)
agent4.set_position([5,2])
agent4.reset()
agent5 = ScrollerAgent(False, ScrollerAgent.GOING_UP, 3, [3,4], [[6,6],[2,7]], True, False, 0, (255.0, 0.0, 0.0), 4)
agent5.set_position([6,4])
agent5.reset()

TRAFFIC_AGENTS = [agent0, agent1, agent2, agent3, agent4, agent5]

class MOTrafficEnv(MOGridworld):

    def __init__(self,
                 from_pixels=True,
                 agent_start=[0,0],
                 agent_color=(0.0, 0.0, 255.0),
                 preference=np.array([-1,-5,+50,-20,-20])):
        super(MOTrafficEnv, self).__init__(map=TRAFFIC_MAP,
                                           object_mapping=TRAFFIC_MAPPING,
                                           from_pixels=from_pixels,
                                           init_agents=TRAFFIC_AGENTS,
                                           agent_start=agent_start,
                                           agent_color=agent_color,
                                           preference=preference,
                                           encounter_other_agents=True)

if __name__=="__main__":
    from gym_mo.envs.gridworlds.mo_gridworld_base import MORandomPlayer
    
    my_grid = MOTrafficEnv()
    player = MORandomPlayer(my_grid, 2)
    player.run()
