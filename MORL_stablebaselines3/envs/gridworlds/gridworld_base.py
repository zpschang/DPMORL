import numpy as np

import time
import copy

from gym import Env, spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

class GridObject():
    """Object that can be placed in a GridWorld.

    """

    def __init__(self, is_walkable, is_consumed, reward_on_encounter, color, idx=None):
        """Initialize GridObject.

        Keyword arguments:
        is_walkable           -- is it possible to walk on the object?
        is_consumed           -- is the object consumed when walked upon?
        reward_on_encounter   -- the reward received when encountering the object
        color                 -- the color of the object
        idx                   -- idx of the object
        """
        self.is_walkable = is_walkable
        self.is_consumed = is_consumed
        self.reward_on_encounter = reward_on_encounter
        self.color = color
        self.idx = idx

class GridAgent(GridObject):
    """Autonomous agent that can act in a GridWorld.

    """

    def __init__(self, is_walkable, is_consumed, reward_on_encounter, color, idx=None):
        """Initialize GridObject.

        Keyword arguments:
        color                 -- the color of the agent
        idx                   -- idx of the agent
        """
        super().__init__(is_walkable, is_consumed, reward_on_encounter, color, idx)
        self.position = None
        self.reward = 0
        self.observation_color = (0.0, 0.0, 255.0)

    def reset_reward(self):
        self.reward = 0

    def set_position(self, position):
        self.position = position

    def reset(self):
        pass

    def step(self, env):
        pass

class PixelRLAgent(GridAgent):
    def __init__(self, act, is_walkable, is_consumed, reward_on_encounter, color, idx):
        self.act = act
        super().__init__(is_walkable, is_consumed, reward_on_encounter, color, idx)

    def step(self, env):
        obs = env.create_image_observation(include_agent=False)
        a = self.act(obs[None])[0]

        new_pos = self.position.copy()
        if a == 1:
            new_pos[0] = new_pos[0] - 1
        elif a == 2:
            new_pos[0] = new_pos[0] + 1
        elif a == 3:
            new_pos[1] = new_pos[1] - 1
        elif a == 4:
            new_pos[1] = new_pos[1] + 1

        if env.is_walkable(new_pos[0], new_pos[1]):
            self.position = new_pos
            self.reward += env.encounter_object(new_pos[0], new_pos[1], False)

    def reset(self):
        self.reward = 0

class HunterAgent(GridAgent):

    def __init__(self, goal_object_idx, is_walkable, is_consumed, reward_on_encounter, color, idx):
        self.goal_object_idx = goal_object_idx
        self.goal_position = None
        self.is_done = False
        super().__init__(is_walkable, is_consumed, reward_on_encounter, color, idx)

    def step(self, env):
        if self.is_done: return

        if self.goal_position is None:
            self.set_goal_position(env)
        elif (env.grid[self.goal_position[1], self.goal_position[0]] is None) or (env.grid[self.goal_position[1], self.goal_position[0]].idx != self.goal_object_idx):
            self.set_goal_position(env)

        new_pos = self.position.copy()
        if new_pos[0] - self.goal_position[0] < 0:
            new_pos[0] = new_pos[0] + 1
        elif new_pos[0] - self.goal_position[0] > 0:
            new_pos[0] = new_pos[0] - 1
        elif new_pos[1] - self.goal_position[1] < 0:
            new_pos[1] = new_pos[1] + 1
        elif new_pos[1] - self.goal_position[1] > 0:
            new_pos[1] = new_pos[1] - 1

        if env.is_walkable(new_pos[0], new_pos[1]):
            self.position = new_pos
            self.reward += env.encounter_object(new_pos[0], new_pos[1], False)

    def set_goal_position(self, env):
        self.goal_position = self.position.copy()
        shortest_distance = None
        for row in range(env.grid.shape[0]):
            for column in range(env.grid.shape[1]):
                if ((env.grid[row, column] is None) or (env.grid[row, column].idx != self.goal_object_idx)): continue
                object_distance = self.grid_distance([column, row], self.position)
                if ((shortest_distance is None) or (object_distance < shortest_distance)):
                    shortest_distance = object_distance
                    self.goal_position = [column, row]

        if shortest_distance is None:
            self.is_done = True

    def grid_distance(self, pos1, pos2):
        return (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    def reset(self):
        self.reward = 0
        self.goal_position = None
        self.is_done = False

class ScrollerAgent(GridAgent):

    GOING_LEFT  = [-1,0]
    GOING_RIGHT = [+1,0]
    GOING_DOWN  = [0,-1]
    GOING_UP    = [0,+1]

    def __init__(self, 
                 random_walk, 
                 direction, 
                 step_delay=0, 
                 step_delay_init=[0,0],
                 random_init=None,
                 is_walkable=True, 
                 is_consumed=False, 
                 reward_on_encounter=0, 
                 color=(255.0, 0.0, 0.0), 
                 idx=0):
        super().__init__(is_walkable, is_consumed, reward_on_encounter, color, idx)
        self.random_walk = random_walk
        self.direction = direction.copy()
        self.step_delay = step_delay
        self.step_delay_init = step_delay_init
        self.stationary_steps = 0
        self.random_init = random_init

    def step(self, env):
        if self.position is None:
            return

        if self.stationary_steps < self.step_delay:
            self.stationary_steps += 1
            return

        self.stationary_steps = 0

        new_pos = [self.position[0] + self.direction[0], self.position[1] + self.direction[1]]

        if env.is_walkable(new_pos[0], new_pos[1]):
            self.position = new_pos
        else:
            self.set_new_direction()

        if self.random_walk and (np.random.randint(100) < 10):
            self.set_new_direction()

    def set_new_direction(self):
        if self.random_walk:
            change = np.random.randint(4)
            if change == 0:
                self.direction = self.GOING_LEFT
            elif change == 1:
                self.direction = self.GOING_RIGHT
            elif change == 2:
                self.direction = self.GOING_DOWN
            else:
                self.direction = self.GOING_UP
        else:
            # Bounce
            self.direction[0] = self.direction[0] * -1
            self.direction[1] = self.direction[1] * -1

    def reset(self):
        if self.random_init is not None:
            self.position[0] = np.random.randint(self.random_init[0][0], self.random_init[0][1] + 1)
            self.position[1] = np.random.randint(self.random_init[1][0], self.random_init[1][1] + 1)
            if (np.random.randint(100) < 10):
                self.set_new_direction()
        self.step_delay = np.random.randint(self.step_delay_init[0], self.step_delay_init[1] + 1)

class Viewport:
    """Class for viewport settings.

    """

    def __init__(self, viewport_width=600, viewport_height=600, view_port_object_delta=0.1):
        self.width        = viewport_width
        self.height       = viewport_height
        self.object_delta = view_port_object_delta

class Gridworld(Env):
    """Base class for gridworld environments.

    """

    def __init__(self,
                 map,
                 object_mapping,
                 viewport=Viewport(),
                 from_pixels=False,
                 inflation=1,
                 random_items=[],
                 random_items_frame=0,
                 init_agents=[],
                 agent_start=[0, 0],
                 max_steps=50,
                 agent_color=(0.0, 0.0, 255.0),
                 encounter_other_agents=False):
        self.map = map
        self.rows = len(self.map)
        self.columns = len(self.map[0])
        self.random_items = random_items
        self.random_items_frame = random_items_frame
        self.init_agents=init_agents
        self.agents = []
        self.agent_start = agent_start
        self.object_mapping = object_mapping
        self.max_steps = max_steps
        self.agent_color = agent_color
        self.encounter_other_agents = encounter_other_agents
        self.viewport = viewport
        self.setup_viewer_configuration()
        self.grid = np.empty([self.rows, self.columns], dtype=object)
        self.agent_pos = [0, 0]
        self.n_actions = 5
        self.n_states = self.rows * self.columns
        self.from_pixels = from_pixels
        self.inflation = inflation
        self.setup_idx_space(object_mapping, init_agents)
        self.setup_observation_space()
        self.action_space = spaces.Discrete(self.n_actions)
        self.seed()
        self.reset()
        #self.create_image_observation()

    def setup_viewer_configuration(self):
        self.SCALE_H = self.viewport.height / (self.rows + 2)
        self.SCALE_W = self.viewport.width / (self.columns + 2)
        self.viewer = None
        self.on_key_press = None
        self.on_key_release = None
        self.render_grid = True

    def setup_observation_space(self):
        if self.from_pixels:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.rows * self.inflation, self.columns * self.inflation, 3),
                dtype='uint8'
            )
        else:
            #self.observation_space = spaces.Discrete(self.rows * self.columns * self.max_idx)
            self.observation_space = spaces.Box(low=0,
                                                high=1,
                                                shape=(self.rows * self.columns * (self.max_idx + 1),),
                                                dtype='uint8')

    def setup_idx_space(self, object_mapping, init_agents):
        self.max_idx = 0
        for _, mapped_object in object_mapping.items():
            if (mapped_object is not None) and (mapped_object.idx > self.max_idx):
                self.max_idx = mapped_object.idx

        for agent in init_agents:
            if agent.idx > self.max_idx:
                self.max_idx = agent.idx
        #print(self.max_idx)

    def create_discrete_observation(self):
        obs = np.zeros(shape=(self.rows, self.columns, self.max_idx + 1), dtype='uint8')

        # Add agent to the first feature plane
        obs[(self.rows - self.agent_pos[1] - 1)][self.agent_pos[0]][0] = 1

        # Add grid objects to feature planes corresponding to their indices
        for row in range(self.grid.shape[0]):
            for column in range(self.grid.shape[1]):
                g_object = self.grid[row, column]
                if g_object is not None:
                    obs[row][column][g_object.idx-1] = 1

        # Add grid agents to feature planes corresponding to their indices
        for agent in self.agents:
            obs[(self.rows - agent.position[1] - 1)][agent.position[0]][0] = 1

        obs = obs.flatten()
        return obs

    def create_image_observation(self, include_agent=True):
        obs = np.zeros(shape=(self.rows * self.inflation, self.columns * self.inflation, 3), dtype='uint8')

        # Add objects
        for row in range(self.grid.shape[0]):
            for column in range(self.grid.shape[1]):
                g_object = self.grid[row, column]
                if g_object is not None:
                    for i in range(self.inflation):
                        for j in range(self.inflation):
                            obs[(self.rows - row - 1) * self.inflation + i, column * self.inflation + j] = g_object.color

        # Add grid agents
        for agent in self.agents:
            for i in range(self.inflation):
                for j in range(self.inflation):
                    obs[(self.rows - agent.position[1] - 1) * self.inflation + i, agent.position[0] * self.inflation + j] = agent.color

        # Add agent
        if include_agent:
            for i in range(self.inflation):
                for j in range(self.inflation):
                    obs[(self.rows - self.agent_pos[1] - 1) * self.inflation + i, self.agent_pos[0] * self.inflation + j] = self.agent_color

        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_callbacks(self, on_key_press, on_key_release):
        self.on_key_press = on_key_press
        self.on_key_release = on_key_release

    def load_map(self, map, object_mapping):
        self.grid = np.empty([self.rows, self.columns], dtype=object)
        rx = 0
        cx = 0
        for row in map:
            for item in row:
                g_object = object_mapping[item]
                self.grid[self.rows - 1 - rx,  cx] = g_object
                #if isinstance(g_object, GridAgent):
                #    g_object.set_position([cx, self.rows - 1 - rx])
                #    self.agents.append(g_object)
                cx += 1
            cx = 0
            rx += 1
        for item in self.random_items:
            self.place_random_item(item, object_mapping)

    def place_random_item(self, item, object_mapping):
        tries = 0
        found_valid_spot = False
        while not (found_valid_spot or tries > 64):
            row = np.random.randint(0+self.random_items_frame,self.rows-self.random_items_frame)
            column = np.random.randint(0+self.random_items_frame,self.columns-self.random_items_frame)
            tries += 1
            if self.grid[self.rows - 1 - row, column] == None:
                found_valid_spot = True
                self.grid[self.rows - 1 - row, column] = object_mapping[item]

    def render(self, mode='human'):
        if self.viewer == None:
            self.viewer = rendering.Viewer(self.viewport.width, self.viewport.height)
            self.viewer.set_bounds(0, self.viewport.width/self.SCALE_W, 0, self.viewport.height/self.SCALE_H)
            if (self.on_key_press is not None) and (self.on_key_release is not None):
                self.viewer.window.on_key_press = self.on_key_press
                self.viewer.window.on_key_release = self.on_key_release

        # Draw background
        self.viewer.draw_polygon( [(0, 0),
                                  (self.viewport.width/self.SCALE_W, 0),
                                  (self.viewport.width/self.SCALE_W, self.viewport.height/self.SCALE_H),
                                  (0,self.viewport.height/self.SCALE_H),
                                  ], color=(0.0, 0.0, 0.0) )

        # Draw grid border
        self.viewer.draw_polyline( [(1, 1),
                                  (1, self.rows + 1),
                                  (self.columns + 1, self.rows + 1),
                                  (self.columns + 1, 1),
                                  (1, 1),
                                  ], color=(255.0, 255.0, 255.0) )

        # Draw grid lines
        if self.render_grid:
            for i in range(2,self.columns+1):
                self.viewer.draw_polyline( [(i, 1), (i, self.rows + 1)], color=(255.0, 255.0, 255.0) )
            for i in range(2,self.rows+1):
                self.viewer.draw_polyline( [(1, i), (self.columns + 1, i)], color=(255.0, 255.0, 255.0) )

        # Draw objects
        for row in range(self.grid.shape[0]):
            for column in range(self.grid.shape[1]):
                g_object = self.grid[row, column]
                if g_object is not None:
                    self.render_object(column + 1, row + 1, g_object.color)

        # Draw grid agents
        for agent in self.agents:
            self.render_object(agent.position[0] + 1, agent.position[1] + 1, agent.color)

        # Draw agent
        self.render_object(self.agent_pos[0] + 1, self.agent_pos[1] + 1, self.agent_color)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def get_new_pos_from_action(self, action, old_pos):
        new_pos = old_pos.copy()
        if action == 1:
            new_pos[0] = new_pos[0] - 1
        elif action == 2:
            new_pos[0] = new_pos[0] + 1
        elif action == 3:
            new_pos[1] = new_pos[1] - 1
        elif action == 4:
            new_pos[1] = new_pos[1] + 1
        return new_pos

    def step(self, action):

        self.step_count += 1

        new_pos = self.get_new_pos_from_action(action, self.agent_pos)

        reward = -1
        if self.is_walkable(new_pos[0], new_pos[1]):
            self.agent_pos = new_pos
        else:
            reward += -5

        # Update grid agents
        for agent in self.agents:
            agent.step(self)

        if self.from_pixels:
            obs = self.create_image_observation()
        else:
            obs = self.create_discrete_observation()
        reward += self.encounter_object(new_pos[0], new_pos[1], self.encounter_other_agents)
        done = self.is_done()

        return (obs, reward, done, "INFO")

    def is_done(self):
        if self.step_count > self.max_steps:
            return True

        for row in range(self.grid.shape[0]):
            for column in range(self.grid.shape[1]):
                if self.grid[row, column] is not None and self.grid[row, column].reward_on_encounter > 0:
                    return False

        for agent in self.agents:
            if agent.reward_on_encounter > 0:
                return False

        return True

    def within_bounds(self, column, row):
        return (row >= 0) and (row < self.rows) and (column >= 0) and (column < self.columns)

    def is_walkable(self, column, row):

        # Check that we are not trying to move to position that is out of bounds
        if not self.within_bounds(column, row):
            return False

        # Check that we are not tryning to walk on unwalkable objects
        if self.grid[row, column] is not None:
                if not self.grid[row, column].is_walkable:
                    return False

        # Check that we are not trying to walk on unwalkable agents
        for agent in self.agents:
            if (agent.position[0]==column) and (agent.position[1]==row):
                if not agent.is_walkable:
                    return False

        return True

    def encounter_object(self, column, row, encounter_agents=True):
        reward = 0
        if self.within_bounds(column, row) and self.grid[row, column] is not None:
            reward += self.grid[row, column].reward_on_encounter
            if self.grid[row, column].is_consumed:
                self.grid[row, column] = None

        if encounter_agents:
            for agent in self.agents:
                if (agent.position[0]==column) and (agent.position[1]==row):
                    reward += agent.reward_on_encounter
                    if agent.is_consumed:
                        self.agents.remove(agent)

        return reward

    def render_object(self, x, y, object_color):
        x1 = x + self.viewport.object_delta
        x2 = x - self.viewport.object_delta + 1
        y1 = y + self.viewport.object_delta
        y2 = y - self.viewport.object_delta + 1 
        self.viewer.draw_polygon( [(x1, y1), (x2, y1), (x2, y2), (x1, y2),], color=object_color )

    def toggle_grid(self):
        self.render_grid = not self.render_grid

    def set_render_grid(self, value):
        self.render_grid = value

    def reset(self):
        self.load_map(self.map, self.object_mapping)
        self.agent_pos = self.agent_start
        self.agents.clear()
        for agent in self.init_agents:
            self.agents.append(copy.deepcopy(agent))
        self.step_count = 0
        if self.from_pixels:
            obs = self.create_image_observation()
        else:
            obs = self.create_discrete_observation()
        return obs

class RandomPlayer:

    def __init__(self, env, num_episodes):
        self.env = env
        self.num_episodes = num_episodes

    def run(self):
        episodes = 0
        while episodes < self.num_episodes:
            done = False
            episode_reward = 0
            self.env.reset()
            while not done:
                self.env.render()
                r, done = self.step_env()
                episode_reward += r
                if done:
                    episodes += 1
                    print("########################")
                    print("Episode done")
                    print("Episode reward: %s" % episode_reward)
                    print("########################")
                time.sleep(0.5)

    def step_env(self):
        _, r, done, _ = self.env.step(self.env.action_space.sample())
        return (r, done)

TEST_MAP = [
' o                  ',
'       o       o    ',
'    #######         ',
'          #         ',
'  p       ###       ',
'               p    ',
'    #  o            ',
'    #          o    ',
'    #     ssss   oo ',
'    #  p            ',
'           #   p    ',
'  p        #        ',
'      o  q      o   ',
'      #########     ',
'o          p  #  p  ',
'    s #       #     ',
'    s # ooooo #     ',
'  o s #########     ',
'                 o  ',
'         p     o    ',
]

TEST_MAPPING = {
    '#': GridObject(False, False, 0, (255.0, 255.0, 255.0), 1),
    'o': GridObject(True, True, 100, (0.0, 255.0, 0.0), 2),
    'p': GridObject(True, True, -20, (255.0, 0.0, 0.0), 3),
    'q': GridObject(True, True, -10, (255.0, 255.0, 0.0), 4),
    's': GridObject(True, False, -10, (255.0, 255.0, 0.0), 5),
    ' ': None
}

agent1 = ScrollerAgent(True, ScrollerAgent.GOING_RIGHT, 0, True, False, 0, (255.0, 0.0, 255.0), 6)
agent1.set_position([16,3])
agent2 = ScrollerAgent(True, ScrollerAgent.GOING_UP, 0, True, False, 0, (255.0, 0.0, 255.0), 6)
agent2.set_position([14,8])

TEST_AGENTS = [agent1, agent2]

if __name__=="__main__":
    my_grid = Gridworld(map=TEST_MAP, object_mapping=TEST_MAPPING, init_agents=TEST_AGENTS)
    player = RandomPlayer(my_grid, 1)
    player.run()
