#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:22:26 2018

@author: arjumand
"""
import matplotlib
import IPython
import numpy as np
import matplotlib.pyplot as plt
import gym

#Toy 2D Grid environment adapted from: https://github.com/dtak/hip-mdp-public/blob/master/grid_simulator/grid.py

goals = [
    [5.0,0],
    [-5.0,0],
    [0,5.0],
    [0,-5.0]
]

reward_var_mat = [[[2.5, 0.0],
                   [0.0, 2.5]],
                  [[1.0, 0.4],
                   [0.4, 1.0]],
                  [[2.5, 0.0],
                   [0.0, 2.5]],
                  [[10.0, 5.0],
                   [5.0, 10.0]]]

in_goal_reward_mean = [[12.0, -1.0],
                       [7.0, 8.0],
                       [-1.0, 12.0],
                       [8.0, 8.0]]

goal_radii = [1.5,1.5,1.5,1.5]
step_size = 0.6

class DiverseGoalEnv(gym.Env):
    """
    This is a 2D Grid environment for simple RL tasks    
    """

    def __init__(self, start_state = [0,0], step_size = step_size, **kw):
        """
        Initialize the environment: creating the box, setting a start and goal region
        Later -- might include other obstacles etc
        """  
        
        self.num_actions = 4
        self.x_range = [-7.0,7.0]
        self.y_range = [-7.0,7.0]
        self.reward_dim = 2
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=np.array([self.x_range[0], self.y_range[0]] * (1 + len(goals))), high=np.array([self.x_range[1], self.y_range[1]] * (1 + len(goals))))
#        self.goal_radius = 0.5
#        self.goal = [1.0,0]
        self.reset(start_state, step_size,**kw)
        
    def reset(self, start_state = [0,0], step_size = step_size, goal = goals, goal_radius = goal_radii, x_std = 0, y_std = 0, **kw):
        """
        Reset Environment
        """
        self.t = 0
        self.step_size = step_size
        self.start_state = start_state
        self.state = start_state
        self.goal_radius = goal_radius
        self.goal = goal
        self.in_goal_reward_mean = np.array(in_goal_reward_mean)
        self.out_goal_reward_mean = np.array([-0.0, -0.0])
        self.out_range_reward_mean = np.array([-10.0, -10.0])
        self.reward_var_mat = np.array(reward_var_mat)
        self.x_std = x_std
        self.y_std = y_std
        return self.observe(), {}
        
    def observe(self):
        obs_list = [self.state]
        for goal in self.goal:
            obs_list.append(self.state - np.array(goal))
        return np.concatenate(obs_list)
        # return self.state
    
    def get_action_effect(self, action):
        """
        Set the effect direction of the action -- actual movement will involve step size and possibly involve action error
        """
        if action == 0:
            return [1,0]
        elif action == 1:
            return [-1,0]
        elif action == 2:
            return [0,1]
        elif action == 3:
            return [0,-1]
        
    def get_next_state(self, state, action):
        """
        Take action from state, and return the next state
        
        """
        action_effect = self.get_action_effect(action)
        new_x = state[0] + (self.step_size * action_effect[0]) + np.random.normal(0, self.x_std)
        new_y = state[1] + (self.step_size * action_effect[1]) + np.random.normal(0, self.y_std)
        
        
        next_state = [new_x, new_y]
        return next_state
    
    def _valid_crossing(self, state=None, next_state=None, action = None):
        if state is None:
            state = self.state
            action = self.action
        if next_state is None:
            next_state = self.get_next_state(state, action)
            
        #Check for moving out of box in x direction 
        if next_state[0] < np.min(self.x_range) or next_state[0] > np.max(self.x_range):
#            print "Going out of x bounds"
            return False
        elif next_state[1] < np.min(self.y_range) or next_state[1] > np.max(self.y_range):
#            print "Going out of y bounds"
            return False
        else:
            return True
        
    def _in_goal(self, state = None):
        if state is None:
            state = self.state  
        each_goal = -1
        for i, (goal_i, radius_i) in enumerate(zip(self.goal, self.goal_radius)):
            if (np.linalg.norm(np.array(state) - np.array(goal_i)) < radius_i):
                each_goal = i
                break
        return each_goal
            
                
    def calc_reward(self, state = None, action = None, next_state = None, **kw):
        if state is None:
            state = self.state
            action = self.action
        if next_state is None:
            next_state = self.get_next_state(state,action)
        cur_goal = self._in_goal(state = next_state)
        in_range = self._valid_crossing(state = state, next_state = next_state, action = action)
        self.in_range = in_range
        if in_range and cur_goal > -1:
            return np.random.multivariate_normal(self.in_goal_reward_mean[cur_goal], self.reward_var_mat[cur_goal])
        elif in_range and cur_goal == -1:
            return self.out_goal_reward_mean
        else: 
            return self.out_range_reward_mean
                
    def step(self, action, **kw):
        self.t += 1
        self.action = action
        reward = self.calc_reward()
        if self._valid_crossing():
            self.state = self.get_next_state(self.state, action)
        # print("Inner env next state = {}, reward = {}, done = {}".format(self.state, reward, self._in_goal() != -1))
        current_goal = self._in_goal()
        # if current_goal != -1:
        #     print('goal', current_goal)
        done = current_goal != -1 or not self.in_range or self.t >= 100
        return self.observe(), reward, done, False, {}
    
def plot_env(env, policy_states, name="", save_path="performance.png"):
    plt.clf()
    theta = np.linspace(0, 2 * np.pi)
    goals, goal_radii = env.goal, env.goal_radius
    for i, (goal, goal_radius) in enumerate(zip(goals, goal_radii)):
        x_circle = goal_radius * np.sin(theta) + goal[0]
        y_circle = goal_radius * np.cos(theta) + goal[1]
        plt.plot(x_circle, y_circle, marker='o', markersize=1, label=f'Goal {i}', linewidth=3.5)
    
    plt.scatter([0.0], [0.0], marker='x', color='r', label='Start', s=[50])
    plt.legend(prop={'size': 10})
    
    xlim = env.x_range
    ylim = env.y_range
    plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
    plt.xlabel('x')
    plt.ylabel('y')
    for path in policy_states:
        path = np.array(path)
        # if count == 0:
        #     plt.plot(path[:,0], path[:,1], 'bo--', markersize = 3)
        # else:
        plt.plot(path[:, 0], path[:, 1], color='b', marker='o', linestyle='-', markersize=3, alpha=0.5, linewidth=3.5)
    plt.title(f'{name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    
            
if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 15})
    env = DiverseGoalEnv()
    goal_size = len(env.goal)
    for cur_goal in range(goal_size):
        points = np.random.multivariate_normal(env.in_goal_reward_mean[cur_goal], env.reward_var_mat[cur_goal], size=200)
        plt.scatter(points[:,0], points[:,1], alpha=0.6, label=f"Goal {cur_goal}")
    plt.xlabel("Reward 0")
    plt.ylabel("Reward 1")
    plt.title("Reward Distribution on Each Goal")
    plt.legend(prop={'size': 12})
    plt.axis('equal')
    plt.savefig("reward-diverse-goal-env.png", dpi=160, bbox_inches='tight')
    
    plot_env(env, [[[0, 0], [-0.6, 0], [-0.6, 0.6], [-1.2, 0.6]]], name='Diverse Goal Environment', save_path='map.png')
