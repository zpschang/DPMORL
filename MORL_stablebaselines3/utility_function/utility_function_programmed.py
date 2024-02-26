import IPython
import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt
import os
import glob
import pickle

np.random.seed(5)

class Utility_Function(nn.Module):
    def __init__(self, reward_shape=2, norm=True, lamda=0.1, function_choice=1):
        super().__init__()
        self.function_choice = function_choice
        self.min_val = torch.full((1, reward_shape,), np.inf)
        self.max_val = torch.full((1, reward_shape,), -np.inf)
        self.params_update_flag = False
        self.norm = norm
        
    def forward(self, xx):
        x = torch.tensor(xx)
        if self.norm and x.shape[0] == 1:
            self.min_val = torch.min(x, self.min_val)
            self.max_val = torch.max(x, self.max_val)
        inputs = torch.cat([self.min_val, self.max_val, x], 0)
        utilities = self.compute_utility(inputs)
        min_util, max_util, util = utilities[0], utilities[1], utilities[2:]
        return (util - min_util) / (max_util - min_util + 1e-6)

    def compute_utility(self, x):
        if self.norm:
            x = (x - self.min_val) / (self.max_val - self.min_val + 1e-5)
        if self.function_choice == 0:
            return torch.mean(x, dim=1)
        elif self.function_choice == 1:
            return torch.max(x, 1)[0]
        elif self.function_choice == 2:
            return x.sum(1)
        elif self.function_choice == 3:
            return 0.2 * x[:, 0] + 0.8 * x[:, 1]
        elif self.function_choice == 4:
            return torch.nn.functional.softplus(x[:, 0]) + torch.pow(torch.nn.functional.softplus(x[:, 1:]), 2).sum(1)
        elif self.function_choice == 5:
            return -torch.pow(x + 1e-7, -1).sum(1)
        elif self.function_choice == 6:
            return torch.nn.functional.softplus(x[:, 0]) + torch.nn.functional.softplus(x[:, 1:]).sum(1)
        elif self.function_choice == 7:
            return torch.pow(torch.nn.functional.softplus(x), 2).sum(1) + torch.pow(torch.nn.functional.softplus(x),
                                                                                    3).sum(1)
        elif self.function_choice == 8:
            return torch.exp(x).sum(1)
        elif self.function_choice == 9:
            return x[:, 0] + torch.pow(x[:, 1:], 2).sum(1)
        else:
            return NotImplementedError

    def plot_distribution(self):
        x = torch.linspace(-10, 10, 1000)
        y = torch.linspace(-10, 10, 1000)
        X, Y = torch.meshgrid(x, y)
        Z = torch.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i, j] = self.forward(torch.tensor([x[i], y[j]]))
        plt.contourf(X, Y, Z, levels=20)
        plt.colorbar()
        plt.xlabel('reward 1')
        plt.ylabel('reward 2')
        plt.show()
        
class Utility_Function_Programmed(nn.Module):
    def __init__(self, reward_shape=2, norm=True, lamda=0.1, function_choice=1, keep_scale=True):
        super().__init__()
        self.function_choice = function_choice
        self.min_val = np.full(reward_shape, np.inf)
        self.max_val = np.full(reward_shape, -np.inf)
        self.params_update_flag = False
        self.norm = norm
        self.keep_scale = keep_scale
        self.reward_shape = reward_shape
        
        self.utility_functions = []
        self.utility_functions.append(lambda x: np.mean(x, 1))
        for i in range(self.reward_shape):
            weights = np.ones(self.reward_shape)
            weights[i] *= 4
            weights = weights / weights.sum()
            self.utility_functions.append(lambda x, w=weights: np.sum(x * np.expand_dims(w, 0), 1))
        
        
    def forward(self, xx, scale_back=True):
        x = np.array(xx)
        if self.norm:
            self.min_val = np.minimum(x.min(0), self.min_val)
            self.max_val = np.maximum(x.max(0), self.max_val)
        if self.keep_scale:
            scale = (self.max_val - self.min_val).max()
            middle_point = (self.max_val + self.min_val) / 2
            min_input = middle_point - scale / 2
            max_input = middle_point + scale / 2
        else:
            min_input = self.min_val
            max_input = self.max_val
            
        inputs = np.concatenate([[min_input], [max_input], x], 0)
        if self.norm:
            inputs = (inputs - np.expand_dims(min_input, 0)) / np.expand_dims(max_input - min_input + 1e-5, 0)
        utilities = self.compute_utility(inputs)
        min_util, max_util, util = utilities[0], utilities[1], utilities[2:]
        util = (util - min_util) / (max_util - min_util + 1e-6)
        if scale_back:
            util *= (max_input - min_input).mean()
        util *= 2
        return util

    def compute_utility(self, x):
        return self.utility_functions[self.function_choice](x)

    def plot_distribution(self):
        x = torch.linspace(-10, 10, 1000)
        y = torch.linspace(-10, 10, 1000)
        X, Y = torch.meshgrid(x, y)
        Z = torch.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i, j] = self.forward(torch.tensor([x[i], y[j]]))
        plt.contourf(X, Y, Z, levels=20)
        plt.colorbar()
        plt.xlabel('reward 1')
        plt.ylabel('reward 2')
        plt.show()

        
class Utility_Function_Linear(nn.Module):
    def __init__(self, reward_shape=2, norm=True, lamda=0.1, function_choice=1, keep_scale=True):
        super().__init__()
        print('initializing linear utility function')
        self.function_choice = function_choice
        self.min_val = np.full(reward_shape, np.inf)
        self.max_val = np.full(reward_shape, -np.inf)
        self.params_update_flag = False
        self.norm = norm
        self.keep_scale = keep_scale
        self.reward_shape = reward_shape
        
        self.utility_functions = []
        
        weights = [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.5, 0.5], 
            [0.3, 0.7],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.98, 0.02],
            [0.02, 0.98],
            [0.95, 0.05],
            [0.05, 0.95],
            [1.0, 0.0], 
            [0.0, 1.0],
        ]

        for weight in weights:
            weight = np.array(weight)
            self.utility_functions.append(lambda x, w=weight: np.sum(x * np.expand_dims(w, 0), 1))
        
    def forward(self, xx):
        x = np.array(xx)
        if self.norm:
            self.min_val = np.minimum(x.min(0), self.min_val)
            self.max_val = np.maximum(x.max(0), self.max_val)
        if self.keep_scale:
            scale = (self.max_val - self.min_val).max()
            middle_point = (self.max_val + self.min_val) / 2
            min_input = middle_point - scale / 2
            max_input = middle_point + scale / 2
        else:
            min_input = self.min_val
            max_input = self.max_val
            
        inputs = np.concatenate([[min_input], [max_input], x], 0)
        if self.norm:
            inputs = (inputs - np.expand_dims(min_input, 0)) / np.expand_dims(max_input - min_input + 1e-5, 0)
        utilities = self.compute_utility(inputs)
        min_util, max_util, util = utilities[0], utilities[1], utilities[2:]
        util = (util - min_util) / (max_util - min_util + 1e-6)
        util *= (max_input - min_input).mean()
        return util

    def compute_utility(self, x):
        return self.utility_functions[self.function_choice](x)
    



class Utility_Function_Diverse_Goal(nn.Module):
    def __init__(self, reward_shape=2, norm=True, lamda=0.1, function_choice=1, keep_scale=True):
        super().__init__()
        self.function_choice = function_choice
        self.min_val = np.full(reward_shape, np.inf)
        self.max_val = np.full(reward_shape, -np.inf)
        self.params_update_flag = False
        self.norm = norm
        self.keep_scale = keep_scale
        self.reward_shape = reward_shape
        
        self.utility_functions = []
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * 2))

        self.utility_functions.append(lambda x: np.clip(x[:, 0] - 7.5, 0.0, np.inf))
        self.utility_functions.append(lambda x: np.clip(x[:, 1] - 7.5, 0.0, np.inf))
        self.utility_functions.append(lambda x: sigmoid((x[:, 0] - 2.5) * 2) * sigmoid((x[:, 1] - 2.5) * 2))
        self.utility_functions.append(lambda x: np.clip(x.mean(1) - 9.0, 0.0, 1.0))
        self.utility_functions.append(lambda x: sigmoid(x[:, 0] - 6.0))
        self.utility_functions.append(lambda x: sigmoid(0.2 * x[:, 0] + 0.8 * x[:, 1] - 2.5) * sigmoid(0.8 * x[:, 0] + 0.2 * x[:, 1] - 2.5))

        
        
    def forward(self, xx):
        x = np.array(xx)
        if self.norm:
            self.min_val = np.minimum(x.min(0), self.min_val)
            self.max_val = np.maximum(x.max(0), self.max_val)
        if self.keep_scale:
            scale = (self.max_val - self.min_val).max()
            middle_point = (self.max_val + self.min_val) / 2
            min_input = middle_point - scale / 2
            max_input = middle_point + scale / 2
        else:
            min_input = self.min_val
            max_input = self.max_val
            
        inputs = np.concatenate([[min_input], [max_input], x], 0)
        if self.norm:
            inputs = (inputs - np.expand_dims(min_input, 0)) / np.expand_dims(max_input - min_input + 1e-5, 0)
        utilities = self.compute_utility(inputs)
        min_util, max_util, util = utilities[0], utilities[1], utilities[2:]
        if self.norm:
            util = (util - min_util) / (max_util - min_util + 1e-6)
            util *= (max_input - min_input).mean()
        return util

    def compute_utility(self, x):
        return self.utility_functions[self.function_choice](x)
