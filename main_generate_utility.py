from MORL_stablebaselines3.utility_function.utility_function_parameterized import Utility_Function_Parameterized
from MORL_stablebaselines3.utility_function.utility_function_programmed import Utility_Function_Programmed
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import glob
from IPython import embed
import sys
import argparse

parser = argparse.ArgumentParser(description="Generating a set of utility functions")
parser.add_argument(
    '--reward_shape',
    help="Dimension of reward function",
    type=int,
    default=2,
)
parser.add_argument(
    '--num_utility_function',
    help="Number of generated utility functions",
    type=int,
    default=100,
)

args = parser.parse_args()

reward_shape, num_utility_function = args.reward_shape, args.num_utility_function

model_dir = f'utility-model/'
plot_dir = f'utility-plot/'

scale = 100
datasets = np.random.uniform(low=0.1 * scale, high=0.9 * scale, size=[1000, reward_shape])
datasets_delta = np.random.uniform(low=-0.1 * scale, high=0.1 * scale, size=[1000, reward_shape])
norm_delta = np.linalg.norm(datasets_delta, -1)
min_val = np.array([0.0] * reward_shape) * scale
max_val = np.array([1.0] * reward_shape) * scale

def compute_grads(utility_function):
    with torch.no_grad():
        value = utility_function(datasets, scale_back=False)
        value_near = utility_function(datasets + datasets_delta, scale_back=False)
        results = []
        for i in range(reward_shape):
            delta = np.zeros_like(datasets_delta)
            delta[:, i] = 0.1
            values_near = utility_function(datasets + delta, scale_back=False)
            results.append((values_near - value) / 0.1)
    grad = np.stack(results, axis=-1)
    return value, value_near, grad

def train_utility(utility_function, values, values_near, values_grad):
    print(f'Start Training #{len(values)}')
    learning_rate = 5e-3
    optim = torch.optim.Adam(utility_function.parameters(), lr=learning_rate)
    epochs = 100
        
    value_tensor = torch.as_tensor(values).cuda()  # [num_others, dataset_size]
    value_near_tensor = torch.as_tensor(values_near).cuda()
    value_grad_tensor = torch.as_tensor(values_grad).cuda()
    value_grad_tensor = value_grad_tensor / (value_grad_tensor.norm(dim=-1).unsqueeze(-1) + 1e-6)
    
    dist_tensor = torch.as_tensor(np.linalg.norm(datasets_delta, axis=-1)).cuda()
    degree_tensor = torch.atan2(value_tensor - value_near_tensor, dist_tensor.unsqueeze(0) / scale)
    
    for _ in range(epochs):
        value_current = utility_function(datasets, return_numpy=False, scale_back=False)
        value_near_current = utility_function(datasets + datasets_delta, return_numpy=False, scale_back=False)
        
        degree_current = torch.atan2(value_current - value_near_current, dist_tensor  / scale)
        dist_value = ((value_tensor - value_current.unsqueeze(0)) ** 2).sum(1)
        dist_degree = ((degree_tensor - degree_current.unsqueeze(0)) ** 2).sum(1)
        dist_grad = torch.zeros_like(dist_degree)

        loss_value = torch.logsumexp(-dist_value, 0)
        loss_degree = torch.logsumexp(-dist_degree, 0)
        loss_grad = torch.logsumexp(-dist_grad, 0)
        loss = 0.1 * loss_value + 0.5 * loss_degree + 0.0 * loss_grad
        if (_+1) % 20 == 0:
            print(f'epoch {_+1}: loss: {loss.item()}, loss_value: {loss_value.item()}, loss_degree: {loss_degree.item()}, loss_grad: {loss_grad.item()}')

        optim.zero_grad()
        loss.backward()
        optim.step()

        # make weight to be positive
        utility_function.make_monotone()
        
    print(f'End Training #{len(values)}')
        
    with torch.no_grad():
        value, value_near, grad = compute_grads(utility_function)
        values.append(value)
        values_near.append(value_near)
        values_grad.append(grad)
        
    return loss.item()

def generate_utility():
    dummy_utility_function = Utility_Function_Programmed(reward_shape=reward_shape, function_choice=0)
    num_deterministic_utility = len(dummy_utility_function.utility_functions)
    utility_function_list = []
    loss_list = []
    for i in range(num_deterministic_utility):
        utility_function = Utility_Function_Programmed(reward_shape=reward_shape, lamda=0.0, function_choice=i)
        utility_function.min_val = min_val.copy()
        utility_function.max_val = max_val.copy()
        utility_function_list.append(utility_function)
        loss_list.append(0.0)
        
    values = []
    values_near = []
    values_grad = []
    for utility_function in utility_function_list:
        with torch.no_grad():
            value, value_near, grad = compute_grads(utility_function)
            values.append(value)
            values_near.append(value_near)
            values_grad.append(grad)
    
    for _ in range(num_utility_function):
        utility_function = Utility_Function_Parameterized(reward_shape=reward_shape, lamda=0.1, max_weight=0.5, size_factor=2).cuda()
        utility_function.min_val = min_val.copy()
        utility_function.max_val = max_val.copy()
        loss = train_utility(utility_function, values, values_near, values_grad)
        utility_function_list.append(utility_function)
        loss_list.append(loss)
        
    plot_utility(utility_function_list[num_deterministic_utility:], loss_list)
    save_utility(utility_function_list[num_deterministic_utility:], start=0)
    
    
def plot_utility(utility_function_list, loss_list):
    X, Y = np.meshgrid(np.linspace(min_val[0], max_val[0], 256), np.linspace(min_val[1], max_val[1], 256))
    XY = np.stack([X, Y], -1).reshape(256 * 256, 2)
    if reward_shape >= 3:
        others = np.zeros([256 * 256, reward_shape - 2])
        XY = np.concatenate([XY, others], 1)
    for i, (utility, loss) in enumerate(zip(utility_function_list, loss_list)):
        if i % 10 == 0:
            print(f'Plotting utility function #{i}/{len(utility_function_list)}')
        utility.eval()
        with torch.no_grad():
            Z = utility(XY)
        Z = Z.reshape(256, 256)
        levels = np.linspace(Z.min()-0.01, Z.max()+0.01, 60)
        
        plt.clf()
        plt.contourf(X, Y, Z, levels=levels)
        plt.colorbar()
        plt.title(f'Utility function #{i}')
        os.makedirs(f'{plot_dir}/dim-{reward_shape}', exist_ok=True)
        plt.savefig(f'{plot_dir}/dim-{reward_shape}/utility-{i}.png', dpi=160)
    
def save_utility(utility_function_list, start):
    os.makedirs(f'{model_dir}/dim-{reward_shape}', exist_ok=True)
    for i in range(start, len(utility_function_list)):
        torch.save(utility_function_list[i].state_dict(), f'{model_dir}/dim-{reward_shape}/utility-{i}.pt')

if __name__ == '__main__':
    generate_utility()