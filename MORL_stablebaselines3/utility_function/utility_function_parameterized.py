import IPython
import numpy as np
from torch import nn
import torch
import matplotlib.pyplot as plt
import os

np.random.seed(5)

class Utility_Function_Parameterized(nn.Module):
    def __init__(self, reward_shape=2, norm=True, lamda=0.05, function_choice=1, max_weight=0.1, keep_scale=True, size_factor=1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(reward_shape, affine=False)
        self.mlp1 = nn.Linear(reward_shape, 24 * size_factor)
        self.mlp2 = nn.Linear(72 * size_factor, 24 * size_factor)
        self.bn2 = nn.BatchNorm1d(72 * size_factor, affine=True)
        self.mlp3 = nn.Linear(72 * size_factor, 24 * size_factor)
        self.bn3 = nn.BatchNorm1d(72 * size_factor, affine=True)
        self.mlp4 = nn.Linear(72 * size_factor, 1)
        self.max_weight = max_weight
        self.final_bias, self.final_normalizer = 0.0, 1.0
        self.min_val = np.full(reward_shape, np.inf)
        self.max_val = np.full(reward_shape, -np.inf)
        self.params_update_flag = False
        self.norm = norm
        self.keep_scale = keep_scale
        self.lamda = lamda
        self.reward_shape = reward_shape
        # Initialize positive weight
        self.make_monotone_init()
        self.make_monotone()

        # Save initialized params
        self.init_params = self.state_dict()
        
    def forward(self, xx, return_numpy=True, scale_back=True):
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

        if return_numpy:
            utilities = self.compute_utility(inputs).detach().cpu().numpy()
        else:
            utilities = self.compute_utility(inputs)

        min_util, max_util, util = utilities[0], utilities[1], utilities[2:]
        util = (1 - self.lamda) * (util - min_util) / (max_util - min_util + 1e-6)
        # Add lambda to keep the gradient of utility function. 
        
        if return_numpy:
            util += self.lamda * (x / (max_input - min_input + 1e-5)).mean(1)
        else:
            util += torch.as_tensor(self.lamda * (x / (max_input - min_input + 1e-5)).mean(1)).cuda()
        # rescale to the original return scale
        if scale_back:
            util *= (max_input - min_input).mean()
        util *= 2
        
        return util

    def compute_utility(self, input_x): 
        input_x = torch.as_tensor(input_x, dtype=torch.float32).cuda()
        input_x = self.bn1(input_x)
        x = self.mlp1(input_x)
        x = torch.cat([x.clamp(min=-0.5), x.clamp(min=-0.5, max=0.5), x.clamp(max=0.5)], -1)
        x = self.bn2(x)
        x = self.mlp2(x)
        x = torch.cat([x.clamp(min=-0.5), x.clamp(min=-0.5, max=0.5), x.clamp(max=0.5)], -1)
        x = self.bn3(x)
        x = self.mlp3(x)
        x = torch.cat([x.clamp(min=-0.5), x.clamp(min=-0.5, max=0.5), x.clamp(max=0.5)], -1)
        x = self.mlp4(x)
        return x[:, 0]

    def random_init(self): # After this, the utility function will be updated
        self.load_state_dict(self.init_params)
        self.params_update_flag = True

    def make_monotone_init(self):
        for layer in [self.mlp1, self.mlp2, self.mlp3, self.mlp4, self.bn2, self.bn3]:
            layer.weight.data = layer.weight.data.abs()

    def make_monotone(self):
        for layer in [self.mlp1, self.mlp2, self.mlp3, self.mlp4, self.bn2, self.bn3]:
            layer.weight.data = torch.maximum(layer.weight.data, torch.tensor(0.0))
        for layer in [self.mlp1, self.mlp2, self.mlp3, self.mlp4, self.bn2, self.bn3]:
            layer.weight.data = torch.minimum(layer.weight.data, torch.tensor(self.max_weight))


def visualize_utility(min_x, max_x, min_y, max_y,
                      model, name_a, name_b):
    os.makedirs('result', exist_ok=True)
    X, Y = np.meshgrid(np.linspace(min_x - 2.0, max_x + 2.0, 256), np.linspace(min_y - 2.0, max_y + 2.0, 256))
    XY = np.stack([X, Y], -1).reshape(256 * 256, 2)
    with torch.no_grad():
        model.eval()
        Z = model(torch.as_tensor(XY, dtype=torch.float32))
        Z = Z.numpy().reshape(256, 256)
        model.train()
    levels = np.linspace(Z.min()-0.01, Z.max()+0.01, 60)
    plt.clf()
    plt.contourf(X, Y, Z, levels=levels)
    plt.colorbar()
    plt.title(f'Utility function: max {name_a}-{name_b}')
    plt.savefig(f'result/utility-max-{name_a}-min-{name_b}.png', dpi=160)
