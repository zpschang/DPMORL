from typing import Any
import numpy as np
import glob
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(0)

envs = ['Hopper', 'HalfCheetah', 'MountainCar', 'DeepSeaTreasure', 'FruitTree', 'Reacher', 'ReacherBullet']
algs = ['gpi-ls', 'gpi-pd', 'ols', 'pgmorl', 'dpmorl']

data_dir = 'test_returns_baselines'

def get_env(alg, sub_dir):
    sub_dir = sub_dir.split('/')[-1]
    sub_dir = sub_dir.split('.')[1]
    return sub_dir

def filter_policy(alg, sub_dir, policies_return_path):
    policy_names = [path.split('/')[-1].split('.')[0].split('_')[-1] for path in policies_return_path]
    policy_names = sorted(policy_names)
    if alg in ['dpmorl', 'dpmorl2']:
        final_policy_name = []
        for policy_name in policy_names:
            number = int(policy_name.split('-')[-1])
            if number < 12:
                final_policy_name.append(policy_name)
        final_policy_paths = [f'{sub_dir}/test_returns_policy_{policy_name}.npz' for policy_name in final_policy_name]
    else:
        final_policy_paths = policies_return_path
    return final_policy_paths

all_data = {env: {alg: [] for alg in algs} for env in envs}

for alg in algs:
    # Read data from data_dir
    sub_dirs = glob.glob(f'{data_dir}/{alg}/*')
    env_map = {get_env(alg, sub_dir): sub_dir for sub_dir in sub_dirs}
    for env in envs:
        if env not in env_map:
            continue
        sub_dir = env_map[env]
        policies_return_path = glob.glob(f'{sub_dir}/test_returns_*.npz')
        
        policies_return_path = filter_policy(alg, sub_dir, policies_return_path)
        
        for path in policies_return_path:
            data = np.load(path)['test_returns'][..., :2]
            all_data[env][alg].append(data.copy())
            
            


env_stats = {}

for env in envs:
    minimum, maximum = [], []
    min_std, max_std = [], []
    for alg in algs:
        data = np.array(all_data[env][alg])[..., :2]
        if len(data) == 0:
            continue
        minimum.append(data.min(0).min(0))
        maximum.append(data.max(0).max(0))
        min_std.append(data.std(1).min(0))
        max_std.append(data.std(1).max(0))
    minimum = np.array(minimum).min(0)
    maximum = np.array(maximum).max(0)
    min_std = np.array(min_std).min(0)
    max_std = np.array(max_std).max(0)
    env_stats[env] = {
        'minimum': minimum,
        'maximum': maximum,
        'min_std': min_std,
        'max_std': max_std
    }
    print(env, min_std, max_std)

def expected_utility(fronts, **kwargs):
    weights = np.stack([np.arange(0, 1.01, 0.01), 1 - np.arange(0, 1.01, 0.01)], axis=1)
    maxs = []
    func = lambda x, y: (x * y).sum()

    for i, weight in enumerate(weights):
        scalarized_fronts = []
        for front in fronts:
            samples = [func(point, weight) for point in front]
            scalarized_fronts.append(np.mean(samples))
        maxs.append(np.max(scalarized_fronts))
    return np.mean(maxs)

def cvar(front, alpha: float = 0.05, **kwargs) -> float:
    """CVaR Metric.

    Conditional Value at Risk of the policies on the PF for various weights.

    Args:
        front: current pareto front to compute the cvar on
        weights_set: weights to use for the utility computation
        alpha: risk level, must be between 0 and 1 (default: 0.05)
        utility: utility function to use (default: dot product)

    Returns:
        float: cvar metric
    """
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"

    weights_set = np.stack([np.arange(0, 1.01, 0.01), 1 - np.arange(0, 1.01, 0.01)], axis=1)

    cvars = []

    for weights in weights_set:
        maxs = []
        for cur_front in front:
            scalarized_front = np.array([np.dot(weights, point) for point in cur_front])
            sorted_returns = np.sort(scalarized_front)
            cutoff_index = int(np.ceil(alpha * len(sorted_returns)))
            maxs.append(np.mean(sorted_returns[:cutoff_index]))
        cvars.append(np.max(maxs))
    return np.mean(cvars)

constraints = {}
variance_constraint = {}

class ReturnConstraint:
    def __init__(self, weights, ts) -> None:
        self.weights = weights
        self.ts = ts
        
    def __call__(self, x, y) -> Any:
        pos = np.array([x, y])
        res = [np.dot(weight, pos) >= t for weight, t in zip(self.weights, self.ts)]
        return all(res)
    

if __name__ == '__main__':
    # Generate return constraints
    for env in envs:
        num_generation = 20
        if env not in constraints:
            constraints[env] = []
        for _ in range(num_generation):
            constraint_number = np.random.randint(1, 3)
            constraint_weights = np.random.uniform(0, 1, size=(constraint_number))
            constraint_weights = np.stack([constraint_weights, 1 - constraint_weights], axis=1)
            constraint_threshold = []
            for w in constraint_weights:
                minval = np.dot(w, env_stats[env]['minimum'])
                maxval = np.dot(w, env_stats[env]['maximum'])
                constraint_threshold.append(np.random.uniform(minval, maxval))
            constraint_threshold = np.array(constraint_threshold)
            constraints[env].append(ReturnConstraint(constraint_weights, constraint_threshold))
            
        if env not in variance_constraint:
            variance_constraint[env] = []
        for _ in range(num_generation):
            min_std = env_stats[env]['min_std']
            max_std = env_stats[env]['max_std']
            weight = np.random.uniform(0, 1)
            weight = np.array([weight, 1 - weight])
            variance_constraint[env].append((np.random.uniform(min_std, max_std), weight))
    

def constraint_satisfaction(front, env_name, env_stats):
    constr = []
    if env_name in constraints:
        constr.extend(constraints[env_name])
    
    results = []
    
    for c in constr:
        probs = []
        for curr_front in front:
            prob = np.mean([c(x, y) for x, y in curr_front])
            probs.append(prob)
        results.append(np.max(probs))
    
    return np.mean(results)

def variance_objective(front, env_name, env_stats):
    all_results = []
    num_exp = 100
    weights = np.random.uniform(0, 1, size=(num_exp, 4))
    weights = weights / weights.sum(axis=1)[:, np.newaxis]
    
    for i in range(num_exp):
        results = []
        for curr_front in front:
            curr_var = curr_front.std(axis=0)
            curr_mean = curr_front.mean(axis=0)
            results.append((curr_mean * weights[i, :2]).sum() - (curr_var * weights[i, 2:]).sum())
        all_results.append(np.max(results))
    
    return np.mean(all_results)
        

metrics = [expected_utility, cvar, constraint_satisfaction, variance_objective]

df = pd.DataFrame(columns=pd.MultiIndex.from_product([envs, [f.__name__ for f in metrics]], names=['Environment', 'Metric']), index=algs)

     
for env in envs:
    for alg in algs:
        data = np.array(all_data[env][alg])
        if len(data) == 0:
            continue
        print(f'Env: {env}, Alg: {alg}, Shape: {data.shape}')
        for metric in metrics:
            result = metric(data, env_name=env, env_stats=env_stats[env]) 
            df.loc[alg, (env, metric.__name__)] = result

num_results_per_env = len(metrics)
column_format = "c|" + "|".join(["c" * num_results_per_env for _ in envs])


print(df.to_latex(float_format=lambda x: "%.2f" % x, column_format=column_format))
