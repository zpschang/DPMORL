import os
import json

import IPython
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import matplotlib.colors as mcolors
import glob
import sys

assert len(sys.argv) >= 2, 'Please run "python plot_utility_returns.py [exp_name]". '
global_save_dir = sys.argv[1]
global_save_dir = 'experiments/' + global_save_dir
plot_history_data = False
plot_learning_curve = True
plot_final = True
batch = [1, 10]

def read_returns(save_dir, env_name, algo_name, seeds):
    all_returns = []
    for seed in seeds:
        file_name = f"{env_name}_{algo_name}_seed{seed}.npz"
        file_path = os.path.join(save_dir, file_name)
        returns = np.load(file_path)['returns']
        all_returns.append(returns)
    return all_returns

def resample_returns(returns_list, target_length):
    resampled_returns = []
    for returns in returns_list:
        indices = np.linspace(0, len(returns)-1, target_length, dtype=int)
        resampled_returns.append(np.array(returns)[indices])
    return resampled_returns

def plot_returns(env_name, algo_names, seeds, save_dir='Returns'):
    plt.figure(figsize=(10, 6))
    print(f"plotting env {env_name}")
    for algo_name in algo_names:
        all_returns = read_returns(save_dir, env_name, algo_name, seeds)

        min_length = min([len(returns) for returns in all_returns])

        resampled_returns = resample_returns(all_returns, min_length)

        returns_mean = np.mean(resampled_returns, axis=0)
        returns_std = np.std(resampled_returns, axis=0)

        x = np.arange(len(returns_mean))
        plt.plot(x, returns_mean, label=algo_name)
        plt.fill_between(x, returns_mean - returns_std, returns_mean + returns_std, alpha=0.2)

    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'{env_name} - Mean Return and Standard Deviation')
    plt.legend()
    plt.savefig(os.path.join(save_dir, env_name + '.png'), bbox_inches='tight')

def plot_scatter(env_name, prefix, save_dir, 
                 batch_size=1, plot_frequency=1):
    save_sub_dir = os.path.join(save_dir, prefix + f".{env_name}.LossNormLamda_0.1")
    print(save_sub_dir)
    min_vals, max_vals = [], []
    for file_path in glob.glob(f'{save_sub_dir}/MORL*.npz'):
        episode_vec_returns = np.load(file_path)['episode_vec_returns']
        min_vals.append(episode_vec_returns.min(0))
        max_vals.append(episode_vec_returns.max(0))
    min_vals = np.min(min_vals, 0)[:2]
    max_vals = np.max(max_vals, 0)[:2]
    import pickle
    with open('normalization_data/data.pickle', 'rb') as file:
        normalization_data = pickle.load(file)
    intervals = max_vals - min_vals
    if plot_learning_curve:
        if plot_history_data:
            for file_path in glob.glob(f'{save_sub_dir}/MORL*.npz'):
                name = '/'.join(file_path.split('/')[-1].split('.')[:-1])
                _, _, algo_name, utility_name, *_ = name.split('_')
                utility_name = utility_name[6:]
                
                if os.path.exists(file_path):
                    print(file_path)
                    episode_vec_returns = np.load(file_path)['episode_vec_returns'] # 100w * 2
                    print('shape:', episode_vec_returns.shape)
                    
                    episode_vec_returns = np.asarray([
                        episode_vec_returns[i] for i in range(0, len(episode_vec_returns), plot_frequency)
                    ])

                    episode_batches = [
                        np.mean(episode_vec_returns[i:i + batch_size], 0) for i in range(0, len(episode_vec_returns), batch_size)
                    ] # [1w, 100] * 2
                    episode_batches = np.asarray(episode_batches)
                    fig, ax = plt.subplots(figsize=(10, 6))

                    colors = cm.rainbow(np.linspace(0, 1, len(episode_batches)))

                    ax.scatter(episode_batches[:, 0], episode_batches[:, 1], c=colors, alpha=0.6)

                    ax.set_xlabel('Return 1')
                    ax.set_ylabel('Return 2')
                    if env_name in normalization_data:
                        x_min, y_min = normalization_data[env_name]['min'][0, :2]
                        x_max, y_max = normalization_data[env_name]['max'][0, :2]
                        if y_max - y_min > x_max - x_min:
                            interval = y_max - y_min
                            mid = (x_max + x_min) / 2
                            x_max = mid + interval / 2
                            x_min = mid - interval / 2
                        else:
                            interval = x_max - x_min
                            mid = (y_max + y_min) / 2
                            y_max = mid + interval / 2
                            y_min = mid - interval / 2
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                    ax.set_title(f'{env_name} - {algo_name} - {utility_name} - 2D Return Scatter')
                    sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=mcolors.Normalize(vmin=0, vmax=len(episode_batches) - 1))
                    cbar = plt.colorbar(sm, ax=ax)
                    cbar.ax.set_ylabel('Episode', rotation=270, labelpad=15)
                    plt.savefig(os.path.join(save_sub_dir, f'{name}_b{batch_size}.png'), bbox_inches='tight')
                    plt.close()
        
        for file_path in glob.glob(f'{save_sub_dir}/test_returns*.npz'):
            plt.clf()
            name = file_path.split('/')[-1].split('.')[0].split('_')[-1]
            number = name.split('-')[-1]
            
            pretrain_type = name.split('-')[0]
            if pretrain_type == 'pretrain':
                number = int(number) + 3
            else:
                number = int(number)
            name = f'pretrain-{number}'
            episode_vec_returns = np.load(file_path)['test_returns']
            
            plt.scatter(episode_vec_returns[:, 0], episode_vec_returns[:, 1], alpha=0.6, label=name, s=80)
            plt.xlabel('Return 1')
            plt.ylabel('Return 2')
            plt.title(rf'Return Distribution of $\pi_{number}$')
            if env_name == 'DiverseGoal':
                plt.xlim(-10, 20)
                plt.ylim(-10, 20)
            plt.savefig(f'{save_sub_dir}/test_returns_{name}.png', dpi=160, bbox_inches='tight')
                
    if plot_final:
        plt.clf()
        paths = sorted(glob.glob(f'{save_sub_dir}/test_returns*.npz'))
        marker = ['o', 'v', '^', 's', 'p']
        for i, file_path in enumerate(paths):
            name = file_path.split('/')[-1].split('.')[0].split('_')[-1]
            number = name.split('-')[-1]
            episode_vec_returns = np.load(file_path)['test_returns']
            
            episode_batches = [
                np.mean(episode_vec_returns[idx:idx + batch_size], 0) for idx in range(0, len(episode_vec_returns), batch_size)
            ]
            episode_batches = np.asarray(episode_batches)
            
            plt.scatter(episode_batches[:, 0], episode_batches[:, 1], alpha=0.6, label=f'Policy #{i}', marker=marker[i // 10])
        plt.title('Return Distribution of Final Policies')
        plt.legend(prop={'size': 12})
        plt.xlabel('Return 1')
        plt.ylabel('Return 2')
        plt.savefig(f'{save_sub_dir}/test_final_batch_{batch_size}.png', dpi=160, bbox_inches='tight')
        
        plt.clf()
        file_paths = sorted(glob.glob(f'{save_sub_dir}/MORL*.npz'))

        sample_returns = []
        max_returns = []
        min_returns = []
        
        for file_path in file_paths:
            episode_vec_returns = np.load(file_path)['episode_vec_returns'] # 100w * 2
            max_returns.append(episode_vec_returns[-500:, :2].max(0))
            min_returns.append(episode_vec_returns[-500:, :2].min(0))
            sample_returns.append(episode_vec_returns[-500:, :2])
        
        selected_policies_first_iter = []
        selected_policies = []

        for i in range(len(max_returns)):
            def not_dominate_sample(current, other):
                return (current - other).max() >= 0
            not_dominated = True
            for j in range(len(min_returns)):
                if j == i: continue
                if not not_dominate_sample(max_returns[i], min_returns[j]):
                    not_dominated = False
                    break
            if not_dominated or True:
                selected_policies_first_iter.append(i)
        
        selected_policies = selected_policies_first_iter
        
        num_plot_policies = len(selected_policies)

        colors = matplotlib.colormaps['gist_rainbow'](np.linspace(0, 1, num_plot_policies))
        
        count = 0
        marker = ['o', 'v', '^', 's', 'p']
        for i, policy_index in enumerate(selected_policies):
            color = colors[i]
            file_path = file_paths[policy_index]
            name = '/'.join(file_path.split('/')[-1].split('.')[:-1])
            _, _, algo_name, utility_name, *_ = name.split('_')
            utility_name = utility_name[6:]
            
            episode_vec_returns = np.load(file_path)['episode_vec_returns'] # 100w * 2
            final_episode_vec_returns = episode_vec_returns[-100 * batch_size:]
            episode_batches = [
                np.mean(final_episode_vec_returns[idx:idx + batch_size], 0) for idx in range(0, len(final_episode_vec_returns), batch_size)
            ]
            episode_batches = np.asarray(episode_batches)
            plt.scatter(episode_batches[:, 0], episode_batches[:, 1], alpha=0.6, marker=marker[count // 10], label=f'policy-{i}')
            count += 1
            
        plt.title(f'{env_name}, batch={batch_size}')
        plt.savefig(os.path.join(save_sub_dir, f'{env_name}_final_batch_{batch_size}.png'), dpi=160, bbox_inches='tight')
        plt.clf()            
        
if __name__ == '__main__':
    prefix = "DPMORL"
    all_subdir = glob.glob(f'{global_save_dir}/*')
    env_names = [s.split('/')[2].split('.')[1] for s in all_subdir]
    print(env_names)

    for env in env_names:
        for batch_size in batch:
            if env == 'DeepSeaTreasure' or env == 'FruitTree':
                plot_frequency = 20
            elif env == 'ResourceGathering':
                plot_frequency = 10
            else:
                plot_frequency = 1
            print(f"Plotting {env}, batch_size={batch_size}")
            plot_scatter(env, prefix, global_save_dir, batch_size=batch_size, plot_frequency=plot_frequency)
