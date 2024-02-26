# DPMORL

Implementations for our paper [*Distributional Pareto-Optimal Multi-Objective Reinforcement Learning*](https://papers.nips.cc/paper_files/paper/2023/hash/32285dd184dbfc33cb2d1f0db53c23c5-Abstract-Conference.html) at NeurIPS 2023. 

## Requirements

To install the environment (except for ReacherBullet), run: 

```
conda create -n dpmorl python=3.8
conda activate dpmorl
pip install -r requirements.txt
```

## Training Policies with DPMORL

### Generating Utility Functions

Before training policies, DPMORL requires first generate utility functions. To generate utility functions, run:

```
python main_generate_utility.py
```

The generated utility functions are saved in `utility-model/dim-2`, and the visualization are saved in `utility-plot/dim-2`. 

You can run 

```
python main_generate_utility.py --reward_shape 3 --num_utility_function 100
```

for configuring the reward dimensions and utility function number for generated utility functions.  

We have provided part of our generated utility functions in `utility-model-selected` and `utility-plot-selected`. 

### Training Policies

To policies by DPMORL in the paper, run this command:

```
python -u main_policy.py --lamda=0.1 --env [env] --reward_two_dim --exp_name [exp_name]
```

The environment name is in `env.txt`. 

Configuration `--reward_two_dim` makes DPMORL run on the first two dimensions of reward functions. To run DPMORL on other dimensions of reward (e.g. 0, 1, 2 dimension), you can change `--reward_two_dim` to `--reward_dim_indices=[0,1,2]`. 

You can also run `. run_policy_parallel.sh` to run DPMORL in all environments in parallel. 

### Evaluate the policies

After training finished, you should evaluate the policies learned by DPMORL by running `. run_test.sh`. 

## Visualize the return distributions of learned policies

To visualize the return distributions of policies learned by DPMORL, run `python plot_utility_returns.py [exp_name]`. The visualization results will be located in the `experiments/[exp_name]` directory. `test_final_*.png` will visualize the return distributions of all learned policies by DPMORL. 

## Compute the evaluation metric

Run `stats.py` to compute all the evaluation metrics for DPMORL and other baseline methods. The implemenetation includes constraint satisfaction and variance objective. 
