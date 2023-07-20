import gym
import torch
import os
import yaml
import json
import tqdm
import random
import pandas as pd
import numpy as np
import torch.multiprocessing as mp

import maml.envs
from maml.utils.helpers import get_policy_for_env, get_input_size
from maml.samplers import MultiTaskSampler
from maml.metalearners import MAMLTRPO
from maml.baseline import LinearFeatureBaseline
from maml.utils.reinforcement_learning import get_returns


def main(args):
    mp.set_start_method('spawn')    #https://blog.csdn.net/woai8339/article/details/105789683
    # 设置多进程的启动方式为spawn，不然会出现cuda错误
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    policy_filename = os.path.join(args.output_folder, 'policy')
    config_filename = os.path.join(args.output_folder, 'config.json')

    with open(config_filename, 'w') as f:
        config.update(vars(args))
        json.dump(config, f, indent=2)

    if args.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
    env.close() # 他的意思是，先创建一个环境，然后关闭它，这样就可以得到环境的一些信息，比如observation_space, action_space等
    # Policy
    policy = get_policy_for_env(env, hidden_sizes=config['hidden-sizes'], nonlinearity=config['nonlinearity'])
    policy.share_memory()
    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))
    # Sampler
    sampler = MultiTaskSampler(config['env-name'], env_kwargs=config.get('env-kwargs', {}), batch_size=config['fast-batch-size'], policy=policy, baseline=baseline, env=env, seed=args.seed, num_workers=args.num_workers)
    metalearner = MAMLTRPO(policy, fast_lr=config['fast-lr'], first_order=config['first-order'], device=args.device)
    num_iterations = 0
    TRAIN = []
    VALID = []
    Trange = tqdm.trange(config['num-batches'])
    for batch in Trange:
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        futures = sampler.sample_async(tasks, num_steps=config['num-steps'], fast_lr=config['fast-lr'], gamma=config['gamma'], gae_lambda=config['gae-lambda'], device=args.device) # ((train_episodes), (valid_episodes))
        # ([[train_episodes1], [train_episodes2]], [valid_episodes]) 这个train的主要取决于num_steps, 有几个num_steps就有几个train_episodes
        logs = metalearner.step(*futures, max_kl=config['max-kl'], cg_iters=config['cg-iters'], cg_damping=config['cg-damping'], ls_max_steps=config['ls-max-steps'], ls_backtrack_ratio=config['ls-backtrack-ratio'])
        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        logs.update(tasks=tasks, num_iterations=num_iterations, train_returns=get_returns(train_episodes[0]), valid_returns=get_returns(valid_episodes))
        TRAIN.append(np.mean(get_returns(train_episodes[0])))
        VALID.append(np.mean(get_returns(valid_episodes)))
        Trange.set_description(f'train: {TRAIN[-1]:.4f}, valid: {VALID[-1]:.4f}')
        # Save data
        if batch % 10 != 0 and batch != config['num-batches'] - 1:
            continue
        a = pd.DataFrame(TRAIN)
        b = pd.DataFrame(VALID)
        a.to_excel(os.path.join(args.output_folder, 'train.xlsx'), index=False)
        b.to_excel(os.path.join(args.output_folder, 'valid.xlsx'), index=False)
        with open(policy_filename + f'{batch}.th', 'wb') as f:
            torch.save(policy.state_dict(), f)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reinforcement learning with Model-Agnostic Meta-Learning (MAML) - Train')
    parser.add_argument('--config', type=str, default="configs/halfcheetah-vel.yaml",help='path to the configuration file.')
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str, default='output')
    misc.add_argument('--seed', type=int, default=None,help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1)
    misc.add_argument('--use-cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu')
    main(args)