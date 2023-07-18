import gymnasium as gym
import torch
import os
import yaml
import json
import tqdm
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
    mp.set_start_method('spawn') # 设置多进程的启动方式为spawn，不然会出现cuda错误
    with open(args.config, 'r') as f:
        config = json.load(f) # 可以通过config['xxx']来获取配置信息
    env = gym.make(config['env-name'], **config.get('env-kwargs',{})) # **表示将字典解包，获取config字典中'env-kwargs'键的值，如果该键不存在，则返回一个空字典{}
    env.close()
    # policy
    policy = get_policy_for_env(env, hidden_sizes=config['hidden-sizes'], nonlinearity=config['nonlinearity'])
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory() # 将policy放到共享内存中，这样多个进程就可以共享这个policy
    # sampler
    baseline = LinearFeatureBaseline(get_input_size(env))
    sampler = MultiTaskSampler(
        config['env-name'], env_kwargs=config.get('env-kwargs',{}), batch_size=config['fast-batch-size'], # 这里的fast-batch-size是指每个任务采样的轨迹数
        num_workers=args.num_workers, policy=policy, env=env, baseline=baseline
    )
    # train
    logs = {'tasks':[]}
    train_returns, valid_returns = [], []
    for batch in tqdm.trange(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size) # 调用env.unwrapped.sample_tasks(num_tasks)生成任务字典
        train_episodes, valid_episodes = sampler.sample(tasks, num_steps=config['num-steps'], device=args.device, fast_lr=config['fast-lr'], gamma=config['gamma'], gae_lambda=config['gae-lambda'])
        logs['tasks'].extend(tasks)
        train_returns.append(np.mean(get_returns(train_episodes[0])))
        valid_returns.append(np.mean(get_returns(valid_episodes)))
    logs['train_returns'] = train_returns
    logs['valid_returns'] = valid_returns
    with open(args.output, 'wb') as f:
        np.savez(f, **logs)





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train MAML-RL')
    parser.add_argument('--config', type=str, default='output/config.json')
    parser.add_argument('--policy', type=str, default='output/policy.th')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-batches', type=int, default=10)
    parser.add_argument('--meta-batch-size', type=int, default=40)
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count()-1)
    parser.add_argument('--output', type=str, default='output/results.npz')
    parser.add_argument('--use-cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu')
    main(args)  