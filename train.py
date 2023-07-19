import gym
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
        config = yaml.load(f, Loader=yaml.FullLoader) # 可以通过config['xxx']来获取配置信息
    # 保存文件和配置信息
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    policy_filename = os.path.join(args.output_folder, 'policy.th')
    config_filename = os.path.join(args.output_folder, 'config.json')
    with open(config_filename, 'w') as f:
        config.update(vars(args)) # vars返回对象args里面的对象和值，update是更新值，如果有了就更新，没有就添加
        json.dump(config, f, indent=2) # indent=2表示缩进2个空格，dump表示将config写入f中
    # environment
    env = gym.make(config['env-name'], **config.get('env-kwargs',{})) # **表示将字典解包，获取config字典中'env-kwargs'键的值，如果该键不存在，则返回一个空字典{}
    env.close()
    # policy
    policy = get_policy_for_env(env, hidden_sizes=config['hidden-sizes'], nonlinearity=config['nonlinearity'])
    policy.share_memory() # 将policy放到共享内存中，这样多个进程就可以共享这个policy
    # sampler
    baseline = LinearFeatureBaseline(get_input_size(env))
    sampler = MultiTaskSampler(
        config['env-name'], env_kwargs=config.get('env-kwargs',{}), batch_size=config['fast-batch-size'], # 这里的fast-batch-size是指每个任务采样的轨迹数
        num_workers=args.num_workers, policy=policy, env=env, baseline=baseline
    )
    # learner
    metalearner = MAMLTRPO(policy=policy, fast_lr=config['fast-lr'], first_order=config['first-order'], device=args.device)
    # train
    num_iterations = 0
    TRAIN = []
    VALID = []
    Trange = tqdm.trange(config['num-batches'])
    for batch in Trange:
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size']) # 调用env.unwrapped.sample_tasks(num_tasks)生成任务字典
        futures = sampler.sample_async(tasks, num_steps=config['num-steps'], device=args.device) 
        # ([[train_episodes1], [train_episodes2]], [valid_episodes]) 这个train的主要取决于num_steps, 有几个num_steps就有几个train_episodes
        metalearner.step(*futures, max_kl=config['max-kl'], cg_iters=config['cg-iters'], cg_damping=config['cg-damping'], ls_max_steps=config['ls-max-steps'], ls_backtrack_ratio=config['ls-backtrack-ratio']) # *futures是train_episodes_futures, valid_episodes_futures
        #!为什么这个地方就可以用step, 二阶梯度是怎么传递的
        train_episodes, valid_episodes = sampler.sample_wait(futures)
        TRAIN.append(np.mean(get_returns(train_episodes[0])))
        VALID.append(np.mean(get_returns(valid_episodes)))
        Trange.set_description(f'train: {TRAIN[-1]:.4f}, valid: {VALID[-1]:.4f}')
        if batch % 10 != 0 and batch != config['num-batches'] - 1:
            continue
        with open(policy_filename, 'wb') as f:
            torch.save(policy.state_dict(), f)
        a = pd.DataFrame(TRAIN)
        b = pd.DataFrame(VALID)
        a.to_excel('train.xlsx', index=False)
        b.to_excel('valid.xlsx', index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train MAML-RL')
    parser.add_argument('--config', type=str, default='configs/halfcheetah-vel.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count()-1)
    parser.add_argument('--output-folder', type=str, default='output')
    parser.add_argument('--use-cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu')
    main(args)  