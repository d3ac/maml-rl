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
from maml.render import RenderSamplerWorker


def main(args):
    mp.set_start_method('spawn')    #https://blog.csdn.net/woai8339/article/details/105789683
    # 设置多进程的启动方式为spawn，不然会出现cuda错误
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.seed is not None:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    config['fast-batch-size'] = 1
    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
    env.close() # 他的意思是，先创建一个环境，然后关闭它，这样就可以得到环境的一些信息，比如observation_space, action_space等
    # Policy
    policy = get_policy_for_env(env, hidden_sizes=config['hidden-sizes'], nonlinearity=config['nonlinearity'])
    with open('policy.th', 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))
    woker = RenderSamplerWorker(config['env-name'], env_kwargs=config.get('env-kwargs', {}), batch_size=config['fast-batch-size'], policy=policy, baseline=baseline, env=env, seed=args.seed, num_workers=args.num_workers)
    params = woker.update_once(config['num-steps'], config['fast-lr'], config['gamma'], config['gae-lambda'], args.device)
    woker.Render(params, config['gamma'], config['gae-lambda'], args.device)

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