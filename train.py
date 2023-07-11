import gymnasium as gym
import torch
import os
import yaml
import json
import tqdm

import maml.envs
from maml.utils.helpers import get_policy_for_env
from maml.samplers import MultiTaskSampler

def main(args):
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
    sampler = MultiTaskSampler(
        config['env-name'], env_kwargs=config.get('env-kwargs',{}), batch_size=config['fast-batch-size'], # 这里的fast-batch-size是指每个任务采样的轨迹数
        num_workers=args.num_workers, policy=policy, env=env
    )
    # learner
    metalearner = MAMLTRPO(policy=policy, fast_lr=config['fast-lr'], first_order=config['first-order'])
    # train
    num_iterations = 0
    for batch in tqdm.trange(config['num-batches']):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size']) # 采样任务
    """
    #TODO 先暂停下，先去写其他的代码，等写完了再回来写这里
    """


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    parser = argparse.ArgumentParser(description='Train MAML-RL')
    parser.add_argument('--config', type=str, default='configs/halfcheetah-vel.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count()-1)
    parser.add_argument('--output-folder', type=str, default='output')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)  