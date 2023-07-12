import torch
import torch.multiprocessing as mp
import threading
import asyncio
import time

from datetime import datetime, timezone
from copy import deepcopy

from maml.samplers.sampler import Sampler,make_env
from maml.envs.utils.sync_vector_env import SyncVectorEnv
from maml.episode import BatchEpisodes
from maml.utils.reinforcement_learning import reinforce_loss

class SamplerWorker(mp.Process):
    """
    """
    def __init__(self, index, env_name, env_kwargs, batch_size, observation_space, action_space, policy, baseline, task_queue, train_queue, valid_queue, policy_lock):
        super(SamplerWorker, self).__init__()
        env_functions = [make_env(env_name, env_kwargs) for _ in range(batch_size)]
        self.envs = SyncVectorEnv(env_functions, observation_space=observation_space, action_space=action_space)
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.task_queue = task_queue
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.policy_lock = policy_lock

    def sample_trajectories(self, params=None):
        # 一个yield的生成器，每次返回一个轨迹, 也就是
        observations, info = self.envs.reset()
        self.envs.dones[:] = False
        with torch.no_grad():
            while not self.envs.dones.all():
                observations_tensor = torch.from_numpy(observations.astype('float32'))
                pi = self.policy(observations_tensor, params=params)
                actions_tensor = pi.sample()
                actions = actions_tensor.cpu().numpy()
                new_observations, rewards, dones, _ = self.envs.step(actions)
                batch_ids = info['batch_ids']
                yield (observations, actions, rewards, dones, batch_ids)
                observations = new_observations
    
    def create_episode(self, params=None, gamma=0.95, gae_lambda=1.0, device='cpu'):
        episodes = BatchEpisodes(self.batch_size, gamma, device)
        episodes.log('_create_episode_at', datetime.now(timezone.utc))
        episodes.log('process_name', self.name)
        t0 = time.time() # 记录开始时间
        for item in self.sample_trajectories(params):
            episodes.append(*item)
        episodes.log('duration', time.time() - t0)
        episodes.compute_advantages(self.baseline, gae_lambda, True)
        return episodes

    def sample(self, index, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        params = None
        for step in range(num_steps): # 采样num_steps个轨迹
            train_episode = self.create_episode(params, gamma, gae_lambda, device) # 创建一个episode
            train_episode.log('_put_train_episode_at', datetime.now(timezone.utc))
            self.train_queue.put((index, step, deepcopy(train_episode))) # deepcopy是为了防止多个进程之间的episode共享内存
            with self.policy_lock:
                loss = reinforce_loss(self.policy, train_episode, params)
                params = self.policy.update_params(loss, params, fast_lr, fast_lr, True)
        valid_episode = self.create_episode(params, gamma, gae_lambda, device)
        valid_episode.log('_put_valid_episode_at', datetime.now(timezone.utc))
        self.valid_queue.put((index, None, deepcopy(valid_episode)))







class MultiTaskSampler(Sampler):
    """

    """
    def __init__(self, env_name, env_kwargs, batch_size, policy, baseline, env=None, num_workers=1):
        super(MultiTaskSampler, self).__init__(env_name, env_kwargs, batch_size, policy, env)
        self.num_workers = num_workers

        self.task_queue = mp.JoinableQueue()
        self.train_episodes_queue = mp.Queue()
        self.valid_episodes_queue = mp.Queue()
        policy_lock = mp.Lock()