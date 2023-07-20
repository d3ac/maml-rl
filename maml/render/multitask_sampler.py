import torch
import torch.multiprocessing as mp
import threading
import asyncio
import time
import numpy as np
import time

from datetime import datetime, timezone
from copy import deepcopy

from maml.samplers.sampler import Sampler,make_env
from maml.envs.utils.sync_vector_env import SyncVectorEnv
from maml.episode import BatchEpisodes
from maml.utils.reinforcement_learning import reinforce_loss

class RenderSamplerWorker(mp.Process):
    # 采样器进程
    # def __init__(self, index, env_name, env_kwargs, batch_size, observation_space, action_space, policy, baseline, task_queue, train_queue, valid_queue, policy_lock, seed=None):
    def __init__(self, env_name, env_kwargs, batch_size, policy, baseline, env=None, num_workers=1, seed=None):
        super(RenderSamplerWorker, self).__init__()
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.env = env

    def sample_trajectories(self, params=None):
        # 一个yield的生成器，每次返回一个轨迹
        observation, info = self.env.reset()
        terminated, truncated = False, False
        with torch.no_grad():
            while not (terminated or truncated): # 不是while True
                observation_tensor = torch.from_numpy(observation.astype('float32')).reshape(1, -1)
                pi = self.policy(observation_tensor, params=params)
                action_tensor = pi.sample()
                action = action_tensor.cpu().numpy()
                action = action.reshape(-1)
                new_observation, reward, terminated, truncated, info = self.env.step(action)
                observation = observation.reshape(1, -1)
                action = action.reshape(1, -1)
                yield (observation, action, (reward,), (0,))
                observation = new_observation
    
    def create_episode(self, params=None, gamma=0.95, gae_lambda=1.0, device='cpu'):
        episodes = BatchEpisodes(self.batch_size, gamma, device)
        for item in self.sample_trajectories(params):
            episodes.append(*item)
        self.baseline.fit(episodes)
        episodes.compute_advantages(self.baseline, gae_lambda, True)
        return episodes

    def update_once(self, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        params = None # 新开一个param, 在采样的时候单独使用
        for step in range(num_steps): # 采样num_steps个轨迹
            train_episode = self.create_episode(params, gamma, gae_lambda, device) # 创建一个episode
            loss = reinforce_loss(self.policy, train_episode, params)
            params = self.policy.update_params(loss, params, fast_lr, True) # 在这个地方更新param并不会影响到policy的param
        return params
    
    def Render(self, params, gamma, gae_lambda, device):
        observation, info = self.env.reset()
        terminated, truncated = False, False
        with torch.no_grad():
            while not (terminated or truncated):
                observation_tensor = torch.from_numpy(observation.astype('float32')).reshape(1, -1)
                pi = self.policy(observation_tensor, params=params)
                action_tensor = pi.sample()
                action = action_tensor.cpu().numpy()
                action = action.reshape(-1)
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.env.render()
                time.sleep(0.03)