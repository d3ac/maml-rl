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

def _create_consumer_thread(queue, futures, loop=None):
    if loop is None:
        loop = asyncio.get_event_loop()

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
        # 一个yield的生成器，每次返回一个轨迹
        observations, info = self.envs.reset()
        self.envs.dones[:] = False
        with torch.no_grad():
            while not self.envs.dones.all(): # 不是while True
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
        for item in self.sample_trajectories(params):
            episodes.append(*item)
        self.baseline.fit(episodes)
        episodes.compute_advantages(self.baseline, gae_lambda, True)
        return episodes

    def sample(self, index, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        params = None
        for step in range(num_steps): # 采样num_steps个轨迹
            train_episode = self.create_episode(params, gamma, gae_lambda, device) # 创建一个episode
            self.train_queue.put((index, step, deepcopy(train_episode))) # deepcopy是为了防止多个进程之间的episode共享内存
            with self.policy_lock:
                loss = reinforce_loss(self.policy, train_episode, params)
                params = self.policy.update_params(loss, params, fast_lr, fast_lr, True)
        valid_episode = self.create_episode(params, gamma, gae_lambda, device)
        self.valid_queue.put((index, None, deepcopy(valid_episode)))
    
    def run(self):
        while True:
            data = self.task_queue.get()
            if data is None:
                self.envs.close()
                self.task_queue.task_done()
                break
            index, task, kwargs = data
            self.envs.reset_task(task)
            self.sample(index, **kwargs) # 采样并且训练
            self.task_queue.task_done() # 通知主进程任务完成





class MultiTaskSampler(Sampler):
    def __init__(self, env_name, env_kwargs, batch_size, policy, baseline, env=None, num_workers=1):
        super(MultiTaskSampler, self).__init__(env_name, env_kwargs, batch_size, policy, env)
        self.num_workers = num_workers

        self.task_queue = mp.JoinableQueue()
        self.train_episodes_queue = mp.Queue()
        self.valid_episodes_queue = mp.Queue()
        policy_lock = mp.Lock()
        self.workers = [
            SamplerWorker(
                index, env_name, env_kwargs, batch_size, self.observation_space, self.action_space, policy,
                deepcopy(baseline), self.task_queue, self.train_episodes_queue, self.valid_episodes_queue, policy_lock
            ) for index in range(num_workers)
        ]
        for worker in self.workers:
            worker.daemon = True # 设置为守护进程
            worker.start()
        self._waiting_sample = False
        self._event_loop = asyncio.get_event_loop() # 创建一个事件循环， asyncio是用于异步编程的库
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def sample_tasks(self, num_tasks):
        return self.env.unwrapped.sample_tasks(num_tasks)
    
    def _start_consumer_threads(self, tasks, num_steps=1):

    
    def sample_asnc(self, tasks, **kwargs):
        if self._waiting_sample:
            raise RuntimeError('Already sampling!')
        for index, task in enumerate(tasks):
            self.task_queue.put((index, task, kwargs))
        num_steps = kwargs.get('num_steps', 1)
        futures = self._start_consumer_threads(tasks, num_steps)