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
    # 采样器进程
    def __init__(self, index, env_name, env_kwargs, batch_size, observation_space, action_space, policy, baseline, task_queue, train_queue, valid_queue, policy_lock, seed=None):
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
        self.envs.seed(None if (seed is None) else seed + index * batch_size)

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
                new_observations, rewards, _, infos = self.envs.step(actions)
                batch_ids = infos['batch_ids']
                yield (observations, actions, rewards, batch_ids)
                observations = new_observations
    
    def create_episode(self, params=None, gamma=0.95, gae_lambda=1.0, device='cpu'):
        episodes = BatchEpisodes(self.batch_size, gamma, device)
        for item in self.sample_trajectories(params):
            episodes.append(*item)
        self.baseline.fit(episodes)
        episodes.compute_advantages(self.baseline, gae_lambda, True)
        return episodes

    def sample(self, index, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        params = None # 新开一个param, 在采样的时候单独使用
        for step in range(num_steps): # 采样num_steps个轨迹
            train_episode = self.create_episode(params, gamma, gae_lambda, device) # 创建一个episode
            self.train_queue.put((index, step, deepcopy(train_episode))) # deepcopy是为了防止多个进程之间的episode共享内存
            with self.policy_lock:
                loss = reinforce_loss(self.policy, train_episode, params)
                params = self.policy.update_params(loss, params, fast_lr, True) # 在这个地方更新param并不会影响到policy的param
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
            self.sample(index, **kwargs)
            self.task_queue.task_done() # 通知主进程任务完成



def _create_consumer(queue, futures, loop):
    while True:
        data = queue.get() # 等待采样器进程的数据, 如果没有数据, 就会阻塞在这里
        if data is None: # 传入None就代表结束
            break
        index, step, episode = data
        future = futures if (step is None) else futures[step]
        if not future[index].cancelled():
            loop.call_soon_threadsafe(future[index].set_result, episode)


class MultiTaskSampler(Sampler):
    def __init__(self, env_name, env_kwargs, batch_size, policy, baseline, env=None, num_workers=1, seed=None):
        super(MultiTaskSampler, self).__init__(env_name, env_kwargs, batch_size, policy, env, seed=seed)
        self.num_workers = num_workers

        self.task_queue = mp.JoinableQueue()
        self.train_episodes_queue = mp.Queue()
        self.valid_episodes_queue = mp.Queue()
        policy_lock = mp.Lock()
        self.workers = [
            SamplerWorker(
                index, env_name, env_kwargs, batch_size, self.env.observation_space, self.env.action_space, policy,
                deepcopy(baseline), self.task_queue, self.train_episodes_queue, self.valid_episodes_queue, policy_lock, seed
            ) for index in range(num_workers)
        ]
        for worker in self.workers:
            worker.daemon = True # 设置为守护进程
            worker.start()
        self._waiting_sample = False
        self._event_loop = asyncio.get_event_loop() # 创建一个事件循环
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def sample_tasks(self, num_tasks):
        return self.env.unwrapped.sample_tasks(num_tasks)

    def _start_consumer_threads(self, tasks, num_steps=1):
        # train
        train_episodes_futures = [[self._event_loop.create_future() for _ in tasks] for _ in range(num_steps)]
        self._train_consumer_thread = threading.Thread(target=_create_consumer, args=(self.train_episodes_queue, train_episodes_futures), kwargs={'loop':self._event_loop})
        self._train_consumer_thread.daemon = True # 设置为守护进程, 因为当主进程结束了, 就不需要继续了
        self._train_consumer_thread.start()
        # valid
        valid_episodes_futures = [self._event_loop.create_future() for _ in tasks]
        self._valid_consumer_thread = threading.Thread(target=_create_consumer, args=(self.valid_episodes_queue, valid_episodes_futures), kwargs={'loop':self._event_loop})
        self._valid_consumer_thread.daemon = True
        self._valid_consumer_thread.start()
        return (train_episodes_futures, valid_episodes_futures)

    def sample_async(self, tasks, **kwargs): # 传入task, 然后传入task_queue进行sample, 最后调用consumer进行set future
        if self._waiting_sample:
            raise RuntimeError('Already sampling!')
        for index, task in enumerate(tasks):
            self.task_queue.put((index, task, kwargs)) # SamplerWorker 已经开始采样了
        num_steps = kwargs.get('num_steps', 1)
        futures = self._start_consumer_threads(tasks, num_steps)
        self._waiting_sample = True
        return futures
    
    @property
    def valid_consumer_thread(self):
        return self._valid_consumer_thread
    
    @property
    def train_consumer_thread(self):
        return self._train_consumer_thread
    
    def sample_wait(self, episodes_futures): # 用来等待所有的异步采样操作完成
        if not self._waiting_sample:
            raise RuntimeError('Not sampling!')
        
        async def _wait(train_futures, valid_futures):
            train_episodes = await asyncio.gather(*[asyncio.gather(*futures) for futures in train_futures])
            valid_episodes = await asyncio.gather(*valid_futures)
            return train_episodes, valid_episodes
        
        samples = self._event_loop.run_until_complete(_wait(*episodes_futures)) # Run the event loop until a Future is done.
        self._join_consumer_threads()
        self._waiting_sample = False
        return samples
    
    def sample(self, tasks, **kwargs):
        futures =  self.sample_async(tasks, **kwargs)
        return self.sample_wait(futures)
    
    def _join_consumer_threads(self): # 等待所有的消费者线程结束, 分别关闭train和valid的消费者线程
        if self._train_consumer_thread is not None:
            self.train_episodes_queue.put(None) # 通知采样器进程结束
            self.train_consumer_thread.join()
        if self._valid_consumer_thread is not None:
            self.valid_episodes_queue.put(None)
            self.valid_consumer_thread.join()
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def close(self):
        if self.closed:
            return
        for _ in range(self.num_workers):
            self.task_queue.put(None) # Put None之后join就不会阻塞了
        self.task_queue.join() # 等待所有的任务完成
        self._join_consumer_threads() # 等待所有的消费者线程结束
        self.closed = True