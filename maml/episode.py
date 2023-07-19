import numpy as np
import torch
import torch.nn.functional as F
from maml.utils.torch_utils import weighted_normalize

class BatchEpisodes(object):
    def __init__(self, batch_size, gamma=0.95, device='cpu'):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        # 保存数据的列表
        self._observations_list = [[] for _ in range(batch_size)] # 创建一个batch_size大小的列表，每个元素是一个空列表 (batch, max_length, obs_dim)
        self._actions_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)] # (batch, max_length)
        # 定义私有变量 (一般以_开头)
        self._observation_shape = None
        self._action_shape = None
        # maxlength 是每个episode的最大长度
        self._observations = None # (max_length, batch_size, obs_dim)
        self._actions = None      # (max_length, batch_size, action_dim)
        self._rewards = None      # (max_length, batch_size)
        self._returns = None      # (max_length, batch_size)
        self._advantages = None
        self._mask = None         # (max_length, batch_size)
        self._lengths = None
        self._logs = {}
    
    @property # 用于将方法转换为属性调用
    def observation_shape(self):
        if self._observation_shape == None:
            self._observation_shape = self.observations.shape[2:] # (batch_size, max_length, obs_dim), max_length是每个episode的最大长度
        return self._observation_shape
    
    @property
    def action_shape(self):
        if self._action_shape == None:
            self._action_shape = self.actions.shape[2:]
        return self._action_shape
    
    @property
    def observations(self):
        if self._observations == None:
            observation_shape = self._observations_list[0][0].shape # (max_length, obs_dim)
            observations = np.zeros((len(self), self.batch_size) + observation_shape, dtype=np.float32)
            # len(self)是episode的数量，在有面有定义, observations的shape是(max_length, batch_size, obs_dim)
            # 注意下observations的第一个维度不是batch_size，而是max_length
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._observations_list[i], axis=0, out=observations[:length, i])
                # stack只stack一个元素，axis=0，就相当于把前面这个变量复制到后面这个observations[:length, i]里面
            self._observations = torch.as_tensor(observations, device=self.device) # as_tensor是把numpy转换成tensor, 共享内存
            del self._observations_list # 删除这个列表，节省内存
        return self._observations
    
    @property
    def actions(self):
        if self._actions == None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size) + action_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._actions_list[i], axis=0, out=actions[:length, i])
            self._actions = torch.as_tensor(actions, device=self.device)
            del self._actions_list
        return self._actions
    
    @property
    def rewards(self):
        if self._rewards == None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._rewards_list[i], axis=0, out=rewards[:length, i])
            self._rewards = torch.as_tensor(rewards, device=self.device)
            del self._rewards_list
        return self._rewards
    
    @property
    def returns(self):
        if self._returns == None:
            self._returns = torch.zeros_like(self.rewards) # zeros_like是创建一个和self.rewards一样shape的tensor，但是值都是0
            return_ = torch.zeros((self.batch_size,), dtype=torch.float32) # (batch_size,)
            for i in range(len(self)-1, -1, -1): # 从len(self)-1到0，步长为1 (倒着来)
                return_ = self.gamma * return_ + self.rewards[i] * self.mask[i] # mask是一个0-1的矩阵
                self._returns[i] = return_
        return self._returns
    
    @property
    def mask(self):
        #  有reward的地方就是1,没有的地方就是0,因为rewards的shape是(max_length, batch_size)，但是不是所有的episode都是max_length
        if self._mask == None:
            self._mask = torch.zeros((len(self), self.batch_size), dtype=torch.float32, device=self.device)
            for i in range(self.batch_size):
                self._mask[:self.lengths[i], i].fill_(1.0)
        return self._mask

    @property
    def advantages(self):
        if self._advantages == None:
            raise ValueError('advantages is not computed yet')
        return self._advantages
    
    @property
    def logs(self):
        return self._logs
    
    def log(self, key, value):
        self.logs_[key] = value

    def append(self, observations, actions, rewards, batch_ids):
        for observation, action, reward, batch_ids in zip(observations, actions, rewards, batch_ids):
            if batch_ids == None:
                continue
            self._observations_list[batch_ids].append(observation.astype(np.float32))
            self._actions_list[batch_ids].append(action.astype(np.float32))
            self._rewards_list[batch_ids].append(reward.astype(np.float32))

    def compute_advantages(self, baseline, gae_lambda=1.0, normalize=True):
        """
        advantage是一个用于评估某个动作相对于平均水平的指标
        """
        values = baseline(self).detach()  # values : (max_length, batch_size)
        values = F.pad(values * self.mask, (0, 0, 0, 1)) # 相当于在最后一行后面加了一排0
        
        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        # deltas 每个元素都是当前时间步的advantage, 计算方式是将当前时间步的reward加上下一个时间步的value，再减去当前时间步的value
        # 这个算法就是GAE里面的算法
        self._advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros((self.batch_size,), dtype=torch.float32)
        for i in range(len(self)-1, -1, -1):
            gae = gae * self.gamma * gae_lambda + deltas[i] 
            self._advantages[i] = gae
        
        if normalize:
            self._advantages =  weighted_normalize(self._advantages, lengths=self.lengths)
        del self._returns
        del self._mask
        return self._advantages

    @property
    def lengths(self): # lengths 是一个列表，每个元素是一个episode的长度
        if self._lengths == None:
            self._lengths = [len(rewards) for rewards in self._rewards_list] #TODO 有个问题就是这个lengths怎么更新呢？
        return self._lengths

    def __len__(self):
        return max(self.lengths)
    
    def __iter__(self):
        return iter(self)