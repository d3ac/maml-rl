import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.distributions import Normal,Independent
from maml.policy.policy import Policy, weight_init

class NormalMLPPolicy(Policy):
    def __init__(self, input_size, output_size, hidden_sizes=(), nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        # hidden_sizes 是一个元组，表示隐藏层的神经元个数 比如(8, 16, 32, 16)
        super(NormalMLPPolicy, self).__init__(input_size, output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = np.log(min_std) # 也就是std的最小值不能小于 -6
        self.num_layers = len(hidden_sizes) + 1 # 还有一个输入层

        layer_sizes = (input_size, ) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)    # 自己学习到均值的合适取值 （比较奇怪） 
        self.sigma = nn.Parameter(torch.Tensor(output_size)) # 标准差
        self.sigma.data.fill_(np.log(init_std)) # 用对数的形式初始化标准差
        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None: # 和Policy的一样，如果没有传入参数，就用自己的参数
            params = OrderedDict(self.named_parameters())
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float32)
        output = input.to(torch.float32) # 初始化输入
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)], bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std)) # 限制标准差的最小值，避免不稳定
        """
        Normal ：是正态分布, 输入是均值和标准差, 输出是一个正态分布
        Independent ：是把一个分布变成独立的分布, 这里的mu和scale都是一个(n,)的向量, 但是Normal输出的东西不是独立的
                    他们之间还有一个相关系数, 所以用Independent把他们变成独立的分布
        """
        return Independent(Normal(loc=mu, scale=scale), 1)