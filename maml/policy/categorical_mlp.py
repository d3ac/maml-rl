import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.distributions import Categorical
from maml.policy.policy import Policy, weight_init

class CategoricalMLPPolicy(Policy):
    def __init__(self, input_size, output_size, hidden_sizes=(), nonlinearity=F.relu):
        super(CategoricalMLPPolicy, self).__init__(input_size, output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1
        
        layer_sizes = (input_size, ) + hidden_sizes + (output_size, )
        for i in range(1, self.num_layers+1):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        self.apply(weight_init)
    
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)], bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        logits = F.linear(output, weight=params['layer{0}.weight'.format(self.num_layers)], bias=params['layer{0}.bias'.format(self.num_layers)])
        # logits 就是最后一层的输出，即将丢到 softmax 中的值
        # Catelogical 就是 Samples are integers from {0,…,K−1} based on the logits provided. K是输出的维度
        return Categorical(logits=logits)