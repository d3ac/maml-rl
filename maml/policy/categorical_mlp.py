import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.distributions import Categorical
from maml.policy.policy import Policy, weight_init

class CategoricalMLPPolicy(Policy):
    def __init__(self, input_size, output_size, hidden_sizes=(), nonlinearity=F.relu):
        pass