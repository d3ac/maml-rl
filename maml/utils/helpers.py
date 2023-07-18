import gymnasium as gym
import torch
from functools import reduce
from operator import mul
from maml.policy import CategoricalMLPPolicy, NormalMLPPolicy

def get_policy_for_env(env, hidden_sizes=(100, 100), nonlinearity='relu'):
    input_size = get_input_size(env)
    # 举个例子 recduce(mul, (4, 5)), 1) = 4 * 5 * 1 = 20
    nonlinearity = getattr(torch, nonlinearity)
    if isinstance(env.action_space, gym.spaces.Box):
        output_size = reduce(mul, env.action_space.shape, 1)
        policy = NormalMLPPolicy(input_size, output_size, hidden_sizes=tuple(hidden_sizes), nonlinearity=nonlinearity)
    else:
        output_size = env.action_space.n
        policy = CategoricalMLPPolicy(input_size, output_size, hiddensizes=tuple(hidden_sizes), nonlinearity=nonlinearity)
    return policy

def get_input_size(env):# reduce 就是把 observation里面的全部元素相乘
    return reduce(mul, env.observation_space.shape, 1)