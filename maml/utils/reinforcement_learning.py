import numpy as np
from maml.utils.torch_utils import weighted_mean, to_numpy

def get_returns(episodes):
    # returns 是一系列动作之后获得的奖励总和
    return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])

def xreinforce_loss(policy, episodes, params=None):
    #!这个还有一点问题，没有弄懂他的维度
    # policy是一个函数, 输入是observation, 输出是动作的分布, 形状是 [action, ]
    pi = policy(episodes.observations.view((-1, *episodes.observation_shape)), params=params) # 每个动作的分布
    # episode.observation的形状是 [trajectory_length, batch_size, observation_dim] -> [trajectory_length * batch_size, observation_dim]
    # pi的形状是 [action,]
    log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape))) # log_probs是distribution的函数
    # episode.actions的形状是 [trajectory_length, batch_size, action_dim] -> [trajectory_length * batch_size, action_dim]
    # log_probs的形状是 [trajectory_length * batch_size, action_dim]
    log_probs = log_probs.view(len(episodes), episodes.batch_size)
    losses = - weighted_mean(log_probs * episodes.advantages, lengths=episodes.lengths) 
    return losses.mean()