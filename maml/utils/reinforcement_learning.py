import numpy as np
from maml.utils.torch_utils import weighted_mean, to_numpy

def get_returns(episodes):
    # returns 是一系列动作之后获得的奖励总和
    return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])

def reinforce_loss(policy, episodes, params=None):
    # policy是一个函数, 输入是observation, 输出是动作的分布, 形状是 [action, ]
    pi = policy(episodes.observations.view((-1, *episodes.observation_shape)), params=params) # 每个动作的分布
    # episode.observation的形状是 [trajectory_length, batch_size, observation_dim] -> [trajectory_length * batch_size, observation_dim]
    # print(pi.sample().shape) # torch.Size([maxlen * meta_batch(2000), 6])
    log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape))) # (maxlen * meta_batch, )
    # log_probs就是计算在给定分布下的概率密度,log_probs的形状是 [trajectory_length * batch_size, ]
    # 因为是那action_dim(6个)同时取到的概率, 所以加起来最后一维action_dim就没有了
    log_probs = log_probs.view(len(episodes), episodes.batch_size)
    losses = - weighted_mean(log_probs * episodes.advantages, lengths=episodes.lengths) 
    return losses.mean()