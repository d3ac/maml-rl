import numpy as np
from maml.utils.torch_utils import weighted_mean, to_numpy

def value_iteration(transitions, rewards, gamma=0.95, theta=1e-5):
    rewards = np.expand_dims(rewards, axis=2) # [trajectory_length, batch_size, 1]
    values = np.zeros(transitions.shape[0], dtype=np.float32) #TODO transitions的形状是什么？
    pass #! 还没写完


def value_iteration_finite_horizon(transitions, rewards, horizon=10, gamma=0.95):
    pass

def get_returns(episodes):
    pass

def xreinforce_loss(policy, episodes, params=None):
    pi = policy(episodes.observations.view((-1, *episodes.observation_shape)), params=params)
    # episode的形状是 [trajectory_length, batch_size, ...], pi的形状是 [trajectory_length, batch_size, action_dim]
    log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
    # log_probs的形状是 [trajectory_length, batch_size]
    #!前面的有点问题，先去写别的