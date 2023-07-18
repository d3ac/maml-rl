from gymnasium.envs.registration import load_env_creator
from gymnasium.wrappers import TimeLimit
from maml.envs.utils.normalized_env import NormalizedActionWrapper

def mujoco_wrapper(entry_point, **kwargs):
    normalization_scale = kwargs.pop('normalization_scale', 1.0)
    max_episode_steps = kwargs.pop('max_episode_steps', 200)
    env_class = load_env_creator(entry_point)
    env = env_class(**kwargs)
    env = NormalizedActionWrapper(env, scale=normalization_scale) # 添加动作归一化
    env = TimeLimit(env, max_episode_steps=max_episode_steps) # 添加最大步数限制
    return env