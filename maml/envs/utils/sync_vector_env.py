import numpy as np
from gym.vector import SyncVectorEnv as SyncVectorEnv_
from gym.vector.utils import concatenate, create_empty_array

class SyncVectorEnv(SyncVectorEnv_):
    def __init__(self, env_fns, observation_space=None, action_space=None, **kwargs):
        super(SyncVectorEnv, self).__init__(env_fns, observation_space=observation_space, action_space=action_space, **kwargs)
        for env in self.envs:
            if not hasattr(env.unwrapped, 'reset_task'):
                raise ValueError('envs must contain a reset_task method for SyncVectorEnv')
        self._dones = np.zeros(len(self.envs), dtype=np.bool_) # 记录是不是每一个环境都结束了

    @property
    def dones(self):
        return self._dones
    
    def reset_task(self, task):
        # self._dones[:] = False
        for env in self.envs:
            env.unwrapped.reset_task(task)
    
    def step_wait(self): # SyncVectorEnv的step会调用step_wait，让每一个环境都step一步
        observations_list, infos = [], []
        batch_ids, j = [], 0
        num_actions = self.action_space.shape[0]
        rewards = np.zeros((num_actions,), dtype=np.float32)
        for i, (env, action) in enumerate(zip(self.envs, self._actions)): # step一轮
            if self._dones[i]:
                continue
            observation, rewards[j], self._dones[i], truncated, info = env.step(action)
            self.dones[i] = self.dones[i] or truncated
            batch_ids.append(i)
            if not self._dones[i]:
                observations_list.append(observation)
                infos.append(info)
            j += 1
        assert num_actions == j
        return (np.array(observations_list), rewards, np.copy(self._dones), {'batch_ids': batch_ids, 'infos': infos})
