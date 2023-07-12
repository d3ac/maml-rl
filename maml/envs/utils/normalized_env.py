import numpy as np
import gymnasium as gym
from gymnasium import spaces

class NormalizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, scale=1.0):
        super(NormalizedActionWrapper, self).__init__(env)
        self.scale = scale
        self.action_space = spaces.Box(low=-self.scale, high=self.scale, shape=self.env.action_space.shape, dtype=np.float32)
    
    def action(self, action):
        """
        把一个动作剪切到[-scale, scale]之间，然后再映射回去[lb, ub]
        """
        action = np.clip(action, -self.scale, self.scale) # 剪切到[-scale, scale]之间
        lb, ub = self.env.action_space.low, self.env.action_space.high # 找到action的最大值和最小值
        if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)): # 如果最大值和最小值都是有限的
            action = lb + (action + self.scale) * (ub - lb) / (2 * self.scale) # 将action加上scale防止小于0
            # 然后乘上(ub-lb)除以2*scale，这样就可以将action映射到[0,1]*(ub-lb)之间，再加上lb映射回去了相当于
            action = np.clip(action, lb, ub)
        return action