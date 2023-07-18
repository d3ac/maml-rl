import gymnasium as gym

class make_env(object):
    """
    这样做的好处是以后在创建环境的时候可以更加方便：
    env_carpole = make_env('CartPole-v1')
    然后后面创建环境的时候都使用: env = env_carpole()
    """
    def __init__(self, env_name, env_kwargs={}, seed=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.seed = seed

    def __call__(self):
        env = gym.make(self.env_name, **self.env_kwargs)
        if hasattr(env, 'seed'):
            env.seed(self.seed)
        return env

class Sampler(object):
    def __init__(self, env_name, env_kwargs, batch_size, policy, env=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.batch_size = batch_size
        self.policy = policy
        if env is None:
            env = gym.make(env_name, **env_kwargs)
        self.env = env
        self.env.close()
        self.closed = False
    
    # 需要子类实现的方法，这里只是定义了接口
    def sample_async(self, args, **kargs):
        raise NotImplementedError()
    def sample(self, args, **kargs):
        return self.sample_async(args, **kargs)