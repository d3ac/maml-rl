import gymnasium as gym

def make_env(env_name, env_kwargs={}):
    """
        这样做的好处是以后在创建环境的时候可以更加方便：
        env_carpole = make_env('CartPole-v1')
        然后后面创建环境的时候都使用: env = env_carpole()
    """
    def _make_env():
        env = gym.make(env_name, **env_kwargs)
    return _make_env

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
    def sample_asnc(self, args, **kargs):
        raise NotImplementedError()
    def sample(self, args, **kargs):
        return self.sample_asnc(args, **kargs)