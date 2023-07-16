import torch
import asyncio

from maml.samplers import MultiTaskSampler

class GradientBasedMetaLearner(object):
    def __init__(self, policy, device='cpu'):
        super(GradientBasedMetaLearner, self).__init__()
        self.device = device
        self.policy = torch.device(device)
        self.policy = policy.to(device)
        self._event_loop = asyncio.get_event_loop()
    
    def adapt(self, episodes, *args, **kwargs):
        raise NotImplementedError
    
    def step(self, train_episodes, valid_episodes, *args, **kwargs):
        raise NotImplementedError
    
    def _async_gather(self, coros): # 传入多个task, 然后运行等待所有完成, 返回多个task的结果
        coro = asyncio.gather(*coros)
        return zip(*self._event_loop.run_until_complete(coro))