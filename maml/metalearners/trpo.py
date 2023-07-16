import torch
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from maml.samplers import MultiTaskSampler
from maml.metalearners.base import GradientBasedMetaLearner
from maml.utils.torch_utils import weighted_mean, detach_distribution, to_numpy, vector_to_parameters
from maml.utils.optimization import conjugate_gradient #!还没写
from maml.utils.reinforcement_learning import reinforce_loss

class MAMLTRPO(GradientBasedMetaLearner):
    def __init__(self, policy, fast_lr=0.5, first_order=False, device='cuda'):
        super(MAMLTRPO, self).__init__(policy, device)
        self.fast_lr = fast_lr
        self.first_order = first_order

    async def adapt(self, train_episode_futures, first_order=None): #! 每一步都懂了,但是在那里调用的呢
        if first_order is None:
            first_order = self.first_order
        params = None
        for train_episode in train_episode_futures:
            inner_loss = reinforce_loss(self.policy, await train_episode, params=params)
            params = self.policy.update_params(inner_loss, lr = self.fast_lr, first_order=first_order)
        return params
    
    def hessian_vector_product(self, kl, damping=1e-2): # 返回一个函数
        # kl是衡量两个分布之间的差异的指标, kl越大,两个分布越不相似
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flag_grad_kl = parameters_to_vector(grads)
        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flag_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters(), retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)
            return flat_grad2_kl + damping * vector
        return _product #!看看这个函数在那里use
    
    async def surrogate_loss(self, train_futures, valid_futures, old_pi=None):
        # 计算一个替代的loss, 用来更新参数, 使用新旧两个分布之间的差异来计算, 会更加稳定
        #!是不使用firtst order有点迷
        first_order = (old_pi is not None) or self.first_order # 
        params = await self.adapt(train_futures, first_order) # 先进行内循环更新
        with torch.set_grad_enabled(old_pi is None): # 如果old_pi为空, 则需要计算梯度
            valid_episodes = await valid_futures 
            pi = self.policy(valid_episodes.observations, params=params)
            if old_pi is None:
                old_pi = detach_distribution(pi) # 如果没有的话就复制一份, 但是不需要梯度
            log_ratio = (pi.log_prob(valid_episodes.actions) - old_pi.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)
            # 和reinforce_loss相比, 这个地方的ratio是两个分布之间的差值, 而不是一个分布下的概率
            losses = - weighted_mean(ratio * valid_episodes.advantages, lengths=valid_episodes.lengths)
            kls = weighted_mean(kl_divergence(pi, old_pi), lengths=valid_episodes.lengths)
        return losses.mean(), kls.mean(), old_pi
    
    def step(self, train_futures, valid_futures, max_kl=1e-3, cg_iters=10, cg_damping=1e-2, ls_max_steps=10, ls_backtrack_ratio=0.5):
        num_tasks = len(train_futures[0])
        logs = {}
        