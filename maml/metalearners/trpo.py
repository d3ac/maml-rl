import torch
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from maml.samplers import MultiTaskSampler
from maml.metalearners.base import GradientBasedMetaLearner
from maml.utils.torch_utils import weighted_mean, detach_distribution, to_numpy, vector_to_parameters
from maml.utils.optimization import conjugate_gradient #!还没写
from maml.utils.reinforcement_learning import reinforce_loss

class MAMLTRPO(GradientBasedMetaLearner):
    def __init__(self, policy, fast_lr=0.5, first_order=False, device='cpu'):
        super(MAMLTRPO, self).__init__(policy, device)
        self.fast_lr = fast_lr
        self.first_order = first_order

    async def adapt(self, train_episode_futures, first_order=None):
        # inner_loop更新一次参数, 然后给最外层outerloop使用loss更新
        if first_order is None:
            first_order = self.first_order
        params = None
        for train_episode in train_episode_futures: # 把一个trajectory拿来更新
            inner_loss = reinforce_loss(self.policy, await train_episode, params=params)
            params = self.policy.update_params(inner_loss, params, lr = self.fast_lr, first_order=first_order)
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
        # 如果没有old_pi就会将valid_episodes丢进policy算一个pi出来
        first_order = (old_pi is not None) or self.first_order # 如果old_pi为空, 就只能依赖于当前的策略梯度, 所以我们不能first_order, 但是如果为不为空,那么说明可以用就策略来计算新策略, 也就是通过一阶梯度策略改进策略
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
        num_tasks = len(train_futures[0]) # 也就是num_steps
        logs = {}
        old_losses, old_kls, old_pis = self._async_gather([self.surrogate_loss(train, valid, old_pi=None) for (train, valid) in zip(zip(*train_futures), valid_futures)])
        # train_futures是一个列表, shape为 (m, n)表示, 每个任务有m个trajectory, 一共有n个不同的任务
        # [ [traj1_task1, traj1_task2, ..., traj1_taskn] ...
        # [trajm_taskm, trajm_task2, ..., trajm_taskn] ]
        # 这里使用 zip(* train_futures)就可以把每个任务的trajectory放在一起, 形成一个列表
        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)
        old_loss = sum(old_losses) / num_tasks 
        old_kl = sum(old_kls) / num_tasks
        grads = torch.autograd.grad(old_loss, self.policy.parameters(), retain_graph=True)
        grads = parameters_to_vector(grads)
        hessian_vector_product = self.hessian_vector_product(old_kl, damping=cg_damping) # 定义的是一个函数
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)
        
        lagrange_multiplier = torch.sqrt(0.5 * torch.dot(stepdir, hessian_vector_product(stepdir, False)) / max_kl)
        step = stepdir / lagrange_multiplier
        old_params = parameters_to_vector(self.policy.parameters())
        
        step_size = 1.0
        for i in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step, self.policy.parameters()) # 就是更新一下参数
            losses, kls, _ = self._async_gather([self.surrogate_loss(train, valid, old_pi=old_pi) for (train, valid, old_pi) in zip(zip(*train_futures), valid_futures, old_pis)])
            improve = sum(losses) / num_tasks - old_losses
            kl = sum(kls) / num_tasks
            if improve.item() < 0.0 and kl.item() < max_kl:
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters()) # 如果for循环正常结束, 则不更新参数
        return logs