import torch
import numpy as np
from torch.distributions import Independent, Normal, Categorical
from torch.nn.utils.convert_parameters import _check_param_device

def weighted_mean(tensor, lengths=None):
    """
    输入tensor的shape是(max_length, batch_size, obs_dim)
    输出的shape是(batch_size, obs_dim)
    返回的是每个episode的平均值
    """
    if lengths is None: # 如果没有传入lengths，就直接求平均值
        return torch.mean(tensor) # 直接返回这个单个episode的平均值
    if tensor.dim() < 2:
        raise ValueError('error at weighted_mean, tensor must be at least 2D')
    for i, lengths in enumerate(lengths):
        tensor[lengths:, i].fill_(0.)
    extra_dims =  (1,) * (tensor.dim() - 2)
    lengths = torch.as_tensor(lengths, dtype=torch.float32)
    out = torch.sum(tensor, dim=0) # 在dim=0上求和，也就是把一个episode的所有值加起来
    out.div_(lengths.view(-1, *extra_dims)) # 然后每个episode都出来除以这个episode的长度
    # 这里lenghths变成了(batch_size, 1, 1, 1, ...)的形式，然后和out做除法，就相当于每个episode都除以这个episode的长度
    return out

def weighted_normalize(tensor, lengths=None, epsilon=1e-8):
    mean = weighted_mean(tensor, lenghth)
    out = tensor - mean.mean() # 现在out的均值为0
    for i, lenghth in enumerate(lengths):
        out[lenghth:, i].fill_(0.)
    std = torch.sqrt(weighted_mean(out ** 2, lengths).mean()) # $$ \sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2} $$
    out.div_(std + epsilon)
    return out

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (tuple, list)):
        return np.stack([to_numpy(t) for t in tensor], axis=0) # 如果使用np.array, 可能会有广播操作，或者数据类型的转换
    else:
        raise NotImplementedError('to_numpy not implemented for type')

def detach_distribution(pi):
    # detach可以把一部分计算图固定住,只更新另一部分
    if isinstance(pi, Independent): # Independent是多个分布组成的, 要分别detach
        return Independent(detach_distribution(pi.base_dist), pi.reinterpreted_batch_ndims)
    elif isinstance(pi, Normal):
        return Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    elif isinstance(pi, Categorical):
        return Categorical(probs=pi.probs.detach())
    else:
        raise NotImplementedError('detach_distribution not implemented for type')

def vector_to_parameters(vector, parameters):
    param_device = None
    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        num_param = param.numel() # 返回参数的元素个数
        param.data.copy_(vector[pointer:pointer + num_param].view_as(param).data)
        pointer += num_param