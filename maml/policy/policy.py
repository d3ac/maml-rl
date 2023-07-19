import torch
from torch import nn
from collections import OrderedDict # 有序字典，按照添加的顺序排序

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size,output_size):
        # 传入了一个module，他的数据都在self.named_parameters()里面
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters
    
    def update_params(self, loss, params=None, lr=0.5, first_order=False):
        # 需要单独把params拿出来，是因为方便计算first_order,second_order的梯度
        if params is None: # 默认是policy自己的参数
            params = OrderedDict(self.named_parameters()) # deep copy, 有序字典
        grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)
        # 如果只需要一阶导数，那么就不需要创建计算图，这样可以节省内存
        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - lr * grad
        return updated_params