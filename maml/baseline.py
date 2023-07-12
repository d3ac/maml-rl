import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class LinearFeatureBaseline(nn.Module):
    def __init__(self, input_size, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        self.weight = nn.Parameter(torch.Tensor(self.feature_size,), requires_grad=True)
        self._eye = torch.eye(self.feature_size, dtype=torch.float32, device=self.weight.device)
    
    @property
    def feature_size(self): # input_size 就是observation的所有维度乘起来
        return 2 * self.input_size + 4 # dim=2这个维度就是这么多，最后会flatten, 所以只用看的dim=2这里, 两个observations就是这个乘2, 剩下的就是每个都是1
    
    def _feature(self, episodes): # 手动构造特征
        ones = episodes.mask.unsqueeze(2) # unsqueeze: add a dimension
        observations = episodes.observations
        time_step = torch.arange(len(episodes)).view(-1, 1, 1) * ones / 100.0
        # 前面先生成一个id(从0~episodes-1), 然后乘上ones，有数字的地方就有id, 除以100.0是为了防止过大
        return torch.cat([observations, observations ** 2, time_step, time_step ** 2, time_step ** 3, ones], dim=2)
    
    def fit(self, episodes):
        # 用feature去拟合returns, 也就是得到了observation去找returns
        featmat = self._feature(episodes).view(-1, self._feature_size) # flatten
        returns = episodes.returns.view(-1, 1)
        flat_mask = episodes.mask.flatten() # (a*b,)
        # 去掉是0的部分
        flat_mask_idx = torch.nonzero(flat_mask)
        featmat = featmat[flat_mask_idx].view(-1, self._feature_size)
        returns = returns[flat_mask_idx].view(-1, 1)
        # 计算
        reg_coeff = self._reg_coeff
        XT_y = torch.matmul(featmat.t(), returns)
        XT_X = torch.matmul(featmat.t(), featmat)
        for i in range(5):
            try:
                coeffs= torch.linalg.lstsq(XT_y, XT_X + reg_coeff * self._eye, driver='gelsy')
                if torch.isnan(coeffs.solution).any() or torch.isinf(coeffs.solution).any():
                    raise RuntimeError
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
            raise RuntimeError('Unable to compute baseline beacause of singular matrix')
        self.weight.copy_(coeffs.solution.flatten())


    def forward(self, episodes):
        # 输入episodes, 输出每个observation的value
        features = self._feature(episodes) # (100, 20, 38)
        values = torch.mv(features.view(-1, self.feature_size), self.weight) # (2000, 38) * (38,) = (2000,)
        return values.view(features.shape[:2]) # (100, 20)