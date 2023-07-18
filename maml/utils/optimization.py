import torch

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    # 共轭梯度, 用于求解Ax=b的解x, 输入的是一个函数f_Ax, 
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b).float() # 初始解x通常取0向量
    rdotr = torch.dot(r, r)
    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break
    return x.detach()