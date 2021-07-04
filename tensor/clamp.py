import torch

"""
张量剪裁，值约束
常用于梯度裁剪，即发生在梯度离散或者梯度爆炸时对梯度的处理
"""

a = torch.rand(2, 2) * 10
print(a)
a = a.clamp(1, 5)
print(a)
