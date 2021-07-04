# 就地操作，即不允许使用临时变量
import torch

a = torch.rand(2, 3)
b = torch.rand(3)

print(a)
print(b)

c = a + b
print(c)
print(c.shape)

a = a * 10
print(a)
print(torch.ceil(a))