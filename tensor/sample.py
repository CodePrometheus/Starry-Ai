import torch

# 约束随机抽样
torch.manual_seed(1)
mean = torch.rand(1, 2)
std = torch.rand(1, 2)

# 正态
print(torch.normal(mean, std))

# 范数
a = torch.rand(1, 1)
b = torch.rand(1, 1)
print(a, b)

print(torch.dist(a, b, p=1))
# 根号(a-b)^2
print(torch.dist(a, b, p=2))
print(torch.dist(a, b, p=3))
print(torch.norm(a, p=0))
