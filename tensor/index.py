import torch

a = torch.rand(4, 4)
b = torch.rand(4, 4)
print(a)
print(b)
# >0.5选择a，否则b
res = torch.where(a > 0.5, a, b)
print("res : ", res)

print("======================")
a = torch.rand(4, 4)
print(a)
res = torch.index_select(a, dim=0, index=torch.tensor([0, 3, 2]))
print("res : ", res)

print("======================")
a = torch.linspace(1, 16, 16).view(4, 4)
print(a)
# 具体维度上的值
res = torch.gather(a, dim=0, index=torch.tensor([[0, 1, 1, 1],
                                                 [0, 1, 2, 2],
                                                 [0, 1, 3, 3]]))
print("res : ", res)

print("======================")
a = torch.linspace(1, 16, 16).view(4, 4)
mask = torch.gt(a, 8)
print(a)
print(mask)
res = torch.masked_select(a, mask)
print("res : ", res)
