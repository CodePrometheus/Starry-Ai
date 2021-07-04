import torch

dev = torch.device("cpu")
a = torch.tensor([2, 2], device=dev)
print(a)

# 稀疏张量
i = torch.tensor([[0, 1, 2], [0, 1, 2]])
v = torch.tensor([1, 2, 3])
a = torch.sparse_coo_tensor(i, v, (4, 4),
                            dtype=torch.float32,
                            device=dev)
b = a.to_dense()
print(a)
print(b)

a = torch.tensor([1, 2])
b = torch.tensor([3, 1])
print(a @ b)
a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 3)
print(a.matmul(b).shape)
