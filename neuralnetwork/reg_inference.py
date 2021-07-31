import re

import numpy as np
import torch


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        # 回归模型
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out


net = torch.load("./model/model.pkl")

file = open("./reg/housing.data").readlines()
data = []

for item in file:
    # 处理为一个空格
    out = re.sub(r"\s{2,}", " ", item).strip()
    # print(out)
    data.append(out.split(" "))

data = np.array(data).astype(float)
# (506, 14)
# print(data.shape)

# 全部倒数第一个数据
Y = data[:, -1]
# 倒数第一个之前的数据
X = data[:, 0:-1]

# 训练集和测试集的划分
X_train = X[0:496, ...]
Y_train = Y[0:496, ...]
X_test = X[496:, ...]
Y_test = Y[496:, ...]

loss_func = torch.nn.MSELoss()

x_data = torch.tensor(X_test, dtype=torch.float32)
y_data = torch.tensor(Y_test, dtype=torch.float32)

pred = net.forward(x_data)
pred = torch.squeeze(pred)
loss_test = loss_func(pred, y_data) * 0.001
print("loss_test:{}".format(loss_test))
