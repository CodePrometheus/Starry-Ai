import re

import numpy as np
import torch

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

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# net
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


net = Net(13, 1)

# loss
loss_func = torch.nn.MSELoss()

# optimizer优化器
# optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# training
for i in range(1000):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32)

    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_train = loss_func(pred, y_data) * 0.001
    # print("pred.shape", pred.shape)
    # print("y_data.shape", y_data.shape)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print("item:{}, loss_train:{}".format(i, loss_train))
    print("预测结果: ", pred[0:10])
    print("真实结果: ", y_data[0:10])

    # test
    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(Y_test, dtype=torch.float32)

    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data) * 0.001
    print("item:{}, loss_test:{}".format(i, loss_test))

torch.save(net, "./model/model.pkl")
# torch.save(net.state_dict(), "model/params.pkl")
