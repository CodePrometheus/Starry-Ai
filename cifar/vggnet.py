import torch.nn.functional as F
from torch import nn


class VggBase(nn.Module):
    def __init__(self):
        super(VggBase, self).__init__()
        # 28 * 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            # 数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 14 * 14
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 7 * 7
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.max_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.max_pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(512 * 4, 10)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv1(x)
        out = self.max_pooling1(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.max_pooling2(out)

        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.max_pooling3(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.max_pooling4(out)

        # 自行计算 batch_size * n 的值
        out = out.view(batch_size, -1)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out


def vgg_net():
    return VggBase
