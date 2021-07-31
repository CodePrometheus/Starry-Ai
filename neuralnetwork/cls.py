import torch.nn
import torch.utils.data as data_utils
import torchvision.datasets as datasets
from torchvision.transforms import transforms

train_data = datasets.MNIST(root="mnist",
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_data = datasets.MNIST(root="mnist",
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

# batch_size 一部分数据进行训练
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle=True)

test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=True)


# net
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


cnn = CNN()
# cnn = cnn.cuda()
cnn = cnn.cpu()

# loss
loss_func = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# training
for epoch in range(1):
    for i, (images, labels) in enumerate(train_loader):
        # images = images.cuda()
        # labels = labels.cuda()
        images = images.cpu()
        labels = labels.cpu()

        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch is {}, item {}/{}, loss is {}"
          .format(epoch + 1, i, len(train_data) // 64,
                  loss.item()))

    # test
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cpu()
        labels = labels.cpu()
        outputs = cnn(images)

        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        accuracy += (pred == labels).sum().item()

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data) // 64)
    print("epoch is {}, accuracy is {}, loss test is {}"
          .format(epoch + 1, accuracy, loss_test.item()))

torch.save(cnn, "./model/mnist.model.pkl")
