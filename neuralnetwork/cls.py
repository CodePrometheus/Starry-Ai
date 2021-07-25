import torchvision.datasets as datasets
from torchvision.transforms import transforms


train_data = datasets.MNIST(root="mnist",
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

train_data = datasets.MNIST(root="mnist",
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)

# batchsize 一部分数据进行训练
# train_data = data_utils