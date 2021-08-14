import glob

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

label_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx


def default_loader(path):
    return Image.open(path).convert("RGB")


# 数据增强
train_transform = transforms.Compose([
    # resize
    # transforms.RandomResizedCrop((28, 28)),
    # 水平翻转
    # transforms.RandomHorizontalFlip(),
    # 垂直翻转
    # transforms.RandomVerticalFlip(),
    # 随机灰度
    # transforms.RandomGrayscale(0.1),
    # 旋转 -90 ~ 90
    # transforms.RandomRotation(90),
    # 改亮度、对比度和饱和度
    # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    transforms.RandomCrop(28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, im_list,
                 transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        images = []

        for im_item in im_list:
            im_label_name = im_item.split("\\")[-2]
            images.append([im_item, label_dict[im_label_name]])

        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path, im_label = self.images[index]
        im_data = self.loader(im_path)
        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.images)


im_train_list = glob.glob("./cifar-10-batches-py/train/*/*.png")
im_test_list = glob.glob("./cifar-10-batches-py/test/*/*.png")

train_dataset = MyDataset(im_train_list,
                          transform=train_transform)
test_dataset = MyDataset(im_test_list,
                         transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=6,
                          shuffle=True,
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=6,
                         shuffle=False,
                         num_workers=4)

print("num_of_train", len(train_dataset))
print("num_of_test", len(test_dataset))
