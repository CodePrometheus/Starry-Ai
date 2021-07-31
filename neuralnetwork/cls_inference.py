import cv2
import torch.nn
import torch.utils.data as data_utils
import torchvision.datasets as datasets
from torchvision.transforms import transforms

# data
test_data = datasets.MNIST(root="mnist",
                           train=False,
                           transform=transforms.ToTensor(),
                           download=False)

test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=True)

for x, y in test_loader:
    # N代表数量， C代表channel，H代表高度，W代表宽度
    print("Shape of x [N, C, H, W]", x.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

cnn = torch.load("./model/mnist.model.pkl")
cnn = cnn.cpu()

# training
loss_test = 0
accuracy = 0
for i, (images, labels) in enumerate(test_loader):
    images = images.cpu()
    labels = labels.cpu()
    outputs = cnn(images)
    _, pred = outputs.max(1)
    accuracy += (pred == labels).sum().item()

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()

    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = labels[idx]
        im_pred = pred[idx]
        im_data = im_data.transpose(1, 2, 0)

        print("label", im_label)
        print("pred", im_pred)

        cv2.imshow("imData", im_data)
        cv2.waitKey(0)

accuracy = accuracy / len(test_data)
print("accuracy: ", accuracy)
