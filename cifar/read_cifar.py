import glob
import os
import pickle

import cv2
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


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

train_list = glob.glob("./cifar-10-batches-py/data_batch_*")
test_list = glob.glob("./cifar-10-batches-py/test_batch")
save_path = "./cifar-10-batches-py/train"
save_path_test = "./cifar-10-batches-py/test"
# print(train_list)

for i in test_list:
    print(i)
    i_dict = unpickle(i)
    print(i_dict.keys())

    for im_idx, im_data in enumerate(i_dict[b'data']):
        # print(im_idx)
        # print(im_data)

        im_label = i_dict[b'labels'][im_idx]
        im_name = i_dict[b'filenames'][im_idx]
        # print(im_label, im_name, im_data)

        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))

        # cv2.imshow("im_data", cv2.resize(im_data, (200, 200)))
        # cv2.waitKey(0)

        if not os.path.exists("{}/{}".format(save_path_test, im_label_name)):
            os.mkdir("{}/{}".format(save_path_test, im_label_name))

        cv2.imwrite("{}/{}/{}".format(save_path_test, im_label_name,
                                      im_name.decode("utf-8")), im_data)
