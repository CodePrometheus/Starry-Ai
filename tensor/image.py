import cv2
import torch

data = cv2.imread(
    "https://dss1.bdstatic.com/6OF1bjeh1BF3odCf/it/u=1811334817,1550207097&fm=74&app=80&f=PNG&size=f121,"
    "121?sec=1880279984&t=644f3bfde029fa5552b9f811f8b89e38")
print(data)

cv2.imshow("test", data)
# cv2.waitKey(0)

res = torch.from_numpy(data)
print(res)

res = torch.flip(res, dims=[0])
data = res.numpy()

cv2.imshow("test1", data)
cv2.waitKey(0)
