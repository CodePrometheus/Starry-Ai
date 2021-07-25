import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5., 5., 0.2)
y = sigmoid(x)

plt.plot(x, y)
plt.show()
