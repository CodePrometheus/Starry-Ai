"""
PCA（Principal Component Analysis） 是一种常见的数据分析方式，常用于高维数据的降维，可用于提取数据的主要特征分量。
将n维特征映射到K维上，这K维是全新的正交特征也被称为主成分，是在原有n维特征的基础上重新构造出来的k维特征
目标：降维后同一纬度的方差最大，不同维度之间的相关性为0，协方差矩阵
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


# Generate datasets
def generate_data():
    """Generate 3 Gaussians samples with the same covariance matrix"""
    n, dim = 512, 3
    np.random.seed(0)
    C = np.array([[1., 0.2, 0], [0.15, 1, 0.2], [0.1, 0.4, 10.0]])
    X = np.r_[
        np.dot(np.random.randn(n, dim), C),
        np.dot(np.random.randn(n, dim), C) + np.array([1, 2, 5]),
        np.dot(np.random.randn(n, dim), C) + np.array([-5, -2, 3]),
    ]
    y = np.hstack((
        np.ones(n) * 0,
        np.ones(n) * 1,
        np.ones(n) * 2,
    ))
    return X, y


class PCA(object):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)


if __name__ == "__main__":
    X, y = generate_data()
    pca = PCA(n_components=2)
    X = torch.FloatTensor(X)
    pca.fit(X)
    trans_X = pca.transform(X)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X.T[0], X.T[1], X.T[2], c=y)
    ax = fig.add_subplot(122)
    ax.scatter(trans_X.T[0], trans_X.T[1], c=y)
    plt.show()
