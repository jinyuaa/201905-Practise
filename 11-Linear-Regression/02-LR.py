# 局部线性加权回归 与 线性回归对比

import numpy as np
import matplotlib.pyplot as plt


def init_data():
    data = np.loadtxt('data2.txt', delimiter='\t')
    # data <class 'numpy.ndarray'>
    return data


# 线性回归 公式解析解
# weight = (x.T * x).I * x.T * y
# 2x200 200x2 2x200 200x1
def func_(x, y):
    XTX = x.T * x
    if np.linalg.det(XTX) == 0.0:
        print('This matrix is singular, cannot do inverse')
    else:
        weights = XTX.I * (x.T * y)
    return weights


# 局部加权线性回归
# weight = (x.T * W * x).I * x.T * W * y
# W 使用高斯核权重 w = exp(- (xi - x)/2 * k **2) k为高斯核参数
def lwlr(x, y, k=1.0):
    xMat = np.mat(x)
    yMat = np.mat(y).T                              # 传入的是 1*m 的数据 将其转置成 m*1
    m = np.shape(xMat)[0]                           # 数据的条数
    y_w = np.zeros(m)                               # 初始化矩阵
    for i in range(m):                              #
        weights = np.mat(np.eye(m))                 # 初始化矩阵为对角阵
        for j in range(m):                          # 遍历所有数据求解每一条数据对应的权重
            diffMat = x[i] - xMat[j, :]
            weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))   # 高斯核
        xTx = xMat.T * (weights * xMat)
        if np.linalg.det(xTx) == 0.0:               # 判断矩阵是否可逆，行列式是否为 0
            print('This matrix is singular,cannot do inverse')
            return
        ws = xTx.I * (xMat.T * (weights * yMat))    # 求解这一条数据的解析解
        y_w[i] = x[i] * ws
    return y_w


if __name__ == '__main__':
    data = init_data()
    x = np.mat(data[:, 0:2])
    y = np.mat(data[:, 2])

    # k = 0.5 欠拟合 标准的线性回归
    # k = 0.01
    # k = 0.003 过多考虑噪声造成过拟合
    y_w = lwlr(x, y, k=0.01)

    XMat = np.mat(x)
    srt_index = XMat[:, 1].argsort(axis=0)
    x_sort = XMat[srt_index][:, 0, :]

    plt.plot(x_sort[:, 1], y_w[srt_index], linewidth=2)
    plt.scatter(data[:, 1], data[:, 2], s=8, c='r')

    plt.show()
