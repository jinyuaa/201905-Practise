# Logistic Regression 二分类 非线性边界
# 加正则化

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

'''
# load the dataset
data = np.loadtxt("data2.txt", delimiter=",")
# 可以看出数据是一个二维数组，维度是100*3
# print(data)
X = data[:, 0:2]
# X存放的是数据的特征，维度是：100*2
# print(X.shape)
y = data[:, 2]
# y存放的是数据的标签，维度是：100*1
# print(y)
pos = np.where(y == 1)
# pos是y中数据等于1的下标索引
# print(pos)
neg = np.where(y==0)
# neg是y中数据等于0的下标索引
# print(neg)
# python中数据可视化函数scatter(数据的横坐标向量，数据的纵坐标向量，marker='0'数据以点的形式显示，c='b'数据点是blue颜色)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')

# 说明二维坐标中o表示Pass,x表示Fail
legend(["y==1", "y==0"])
show()
'''


def load_data_set():
    # load the data_set
    data = np.loadtxt("data2.txt", delimiter=",")
    # 拿到X和y
    y = np.c_[data[:, 2]]
    x = data[:, 0:2]
    return data, x, y


def map_feature(x1, x2):
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    # accuracy max d=6
    degree = 6
    mapped_fea = np.ones(shape=(x1[:, 0].size, 1))
    for i in range(0, degree):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            mapped_fea = np.append(mapped_fea, r, axis=1)
    return mapped_fea


# 计算Sigmoid函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 定义损失函数
def loss_function(theta, x, y, l):
    m = y.size
    h = sigmoid(x.dot(theta))
    loss = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) \
        + (l / (2.0 * m)) * np.sum(np.square(theta[1:]))
    # print("---------------")
    # print(theta[1:])
    if np.isnan(loss[0]):
        return np.inf
    return loss[0]


# 计算梯度
def compute_grad(theta, x, y, l):
    m = y.size
    h = sigmoid(x.dot(theta.reshape(-1, 1)))
    grad = (1.0 / m) * x.T.dot(h - y) + (l / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    return grad.flatten()


# 梯度下降并优化
def grad_descent(xx, y, l):
    initial_theta = np.zeros(xx.shape[1])
    cost = loss_function(initial_theta, xx, y, l)
    print('Cost: ', cost)
    # 最优化 costFunctionReg
    res = minimize(loss_function, initial_theta, args=(xx, y, l), jac=compute_grad, options={'maxiter': 3000})
    return res


# 画出最终分类的图
def plot_best_fit(data, res, x, accuracy, l, axes):
    # 对X,y的散列绘图
    plot_data(data, 'feature_1', 'feature_2', 'sign 1', 'sign 0', axes=None)
    # 画出决策边界
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max(),
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(map_feature(xx1.ravel(), xx2.ravel()).dot(res.x))
    h = h.reshape(xx1.shape)
    if axes is None:
        axes = plt.gca()
    axes.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    axes.set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), l))
    plt.show()


def plot_data(data, label_x, label_y, label_pos, label_neg, axes):
    # 获得正负样本的下标
    negative = data[:, 2] == 0
    positive = data[:, 2] == 1
    if axes is None:
        axes = plt.gca()
    axes.scatter(data[positive][:, 0], data[positive][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[negative][:, 0], data[negative][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)


# 预测标签
def predict(theta, x):
    m, n = x.shape
    p = np.zeros(shape=(m, 1))
    h = sigmoid(x.dot(theta.T))
    for it in range(0, h.shape[0]):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0
    return p


def main():
    data, x, y = load_data_set()
    # 对给定的两个feature做一个多项式特征的映射
    mapped_fea = map_feature(x[:, 0], x[:, 1])
    print("m_f:", mapped_fea.shape)
    # 决策边界
    # Lambda = 0 : 就是没有正则化，过拟合
    # Lambda = 1
    # Lambda = 100 : 正则化项太激进，导致基本就没拟合出决策边界，欠拟合
    line = 1

    res = grad_descent(mapped_fea, y, line)
    print(res)

    # 准确率
    accuracy = y[np.where(predict(res.x, mapped_fea) == y)].size / float(y.size)*100.0

    # 画决策边界
    plot_best_fit(data, res, x,  accuracy, line, axes=None)


if __name__ == '__main__':
    main()
