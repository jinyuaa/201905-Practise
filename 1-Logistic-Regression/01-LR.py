# Logistic Regression 典型的线性分类器(二分类) 线性判别边界
# 构造损失函数 用极大似然法对其进行估计 取对数 转换 求解最优化问题
# 梯度下降法 优化参数 w
# 区别LR：把 LR 的 wx+b 通过 sigmoid 函数映射到(0,1)上，并划分一个阈值，
# 大于阈值的分为一类，小于等于分为另一类，可以用来处理二分类问题
# 最大迭代次数  学习率????
# 没有预测真确率????
# theta 初始化为[0, 0, 0]还是为[1, 1, 1]?

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

"""
# 将数据可视化 
data = np.loadtxt('data1.txt', delimiter=",")
X = data[:, 0:2]
print(X.shape)
y = data[:, 2]
# y数据中等于1的下标索引, tuple元组类型
positive_index = np.where(y == 1)
negative_index = np.where(y == 0)
plt.scatter(X[positive_index, 0], X[positive_index, 1], marker='o', c='b')
plt.scatter(X[negative_index, 0], X[negative_index, 1], marker='x', c='r')
plt.xlabel("feature_1")
plt.ylabel("feature_2")
plt.legend(["1", "0"])
plt.show()
"""


def load_data_set():
    # load the data_set
    data = np.loadtxt("data1.txt", delimiter=",")
    # data = np.loadtxt("data3.txt", delimiter="\t")    # data3: success:False
    # np.c_按列来组合array
    x = np.c_[np.ones((data.shape[0], 1)), data[:, 0:2]]
    print(type(x))   # tuple? (100,3)
    # 数组形式 [[],[],[]...,[]] y (100, 1)
    # y = data[:, 2].reshape(-1, 1)
    y = np.c_[data[:, 2]]
    print(y.shape)
    return data, x, y


# 定义 sigmoid 函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 定义损失函数
def loss_function(theta, x, y):
    # y 的长度 100
    m = y.size
    # h (100, )
    h = sigmoid(x.dot(theta))
    # print("h为000：", h[0])
    loss = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))
    # print("loss为:", loss)
    if np.isnan(loss[0]):
        return np.inf
    return loss[0]


# 求解梯度
def compute_grad(theta, x, y):
    m = y.size
    # print(theta.reshape(-1, 1)) 3x1
    h = sigmoid(x.dot(theta.reshape(-1, 1)))
    # h (100, 1)
    # print("h为：", h[0])
    grad = (1.0 / m) * x.T.dot(h-y)
    return grad.flatten()


# 进行迭代
def grad_descent(x, y):
    # theta [0. 0. 0.] <class 'numpy.ndarray'> (3, )
    initial_theta = np.zeros(x.shape[1])
    print("initial_theta的形状为：", initial_theta)
    cost = loss_function(initial_theta, x, y)
    grad = compute_grad(initial_theta, x, y)
    print("cost: {}".format(cost))
    print("grad: {}".format(grad))
    # 求解局部最小
    # scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None,
    # bounds=None, constraints=(), tol=None, callback=None, options=None)
    # fun 最小值的目标函数, x0 变量初始值, args() 常数值, method 求极值法, 一般默认
    # jac 雅各比矩阵(一阶导数矩阵)
    # options {maxiter: int} 要执行的最大迭代次数
    # 返回的是一个 optimization 对象
    res = minimize(loss_function, initial_theta, args=(x, y), jac=compute_grad, options={'maxiter': 1000})
    return res


def plot_best_fit(data, res, x, score):  # 画出最终分类的图
    plt.scatter(score[1], score[2], s=60, c='r', marker='v', label='('+str(score[1])+','+str(score[2])+')')
    plot_data(data, 'feature_1', 'feature_2', 'sign 1', 'sign 0')
    x1_min, x1_max = x[:, 1].min(), x[:, 1].max()
    x2_min, x2_max = x[:, 2].min(), x[:, 2].max()
    # 设置坐标范围
    # np.meshgrid(A, B) 从坐标向量中返回坐标矩阵，类型为 list，两个元素，第一个是x轴的取值，第二个是y轴的取值
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    # xx1: (50,50) xx2:(50,50)
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(res.x))
    # 上述 h (2500, )
    h = h.reshape(xx1.shape)
    # print("h", h.shape) (50, 50)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.show()


def plot_data(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    negative = data[:, 2] == 0
    # print("下标是：", data[negative][:, 1])
    positive = data[:, 2] == 1
    # 判断当前坐标轴不存在 或者不是一个极轴 则创建合适的坐标轴 返回
    if axes is None:
        axes = plt.gca()
    axes.scatter(data[positive][:, 0], data[positive][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[negative][:, 0], data[negative][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(loc='lower left', frameon=True, fancybox=True)


def predict(theta, in_put, threshold=0.5):
    p = sigmoid(in_put.dot(theta.T)) >= threshold
    return p.astype('int')


def main():
    data, x, y = load_data_set()
    res = grad_descent(x, y)
    print("res是：\n", res)
    print("请输入您要预测的值(用空格隔开)(x:[30,100],y:[30:100])：")
    input_score = input()
    num = [int(n) for n in input_score.split()]
    score = np.array(num, dtype=int)
    # res.x  [-25.16131634   0.2062316    0.20147143]
    print("您预测的数据是: %d" % predict(res.x, score))
    plot_best_fit(data, res, x, score)


if __name__ == '__main__':
    main()





