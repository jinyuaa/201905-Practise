#  一元线性回归
# 最小二乘法 -- 梯度下降法求解

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def init_data():
    data = np.loadtxt('data1.txt', delimiter=',')
    return data


def linear_regression():
    learning_rate = 0.01  # 步长
    initial_b = 0
    initial_m = 0
    num_iter = 1000  # 迭代次数

    data = init_data()
    [b, m] = optimizer(data, initial_b, initial_m, learning_rate, num_iter)
    plot_data(data, b, m)
    print('y = {} + {} * x'.format(b, m))
    return b, m


# 方法1：梯度下降法求解, 已知梯度下降公式情况下 GD求解
def optimizer(data, initial_b, initial_m, learning_rate, num_iter):
    b = initial_b
    m = initial_m

    for i in range(num_iter):
        b, m = compute_gradient(b, m, data, learning_rate)
        if i % 100 == 0:
            print(i, compute_error(b, m, data))
    return [b, m]


def compute_gradient(b_cur, m_cur, data, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(data))
    # 偏导数，梯度
    for i in range(0, len(data)):
        x = data[i, 1]
        y = data[i, 2]
    # 已知偏导数的公式，梯度公式的情况下
        b_gradient += -(2 / N) * (y - ((m_cur * x) + b_cur))
        m_gradient += -(2 / N) * x * (y - ((m_cur * x) + b_cur))

    new_b = b_cur - (learning_rate * b_gradient)
    nwe_m = m_cur - (learning_rate * m_gradient)

    return [new_b, nwe_m]


def compute_error(b, m, data):
    # total_err = 0
    x = data[:, 1]
    y = data[:, 2]
    total_err = (y - m * x - b) ** 2
    total_err = np.sum(total_err, axis=0)
    return total_err / len(data)


'''
# 方法2：梯度下降求解，当不知道梯度公式时，利用导数定义
def optimizer2(data, initial_b, initial_m, learning_rate, num_iter):
    b = initial_b
    m = initial_m

    while True:
        before_error = compute_error(b, m, data)
        # print('before_error', before_error )
        b, m = compute_gradient2(b, m, data, learning_rate)
        after_error = compute_error(b, m, data)
        # print('after_error', after_error)
        if abs(after_error - before_error) < 0.0000001:
            break
    return [b, m]


def compute_gradient2(b_cur, m_cur, data, learning_rate):
    b_gradient = 0
    m_gradient = 0

    N = float(len(data))

    delta = 0.0000001

    for i in range(len(data)):
        x = data[i, 1]
        y = data[i, 2]
        # 利用导数来定义梯度
        b_gradient = (error(x, y, b_cur + delta, m_cur) - error(x, y, b_cur - delta, m_cur)) / (2 * delta)
        m_gradient = (error(x, y, b_cur, m_cur + delta) - error(x, y, b_cur, m_cur - delta)) / (2 * delta)

    b_gradient = b_gradient / N
    m_gradient = m_gradient / N

    new_b = b_cur - (learning_rate * b_gradient)
    new_m = m_cur - (learning_rate * m_gradient)

    return [new_b, new_m]


def error(x, y, m, b):
    return (y - m * x - b) ** 2
'''


# 方法3：sklearn中方法
def fun_skl():
    data = init_data()
    x = data[:, 1]
    y = data[:, 2]
    x = (x.reshape(-1, 1))
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    # 截距，系数
    print('调用库函数结果: {} {}'.format(lin_reg.intercept_, lin_reg.coef_))


# 方法4：矩阵表示 公式解 weight = (x.T * x).I * x.T * y
# 2x41 41x2 2x41 41x1
# 有局限：要求 x 是列满秩矩阵，XTX 是满秩矩阵 可逆
def func_():
    data = init_data()
    x = np.mat(data[:, 0:2])
    y = np.mat(data[:, 2]).T
    XTX = x.T * x
    if np.linalg.det(XTX) == 0.0:  # 判断矩阵是否可逆，|det| != 0 ?
       print('This matrix is singular, cannot do inverse')
    else:
        weights = XTX.I * (x.T * y)
        print('公式法结果: ', weights)


def plot_data(data, b, m):
    x = data[:, 1]
    y = data[:, 2]
    y_pre = m * x + b
    plt.plot(x, y, 'o')
    plt.plot(x, y_pre, 'r')
    plt.show()


if __name__ == '__main__':
    linear_regression()
    fun_skl()
    func_()
