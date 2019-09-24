# 梯度下降法求多元函数 已知 x0^5 + e^x1 + x0^3 + x0 + x1 - 5 = 0 求 x0 和 x1

import numpy as np


def problem(x):
    e = 2.71828182845
    # return x[0]**5 + e**x[1] + x[0]**3 + x[0] + x[1] - 5
    return x[0]*x[1] + x[0]*(x[2]**2) + x[1] + x[1]**3 + x[2] + x[2]**5 + x[3] + x[3]**7 - 15


def error(x):
    # x = [x0, x1] 为一组数
    return (problem(x) - 0)**2

'''
def gradient_descent(x):
    delta = 0.00000001
    # 导数的定义 f'(x0)=lim x0->0 [f(x0+alpha)-f(x0-alpha)] /2*alpha
    derivative_x0 = (error([x[0] + delta, x[1]]) - error([x[0] - delta, x[1]])) / (delta * 2)
    derivative_x1 = (error([x[0], x[1] + delta]) - error([x[0], x[1] - delta])) / (delta * 2)
    # 步长
    alpha = 0.01
    # 迭代更新
    x[0] = x[0] - derivative_x0 * alpha
    x[1] = x[1] - derivative_x1 * alpha
    return [x[0], x[1]]
'''


# 用 numpy 实现
def gd(x):
    delta = 0.00000001
    # 梯度向量
    gradient = np.zeros(x.shape)

    for i in range(len(gradient)):
        deltavector = np.zeros(x.shape)
        deltavector[i] = delta

        gradient[i] = (error(x+deltavector) - error(x-deltavector)) / (delta * 2)

    # alpha = 0.001
    alpha = 0.001
    # 迭代更新
    x = x - gradient * alpha
    return x


if __name__ == "__main__":
    # x = [0.0, 0.0]
    x = np.array([0.0, 0.0, 0.0, 0.0])
    for i in range(50):
        # x = gradient_descent(x)
        # print('x = {:6f}, {:6f}, problem(x) = {:6f}'.format(x[0], x[1], problem(x)))
        x = gd(x)
        print('x = {}, problem(x) = {:6f}'.format(x, problem(x)))
