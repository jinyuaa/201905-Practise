# 统计学习-李航 书中例子
# 数据集为
# x  0  1  2  3  4  5  6  7  8  9
# y  1  1  1 -1 -1 -1  1  1  1  -1

# 弱分类器规则，G(x)= 1  , x < target
#                   -1 , x > target

# 1、得到样本，计算错误率最小找出弱分类器的 阈值 target=flag
# 2、就能计算 错误率，计算此分类器的权重
# 3、更新样本权重，重复 1
# 4、直到误差率为 0 时，停止弱分类器训练

import numpy as np
import matplotlib.pyplot as plt

value_enable = 1
value_unable = -1


# 定义基础分类器
# 返回的是 误差率 最小值 在此最下的分类器阈值 target
def base_model(data, weight_list, times, boundary):
    m = data.shape[0]
    target = 0
    min_error = 1
    for i in range(m):
        error = error_ratio(data, i, weight_list, times, boundary)
        print('阈值为{}, 误差率为 {}'.format(i, error))
        if error < min_error:
            target = i
            min_error = error

    print('最佳分类阈值为：{}，误差率为：e={}'.format(target + 0.5, min_error))
    return target, min_error


# 计算错误率
# 输入是样本数据 和 i, 样本权重系数，输出的是 错误样本的权重之和
def error_ratio(data, target, weight_list, times, boundary):
    m = data.shape[0]
    error_num = []
    for i in range(m):
        if times <= boundary:
            error = (data[i][0] <= target and data[i][1] != value_enable) \
                    or (data[i][0] > target and data[i][1] != value_unable)

        else:
            error = (data[i][0] <= target and data[i][1] != value_unable) \
                    or (data[i][0] > target and data[i][1] != value_enable)

        if error:
            error_num.append(weight_list[i])
    err_sum = np.sum(error_num)
    return err_sum


# 计算更新权重 target=flag
# 输入数据为 data 原始数据; min_err 最下误差率; target 弱分类器阈值; w 样本的权重
def ada_boost(data, min_err, target, weight_list, times, boundary):
    m = data.shape[0]  # m = 10
    weight_result = []
    # 当前分类器的权重系数
    alpha = 0.5 * np.log((1 - min_err) / min_err)
    # 样本权重更新
    factor_sum = get_factor_sum(times, alpha, data, m, target, weight_list, boundary)
    for i in range(m):
        if times <= boundary:
            if data[i][0] <= target:
                Gx = value_enable
            else:
                Gx = value_unable
        else:
            if data[i][0] <= target:
                Gx = value_unable
            else:
                Gx = value_enable

        weight = weight_list[i] * np.exp(-data[i][1] * alpha * Gx) / factor_sum
        weight_result.append(weight)
    print('当前分类器权重：{}，样本权重：w={}'.format(alpha, weight_result))
    return weight_result, alpha


# 求权重因子
def get_factor_sum(times, alpha, data, m, target, weight_list, boundary):
    factor_sum = []
    for i in range(m):
        if times <= boundary:
            if data[i][0] <= target:
                Gx = value_enable
            else:
                Gx = value_unable
        else:
            if data[i][0] <= target:
                Gx = value_unable
            else:
                Gx = value_enable
        factor = weight_list[i] * np.exp(-data[i][1] * alpha * Gx)
        factor_sum.append(factor)
    factor_sum = np.sum(factor_sum)
    return factor_sum


def ada_boost_result(data, times, boundary):
    alpha_array = []
    target_array = []
    weight = np.ones(data.shape[0]) * (1 / data.shape[0])
    print('w:', weight)
    for i in range(times):
        targets, min_error = base_model(data, weight, i + 1, boundary)
        weight, alphas = ada_boost(data, min_error, targets, weight, i + 1, boundary)
        print('第{}个分类器的权重为{}'.format(i + 1, alphas))
        alpha_array.append(alphas)
        target_array.append(targets)
    return alpha_array, target_array


if __name__ == '__main__':
    data = np.array([[0, 1], [1, 1], [2, 1], [3, -1], [4, -1], [5, -1],
                     [6, 1], [7, 1], [8, 1], [9, -1]], dtype=np.int)
    print('样本数据为：', data)
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 1], cmap=plt.cm.Paired)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    alpha_result, target_result = ada_boost_result(data, 3, 2)

    # 最终分类器

    Y_pre = []
    print('y的真实值为：', data[:, 1])
    for x in range(data.shape[0]):
        # G1(X)
        if x <= target_result[0]:
            G1_X = value_enable
        else:
            G1_X = value_unable
        # G2(x)
        if x <= target_result[1]:
            G2_X = value_enable
        else:
            G2_X = value_unable
        # G3(x)
        if x > target_result[2]:
            G3_X = value_enable
        else:
            G3_X = value_unable
        y_pre = np.sign(alpha_result[0] * G1_X + alpha_result[1] * G2_X + alpha_result[2] * G3_X)
        Y_pre.append(y_pre)
    print('y的预测值为：', Y_pre)
