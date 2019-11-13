# 按照原理实现 k-means 算法 数据data.txt
"""
算法流程：
(1) 首先定义样本和每个聚类中心的相似性度量：欧氏距离(平方)
(2) 随机初始化 k个聚类中心的值  center = min+rand(0,1)*(max-min)
(3) 初始化 sub_center(80, 2) 全0矩阵
(3a) 分别计算每个样本数据和 k个聚类中心的欧氏距离平方 并根据 min 原则  存储该样本所属类别和最小距离  更新 sub_center
(3b) 统计每个聚类簇中样本的个数  计算这些样本坐标的和[x_sum, y_sum] / 个数 = 坐标均值 [x_mean, y_mean] 即为更新的 center 循环
(4) 直到 center坐标不在变换 统计迭代次数 输出聚类中心的值和结果
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.txt', delimiter='\t')


# 定义欧式距离的平方
def distance(vec_a, vec_b):
    dist = (vec_a - vec_b) * (vec_a - vec_b).T
    return dist[0, 0]


# 随机初始化聚类中心
def random_center(data, k):
    n = np.shape(data)[1]               # 特征个数 n=2
    center = np.mat(np.zeros((k, n)))     # 初始化 k 个聚类中心 (4,2)
    for j in range(n):                  # 初始化聚类中心的每一维的坐标
        min_j = np.min(data[:, j])
        range_j = np.max(data[:, j]) - min_j
        center[:, j] = min_j * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * range_j
        #            1,1      4,1       4,1   1,1  cent=min+rand(0,1)*(max-min)
    return center   # (4,2) 矩阵


# 参数：data 数据 k 聚类中心点的个数 cent 随机初始化的距离类中心
# 返回：cent 训练完成的聚类中心 sub_certer 每个样本所属类别
def k_means(data, k, center):
    m, n = np.shape(data)  # 特征维度  m=80, n=2
    sub_center = np.mat(np.zeros((m, 2)))   # 初始化每个样本所属类别
    change = True       # 判断是否需要重新计算聚类中心
    num = 0
    while change == True:
        change = False    # 重置
        for i in range(m):
            min_distance = np.inf   # 设置样本与聚类中心的最小距离，初始值为无穷大
            min_index = 0       # 所属类别
            for j in range(k):   # 分别计算每个样本数据和 k个聚类中心的距离
                dist = distance(data[i, :], center[j, :])
                if dist < min_distance:
                    min_distance = dist
                    min_index = j
            # ??
            # 判断是否需要改变
            if sub_center[i, 0] != min_index:  # 需要改变
                change = True
                sub_center[i, :] = np.mat([min_index, min_distance])
        # 标记
        num += 1

        # 重新计算聚类中心
        for j in range(k):
            sum_all = np.mat(np.zeros((1, n)))
            r = 0     # 每个类别中的样本数
            for i in range(m):
                if sub_center[i, 0] == j:
                    sum_all += data[i, :]
                    r += 1
            for z in range(n):
                try:
                    center[j, z] = sum_all[0, z] / r
                except:
                    print("division by zero!")
    return sub_center, center, num


if __name__ == "__main__":
    k = 4
    centroids = random_center(data, k)
    sub_center, center, num = k_means(data, k, centroids)
    print("迭代的次数为：" + str(num))
    # 将矩阵转化为数组
    sub_center = np.array(sub_center)
    center = np.array(center)

    y_label = sub_center[:, 0]

    # 画图
    plt.scatter(data[:, 0], data[:, 1], c=y_label, cmap='Dark2', alpha=0.5, s=30, marker='x')
    plt.scatter(center[:, 0], center[:, 1], c=[0, 1, 2, 3], cmap='Dark2', s=70, marker='s')
    plt.show()
