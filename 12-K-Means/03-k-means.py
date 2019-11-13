# 按照原理另一种实现 k-means 算法  数据data.txt
"""
与01区别：(1)聚类中心初始值 随机为样本数据点(索引)
         (2)画图 保存成文件形式 并且迭代的每一帧显示图过程保存成 gif 动图
"""


import numpy as np
import matplotlib.pyplot as plt
import random as rd
import imageio

data = np.loadtxt('data.txt', delimiter='\t')


# 计算平面两点的欧氏距离
step = 0
color = ['.r', '.g', '.b', '.y']  # 颜色种类
dcolor = ['sr', 'sg', 'sb', 'sy']  # 颜色种类
frames = []


# 欧式距离的平方
def distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# K均值算法
def k_means(x, y, k_count):
    count = len(x)  # 点的个数  80
    # 随机选择 k个点
    k = rd.sample(range(count), k_count)
    k_point = [[x[i], [y[i]]] for i in k]  # 保证有序
    k_point.sort()
    global frames
    global step
    while True:
        km = [[] for i in range(k_count)]  # 存储每个簇的索引

        # 遍历所有点
        for i in range(count):
            cp = [x[i], y[i]]  # 当前点
            # 计算cp点到所有质心的距离
            _sse = [distance(k_point[j], cp) for j in range(k_count)]
            # cp点到那个质心最近
            min_index = _sse.index(min(_sse))
            # 把cp点并入第i簇
            km[min_index].append(i)
        # 更换聚类中心
        step += 1
        k_new = []
        for i in range(k_count):
            _x = sum([x[j] for j in km[i]]) / len(km[i])
            _y = sum([y[j] for j in km[i]]) / len(km[i])
            k_new.append([_x, _y])
        k_new.sort()  # 排序

        # 使用 Matplotlab画图
        plt.figure()
        plt.title("N=%d,k=%d  iteration:%d" % (count, k_count, step))
        for j in range(k_count):
            plt.plot([x[i] for i in km[j]], [y[i] for i in km[j]], color[j % 4])
            plt.plot(k_point[j][0], k_point[j][1], dcolor[j % 4])
        plt.savefig("03.jpg")
        frames.append(imageio.imread('03.jpg'))
        if k_new != k_point:  # 一直循环直到聚类中心没有变化
            k_point = k_new
        else:
            return km


x, y = np.loadtxt('data.txt', delimiter='\t', unpack=True)
k_count = 4

# 聚类结果
km = k_means(x, y, k_count)
print(km)
print("迭代次数为：" + str(step))

imageio.mimsave('k-means.gif', frames, 'GIF', duration=2)