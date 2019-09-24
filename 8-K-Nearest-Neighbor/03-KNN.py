# 带标记的训练集(分布、可视化) 新样本进来调用 KNN 自定义KNN函数
# KNN：定K值 距离函数  新样本与训练样本的距离，选前K个最近的点，其所属类别计数按大小排列
# 返回max的类别数即为新样本的类别

import numpy as np
from math import sqrt
import operator as opt
import matplotlib.pyplot as plt


# 对数据进行最大最小标准化 = (x - min) / (max - min)
def normal_data(dataset):
    maxvals = dataset.max(axis=0)
    minvals = dataset.min(axis=0)
    ranges = maxvals - minvals
    retdata = (dataset - minvals) / ranges
    return retdata, ranges, minvals


# 定义 K近邻算法 利用标准化之后的数据进行计算
def knn(dataset, label, testdata, k):
    # 欧氏距离
    # 计算差值的平方
    distsquaremat = (dataset - testdata) ** 2
    # 求每一行的差值平方和
    distsquaresum = distsquaremat.sum(axis=1)
    # 开根号，得出每个样本到测试点的距离
    distances = distsquaresum ** 0.5
    # 排序，得到排序后的下标
    sortedindices = distances.argsort()
    # 取最小的 K个
    indices = sortedindices[:k]
    print(indices)
    # 存储每个label的出现次数
    labelcount = {}
    for i in indices:
        label = label[i]
        print(label)
        # 次数 + 1
        labelcount[label] = labelcount.get(label, 0) + 1
    # 对label出现的次数从大到小进行排序
    sortedcount = sorted(labelcount.items(), key=opt.itemgetter(1), reverse=True)
    # 返回出现最大次数的 label
    return sortedcount[0][0]


if __name__ == "__main__":
    dataset = np.array([[2, 3], [6, 8], [4, 5], [3, 4], [5, 9]])
    # print(dataset)
    normaldata, ranges, minvals = normal_data(dataset)
    # print(normaldata, ranges, minvals)
    labels = ['a', 'b', 'b', 'a', 'b']
    testdata = np.array([3.9, 5.5])
    normaltestdata = (testdata - minvals) / ranges
    result = knn(normaldata, labels, normaltestdata, 1
                 )
    print(result)


'''
# 用 make_blobs生成分类数据
# 分类可视化 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 生成样本数为500，分类数为5的数据集
# make_blobs常用来生成聚类算法的测试数据
# 生成的样本点数为500，每个样本的特征数为2，类别数为5，每类的方差为 1
data = make_blobs(n_samples=500, n_features=2, centers=5, cluster_std=1.0, random_state=8)
X, Y = data

# 将生成的数据集进行可视化
plt.scatter(X[:, 0], X[:, 1], s=80, c=Y, cmap=plt.cm.spring, edgecolors='k')
# plt.legend(['0', '1', '2', '3', '4'], loc='upper left') 不能添加图例？
plt.show()

clf = KNeighborsClassifier()
clf.fit(X, Y)

# 绘制图形
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

z = z.reshape(xx.shape)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Pastel1)
plt.scatter(X[:, 0], X[:, 1], s=80, c=Y, cmap=plt.cm.spring, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classifier:KNN")

# 把待分类的数据点用五星表示出来
x, y = 2.5, 10
plt.scatter(x, y, marker='*', c='red', s=200)

# 对待分类的数据点的分类进行判断
res = clf.predict([[x, y]])
# plt.text(0.2, 4.6, 'Classification flag: '+ str(res))
# plt.text(3.75, -13, 'Model accuracy: {:.2f}'.format(clf.score(X, Y)))

plt.show()
print("测试样本{}属于{}类".format([x, y], str(res)))
print("模型的准确率为{}".format(clf.score(X, Y)))
'''