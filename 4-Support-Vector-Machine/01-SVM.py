# sklearn.datasets自带数据集 Iris鸢尾花植物 数据中没有缺失值和异常值
# 3类数据的4个特征  非线性多分类
# 3类为[Iris-Setosa, Iris-Versicolour, Iris-Virginica]
# 4类特征为[sepal length, sepal width, petal length, petal width](cm)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# 1、加载鸢尾花数据集
iris = datasets.load_iris()
# dict_keys(['data','target','DESCR','target_names','feature_names'])
# print(iris.keys())
# 取所有行，为了方便绘图仅选择两个特征，第1列第2列
X = iris.data[:, :2]
print(X)
y = iris.target
print(y)

# 测试样本（绘制分类区域）
# np.linspace(X,Y,N)在X和Y之间产生N个等间距的数列
xlist1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
xlist2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
# np.meshgrid()从一个坐标向量中返回一个坐标矩阵
# xlist1变成XGrid1的行向量, xlist2变成XGrid2的列向量
# XGrid1和XGrid2的维数是一样的，是网格矩阵也就是坐标矩阵
XGrid1, XGrid2 = np.meshgrid(xlist1, xlist2)

# 2、搭建模型，训练SVM分类器
# 非线性SVM：RBF核，超参数为0.5，正则化系数为1，SMO迭代精度1e-5, 内存占用1000MB
# 用全部数据进行训练，也可将数据划分为训练集和测试集
svc = svm.SVC(kernel='rbf', C=1, gamma=0.5, tol=1e-5, cache_size=1000).fit(X, y)
# 预测并绘制结果
# ravel()函数是将矩阵变成一个一维的数组
Z = svc.predict(np.vstack([XGrid1.ravel(), XGrid2.ravel()]).T)
Z = Z.reshape(XGrid1.shape)

# 3、画图
plt.contourf(XGrid1, XGrid2, Z, cmap=plt.cm.hsv)
plt.contour(XGrid1, XGrid2, Z, colors=('k',))
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.title('Iris SVM-two feature', fontsize=15)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', linewidth=1, cmap=plt.cm.hsv)
plt.show()