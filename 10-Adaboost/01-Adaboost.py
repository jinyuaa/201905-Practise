# 使用 sklearn库的 AdaBoostClassifier方法进行实现

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


# 用 make_gaussian_quantiles生成多组多维正态分布的数据
# 这里生成2维正态分布，设定样本数1000，协方差2
x1, y1 = make_gaussian_quantiles(cov=2., n_samples=200, n_features=2, n_classes=2, shuffle=True, random_state=1)
# 为了增加样本分布的复杂度，再生成一个数据分布
x2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, shuffle=True, random_state=1)
# 合并
X = np.vstack((x1, x2))
y = np.hstack((y1, 1-y2))
# plt.scatter(X[:,0],X[:,1],c=Y)
# plt.show()

# 设定弱分类器CART 决策树的最大深度：max_depth=1
weakClassifier = DecisionTreeClassifier(max_depth=1)

# 构建模型
# 参数：base_estimator 基分类器 默认是决策树;
# algorithm 模型提升准则 两种 SAMME(对样本集错误概率) SAMME.R(对样本集错误比例) 区别是对弱分类器权重的衡量;
# n_estimators 基分类器提升(循环)次数 默认50 值过大容易过拟合 值过小容易欠拟合
# learning_rate 学习率 梯度收敛速度 默认1
clf = AdaBoostClassifier(base_estimator=weakClassifier, algorithm='SAMME', n_estimators=300, learning_rate=0.8)
clf.fit(X, y)

# 绘制分类效果
x1_min = X[:, 0].min() - 1
x1_max = X[:, 0].max() + 1
x2_min = X[:, 1].min() - 1
x2_max = X[:, 1].max() + 1
x1_, x2_ = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))

y_ = clf.predict(np.c_[x1_.ravel(), x2_.ravel()])
y_ = y_.reshape(x1_.shape)
plt.contourf(x1_, x2_, y_, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
