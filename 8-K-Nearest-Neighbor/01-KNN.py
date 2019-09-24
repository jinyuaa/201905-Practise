# KNN 实现分类
# 使用 scikit-learn 自带的手写识别数据集 Digit Dataset
# 该数据集由 1797张样本图片组成，每张图片都是 8x8 大小的手写数字位图
# scikit-learn中提供了一个 KNeighborsClassifier 类来实现 K 近邻法分类模型
# 在 scikit-learn中 KNN算法的 K 值是通过 n_neighbors 参数来调节的，默认值是 5

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split


# 加载数据集
def load_data():
    digits = datasets.load_digits()
    X_train = digits.data
    y_train = digits.target
    return train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


def test_knn_classifier(*data):
    x_train, x_test, y_train, y_test = data
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    print("Train Score: %f " % clf.score(x_train, y_train))
    print("Testing Score: %f " % clf.score(x_test, y_test))


# sklearn.neighbors.KNeighborsClassifier类实现 K 近邻法分类模型
# 分析函数在使用不同的投票策略['uniform', 'distance']时，随 K 增长，分类器预测性能变化
# k 值及投票策略对预测性能的影响 定义测试函数
def test_k_w(* data):
    x_train, x_test, y_train, y_test = data
    K = np.linspace(1, y_train.size, num=100, endpoint=False, dtype='int')
    print(K)
    weights = ['uniform', 'distance']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for weight in weights:
        training_score = []
        testing_score = []
        for k in K:
            clf = neighbors.KNeighborsClassifier(weights=weight, n_neighbors=k)
            clf.fit(x_train, y_train)
            testing_score.append(clf.score(x_test , y_test))
            training_score.append(clf.score(x_train, y_train))
        ax.plot(K, testing_score, label="testing score: weight=%s" % weight)
        ax.plot(K, training_score, label="training score: weight=%s" % weight)
    ax.legend(loc='best')
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNN-Classifier")
    plt.show()


# 考察 p 值 (即距离函数的形式)对于预测性能的影响
# p=1 对应曼哈顿距离，p=2 对应欧拉距离
def test_k_p(* data):
    x_train, x_test, y_train, y_test = data
    K = np.linspace(1, y_train.size, endpoint=False, dtype='int')
    P = [1, 2, 10]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for p in P:
        training_score = []
        testing_score = []
        for k in K:
            clf = neighbors.KNeighborsClassifier(p=p, n_neighbors=k)
            clf.fit(x_train, y_train)
            testing_score.append(clf.score(x_test , y_test))
            training_score.append(clf.score(x_train, y_train))
        ax.plot(K, testing_score, label="testing score: p=%s" % p)
        ax.plot(K, training_score, label="training score: p=%s" % p)
    ax.legend(loc='best')
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNN-Classifier")
    plt.show()


x_train, x_test, y_train, y_test = load_data()
test_knn_classifier(x_train, x_test, y_train, y_test)
test_k_w(x_train, x_test, y_train, y_test)
test_k_p(x_train, x_test, y_train, y_test)
