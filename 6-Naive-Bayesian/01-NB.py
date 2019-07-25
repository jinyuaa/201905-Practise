# 使用的数据集是 scikit-learn 自带的手写识别数据集 Digit Dataset
# 在scikit-learn库中有贝叶斯的程序包，常用的朴素贝叶斯分类器有：
# 1、高斯贝叶斯分类器，2、多项式贝叶斯分类器，3、伯努利贝叶斯分类器

from sklearn import datasets, naive_bayes
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    digits = datasets.load_digits()
    data = digits.data
    label = digits.target
    return train_test_split(data, label, test_size=0.25, random_state=0)


def test_GaussianNB():
    train_x, test_x, train_y, test_y = load_data()
    cls = naive_bayes.GaussianNB()
    cls.fit(train_x, train_y)
    print("Training Score: %.2f" % cls.score(train_x, train_y))
    print("Testing Score: %.2f" % cls.score(test_x, test_y))


def test_MultinomialNB():
    train_x, test_x, train_y, test_y = load_data()
    cls = naive_bayes.MultinomialNB()
    cls.fit(train_x, train_y)
    print("Training Score: %.2f" % cls.score(train_x, train_y))
    print("Testing Score: %.2f" % cls.score(test_x, test_y))


# 检验不同 alpha对多项式贝叶斯分类器的预测性能的影响
def test():
    train_x, test_x, train_y, test_y = load_data()
    alphas = np.logspace(-2, 5, num=200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        cls = naive_bayes.MultinomialNB(alpha=alpha)
        cls.fit(train_x, train_y)
        train_scores.append(cls.score(train_x, train_y))
        test_scores.append(cls.score(test_x, test_y))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, train_scores)
    ax.plot(alphas, test_scores)
    ax.legend(['Training Score', 'Testing Score'], loc='lower right')
    ax.set_xlabel("alpha")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("MultinomialNB")
    ax.set_xscale("log")
    plt.show()


def test_BernoulliNB():
    train_x, test_x, train_y, test_y = load_data()
    cls = naive_bayes.BernoulliNB()
    cls.fit(train_x, train_y)
    print("Training Score: %.2f" % cls.score(train_x, train_y))
    print("Testing Score: %.2f" % cls.score(test_x, test_y))


if __name__ == "__main__":
    print("使用高斯贝叶斯分类器结果：")
    test_GaussianNB()
    print("使用多项式贝叶斯分类器结果：")
    test_MultinomialNB()
    print("使用伯努利贝叶斯分类器结果：")
    test_BernoulliNB()

    test()

