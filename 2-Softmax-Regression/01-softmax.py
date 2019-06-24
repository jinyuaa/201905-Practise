# softmax 回归实例
# sklearn.datasets 自带的小数据集 手写体数据集 y:{0 1 2 3 4 5 6 7 8 9}
# (1797，64) data 8x8大小

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def load_data():
    digits = load_digits()
    x_data = digits.data
    y_label = digits.target
    return np.mat(x_data), y_label


# data, label = load_data()
# print(data[0, :].reshape(-1, 8))
# print(label[0])


def gradient_descent(train_x, train_y, k, maxCycle, alpha):
    # k 为类别数
    num_samples, num_features = np.shape(train_x)
    weights = np.mat(np.ones((num_features, k)))
    i = 0
    for i in range(maxCycle):
        value = np.exp(train_x * weights)
        # if i % 100 == 0:
        # print("----iter: ", i, ", cost: ", cost(value, train_y))
        rowsum = value.sum(axis=1)  # 横向求和
        rowsum = rowsum.repeat(k, axis=1)  # 横向复制扩展
        err = - value / rowsum  # 计算出每个样本属于每个类别的概率
        for j in range(num_samples):
            err[j, train_y[j]] += 1
        weights = weights + (alpha / num_samples) * (train_x.T * err)

    return weights




'''
def cost(err, train_y):
    m = np.shape(err)[0]
    sum_cost = 0.0
    for i in range(m):
        if err[i, train_y[i, 0]] / sum_cost(err[i, :]) > 0:
            sum_cost -= np.log(err[i, train_y[i, 0]] / np.sum(err[i, :]))
        else:
            sum_cost -= 0
    return sum_cost / m

'''


def test_model(test_x, test_y, weights):
    results = test_x * weights
    predict_y = results.argmax(axis=1)
    count = 0
    for i in range(np.shape(test_y)[0]):
        if predict_y[i, ] == test_y[i, ]:
            count += 1
    return count / len(test_y), predict_y


if __name__ == "__main__":
    data, label = load_data()
    # data = preprocessing.minmax_scale(data, axis = 0)
    # 数据处理之后识别率降低了
    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0)
    k = len(np.unique(label))
    print("k= ", k)

    weights = gradient_descent(train_x, train_y, k, 800, 0.01)
    # w (64,10)
    print(weights.shape)
    accuracy, predict_y = test_model(test_x, test_y, weights)
    print("Accuracy:", accuracy)

    print(test_x[2, :].reshape(8, 8))
    print(test_y[2])
    A = test_x[2, :] * weights
    p = np.exp(A) / np.sum(np.exp(A))
    print(type(p))
    print("预测值为：", p.argmax(axis=1))
