# KNN 实现回归
# 随机生成了 100 个 样本数据

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split


# 加噪
def create_data(n):
    X = 5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    y[::5] += 1 * (0.5 - np.random.rand(int(n/5)))
    return train_test_split(X, y, test_size=0.25, random_state=0)


def test_knn_regression(*data):
    x_train, x_test, y_train, y_test = data
    reg = neighbors.KNeighborsRegressor()
    reg.fit(x_train, y_train)
    print("Train Score: %f " % reg.score(x_train, y_train))
    print("Testing Score: %f " % reg.score(x_test, y_test))


x_train, x_test, y_train, y_test = create_data(1000)
test_knn_regression(x_train, x_test, y_train, y_test)
