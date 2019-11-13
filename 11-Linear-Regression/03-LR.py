# 实现 线性回归、多项式回归、岭回归和 Lasso 回归
# 数据源：随机生成一组测试用例 : x**2 + 2*x + 3  并加入高斯噪声

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso


x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = x**2 + 2 * x + 3 + np.random.normal(0, 1, 100)
plt.scatter(x, y)
plt.title('original data')
plt.figure()

# 使用线性回归     y=_x+_b
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict = lin_reg.predict(X)
plt.title('linear-regression')
plt.scatter(x, y)
plt.plot(X, y_predict, color='r')
plt.figure()

# 在上述基础上添加一个特征 X**2     y=_x**2+_x+_b
# 原来所有的数据都在 X 中 现在对每个 X 中每个数据都进行平方，再将得到数据集与原来数据集拼接，用新的数据集进行线性回归
# 形成一条曲线，对原来的数据拟合程度更好的(相比线性回归)
X2 = np.hstack([X, X**2])
lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)
plt.title('add x**2 feature-linreg')
plt.scatter(x, y)
# x是乱序的，进行排序
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
plt.figure()
# lin_reg2.coef_第一个系数是x的系数，第二个是x**2系数
print(lin_reg2.coef_)
# 常数项
print(lin_reg2.intercept_)


# 采用多项式回归
poly = PolynomialFeatures(degree=2)
poly.fit(X, y)
X3 = poly.transform(X)

lin_reg3 = LinearRegression()
lin_reg3.fit(X3, y)
y_predict3 = lin_reg3.predict(X3)

# 与上述拟合系数一致
print(lin_reg3.coef_)
print(lin_reg3.intercept_)
plt.title('polynomial-regression')
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict3[np.argsort(x)], c='r')
plt.show()


# 岭回归
def ridge_regression(degree, alpha):
    return Pipeline([("poly", PolynomialFeatures(degree=degree)),
                     ("std_scaler",StandardScaler()),
                     ("ridge_reg",Ridge(alpha=alpha))
                    ])


def plot_module(module):
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_plot = module.predict(X_plot)
    plt.scatter(x, y)
    plt.plot(X_plot[:, 0], y_plot, color='r')
    plt.axis([-3.2, 3.2, -1.2, 20])
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y)
ridge1_reg = ridge_regression(degree=20, alpha=0.001)
ridge1_reg.fit(X_train, y_train)
ridge1_predict = ridge1_reg.predict(X_test)
mean_squared_error(y_test, ridge1_predict)
plt.title('ridge-regression')
plot_module(ridge1_reg)


# Lasso回归
def lasso_regression(degree,alpha):
    return Pipeline([("poly", PolynomialFeatures(degree=degree)),
                     ("std_scatter", StandardScaler()),
                     ("lasso_reg", Lasso(alpha=alpha))
                    ])


lasso1_reg = lasso_regression(degree=20, alpha=0.01)
lasso1_reg.fit(X_train, y_train)
lasso1_predict = lasso1_reg.predict(X_test)
mean_squared_error(lasso1_predict, y_test)
plt.title('lasso-regression')
plot_module(lasso1_reg)
