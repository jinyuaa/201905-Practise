# 使用 封装好的函数
# 拟合出分类直线
# 需要对数据进行 特征样本归一化

from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt


# 同一个样本，可能具备不同类型的特征，各特征的数值大小范围不一致 变到一个量级
# 特征归一化 默认将每种特征值都归一化到[0,1]之间，下式则是[-1,1]之间
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
df = pd.read_csv('data1.csv', header=0)
# df 类型 <class 'pandas.core.frame.DataFrame'>
# print(df)
df.columns = ["grade1", "grade2", "label"]
# ------方法1
# df.iloc 基于位置选择的纯整数位置索引，.values返回 DataFrame 的 Numpy 表示形式
# 选择 第0列和第1列  [0,1] 注意区分 [0:1]
# X = df.iloc[:, [0, 1]].values
# y = df.iloc[:, 2].values
# ------方法2
# 取 "grade1", "grade2" 这两列数据  转化为 numpy.ndarray 类型
x = df[["grade1", "grade2"]]
x = np.array(x)
y = df["label"]
y = np.array(y)

x = min_max_scaler.fit_transform(x)
# standard_scale = preprocessing.StandardScaler()
# x = standard_scale.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# feature scaling
# standard_scale = preprocessing.StandardScaler()
# x_train = standard_scale.fit_transform(x_train)
# x_test = standard_scale.transform(x_test)

# 将逻辑回归拟合到训练集
clf = LogisticRegression()
clf.fit(x_train, y_train)
# 预测测试集结果  88%
print("预测准确率为： ", clf.score(x_test, y_test))
print(clf.coef_)
print("迭代的次数为：", clf.n_iter_)

theta = clf.coef_
b = clf.intercept_
# 分类线为 theta0 * x1 + theta1 * x2 + b = 0
x_plot = np.arange(-1, 1, 0.1)
y_plot = - theta[0, 0] / theta[0, 1] * x_plot - b / theta[0, 1]
print(b / theta[0, 1])
plt.plot(x_plot, y_plot, c='g')


positive = np.where(y == 1)
negative = np.where(y == 0)
plt.scatter(x[positive, 0], x[positive, 1], marker='o', c='b')
plt.scatter(x[negative, 0], x[negative, 1], marker='+', linewidth=2, c='r')
plt.xlabel("grade1")
plt.ylabel("grade2")
plt.legend(['sign 1', 'sign 0'])
plt.title("Logistic Regression")
plt.show()



'''
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                     np.arange(start=X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))

# print(X1, X2)
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('black', 'blue'))(i), label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
'''