import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
data = np.loadtxt('data1.txt', delimiter="\t")
X = data[:, 0:2]
print(X.shape)
y = data[:, 2]
# y数据中等于1的下标索引, tuple元组类型
positive_index = np.where(y == 1)
negative_index = np.where(y == 0)
plt.scatter(X[positive_index, 0], X[positive_index, 1], marker='o', c='b')
plt.scatter(X[negative_index, 0], X[negative_index, 1], marker='x', c='r')
plt.xlabel("feature_1")
plt.ylabel("feature_2")
plt.legend(["1", "0"])
plt.show()
'''


def load_data(filename):
    # 读取数据 DataFrame 格式
    df = pd.read_csv(filename, sep="\t", names=['1', '2', 'target'])
    print(type(df))
    data = []
    target = []
    for i in range(len(df)):
        d = df.iloc[i]
        ds = [d['1'], d['2']]
        t = int(d['target'])
        if t == 0:
            t = -1
        data.append(ds)
        target.append(t)
    return np.mat(data), np.mat(target).tolist()[0]


# 计算准确度
def get_accuracy(prediction, classlabel):
    score = 0
    for i in range(len(prediction)):
        if prediction[i] > 0.5:
            prediction[i] = 1
        else:
            prediction[i] = -1
        if prediction[i] == classlabel[i]:
            score += 1
    print("Accuracy: ", score/len(prediction))


# 初始化权重 w 和交叉项权重 v
def initialize(n, k):
    v = np.mat(np.zeros((n, k)))
    for i in range(n):
        for j in range(k):
            # 把 v 中的值变为服从 N(0, 0.2) 的正态分布数值
            v[i, j] = np.random.normal(0, 0.2)
    print("v的type: ", v)
    return v


def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))


# 定义误差损失函数 loss(y', y)= sum(-ln[sigmoid(y'*y)])
def get_cost(predict, classlables):
    m = len(predict)
    error = 0.0
    for i in range(m):
        error -= np.log(sigmoid(predict[i]*classlables[i]))
    return error


# 定义预测结果函数
def get_prediction(datamatrix, w0, w, v):
    m = np.shape(datamatrix)[0]
    result = []
    for x in range(m):
        inter_1 = datamatrix[x] * v
        inter_2 = np.multiply(datamatrix[x], datamatrix[x]) * np.multiply(v, v)
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2
        p = w0 + datamatrix[x] * w + interaction
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result


# 用梯度下降法求解模型参数 训练 FM 模型
# 参数(输入数据集特征,对应标签,交叉项矩阵维度,最大迭代次数,学习率)
def stoc_grad_descent(datamatrix, classlabels, k, max_iter, alpha):
    # initialize parameters
    m, n = np.shape(datamatrix)
    w = np.zeros((n, 1))
    w0 = 0
    # 初始化参数
    v = initialize(n, k)
    # training
    for it in range(max_iter):
        for x in range(m):
            # 1xn nxk  1xk(inter_1)
            # 随机优化，对每个样本进行
            inter_1 = datamatrix[x] * v
            # 对应元素相乘
            inter_2 = np.multiply(datamatrix[x], datamatrix[x])*np.multiply(v, v)
            # 交叉项
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2)/2
            # 计算预测输出
            p = w0 + datamatrix[x]*w + interaction
            loss = sigmoid(classlabels[x] * p[0, 0]) - 1
            w0 = w0 - alpha*loss*classlabels[x]
            for i in range(n):
                if datamatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha*loss*classlabels[x]*datamatrix[x, i]
                for j in range(k):
                    v[i, j] = v[i, j] - alpha*loss*classlabels[x]*(datamatrix[x, i]*inter_1[0, j]-v[i, j]*datamatrix[x, i]*datamatrix[x, i])
        if it % 1000 == 0:
            print('-----iter: ', it, ', cost: ', get_cost(get_prediction(np.mat(datamatrix), w0, w, v), classlabels))
            get_accuracy(get_prediction(np.mat(datamatrix), w0, w, v), classlabels)
    return w0, w, v


if __name__ == '__main__':
    '''
    data = np.loadtxt('data1.txt', delimiter="\t")
    data_matrix = data[:, 0:2]
    print(type(data_matrix))
    target = data[:, 2]
    '''
    # 1、导入数据
    data_matrix, target = load_data('data1.txt')
    print(data_matrix)
    print(target)
    # 2、利用梯度下降训练模型
    w0, w, v = stoc_grad_descent(data_matrix, target, 5, 5000, 0.01)
    print(w0)
    print("w0大小：", w0.shape)
    print(w)
    print("w大小：", w.shape)
    print(v)
    print("v大小：", v.shape)
    # predict_result = get_prediction(np.mat(data_matrix), w0, w, v)


