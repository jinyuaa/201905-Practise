# 样本数据生成 生成一份螺旋状的分布样本点
# 用 softmax分类器，准确率在 0.52 原因，线性分类器 对非线性数据效果不好
# 用 neural network 准确率在 0.99
# 3神经网络：输入层、隐层、输出层

import numpy as np
import matplotlib.pyplot as plt

# 每类样本中的点数
N = 100
# 样本维度
D = 2
# 样本类别个数
K = 3
X = np.zeros((N*K, D))
y = np.zeros(N*K, dtype='uint8')

# 生成螺旋状分布样本
for i in range(K):
    ix = range(N*i, N*(i+1))
    # np.linspace（start, stop, N）在指定间隔返回均匀间隔数字，均匀分布样本
    r = np.linspace(0.0, 1, N)
    t = np.linspace(i*4, (i+1)*4, N) + np.random.rand(N)*0.2
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = i

# cmap = plt.cm.Spectral 实现的功能是给每类样本不同的颜色 c label:1,2,3
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
plt.show()

'''
# 绘制决策边界
def plot_decision_boundary(pred_func):
    # 设置最大值最小值并给它一些填充
    x_min, x_max = X[:, 0].min()-.5, X[:, 0].max()+.5
    y_min, y_max = X[:, 1].min()-.5, X[:, 1].max()+.5
    h = 0.01
    # 生成一个距离为 h的点网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print(xx.ravel())
    # 预测整个 gid的功能值
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 绘制轮廓和训练数据, contourf是等值线图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
'''

# 1、使用 softmax 训练线性分类器   300x2  2x3
# softmax：使用的是一个线性得分函数，损失函数使用交叉熵损失
W = 0.01 * np.random.rand(D, K)
b = np.zeros((1, K))
# 设置步长和正则化系数
step_size = 1e-0
reg = 1e-3
# 梯度下降迭代循环 300
num_examples = X.shape[0]
for i in range(200):
    # 计算得分函数
    scores = np.dot(X, W) + b
    # 用指数函数还原 计算类别概率
    exp_scores = np.exp(scores)
    # 归一化
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # 计算log概率和互熵损失
    corect_log = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_log) / num_examples
    # 正则化项，L2
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    if i % 10 == 0:
        print("iteration %d: loss %f" % (i, loss))

    # 计算得分上的梯度
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # 计算和回传梯度
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)

    dW += reg * W  # 正则化梯度

    # 参数更新
    W += -step_size * dW
    b += -step_size * db

'''
def predict(model, x):
    scores = np.dot(X, W) + b
    # 用指数函数还原 计算类别概率
    exp_scores = np.exp(scores)
    # 归一化
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # np.argmax 取出 probs中最大值所对应的索引
    return np.argmax(probs, axis=1)
'''

# 评估准确度
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print('softmax linear classifier training accuracy: %.2f' % (np.mean(predicted_class == y)))

# plot_decision_boundary(predicted_class)
# plt.title("Logistic Regression")

# 使用神经网络
# 初始化参数  D=2 K=3

h = 100
W = 0.01*np.random.rand(D, h)
b = np.zeros((1, h))
W2 = 0.01*np.random.rand(h, K)
b2 = np.zeros((1, K))

for i in range(10000):
    # 2层神经网络的向前传播
    # 激活函数 ReLU函数
    hidden_layer = np.maximum(0, np.dot(X, W) + b)  # 使用的ReLU神经元
    scores = np.dot(hidden_layer, W2) + b2

    # 计算类别概率
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]
    # 计算互熵损失与正则化项
    corect_log = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_log) / num_examples
    reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("iteration %d: loss %f" % (i, loss))

    # 计算梯度
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # 梯度回传
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    dhidden = np.dot(dscores, W2.T)

    dhidden[hidden_layer <= 0] = 0

    # 拿到最后W,b上的梯度
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

    # 加上正则化梯度部分
    dW2 += reg * W2
    dW += reg * W

    # 参数迭代与更新
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

# 计算分类准确度
hidden_layer = np.maximum(0, np.dot(X, W) + b)  # 使用的ReLU神经元
scores1 = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores1, axis=1)
print('Neural Network training accuracy: %.2f' % (np.mean(predicted_class == y)))

