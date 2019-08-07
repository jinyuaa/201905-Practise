# 神经网络 B-P算法训练网络参数（基于梯度下降策略）
# 与 Logistic Regression进行对比
# 二分类和三分类对比

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.linear_model


'''
# 1、生成数据集并绘制出来
# 产生相同的随机数，参数只是确实能够随机数的其实位置，使得随机数可预测
np.random.seed(0)
# 通过make_moons函数生成数据集X,y,其中X是数据样本，y是数据样本对应的标签
# make_moons的第一个参数200指定了生成数据点的个数，第二个参数指定了数据点所服从的高斯噪声的标准差
X, y = sklearn.datasets.make_moons(200, noise=0.20)
# cmap=plt.cm.Spectral 实现的功能是给 label为1的点一种颜色，给 label为0的点另一种颜色
plt.scatter(X[:, 0], X[:, 1], s=50, c=y, cmap=plt.cm.Spectral)
plt.show()
# scikit-learn里面的逻辑回归分类器
# 生成一个LogisticRegressionCV对象
clf = sklearn.linear_model.LogisticRegressionCV()
# 调用fit方法训练LogisticRegression
clf.fit(X, y)
'''
K = 3
N = 100
D = 2
X = np.zeros((N*K, D))
y = np.zeros(N*K, dtype='uint8')
for i in range(K):
    ix = range(N*i, N*(i+1))
    # np.linspace（start, stop, N）在指定间隔返回均匀间隔数字，均匀分布样本
    r = np.linspace(0.0, 1, N)
    t = np.linspace(i*4, (i+1)*4, N) + np.random.rand(N)*0.2
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = i


clf = sklearn.linear_model.LogisticRegressionCV()
# 调用fit方法训练LogisticRegression
clf.fit(X, y)

# cmap = plt.cm.Spectral 实现的功能是给每类样本不同的颜色 c label:1,2,3
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
plt.show()


# 辅助函数绘制决策边界
def plot_decision_boundary(pred_func):
    # 设置最大值最小值并给它一些填充
    x_min, x_max = X[:, 0].min()-.5, X[:, 0].max()+.5
    y_min, y_max = X[:, 1].min()-.5, X[:, 1].max()+.5
    h = 0.01
    # 生成一个距离为 h的点网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 预测整个 gid的功能值
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 绘制轮廓和训练数据, contourf是等值线图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


# lambda实际上是一种函数，当你想运行一个函数而又毫不关心它的函数名时，可以称之为 lambda
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.show()


# 训练一个神经网络，搭建一个输入层、一个隐藏层、一个输出层组成的三层神经网络

# 输入层中的节点数由数据的维度来决定，也就是2个。相应的，输出层的节点数则是由类的数量来决定，
# 也是2个。（因为我们只有一个预测0和1的输出节点，所以我们只有两类输出，实际中，两个输出节
# 点将更易于在后期进行扩展从而获得更多类别的输出）。以x，y坐标作为输入，输出的则是两种概率，
# 一种是0，另一种是1

# num_examples = len(X)   # 训练集的大小
num_examples = X.shape[0]
nn_input_dim = 2        # 输入层维度
# nn_output_dim = 2
nn_output_dim = 3       # 输出层维度

# 梯度下降的参数
epsilon = 0.01          # 梯度下降的学习率
reg_lambda = 0.01       # 正则化强度


# 在数据集上估算总体的损失函数
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 正向传播计算预测  使用 tanh激活函数
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    # 用指数函数还原 计算类别概率
    exp_scores = np.exp(z2)

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # 计算损失
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # 在损失中增加正则项 L2
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss


# 预测输出 0或1
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 正向传播
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    # 用指数函数还原 计算类别概率
    exp_scores = np.exp(z2)

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # np.argmax 取出 probs中最大值所对应的索引
    return np.argmax(probs, axis=1)


# 神经网络学习参数并返回模型
# nn_hdim 隐藏层节点数（神经元个数）
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    # 用随机值初始化参数
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # 最终要返回的数据
    model = {}

    # 梯度下降
    for i in range(0, num_passes):

        # 正向传播
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 反向传播
        delta3 = probs

        delta3[range(num_examples), y] -= 1

        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # 添加正则项（ b1和 b2项）
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # 梯度下降更新参数
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # 为模型分配新参数
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # 打印损失
        if print_loss and i % 1000 == 0:
          print("Loss after iteration %i: %f" % (i, calculate_loss(model)))

    return model


# 一个隐藏层规模为3的网络
model = build_model(15, print_loss=True)
# 画出决策边界

plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
plt.show()

'''
model = build_model(20, print_loss=False)
# 画出决策边界
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 20")
plt.show()


plt.figure(figsize=(16, 16))
# 隐层分别为1、2、3、4、20、50
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer size %d' % nn_hdim)
    model = build_model(nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x))
'''