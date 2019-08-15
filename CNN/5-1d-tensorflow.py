# 训练样本过少，模型参数过多，容易过拟合
# 表现：训练集损失小，准确率很高，测试集损失大，准确率低
# Tensorboard 可视化网络结构图，损失及精度，调入接口 tf.summary

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 数据预处理（规范化） ？

Data_man = pd.read_excel("radar_data/man_dopplor(120x1).xlsx")
Data_car = pd.read_excel("radar_data/car_dopplor(120x1).xlsx")
data_man = np.array(Data_man)[0:600, 10:124]
data_car = np.array(Data_car)[0:600, 10:124]
# print(data_car[0])

DATA = np.vstack((data_man, data_car))

# print(DATA[0].shape)

# 划分训练集和测试集
# data = input_data.read_data_sets('MNIST_data', one_hot=True)

# 训练数据
x = tf.placeholder(tf.float32, shape=[None, 112*1], name='x')
# 训练标签数据
y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y')
# 把 x 更改为 4 维张量 第一维代表样本数量，第二维和第三维代表图像长宽，第四维代表通道数 1表示黑白
x_image = tf.reshape(x, [-1, 112, 1, 1])


# 初始化权重
def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


# 初始化偏置
def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积层
def convolution_2d(x, W):
    # 移动步长为1，使用全0填充 SAME  （不填充 卷积之后改变尺寸）
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化层
def max_pooling_2x2(x):
    # 池化层过滤器的大小为 2x2 移动步长为 2  不填充
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')


# 第一层卷积： conv + relu + max_pool
# 输入 112 卷积填充 尺寸不变
w_conv1 = weights([3, 1, 1, 16])
b_conv1 = bias([16])
h_conv1 = tf.nn.relu(convolution_2d(x_image, w_conv1) + b_conv1)
# 56x1
h_pool1 = max_pooling_2x2(h_conv1)

# 第二层卷积： conv + relu + max_pool
# 输入 56x1
w_conv2 = weights([3, 1, 16, 32])
b_conv2 = bias([32])
h_conv2 = tf.nn.relu(convolution_2d(h_pool1, w_conv2) + b_conv2)
# 28x1
h_pool2 = max_pooling_2x2(h_conv2)

# 第三层卷积： conv + relu + max_pool
# 输入 28x1
w_conv3 = weights([3, 1, 32, 64])
b_conv3 = bias([64])
h_conv3 = tf.nn.relu(convolution_2d(h_pool2, w_conv3) + b_conv3)
# 14x1
h_pool3 = max_pooling_2x2(h_conv3)


# 第四层卷积： conv + relu + max_pool
# 输入 14x1
w_conv4 = weights([3, 1, 64, 128])
b_conv4 = bias([128])
h_conv4 = tf.nn.relu(convolution_2d(h_pool3, w_conv4) + b_conv4)
# 7x1
h_pool4 = max_pooling_2x2(h_conv4)

h_pool4_flat = tf.reshape(h_pool4, [-1, 7*1*128])


# 全连接层 1  全连接层不是用relu非线性激活函数
# 7*1*128=896  把前一层的输出变成特征向量
w_fc1 = weights([7*1*128, 120])
b_fc1 = bias([120])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool4_flat, w_fc1) + b_fc1)

# 全连接层2
w_fc2 = weights([120, 84])
b_fc2 = bias([84])
h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, w_fc2) + b_fc2)


# Dropout层 目的：为了减少过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc2, keep_prob)

# 输出层：全连接层2
# 神经元节点数 84，分类节点
w_fc3 = weights([84, 2])
b_fc3 = bias([2])
h_fc3 = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc3) + b_fc3)

# 定义交叉熵损失函数
# cross_entropy = -tf.reduce_sum(y_true * tf.log(h_fc3)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(h_fc3), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=h_fc3))
# 选择优化器，并让优化器最小化损失函数/收敛，反向传播
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(cross_entropy)
# 预测 判断参数是否相等 实现准确率的验证
# tf.argmax(x, 1)返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真实值
# 判断预测值y和真实值y_中最大数的索引是否一致，y的值为1-10概率
correct_prediction = tf.equal(tf.argmax(h_fc3, 1), tf.argmax(y_true, 1))
# 用平均值来统计测试准确率
# tf.cast 表示数值类型转化，将correct_prediction中的值转化为 float32 类型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(50):
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(DATA[:, 0:112], DATA[:, 112:124], test_size=.5, random_state=1)
    x_train = x_train.reshape(x_train.shape[0], 112)
    # 对数据进行规范化操作 之后 训练和测试准确率都变低
    # x_train -= np.mean(x_train, axis=0)
    # x_train /= np.std(x_train, axis=0)
    # x_test = x_test / 5000
    # print(x_train)
    y_train = y_train.reshape(y_train.shape[0], 2)

    if i % 10 == 0:
        # 评估阶段不使用 Dropout  keep_prob: 1.0
        # eval 是 sess.run() 的另一种表达，可改成 sess.run(accuracy, feed_dict={})
        train_accuracy = accuracy.eval(feed_dict={x: x_train, y_true: y_train, keep_prob: 1.0 })
        print("step {}, training accuracy {}".format(i, train_accuracy))
    # 训练阶段使用50%的 Dropout  keep_prob: 保留概率，保留结果所占比例    =1则为所有元素全部保留
    train_step.run(feed_dict={x: x_train, y_true: y_train, keep_prob: 0.5})

print('test accuracy: %g' % accuracy.eval(feed_dict={x: x_test, y_true: y_test, keep_prob: 0.7}))



