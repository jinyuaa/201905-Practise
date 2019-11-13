# 基本实现  AE MNIST数据集  将784维数据压缩成128维 实现输入数据的低维重构
# 784 -> [256 -> 128 -> 256] -> 784
#        ==================  栈式自编码器  一般对称结构

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('mnist_data/', one_hot=True)

learning_rate = 0.01
# 第一次256个节点
n_hidden_1 = 256
# 第二层128个节点
n_hidden_2 = 128
n_input = 784

x = tf.placeholder("float", [None, n_input])
# 输出维度等于输入维度
y = x

weights = {"encoder_h1": tf.Variable(tf.random_normal([n_input, n_hidden_1])),
           "encoder_h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           "decoder_h1": tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
           "decoder_h2": tf.Variable(tf.random_normal([n_hidden_1, n_input]))}

biases = {"encoder_b1": tf.Variable(tf.zeros([n_hidden_1])),
          "encoder_b2": tf.Variable(tf.zeros([n_hidden_2])),
          "decoder_b1": tf.Variable(tf.zeros([n_hidden_1])),
          "decoder_b2": tf.Variable(tf.zeros([n_input]))}


# 编码
def encoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["encoder_h1"]), biases["encoder_b1"]))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights["encoder_h2"]), biases["encoder_b2"]))
    return layer2


# 解码  编码的逆过程 完全对称 sigmoid非线性激活函数
def decoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["decoder_h1"]), biases["decoder_b1"]))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights["decoder_h2"]), biases["decoder_b2"]))
    return layer2


# 输出节点
encoder_out = encoder(x)
pred = decoder(encoder_out)

# 设置代价函数
# 对所有元素求和求平均
cost = tf.reduce_mean(tf.square(y-pred))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 训练参数
# 迭代轮数
training_epochs = 20
# 小批量大小
batch_size = 256
disply_step = 5


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 计算一轮迭代多少次
    total_batch = int(mnist_data.train.num_examples / batch_size)
    # 开始训练 迭代
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_x, batch_y = mnist_data.train.next_batch(batch_size)
            _, loss = sess.run([optimizer, cost], {x: batch_x})
        if epoch % disply_step == 0:
            print("Epoch:", "%02d" % (epoch + 1), "Cost = ", "{:.9f}".format(loss))

    # 解码器还原可视化
    show_num = 10
    reconstruction = sess.run(pred, feed_dict={x: mnist_data.test.images[:show_num]})
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist_data.test.images[i], (28, 28)), cmap='gray')
        a[1][i].imshow(np.reshape(reconstruction[i], (28, 28)), cmap='gray')
    plt.draw()
    plt.show()
