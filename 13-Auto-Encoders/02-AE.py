# 基本实现  AE MNIST数据集  实现分类的功能
# 784 -> 256 -> 128 -> 10

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
n_output = 10


weights = {"encoder_h1": tf.Variable(tf.random_normal([n_input, n_hidden_1])),
           "encoder_h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           # "decoder_h1": tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
           # "decoder_h2": tf.Variable(tf.random_normal([n_hidden_1, n_input])),
           "full_connect": tf.Variable(tf.random_normal([n_hidden_2, n_output]))}

biases = {"encoder_b1": tf.Variable(tf.zeros([n_hidden_1])),
          "encoder_b2": tf.Variable(tf.zeros([n_hidden_2])),
          # "decoder_b1": tf.Variable(tf.zeros([n_hidden_1])),
          # "decoder_b2": tf.Variable(tf.zeros([n_input])),
          "full_connect": tf.Variable(tf.zeros([n_output]))}


def full_connection(x):
    out_layer = tf.nn.softmax(tf.add(tf.matmul(x, weights["full_connect"]), biases["full_connect"]))
    return out_layer


# 编码
def encoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["encoder_h1"]), biases["encoder_b1"]))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights["encoder_h2"]), biases["encoder_b2"]))
    return layer2


# 解码  编码的逆过程 完全对称 sigmoid非线性激活函数
# def decoder(x):
#     layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["decoder_h1"]), biases["decoder_b1"]))
#     layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights["decoder_h2"]), biases["decoder_b2"]))
#     return layer2

x = tf.placeholder(tf.float32, [None, n_input])
y_true = tf.placeholder(tf.float32, [None, 10])

encoder_out = encoder(x)
y_pred = full_connection(encoder_out)


loss = -tf.reduce_sum(y_true * tf.log(y_pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
correct_pred = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 训练参数
# 迭代轮数
training_epochs = 20
# 小批量大小
batch_size = 50
disply_step = 5


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 计算一轮迭代多少次
    total_batch = int(mnist_data.train.num_examples / batch_size)
    # 开始训练 迭代
    for epoch in range(0, training_epochs):
        for i in range(0, total_batch):
            # batchsize = 192
            batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
            _, f_l = sess.run([optimizer, loss], feed_dict={x: batch_xs, y_true: batch_ys})
        y_pred1, acc = sess.run([y_pred, accuracy], feed_dict={x: batch_xs, y_true: batch_ys})
        print('finetune---Epoch:' + str(epoch) + ' loss = ' + str(f_l) + ', training accuracy = ' + str(acc))

    print('\nFinetune Finished!\n')

    batch_x, batch_y = mnist_data.test.next_batch(batch_size)
    test_accuracy = sess.run([accuracy], feed_dict={x: batch_x, y_true: batch_y})
    print('test accuracy:' + str(test_accuracy))
