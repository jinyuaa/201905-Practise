# 原始 LeNet-5网络实现 参数与架构一致
# 未入dropout 最后一层用 softmax
# 非线性激活函数若使用 sigmoid 准确率达 65%
# 使用 relu 准确率达 92%

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os


## 0、准备数据集并且定义变量
data = input_data.read_data_sets('MNIST_data', one_hot=True)
print(type(data.train.labels))  #(55000, 10)

x = tf.placeholder(tf.float32, shape=[None, 28*28], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y')
# reshape the data to 4_D  读取数据集之后 x 2维张量[10000, 784]--> x_image 4维张量[10000, 28, 28, 1]
x_image = tf.reshape(x, [-1, 28, 28, 1])


## 1、定义功能层及初始化参数
# 初始化权重
def weights(shape):
    # tf.random_normal() tf.truncated_normal()  都是从给定均值和方差的正太分布中输出随机变量
    # tf.truncated_normal() 截取的是两个标准差以内的部分，截取的随机变量值更接近与均值 stddev=0.1正太分布的标准差
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置
def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积层
def convolution_2d(x, W):
    # 要求 x 输入是一个 4 维张量
    # 卷积层中 padding='SAME' 进行填充 图像的上下左右行全补 0
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化层
def max_pooling_2x2(x):
    # 池化层为什么要补 0 ？
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    # value：表示数据输入形式
    # data_format：表示数据的输入格式，需要和输入数据格式保持一致，'NHWC' [batch, in_hight, in_wide, in_channels]
    # ksize：表示池化的大小，是一个4个元素列表  跟输入数据格式对应 [batch, in_hight, in_wide, in_channels]，目前 batch in_channels都为 1
    # strides：表示在特征图上移动，是一个4个元素列表  跟输入数据格式对应 [batch, in_hight, in_wide, in_channels]
    # padding:是否填充，'SAME'：得到的输出特征图跟输入特征图相同；'VALID'：得到的输出特征图跟输入特征图不同
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


## 2、设计网络结构
# 第一层卷积： conv + relu + max_pool
# 原图 28x28
# 6个5x5 的卷积核  得到 6个特征图  尺寸 28x28  因为进行填充，故图像尺寸未改变
# [5, 5, 1, 6] 前两个数为卷积核大小，1为通道数，6为卷积过后输出特征图数量
w_conv1 = weights([5, 5, 1, 6])
b_conv1 = bias([6])
# 原生网络用的是 sigmoid 函数
h_conv1 = tf.nn.relu(convolution_2d(x_image, w_conv1) + b_conv1)
# 14x14
h_pool1 = max_pooling_2x2(h_conv1)


# 第二层卷积： conv + relu + max_pool
# 再次进行卷积，填充，16个特征图  尺寸 14x14
w_conv2 = weights([5, 5, 6, 16])
b_conv2 = bias([16])
h_conv2 = tf.nn.relu(convolution_2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pooling_2x2(h_conv2)
# 7*7*16
print(h_pool2.shape)
# 拉伸
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])


# 第三层：全连接层 1
w_fc1 = weights([7*7*16, 120])
b_fc1 = bias([120])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

w_fc2 = weights([120, 84])
b_fc2 = bias([84])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

# Dropout 层
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层2
w_fc3 = weights([84, 10])
b_fc3 = bias([10])
h_fc3 = tf.nn.softmax(tf.matmul(h_fc2, w_fc3) + b_fc3)


## 3、模型训练过程
cross_entropy = -tf.reduce_sum(y_true * tf.log(h_fc3))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=h_fc3))
# 随机梯度下降
learn_rate = 1e-4
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(cross_entropy)
# 预测 判断参数是否相等
correct_prediction = tf.equal(tf.argmax(h_fc3, 1), tf.argmax(y_true, 1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## 4、创建实例对象，输入数据并迭代输出精度
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# 创建一个saver类 并实例化对象
model_saver = tf.train.Saver()

for i in range(1000):
    batch_x, batch_y = data.train.next_batch(50)
    train_accuracy, train_optimizer=sess.run([accuracy, optimizer], feed_dict={x: batch_x, y_true: batch_y})

    if i % 100 == 0:
        # train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch_x, y_true: batch_y})
        print("step {}, training accuracy {}".format(i, train_accuracy))
    # optimizer.run(session=sess, feed_dict={x: batch_x, y_true: batch_y})

# print('test accuracy: %g' % accuracy.eval(session=sess, feed_dict={x: data.test.images, y_true: data.test.labels}))


## 5、保存模型
model_dir = "mnist_model"
model_name = "my.ckpt"
model_path = os.path.join(model_dir, model_name)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# 调用saver的save方法，需传递两个参数(1、训练的session，2、文件存储路径)
# 存储心训练好的 variables
save_path = model_saver.save(sess, model_path)
print("model saved successfully!Model saved in file: %s" % save_path)





'''
# 由于GPU显存不足导致报错，将test_set分成几个batch分别测试，最后求平均精度
accuracy_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
good = 0
total = 0
for j in range(10):
    testSet = data.test.next_batch(50)
    good += accuracy_sum.eval(feed_dict={x: testSet[0], y_true: testSet[1]})
    total += testSet[0].shape[0]
print("test accuracy {}".format(good/total))

'''
'''
# 单层网络 达到91%
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 100
X_holder = tf.placeholder(tf.float32)
y_holder = tf.placeholder(tf.float32)

Weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([1, 10]))
predict_y = tf.nn.softmax(tf.matmul(X_holder, Weights) + biases)
loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

for i in range(1000):
    images, labels = mnist.train.next_batch(batch_size)
    session.run(train, feed_dict={X_holder: images, y_holder: labels})
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_value = session.run(accuracy, feed_dict={X_holder: mnist.test.images, y_holder: mnist.test.labels})
        print('step:%d accuracy:%.4f' % (i, accuracy_value))

print(mnist.test.labels[1])
image = mnist.test.images[1].reshape(28, 28)
plt.imshow(image)
plt.show()

'''