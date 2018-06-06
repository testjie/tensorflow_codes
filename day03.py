# -*- coding:utf-8 -*-
__author__ = 'snake'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
1. 卷积神经网络


"""
# coding:utf-8
# 简单的卷积网络 - MNIST数据集分类

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 读取MNIST数据集
mnist = input_data.read_data_sets('mnist/MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()


# 权重初始化函数
def weight_variable(shape):
    # 添加一些随机噪声来避免完全对称，使用截断正态分布，标准差为0.1
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial_value=initial)


# 偏置量初始化函数
def bias_variable(shape):
    # 为偏置量增加一个很小的正值(0.1)，避免死亡节点
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial)


# 卷积函数
def conv2d(x, W):
    # x: 输入
    # W: 卷积参数，例如[5,5,1,32]：5,5代表卷积核尺寸、1代表通道数：黑白图像为1，彩色图像为3、32代表卷积核数量也就是要提取的特征数量
    # strides: 步长，都是1代表会扫描所有的点
    # padding: SAME会加上padding让卷积的输入和输出保持一致尺寸
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 最大池化函数
def max_pool_2x2(x):
    # 使用2*2进行最大池化，即把2*2的像素块降为1*1，保留原像素块中灰度最高的一个像素，即提取最显著特征
    # 横竖两个方向上以2为步长
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 输入
x = tf.placeholder(tf.float32, [None, 784])
# 对应的label
y_ = tf.placeholder(tf.float32, [None, 10])
# 将一维的输入转成二维图像结棍
# -1: 数量不定
# 28*28: 图像尺寸
# 1: 通道数，黑白图像为1
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层
# 共享权重，尺寸：[5, 5, 1, 32]，5*5的卷积核尺寸、1个通道(黑白图像)、32个卷积核数量，即特征数量
W_conv1 = weight_variable([5, 5, 1, 32])
# 共享偏置量，32个=特征数量
b_conv1 = bias_variable([32])
# 激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 池化
h_pool1 = max_pool_2x2(h_conv1)

# 定义第二个卷积层
# 共享权重[5, 5, 32, 64]，5*5的卷积核尺寸、32个通道(第一个卷积层的输出特征数)、64个卷积核数量，即特征数量
W_conv2 = weight_variable([5, 5, 32, 64])
# 共享偏置量，64个=特征数量
b_conv2 = bias_variable([64])
# 激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 池化
h_pool2 = max_pool_2x2(h_conv2)

# 构建全连接层，设定为1024个神经元
# 经历两次池化之后输出变成了 7*7 共有 64个特征，共计7*7*64
# 权重
W_fc1 = weight_variable([7*7*64, 1024])
# 偏置
b_fc1 = bias_variable([1024])
# 将第二个卷积层的池化输出转换为一维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout层，避免过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# 最后连接至Softmax层，得到最后的概率输出
# 权重，1024输入，10输出
W_fc2 = weight_variable([1024, 10])
# 偏置 10个神经元(输出)
b_fc2 = bias_variable([10])
# 使用softmax激活函数，输出概率值
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 损失函数-交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# 优化器，使用了比较小的学习速率
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 评估准确率的tensor
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练过程，使用了交互式session
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('Step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 进行测试
print('Test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))