# -*- coding:utf-8 -*-
__author__ = 'snake'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def test01_liner():
    # 用numpy生成100个点
    x_data = np.random.rand(100)
    y_data = x_data * 0.1 + 0.2

    # 构造一个线程模型
    # k:斜率; b:偏置值
    b = tf.Variable(0.)
    k = tf.Variable(0.)
    y = k * x_data + b

    # 定义二次方差损失函数，用于优化计算结果，机器学习理论部分
    # 求得预测值和实际值的平方差，用于判断计算结果的损失
    loss = tf.reduce_mean(tf.square(y_data - y))

    # 定义梯度下降法来进行训练的优化器，0.1参数为训练步长
    optimizer = tf.train.GradientDescentOptimizer(0.1)

    # 定义最小化代价函数，梯度下降的目的是把loss降到最小
    # loss越小代表预测值与真实值越接近,
    # k,b越接近0.1和0.2
    train = optimizer.minimize(loss)

    # 初始化变量，一旦有变量，则需要初始化变量
    init = tf.global_variables_initializer()

    # run
    with tf.Session() as sess:
        sess.run(init)
        for step in range(8001):
            sess.run(train)
            if step % 20 == 0:
                print(step, sess.run([k, b]))

    """运行过程:
        1. sess.run(train) -> optimizer.minimize(loss) :使用梯度下降法计算最小化的loss:
        原理: 深度学习06-logistic
        2. 然后计算y_data 和y 的平方差，y_data为x_data的固定值；根据y的模型来计算斜率k和偏置量b
        3. 当k和b最接近y_data模型中的斜率和偏置量，则说明模型最优化
        4. 为什么要使用确定的值来训练模型？
            求预测模型最接近真实模型的算法？？？还是没搞懂。。

    """


def test02_notliner():
    # 使用numpy生成200个随机点
    x_data = np.linspace(-0.5, 0.5, 200).reshape((200,1))   # x_data二维数据，维度为(200, 1)，200行1列
    noise = np.random.normal(0, 0.02, x_data.shape)         # 生成干扰项
    y_data = np.square(x_data) + noise                      # x_data * x_data + noise = 实际值

    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    # 定义神经网络中间层
    # x * W + b = pre_y
    Weights_L1 = tf.Variable(tf.random_normal([1, 10]))     # 10个神经元
    bases_L1 = tf.Variable(tf.zeros([1, 10]))
    Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + bases_L1
    L1 = tf.nn.tanh(Wx_plus_b_L1)                           # 双曲正切函数作为激活函数

    # 定义神经网络输出层
    # 中间层的输出就是下一层的输入
    Weights_L2 = tf.Variable(tf.random_normal([10, 1]))     # 输出层1个神经元
    bases_L2 = tf.Variable(tf.zeros([1, 1]))                # 设置偏置值
    Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + bases_L2
    prediction = tf.nn.tanh(Wx_plus_b_L2)                   # 预测值
    
    # 代价函数
    loss = tf.reduce_mean(tf.square(y-prediction))
    # 使用梯度下降法最小化loss，学习速率（步长）为0.1
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as sess:
        # 初始化变量
        sess.run(tf.global_variables_initializer())     # 初始化变量，在TensorFlow中用到Varibals就需要初始化

        # 循环2000次
        for _ in range(2000):
            # 运行梯度下降训练，训练数据为x_data, 实际值为y_data：
            # 实际上是监督学习：通过事先准备好的值来训·练模型
            sess.run(train_step, feed_dict={x: x_data, y: y_data})

        # 保存图，使用tensorboard打开观察数据流
        writer = tf.summary.FileWriter("./logs", tf.get_default_graph())
        writer.close()

        # 获取预测值并画图
        prediction_value = sess.run(prediction, feed_dict={x: x_data, y: y_data})
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, prediction_value, 'r-', lw=5)
        plt.show()


def test03_mnist():
    # 载入数据，将数据进行one_hot向量化
    mnist = input_data.read_data_sets('mnist/MNIST_data', one_hot=True)

    # 每个批次的大小
    batch_size = 200
    # 计算共有多少个批次
    n_batch = mnist.train.num_examples // batch_size

    # 定义两个placeholder
    # None:任意值：每个批次一次一次的传值到placeholder中，实现动态传值，比如传100行数据进去，None就变为100
    # 784列:每个图片的为28 * 28的格式，将图片转换为一维数组的方式，所有有 None行、874列
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])    # 一共有10个数字:0-9

    # 创建神经网络-输入层
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # 定义激活函数
    prediction = tf.nn.softmax(tf.matmul(x, W) + b)

    # 定义二次代价函数
    loss = tf.reduce_mean(tf.square(y-prediction))
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 结果存放在一个bool列表中
    # argmax返回一个一维张量中最大的值所在的位置
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(init)
        for e in range(200):
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Iter" + str(e) + ", Testing Accuracy " + str(acc))


if __name__ == "__main__":
    #test01_liner()
    #test02_notliner()
    test03_mnist()