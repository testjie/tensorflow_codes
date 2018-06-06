# -*- coding:utf-8 -*-
__author__ = 'snake'

import tensorflow as tf
import numpy as np

""" TensorFlow基本概念

1. 使用图（graphs）来表示计算任务
2. 在被称之为会话（Session）的上下文（context）中执行图
3. 使用tensor表示数据
4. 通过变量（Variable）维护状态
5. 使用feed和fetch可以为任意的操作赋值或者从其中获取数据

"""


def test02():
    # 定义变量
    x = tf.Variable([1, 2])
    # 定义常量
    a = tf.constant([3, 3])

    # 增加一个减法op
    sub = tf.subtract(x, a)
    # 增加一个加法op
    add = tf.add(x, sub)

    # 初始化变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(sub))
        print(sess.run(add))

    # 创建一个变量初始化为0
    state = tf.Variable(0, name="counter")
    # 创建一个op，作用是state+1
    new_value = tf.add(state, 1)
    # 赋值op
    update = tf.assign(state, new_value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(state))
        for _ in range(5):
            sess.run(update)
            print(sess.run(state))


def fetch():
    # fetch
    input1 = tf.constant(3.0)
    input2 = tf.constant(4.0)
    input3 = tf.constant(5.0)

    # 定义加法和乘法op
    add = tf.add(input2, input3)
    mul = tf.multiply(input1, add)

    with tf.Session() as sess:
        result = sess.run([mul, add])
        print(result)


def feed():
    # Feed
    # 创建占位符
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        # feed的数据以字典的形式传入
        print(sess.run(output, feed_dict={input1: [7.0], input2: 2.0}))



if __name__ == "__main__":
    # test02()
    # fetch()
    feed()