import tensorflow as tf
import numpy as np
from PIL import Image
import random, datetime


# 字符长度,验证码长度, 图片高宽
CHAR_SET_LEN = 10
MAX_CAPTCHA, IMAGE_HEIGHT, IMAGE_WIDTH = 4, 80, 200

# 日志和模型保存目录
TRAIN_LOG_PATH, TRAIN_MODEL_PATH = "logs/", "model/fuck_captche.model-1800"

# 训练库位置
TRAIN_LABLE_PATH, TRAIN_IMGS_PATH = "D:\\verifies\\train\\verfiycodes.txt", "D:\\verifies\\train\\"
# 测试库位置
TEST_LABLE_PATH, TEST_IMGS_PATH = "D:\\verifies\\test\\verfiycodes.txt", "D:\\verifies\\test\\"

# x y占位符
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, CHAR_SET_LEN * MAX_CAPTCHA])
keep_prob = tf.placeholder(tf.float32)


def _convert2gray(img):
    """
    彩色图片转灰色图片
    :param img:
    :return:
    """
    if len(img.shape)>2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def _char2pos(c):
    """
    字符串pos
    :param c:
    :return:
    """
    if c == '_':
        k = 62
        return k
    k = ord(c) - 48
    if k > 9:
        k = ord(c) - 55
        if k > 35:
            k = ord(c) - 61
            if k > 61:
                raise ValueError('No Map')
    return k


def _text2vec(text):
    """
    文本转向量
    :param text:
    :return:
    """
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + _char2pos(c)
        vector[idx] = 1
    return vector


def _get_captcha_texts(path=TRAIN_LABLE_PATH):
    """
    获取所有训练标签
    :param path:
    :return:
    """
    texts = []
    for i in open(path, 'r'):
        texts.append(i.strip('\n'))

    return texts


def _get_captcha_text_and_image(index):
    """
    获取验证码和对应的标签
    :param index:
    :return:
    """
    if IS_TRAIN:
        imgs_path = TRAIN_IMGS_PATH + str(index)+".jpg"
        lable = TRAIN_MODEL_TEXTS[index-1]
    else:
        imgs_path = TEST_IMGS_PATH + str(index)+".jpg"
        lable = TEST_MODEL_TEXTS[index-1]

    image = np.array(Image.open(imgs_path))
    return lable, image


def _get_next_batch(batch_size=60):
    """
    生成一个batch
    :param batch_size:
    :return:
    """
    batch_x = np.zeros([BATCH, 80 * 200])
    batch_y = np.zeros([BATCH, 40])

    batch_index = 0
    end_index = batch_size + 1
    start_index = end_index - BATCH

    # 测试
    if IS_TRAIN:
        path = TRAIN_IMGS_PATH
    else:
        path = TEST_IMGS_PATH
    for i in range(start_index, end_index):
        text, image = _get_captcha_text_and_image(i)
        image = _convert2gray(image)

        # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
        batch_x[batch_index, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[batch_index, :] = _text2vec(text)
        batch_index = batch_index + 1

    # 返回该训练批次
    return batch_x, batch_y


def cnn(b_alpha=0.1):
    """
    3层卷积神经网络
    :param b_alpha:
    :return:
    """
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    wc1 = tf.get_variable(name='wc1', shape=[3, 3, 1, 32],
                          dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    bc1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1)) # 输出大小不变
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64],
                          dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    bc2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128],
                          dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    bc3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    wd1 = tf.get_variable(name='wd1', shape=[10*25*128, 1024],
                          dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    bd1 = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, 10*25*128])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
    dense = tf.nn.dropout(dense, keep_prob)

    wout = tf.get_variable('name', shape=[1024, MAX_CAPTCHA * CHAR_SET_LEN],
                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    bout = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, wout), bout)
    return out


def train():
    """
    训练模型函数
    :return:
    """
    output = cnn()
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.graph.finalize()
        for step in range(1, 2001):
            start_time = datetime.datetime.now()

            batch_x, batch_y = _get_next_batch(BATCH * step)
            _, cost_ = sess.run([optimizer, cost],
                                feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})

            end_time = datetime.datetime.now()
            print("step=%s, cost=%s, spending times=%.2fs"
                  % (step, cost_, (end_time-start_time).microseconds / 1000000))

            # 每100步测试一下准确率
            if step % 100 == 0:
                # 测试数据集使用下一次的数据集
                batch_x_test, batch_y_test = _get_next_batch(BATCH * (step+1))
                acc = sess.run(accuracy,
                               feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 0.75})
                print("step=%d, 准确率 ------------> %s" % (step, acc))

                # 达到99%准确率就保存model并退出
                if acc > 0.99:
                    saver.save(sess, TRAIN_MODEL_PATH, global_step=step)
                    break


def _fuck_captcha(sess, predict, captcha_image):
    """
    破解验证码方法
    :param captcha_image:
    :return:
    """
    text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

    text = text_list[0].tolist()
    text = "".join(list(map(str, text)))

    return text


def test():
    """
    测试函数
    :return:
    """
    output = cnn()
    saver = tf.train.Saver()
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint("model/"))

        task_count = 5000
        right_count = 0
        for i in range(task_count):
            text, image = _get_captcha_text_and_image(i+1)
            image = _convert2gray(image)
            image = image.flatten() / 255
            predict_text = _fuck_captcha(sess, predict, image)
            if str(text) == predict_text:
                right_count += 1
            else:
                print("【错误】: \t正确值: {}  预测值: {}".format(text, predict_text))

        print('正确/共计-----', right_count, '/', task_count)


if __name__ == '__main__':
    # 获取所有指定库的标签
    IS_TRAIN = True
    IS_TRAIN = False
    TRAIN_MODEL_TEXTS, TEST_MODEL_TEXTS = [], []

    if IS_TRAIN:
        BATCH = 100  # 每次取batch条数据作为训练集
        TRAIN_MODEL_TEXTS = _get_captcha_texts(TRAIN_LABLE_PATH)
        train()
    else:
        BATCH = 2000
        TEST_MODEL_TEXTS = _get_captcha_texts(TEST_LABLE_PATH)
        test()
