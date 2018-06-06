# -*- coding:utf-8 -*-
__author__ = 'snake'
from PIL import Image


def binaryzation(im, threshold):
    """ 二值化：得到黑白图像 """
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    return im.point(table, "1")


def image_decode(file_name, count):
    for i in range(20645, count+1):
        file = file_name + str(i) + ".jpg"
        im = Image.open(file)
        im = im.convert("L")    # 转换为灰度模式
        im = im.convert("P")  # 二值化图片转为8位像素模式

        im = im.resize((160, 60))
        print("正在保存第%s张图片..." % str(i))
        im.save("D:\\verifies\\train_mini\\" + str(i) + ".png")

    return True


if __name__ == "__main__":
    file = "D:\\verifies\\train\\"
    image_decode(file, 200000)