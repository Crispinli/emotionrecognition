from __future__ import division
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


def get_pix(file_name):
    '''
    :author: lwp
    :description: 若为无效图像，则会返回True，否则返回False
    :param file_name: 图像的绝对路径
    :return: boolean
    '''
    img = Image.open(file_name)
    pixdata = img.load()
    tsize = 0
    try:
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if pixdata[x, y][0] > 250 and pixdata[x, y][0] > 250 and pixdata[x, y][0] > 250:
                    tsize = tsize + 1
        allsize = img.size[0] * img.size[1]
        percentage = float(tsize) / float(allsize)
        if (percentage >= 0.9):
            return True
        else:
            return False
    except TypeError:
        #--无效图像会产生异常--#
        return True


def get_labels(file_name):
    '''
    :author: lwp
    :description: 用于获取训练数据和测试数据对应的类标签
    :param file_name: 包含类标签的txt文件的绝对路径
    :return: object
    '''
    train_labels = {}
    file = open(file_name)
    for line in file:
        line = line.rstrip()
        cols = line.split()
        train_labels[cols[0]] = cols[1]
    file.close()
    return train_labels


def labels_to_indexs(labels):
    '''
    :author: lwp
    :description: 将情感标签换算成对应的数字
    :param labels: 需要换算的字典对象
    :return: 换算完的字典对象
    '''
    for key in labels:
        label = labels[key]
        labels[key] = one_hot_list[label_index[label]]
    return labels


def _bytes_feature(value):
    '''
    :author: lwp
    :description: 将图像转换成二进制特征
    :param value: 图像
    :return: 二进制的图像特征
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    '''
    :author: lwp
    :description: 将数字标签转换成64位整型特征
    :param value: 数字标签
    :return:  64位整型的特征
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_data_to_tfrecord(path, labels, filename):
    '''
    :author: lwp
    :description: 将数据写入tfrecord文件
    :param path: 存储数据的绝对路径
    :param labels: 存储数字标签的字典
    :param filename: 存储tfrecord文件的绝对路径
    :return: void
    '''
    writer = tf.python_io.TFRecordWriter(filename)
    for img_floder in os.listdir(path):
        img_path = path + os.sep + img_floder
        label = one_hot_list[labels[str(img_floder)]]
        label_raw = label.tostring()
        for img_file in os.listdir(img_path):
            dir_path = img_path + os.sep + img_file
            try:
                if not get_pix(dir_path):
                    img = Image.open(dir_path).convert('L')
                    img_raw = img.tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image_raw': _bytes_feature(img_raw),
                        'label': _bytes_feature(label_raw),
                        'img_floder': _bytes_feature(bytes(img_floder, encoding="utf8"))
                    }))
                    writer.write(example.SerializeToString())
            except OSError:
                print('OSError: cannot identify image file %s' % dir_path)
                continue
    writer.close()
    print(filename, "is completed!")


one_hot_list = [
    np.array([1, 0, 0, 0, 0, 0, 0, 0], "float"),
    np.array([0, 1, 0, 0, 0, 0, 0, 0], "float"),
    np.array([0, 0, 1, 0, 0, 0, 0, 0], "float"),
    np.array([0, 0, 0, 1, 0, 0, 0, 0], "float"),
    np.array([0, 0, 0, 0, 1, 0, 0, 0], "float"),
    np.array([0, 0, 0, 0, 0, 1, 0, 0], "float"),
    np.array([0, 0, 0, 0, 0, 0, 1, 0], "float"),
    np.array([0, 0, 0, 0, 0, 0, 0, 1], "float")
]

# 情感标签以及对应的数字
label_index = {
    "angry": 0,
    "anxious": 1,
    "disgust": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
    "worried": 7
}

# 存储数据的根目录以及根目录下的文件和文件夹
root_url = r"C:\Users\lwp\Desktop\data"
name_list = os.listdir(root_url)

# 存储对应的名称
train_set_floder = str(name_list[3])
train_label_txt = str(name_list[2])
test_set_floder = str(name_list[1])
test_label_txt = str(name_list[0])

# 定义对应的绝对路径
train_set_path = root_url + os.sep + train_set_floder
train_labels_path = root_url + os.sep + train_label_txt
test_set_path = root_url + os.sep + test_set_floder
test_labels_path = root_url + os.sep + test_label_txt

# 获取训练数据和测试数据对应的类标签
train_labels = get_labels(train_labels_path)
test_labels = get_labels(test_labels_path)

# 将情感标签换算成对应的数字
train_labels = labels_to_indexs(train_labels)
test_labels = labels_to_indexs(test_labels)

# np.save("./npy_data/train_label.npy", train_labels)
# np.save("./npy_data/test_label.npy", test_labels)
#
# print("the labels are completed!")

# 存储数据的tfrecord文件
# tfrecords_filename_train = './data/Face_train.tfrecords'
# tfrecords_filename_test = './data/Face_test.tfrecords'

# write_data_to_tfrecord(
#     path=train_set_path,
#     labels=train_labels,
#     filename=tfrecords_filename_train
# )
#
# write_data_to_tfrecord(
#     path=test_set_path,
#     labels=test_labels,
#     filename=tfrecords_filename_test
# )
