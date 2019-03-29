#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os
import random
import cv2
import matplotlib.pyplot as plt


# %%
# 生成整数型的属性
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成实数型的属性
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# %%
def load_txt_file(filename, labels_num=1, shuffle=True):
    """
    载入txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1, 标签2，如：test_image/1.jpg 0 2
    :param filename: txt文件名
    :param labels_num:
    :param shuffle:
    :return:
    """
    images_list = []
    labels_list = []
    with open(filename) as f:
        lines_list = f.readlines()
        if shuffle:
            random.shuffle(lines_list)

        for lines in lines_list:               # lines: 'flower/image_01.jpg 0\n'
            line = lines.rstrip().split(' ')   # line: ['flower/image_01.jpg', '0']
            label = []
            for i in range(labels_num):
                label.append(int(line[i + 1]))
            images_list.append(line[0])
            labels_list.append(label)
    return images_list, labels_list


# %%
def show_image(title, image):
    """

    :param title: 图像标题
    :param image: 图像数据
    :return:
    """
    plt.imshow(image)
    plt.axis('on')
    plt.title(title)
    plt.show()


# %%
def read_image(filename, resize_height, resize_width, normalization=False):
    """
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename: 图片名
    :param resize_height:
    :param resize_width:
    :param normalization: 是否归一化到[0.,1.0]
    :return:
    """
    bgr_image = cv2.imread(filename)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)    # 将bgr图像转换成rgb图像

    if resize_width > 0 and resize_height > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image)

    if normalization:
        rgb_image = rgb_image / 255.0
    return rgb_image


# %%
def create_tfrecords(image_dir, txt_file, output_dir, resize_height, resize_width, shuffle, log=5):
    """
    实现将图像原始数据,label,长,宽等信息保存为tfrecords文件
    :param image_dir: 原始图像的目录
    :param txt_file: txt文件名
    :param output_dir: 保存tfrecords文件的路径
    :param resize_height:
    :param resize_width:
    :param shuffle:
    :param log: 信息打印间隔
    :return:
    """
    images_list, labels_list = load_txt_file(txt_file, 1, shuffle)

    writer = tf.python_io.TFRecordWriter(output_dir)
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print('Error: no image', image_path)
            continue
        image = read_image(image_path, resize_width, resize_height)
        image_raw = image.tostring()
        if i % log == 0 or i == len(images_list) - 1:
            print('--------------process %d-th-------------' % i)
            print('current image_path=%s' % image_path, 'shape:{}'.format(image.shape), 'labels:{}'.format(labels))

        # 这里仅保存一个label,多label适当增加"'label': _int64_feature(label)"项
        label = labels[0]
        example = tf.train.Example(features=tf.train.Features(feature={'image_raw': bytes_feature(image_raw),
                                                                       'image height': int64_feature(image.shape[0]),
                                                                       'image width': int64_feature(image.shape[1]),
                                                                       'image depth': int64_feature(image.shape[2]),
                                                                       'image label': int64_feature(label)}
                                                              ))
        writer.write(example.SerializeToString())
    writer.close()


# %%
def read_tfrecords(tfrecords_file, resize_height, resize_width, output_model=None):
    """

    :param tfrecords_file: The tfrecords file
    :param output_model: 选择图像数据的返回类型
                        None:默认将uint8-[0,255]转为float32-[0,255]
                        normalization:归一化float32-[0,1]
                        centralization:归一化float32-[0,1],再减均值中心化
    :return:
    """
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    reader = tf.TFRecordReader()

    key, serialized_example = reader.read(filename_queue)    # key: tfrecords文件名
    features = tf.parse_single_example(serialized_example, features={'image_raw': tf.FixedLenFeature([], tf.string),
                                                                     'image height': tf.FixedLenFeature([], tf.int64),
                                                                     'image width': tf.FixedLenFeature([], tf.int64),
                                                                     'image depth': tf.FixedLenFeature([], tf.int64),
                                                                     'image label': tf.FixedLenFeature([], tf.int64)})
    images = tf.decode_raw(features['image_raw'], tf.uint8)   # 获得原始图像数据s
    labels = features['image label']

    images = tf.reshape(images, [resize_height, resize_width, 3])
    if output_model is None:
        images = tf.cast(images, tf.float32)      # 为了训练,要将数据转换为浮点型
    elif output_model == 'normalization':
        images = tf.cast(images, tf.float32) / 255.0
    elif output_model == 'centralization':
        images = tf.cast(images, tf.float32) / 255.0 - 0.5     # 假定中间值是0.5

    return images, labels


# %%
def get_batch_images(images, labels, batch_size, num_classes, one_hot=False, shuffle=False, num_threads=16):
    """

    :param images:
    :param labels:
    :param one_hot:
    :param shuffle:
    :param num_threads:
    :return:
    """
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size    # 保证capacity必须大于min_after_dequeue参数值
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images, labels],
                                                            batch_size=batch_size,
                                                            num_threads=num_threads,
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
    else:
        images_batch, labels_batch = tf.train.batch([images, labels],
                                                    batch_size=batch_size,
                                                    num_threads=num_threads,
                                                    capacity=capacity)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, num_classes, on_value=1, off_value=0)

    return images_batch, labels_batch


# %%
def get_example_nums(tf_records_file):
    """
    统计tf_records图像的个数(example)个数
    :param tf_records_file:
    :return:
    """
    nums = 0
    for record in tf.python_io.tf_record_iterator(tf_records_file):
        nums += 1
    return nums


# %%
def batch_test(tfrecords_file, resize_height, resize_width):
    """

    :param tfrecords_file:
    :param resize_height:
    :param resize_width:
    :return:
    """
    images, labels = read_tfrecords(tfrecords_file, resize_height, resize_width)
    images_batch, labels_batch = get_batch_images(images, labels, batch_size=4, num_classes=5, one_hot=True, shuffle=True)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        four_images, four_labels = sess.run([images_batch, labels_batch])

        for i in range(4):
            show_image('image', four_images[i])
        print('shape:{}, type:{}, labels:{}'.format(four_images.shape, four_images.dtype, four_labels))

        coord.request_stop()
        coord.join(threads)


# %%
if __name__ == '__main__':
    # 参数设置

    # resize_height = 224    # 指定存储图片高度
    # resize_width = 224     # 指定存储图片宽度
    # shuffle = True
    # log = 5
    # # 产生train.record文件
    # image_dir = 'dataset/train'
    # train_labels = 'dataset/train.txt'    # 图片路径
    # train_record_output = 'dataset/record/train_{}.tfrecords'.format(resize_height)
    # create_tfrecords(image_dir, train_labels, train_record_output, resize_height, resize_width, shuffle, log)
    # train_nums = get_example_nums(train_record_output)
    # print("save train example nums={}".format(train_nums))
    #
    # # 产生val.record文件
    # image_dir = 'dataset/val'
    # val_labels = 'dataset/val.txt'       # 图片路径
    # val_record_output = 'dataset/record/val_{}.tfrecords'.format(resize_height)
    # create_tfrecords(image_dir, val_labels, val_record_output, resize_height, resize_width, shuffle, log)
    # val_nums = get_example_nums(val_record_output)
    # print("save val example nums={}".format(val_nums))

    resize_height = 224
    resize_width = 224
    train_tfrecords = 'dataset/record/train_224.tfrecords'
    batch_test(train_tfrecords, resize_height, resize_width)

