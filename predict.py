#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import cv2
import os
import glob
import slim.nets.inception_v1 as inception_v1

from create_tfrecords_files import *
import tensorflow.contrib.slim as slim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def predict_images():

    models_path = './logs/model.ckpt-11999'
    images_dir = './test_images'
    labels_txt_file = './dataset/label.txt'

    num_calsses = 5
    resize_height = 224
    resize_width = 224
    channels = 3

    images_list = glob.glob(os.path.join(images_dir, '*.jpg'))  # 返回匹配路径名模式的路径列表

    # delimiter='\t'表示以空格隔开
    labels = np.loadtxt(labels_txt_file, str, delimiter='\t')   # labels = ['flower' 'guitar' 'animal' 'houses' 'plane']
    intput_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, channels], name='input')

    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        out, end_points = inception_v1.inception_v1(inputs=intput_images,
                                                    num_classes=num_calsses,
                                                    dropout_keep_prob=1.0,
                                                    is_training=False)
    score = tf.nn.softmax(out)
    class_id = tf.argmax(score, axis=1)     # 最大score的id值

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        saver = tf.train.Saver()
        saver.restore(sess, models_path)

        for image_name in images_list:
            image = read_image(image_name, resize_height, resize_width, normalization=True)
            image = image[np.newaxis, :]   # 给数据增加一个新的维度
            predict_score, predict_id = sess.run([score, class_id], feed_dict={intput_images: image})
            max_score = predict_score[0, predict_id]      # id相对应的得分(得到的score是二维的)
            print("{} is: label:{},name:{} score: {}".format(image_name, predict_id, labels[predict_id], max_score))


if __name__ == '__main__':

    predict_images()
