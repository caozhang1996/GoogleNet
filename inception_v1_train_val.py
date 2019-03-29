#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import pdb
import math
import slim.nets.inception_v1 as inception_v1
from create_tfrecords_files import *
import tensorflow.contrib.slim as slim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


num_classes = 5
batch_size = 16              # batch_size不宜过大，否者会出现内存不足的问题
resize_height = 224
resize_width = 224
channels = 3
data_shape = [batch_size, resize_height, resize_width, channels]

input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, channels])
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, num_classes])

keep_prob = tf.placeholder(dtype=tf.float32)
is_training = tf.placeholder(dtype=tf.bool)


def evaluation(sess, loss, accuracy, val_batch_images, val_batch_labels, val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for i in xrange(val_max_steps):
        val_x, val_y = sess.run([val_batch_images, val_batch_labels])

        val_loss, val_acc = sess.run([loss, accuracy], feed_dict={input_images: val_x,
                                                                  input_labels: val_y,
                                                                  keep_prob: 1.0,
                                                                  is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc


def train(train_tfrecords_file,
          base_lr,
          max_steps,
          val_tfrecords_file,
          num_classes,
          data_shape,
          train_log_dir,
          val_nums):
    """

    :param train_tfrecords_file: 训练数据集的tfrecords文件
    :param base_lr: 学习率
    :param max_steps: 迭代次数
    :param val_tfrecords_file: 验证数据集的tfrecords文件
    :param num_classes: 分类个数
    :param data_shape: 数据形状[batch_size, resize_height, resize_width, channels]
    :param train_log_dir: 模型文件的存放位置
    :return:
    """
    [batch_size, resize_height, resize_width, channels] = data_shape

    # 读取训练数据
    train_images, train_labels = read_tfrecords(train_tfrecords_file,
                                                resize_height,
                                                resize_width,
                                                output_model='normalization')
    train_batch_images, train_batch_labels = get_batch_images(train_images,
                                                              train_labels,
                                                              batch_size=batch_size,
                                                              num_classes=num_classes,
                                                              one_hot=True,
                                                              shuffle=True)
    # 读取验证数据,验证数据集可以不用打乱
    val_images, val_labels = read_tfrecords(val_tfrecords_file,
                                            resize_height,
                                            resize_width,
                                            output_model='normalization')
    val_batch_images, val_batch_labels = get_batch_images(val_images,
                                                          val_labels,
                                                          batch_size=batch_size,
                                                          num_classes=num_classes,
                                                          one_hot=True,
                                                          shuffle=False)

    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):  # inception_v1.inception_v1_arg_scope()括号不能掉，表示一个函数
        out, end_points = inception_v1.inception_v1(inputs=input_images,
                                                    num_classes=num_classes,
                                                    is_training=is_training,
                                                    dropout_keep_prob=keep_prob)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32)) * 100.0

    optimizer = tf.train.MomentumOptimizer(learning_rate=base_lr, momentum=0.9)  # 这里可以使用不同的优化函数

    # 在定义训练的时候, 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
    # 正常的训练过程不包括更新，需要我们去手动像下面这样更新
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):    # 执行完更新操作之后，再进行训练操作
        train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for steps in np.arange(max_steps):
            input_batch_images, input_batch_labels = sess.run([train_batch_images, train_batch_labels])
            _, train_loss = sess.run([train_op, loss], feed_dict={input_images: input_batch_images,
                                                                  input_labels: input_batch_labels,
                                                                  keep_prob: 0.8,
                                                                  is_training: True})
            # 得到训练过程中的loss， accuracy值
            if steps % 50 == 0 or (steps + 1) == max_steps:
                train_acc = sess.run(accuracy, feed_dict={input_images: input_batch_images,
                                                          input_labels: input_batch_labels,
                                                          keep_prob: 1.0,
                                                          is_training: False})
                print ('Step: %d, loss: %.4f, accuracy: %.4f' % (steps, train_loss, train_acc))

            # 在验证数据集上得到loss, accuracy值
            if steps % 200 == 0 or (steps + 1) == max_steps:
                val_loss, val_acc = evaluation(sess, loss, accuracy, val_batch_images, val_batch_labels, val_nums)
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (steps, val_loss, val_acc))

            # 每隔2000步储存一下模型文件
            if steps % 2000 == 0 or (steps + 1) == max_steps:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=steps)

        coord.request_stop()
        coord.join(threads)


# %%
if __name__ == "__main__":

    train_tfrecords_file = './dataset/record/train_224.tfrecords'
    val_tfrecords_file = './dataset/record/val_224.tfrecords'
    train_log_dir = './logs/'
    base_lr = 0.01
    max_steps = 10000
    val_nums = get_example_nums(val_tfrecords_file)

    train(train_tfrecords_file, base_lr, max_steps, val_tfrecords_file,
          num_classes, data_shape, train_log_dir, val_nums)


