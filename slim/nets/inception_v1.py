#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from slim.nets import inception_utils

slim = tf.contrib.slim
truncated_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


# %%
def inception_v1_base(inputs, final_endpoint='Mixed_5c', scope_name='InceptionV1'):
    """
    Defines the Inception V1 base architecture.
    :param inputs: a tensor of size [batch_size, height, width, channels].
    :param final_endpoint: specifies the endpoint to construct the network up to. It
                           can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
                           'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
                           'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
                           'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    :param scope_name: Optional variable_scope.
    :return:resize_height
    """
    end_points = {}     # 字典，key为这一层的网络名， value为这一层网络的输出值
    with tf.variable_scope(scope_name, 'InceptionV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_initializer=truncated_normal(0.01)):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):
                end_point = 'Conv2d_1a_7x7'
                net = slim.conv2d(inputs, 64, [7, 7], stride=2, scope=end_point)   # output size: 112x112x64
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'MaxPool_2a_3x3'
                net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Conv2d_2b_1x1'
                net = slim.conv2d(net, 64, [1, 1], scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Conv2d_2c_3x3'
                net = slim.conv2d(net, 192, [3, 3], scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'MaxPool_3a_3x3'
                net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Mixed_3b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                    # 深度上的连接, feature map的shape是[batch_size, width, height, depth], 所以axis=3
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Mixed_3c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'MaxPool_4a_3x3'
                net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Mixed_4b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Mixed_4c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Mixed_4d'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Mixed_4e'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Mixed_4f'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'MaxPool_5a_2x2'
                net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Mixed_5b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Mixed_5c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


# %%
def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope_name='InceptionV1'):
    """
    Defines the Inception V1 architecture.
    :param inputs: a tensor of size [batch_size, height, width, channels].
    :param num_classes:
    :param is_training: whether is training or not.
    :param dropout_keep_prob: 保留的激活值的百分比。
    :param prediction_fn: a function to get predictions out of logits.
    :param spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
                            shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    :param reuse: whether or not the network and its variables should be reused. To be
                  able to reuse 'scope' must be given.
    :param scope_name:
    :return:
    """
    # Final pooling and prediction
    with tf.variable_scope(scope_name, 'InceptionV1', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = inception_v1_base(inputs, scope_name=scope)
            with tf.variable_scope('Logits'):
                # Pooling with a fixed kernel size.
                net = slim.avg_pool2d(net, [7, 7], stride=1, scope='AvgPool_0a_7x7')
                end_points['AvgPool_0a_7x7'] = net

                if not num_classes:           # 如果num_classes是0或者None，则返回logits层的特性
                    return net, end_points

                net = slim.dropout(net, dropout_keep_prob, scope='Dropout_0b')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_0c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                end_points['Logits'] = logits
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


inception_v1.default_image_size = 224
inception_v1_arg_scope = inception_utils.inception_arg_scope
