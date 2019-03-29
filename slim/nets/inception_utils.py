#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


# %%把多scope封装成函数
def inception_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        activation_fn=tf.nn.relu,
                        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
    """

    :param weight_decay: The weight decay to use for regularizing the model.
    :param use_batch_norm: If `True`, batch_norm is applied after each convolution.
    :param batch_norm_decay: Decay for batch norm moving average.
    :param batch_norm_epsilon: Small float added to variance to avoid dividing by zero
                               in batch norm.
    :param activation_fn: Activation function for conv2d.
    :param batch_norm_updates_collections: Collection for the update ops for batch norm.
    :return: An `arg_scope` to use for the inception models.
    """

    batch_norm_params = {'decay': batch_norm_decay,      # Decay for the moving averages.
                         'epsilon': batch_norm_epsilon,  # epsilon to prevent 0s in variance.
                         'updates_collections': batch_norm_updates_collections,  # collection containing update_ops.
                         'fused': None, }                # use fused batch norm if possible.

    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params    # 归一化参数需要以字典的形式传入
    else:
        normalizer_fn = None
        normalizer_params = {}

    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):  # l2正则化防止过拟合
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=slim.variance_scaling_initializer(),   # 卷积层的权重初始化:
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params
                            ) as sc:
            return sc


