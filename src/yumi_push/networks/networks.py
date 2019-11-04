#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Patrick Pfreundschuh, Simon Schaefer
# Description : Network definition extending OpenAI Baselines standard models.
# =============================================================================
import numpy as np

import tensorflow as tf

from baselines.a2c.utils import conv, fc, conv_to_fc
from baselines.common.models import register

@register("cnn_smaller")
def cnn_smaller(**net_kwargs):

    def cnn_pool(name, input_tensor, filters=16):
        h = conv(input_tensor, name, nf=filters, rf=3, stride=1, pad="SAME")
        h = tf.nn.relu(h)
        y = tf.layers.max_pooling2d(inputs=h, pool_size=[2, 2], strides=2)
        return y

    def network_fn(X):
        h = cnn_pool("c1", X, filters=8)
        h = tf.nn.relu(h)
        tf.layers.max_pooling2d(inputs=h, pool_size=[2, 2], strides=2)
        h = cnn_pool("c2", h, filters=8)
        h = tf.nn.relu(h)
        h = conv_to_fc(h)
        h = tf.nn.relu(fc(h, 'fc2', nh=8))
        h = fc(h, 'fc3', nh=net_kwargs["nactions"])
        return h
    return network_fn

@register("cnn_med")
def cnn_med(**net_kwargs):

    def network_fn(X):
        h = conv(X,'c1', nf=16, rf=8, stride=4, pad="VALID")
        h = tf.nn.relu(h)
        h = conv(h, 'c2', nf=32, rf=4, stride=2, pad="VALID")
        h = tf.nn.relu(h)
        h = conv_to_fc(h)
        h = tf.nn.relu(fc(h, 'fc1', nh=30))
        h = fc(h, 'fc3', nh=net_kwargs["nactions"])
        return h
    return network_fn

@register("cnn_large")
def cnn_large(**net_kwargs):

    def network_fn(X):
        h = tf.nn.relu(conv(X,'c1', nf=16, rf=4, stride=2, pad="VALID"))
        h = tf.nn.relu(conv(h,'c2', nf=32, rf=4, stride=2, pad="VALID"))
        h = conv_to_fc(h)
        h = tf.nn.relu(fc(h, 'fc1', nh=128))
        h = tf.nn.relu(fc(h, 'fc2', nh=64))
        h = tf.nn.relu(fc(h, 'fc3', nh=32))
        h = tf.nn.relu(fc(h, 'fc4', nh=16))
        h = fc(h, 'fc5', nh=net_kwargs["nactions"])
        return h
    return network_fn

@register("cnn_256_32")
def cnn_256_32(**net_kwargs):

    def network_fn(X):
        h = conv(X,'c1', nf=16, rf=8, stride=4, pad="VALID")
        h = tf.nn.relu(h)
        h = conv(h, 'c2', nf=32, rf=4, stride=2, pad="VALID")
        h = tf.nn.relu(h)
        h = conv_to_fc(h)
        h = tf.nn.relu(fc(h, 'fc1', nh=256))
        h = tf.nn.relu(fc(h, 'fc2', nh=64))
        h = fc(h, 'fc3', nh=net_kwargs["nactions"])
        return h
    return network_fn

@register("cnn_r3")
def cnn_r3(**net_kwargs):

    def network_fn(X):
        h = tf.nn.relu(conv(X,'c1', nf=64, rf=3, stride=2, pad="VALID"))
        h = tf.nn.relu(conv(h,'c2', nf=64, rf=3, stride=2, pad="VALID"))
        h = conv_to_fc(h)
        h = tf.nn.relu(fc(h, 'fc1', nh=128))
        h = tf.nn.relu(fc(h, 'fc2', nh=64))
        h = tf.nn.relu(fc(h, 'fc4', nh=32))
        h = fc(h, 'fc5', nh=net_kwargs["nactions"])
        return h
    return network_fn

@register("fc_small")
def fc_small(**net_kwargs):

    def network_fn(X):
        X = tf.contrib.layers.flatten(X)
        h = tf.nn.relu(fc(X, 'fc1', nh=64))
        h = tf.nn.relu(fc(h, 'fc2', nh=64))
        h = fc(h, 'fc3', nh=net_kwargs["nactions"])
        return h
    return network_fn

@register("fc_large")
def fc_large(**net_kwargs):

    def network_fn(X):
        X = tf.contrib.layers.flatten(X)
        h = tf.nn.relu(fc(X, 'fc1', nh=128))
        h = tf.nn.relu(fc(h, 'fc2', nh=64))
        h = tf.nn.relu(fc(h, 'fc3', nh=64))
        h = tf.nn.relu(fc(h, 'fc4', nh=32))
        h = tf.nn.relu(fc(h, 'fc5', nh=16))
        h = fc(h, 'fc6', nh=net_kwargs["nactions"])
        return h
    return network_fn

@register("fc_very_large")
def fc_very_large(**net_kwargs):

    def network_fn(X):
        X = tf.contrib.layers.flatten(X)
        h = tf.nn.relu(fc(X, 'fc1', nh=256))
        h = tf.nn.relu(fc(h, 'fc2', nh=128))
        h = tf.nn.relu(fc(h, 'fc3', nh=64))
        h = tf.nn.relu(fc(h, 'fc4', nh=64))
        h = tf.nn.relu(fc(h, 'fc5', nh=32))
        h = tf.nn.relu(fc(h, 'fc6', nh=32))
        h = tf.nn.relu(fc(h, 'fc7', nh=16))
        h = fc(h, 'fc8', nh=net_kwargs["nactions"])
        return h
    return network_fn
