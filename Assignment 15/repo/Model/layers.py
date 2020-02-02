#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ..imports import *

import tensorflow as tf
import numpy as np
import math

def init_pytorch(shape, dtype=tf.float32, partition_info=None):
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)


class ConvBN(tf.keras.Model):
    def __init__(self, c_out):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.drop = tf.keras.layers.Dropout(0.1)

    def call(self, inputs):
        return tf.nn.relu(self.bn(self.drop(self.conv(inputs))))

