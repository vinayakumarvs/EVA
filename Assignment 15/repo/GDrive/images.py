#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ..imports import *

import cv2
import tensorflow as tf
def replace_slice(input_: tf.Tensor, replacement, begin) -> tf.Tensor:
    """
    replace slice. Randomly cuts a h by w whole in the image, and fill the whole with given pixels.
    :param input_: Input image.
    :param replacement: replacement pixels
    :param begin: pixel from where the replacement will start
    :return: Transformed image.
    """
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)


def load_image(addr):
    """
    load image. read an image and resize to (32, 32), cv2 load images as BGR, convert it to RGB
    :param address: Input image address
    :return: image.
    """
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

