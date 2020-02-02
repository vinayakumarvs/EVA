#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ..imports import *
from ..GDrive import image as imgs

import tensorflow as tf
import numpy as np

def cutout(x: tf.Tensor, h: int, w: int, c: int = 3) -> tf.Tensor:
    """
    Cutout data augmentation. Randomly cuts a h by w whole in the image, and fill the whole with zeros.
    :param x: Input image.
    :param h: Height of the hole.
    :param w: Width of the hole
    :param c: Number of color channels in the image. Default: 3 (RGB).
    :return: Transformed image.
    """
    shape = tf.shape(x)
    x0 = tf.random.uniform([], 0, shape[0] + 1 - h, dtype=tf.int32)
    y0 = tf.random.uniform([], 0, shape[1] + 1 - w, dtype=tf.int32)
    x = imgs.replace_slice(x, tf.zeros([h, w, c]), [x0, y0, 0])
    return x


def normalize(data_set, padding: bool = False):
    # Null check
    if data_set is None:
        return None

    mean = np.mean(data_set, axis=(0, 1, 2))
    std = np.std(data_set, axis=(0, 1, 2))

    data_set = ((data_set - mean) / std).astype('float32')

    if padding:
        data_set = pad4(data_set)

    return data_set


def pad4(image):
    """
    Pad image with 4 pixels from each side
    :param image: image
    :return: padded image
    """
    return np.pad(image, [(0, 0), (4, 4), (4, 4), (0, 0)], mode='reflect')

