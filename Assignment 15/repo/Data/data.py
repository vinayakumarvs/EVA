#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ..imports import *
from ..GDrive import  gdrive as gd
from ..GDrive import images as imgs
from ..aug import *

import tensorflow as tf
import os
import sys
from keras.datasets import cifar10

def int64_feature(value) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def input_fn(file_names, batch_size: int, is_train_file=True):

    def parser(record, is_train):
        keys_to_features = {
            "image_raw": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed["image_raw"], tf.float32)
        label = tf.cast(parsed["label"], tf.int64)
        image = tf.reshape(image, [32, 32, 3])

        # Random crop and random flip from left to right in case of training images
        if is_train:
            image = tf.image.random_flip_left_right(tf.random_crop(image, [32, 32, 3]))
            image = cutout(image, 10, 10)

        return image, label

    tf_record_data_set = tf.data.TFRecordDataset(filenames=file_names, num_parallel_reads=40)
    tf_record_data_set = tf_record_data_set.map(lambda x: parser(x, is_train_file)).shuffle(True).batch(batch_size)

    tf_record_iterator = tf_record_data_set.make_one_shot_iterator()
    return tf_record_iterator


def get_tf_record_count(file_name):
    return sum(1 for _ in tf.python_io.tf_record_iterator(file_name))


class DataSet(object):
    data_path: str

    def __init__(self):
        gd.mount_google_drive()
        os.listdir('../content/drive/My Drive/datasets/')
        self.data_path = '../content/drive/My Drive/datasets/'

    def create_data_record(self, out_filename, addrs, labels, is_image: bool = False):
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(out_filename)

        for i in range(len(addrs)):
            # print how many images are saved every 1000 images
            if not i % 1000:
                print('{} data: {}/{}'.format(out_filename, i, len(addrs)))
                sys.stdout.flush()

            # Load the image
            if is_image:
                img = addrs[i]
            else:
                img = imgs.load_image(addrs[i])

            if img is None:
                continue

            label = labels[i]

            # Create a feature
            feature = {
                'image_raw': bytes_feature(img.tostring()),
                'label': int64_feature(label)
            }
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

    def get_tf_records(self, data_set_name):

        if data_set_name is None:
            return

        train_file_path = self.data_path + data_set_name + '.train.tfrecords'
        test_file_path = self.data_path + data_set_name + '.test.tfrecords'

        if 'cifar10' in data_set_name:
            if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
                (train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
                self.create_data_record(train_file_path, normalize(train_features), train_labels, True)
                self.create_data_record(test_file_path, normalize(test_features), test_labels, True)
        elif 'mnist' in data_set_name:
            if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
                (train_features, train_labels), (test_features, test_labels) = mnist.load_data()
                self.create_data_record(train_file_path, normalize(train_features), train_labels, True)
                self.create_data_record(test_file_path, normalize(test_features), test_labels, True)
        else:
            raise Exception("The data set {} is not supported.".format(data_set_name))

        return train_file_path, test_file_path

