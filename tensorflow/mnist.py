import struct
import time

import numpy as np
import matplotlib.pyplot as plt
import random
from perceptrons import multiple
import datetime

import tensorflow as tf


def read_mnist_data(mnist_dir):
    train_images_path = mnist_dir + 'train-images-idx3-ubyte'
    train_labels_path = mnist_dir + 'train-labels-idx1-ubyte'
    test_images_path = mnist_dir + 't10k-images-idx3-ubyte'
    test_labels_path = mnist_dir + 't10k-labels-idx1-ubyte'

    return decode_idx3_images(train_images_path), decode_idx1_labels(train_labels_path), \
           decode_idx3_images(test_images_path), decode_idx1_labels(test_labels_path)


def decode_idx3_images(path):
    bin_data = open(path, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print(path, '魔数', magic_number, '，图片数量：', num_images, ', 图片大小：', num_rows, '*', num_cols)

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'

    images = np.empty((num_images, num_rows, num_cols))

    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_labels(path):
    bin_data = open(path, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # print(path, '魔数', magic_number, '，图片数量：', num_images)

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def normalization(images, labels):
    shape = np.array(images).shape
    if not len(shape) == 3:
        return []
    result = []
    label_result = []
    for i in range(shape[0]):
        image_result = []
        for j in range(shape[1]):
            for k in range(shape[2]):
                image_result.append(images[i][j][k] / 255.0)
        result.append(image_result)
        label = labels[i]
        label_result.append([1 if label == i else 0 for i in range(0, 10)])
    return result, label_result


def get_random_data(train_images, train_labels, num):
    if not len(train_images) == len(train_labels) or num < 1 or num > len(train_images):
        return False, False
    total_index = list(range(0, len(train_images)))
    random.shuffle(total_index)

    random_images = []
    random_labels = []
    for i in range(num):
        random_images.append(train_images[total_index[i]])
        random_labels.append(train_labels[total_index[i]])
    return random_images, random_labels


def train_model(train_images, train_labels, test_images, test_labels, times=1000, batch=100):
    # 2 layer
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))  # output layer weights
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b

    # 3 layer
    # hide_layer_num = 1024
    # x = tf.placeholder(tf.float32, [None, 784])
    # w = tf.Variable(tf.zeros([784, hide_layer_num]))  # output layer weights
    # b = tf.Variable(tf.zeros([hide_layer_num]))
    #
    # middle_layer_num = 512
    # x2 = tf.matmul(x, w) + b
    # w2 = tf.Variable(tf.zeros([hide_layer_num, middle_layer_num]))
    # b2 = tf.Variable(tf.zeros([middle_layer_num]))
    # x3 = tf.matmul(x2, w2) + b2
    # w3 = tf.Variable(tf.zeros([middle_layer_num, 10]))
    # b3 = tf.Variable(tf.zeros([10]))
    # y = tf.matmul(x3, w3) + b3

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.relu(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)))
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    loss_results = []
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()
        for i in range(times):
            x_images, y_label = normalization(*get_random_data(train_images, train_labels, batch))
            loss, _ = sess.run([cross_entropy, train_step], {
                x: x_images,
                y_: y_label,
            })
            loss_results.append([i, loss])

        x_test_images, y_test_labels = normalization(test_images, test_labels)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
        rate = sess.run(accuracy, feed_dict={
            x: x_test_images,
            y_: y_test_labels,
        })
        print('accuracy rate', rate)
    return loss_results, []


def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variables(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


def train_model_conv(train_images, train_labels, test_images, test_labels, times=1000, batch=100):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    x_images = tf.reshape(x, [-1, 28, 28, 1])

    # conv1
    W_conv1 = weight_variables([5, 5, 1, 32])
    b_conv1 = bias_variables([32])
    h_conv1 = tf.nn.relu(conv2d(x_images, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # conv2
    W_conv2 = weight_variables([5, 5, 32, 64])
    b_conv2 = bias_variables([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # fc1
    W_fc1 = weight_variables([7 * 7 * 64, 1024])
    b_fc1 = bias_variables([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # drop
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # out
    W_fc2 = weight_variables([1024, 10])
    b_fc2 = bias_variables([10])
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    )
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss_results = []
    rate_results = []
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()
        for i in range(times):
            x_images, y_label = normalization(*get_random_data(train_images, train_labels, batch))
            rate, loss, _ = sess.run([accuracy, cross_entropy, train_step], {
                x: x_images,
                y_: y_label,
                keep_prob: 1.0,
            })
            loss_results.append([i, loss])
            rate_results.append([i, rate])

        x_test_images, y_test_labels = normalization(test_images, test_labels)
        rate = sess.run(accuracy, feed_dict={
            x: x_test_images,
            y_: y_test_labels,
            keep_prob: 1.0,
        })
        print('final accuracy rate', rate)
    return loss_results, rate_results


def norm(x, lsize=4):
    return tf.nn.lrn(x, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


def get_alex_conv(size, input):
    W_conv = weight_variables(size)
    b_conv = bias_variables([size[-1]])
    h_conv = tf.nn.relu(conv2d(input, W_conv) + b_conv)
    h_pool = max_pool_3x3(h_conv)
    h_normal = norm(h_pool, lsize=4)

    return h_normal


def train_model_alex(train_images, train_labels, test_images, test_labels, times=1000, batch=100, save_path="mnist-result/alex-net.ckpt"):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_images = tf.reshape(x, [-1, 28, 28, 1])

    # conv 1 2 3
    conv1 = get_alex_conv([3, 3, 1, 64], x_images)
    conv2 = get_alex_conv([3, 3, 64, 128], conv1)
    conv3 = get_alex_conv([3, 3, 128, 256], conv2)

    # fc1 + dropout
    W_fc1 = weight_variables([4*4*256, 1024])
    b_fc1 = bias_variables([1024])
    h_norm3_flat = tf.reshape(conv3, [-1, 4*4*256])
    h_fc1 = tf.nn.relu(tf.matmul(h_norm3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fc2 + dropout
    W_fc2 = weight_variables([1024, 1024])
    b_fc2 = bias_variables([1024])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # output
    W_fc3 = weight_variables([1024, 10])
    b_fc3 = bias_variables([10])
    y = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    )
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss_results = []
    rate_results = []
    saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()
        for i in range(times):
            x_images, y_label = normalization(*get_random_data(train_images, train_labels, batch))
            rate, loss, _ = sess.run([accuracy, cross_entropy, train_step], {
                x: x_images,
                y_: y_label,
                keep_prob: 1.0,
            })
            loss_results.append([i, loss])
            rate_results.append([i, rate])

        x_test_images, y_test_labels = normalization(test_images, test_labels)
        rate = sess.run(accuracy, feed_dict={
            x: x_test_images,
            y_: y_test_labels,
            keep_prob: 1.0,
        })
        print('final accuracy rate', rate)
        saver.save(sess, save_path)
    return loss_results, rate_results


save_path = 'mnist-result/alex-net.ckpt'

def main():
    start = datetime.datetime.now()
    train_images, train_labels, test_images, test_labels = read_mnist_data('./mnist-data/')
    end = datetime.datetime.now()

    print('load data times:', (end - start))
    # print(np.array(train_images).shape)
    # print(np.array(train_labels).shape)
    # print(np.array(test_images).shape)
    # print(np.array(test_labels).shape)
    #
    # for i in range(10):
    #     print(train_labels[i])
    #     plt.imshow(train_images[i], cmap='gray')
    #     plt.show()

    start = datetime.datetime.now()
    loss_result, rate_result = train_model_alex(train_images, train_labels, test_images, test_labels, 1000, 300, save_path)
    end = datetime.datetime.now()
    print('train times:', (end - start))
    multiple.show_line_graph([
        [loss_result, 'r-', 'loss'],
        [rate_result, 'b-', 'rate'],
    ], 'epoch', 'data')


if __name__ == '__main__':
    main()
