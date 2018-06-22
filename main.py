# -*- coding: utf-8 -*-

"""
    项目名称：基于TensorFlow的类图像识别
"""
import cifar10
import cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

max_steps = 3000
batch_size = 128
disp_step = 10
dataset_dir = '/tmp/cifar10_data/cifar-10-batches-bin'


def variable_with_weight_loss(shape, stddev, weight_loss_val):
    """
        添加了L2正则化的初始化weight函数
    """
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if weight_loss_val is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), weight_loss_val, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


def build_cnn_network(images):
    """
        创建网络结构
    """
    # 第一个卷积层
    weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, weight_loss_val=0.0)
    kernel1 = tf.nn.conv2d(images, weight1, [1, 1, 1, 1], padding='SAME')
    bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # 第二个卷积层
    weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, weight_loss_val=0.0)
    kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 全连接层
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, weight_loss_val=0.004)
    bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

    # 全连接层
    weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, weight_loss_val=0.004)
    bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
    local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

    # 输出层
    weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, weight_loss_val=0.0)
    bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
    logitis = tf.add(tf.matmul(local4, weight5), bias5)

    return logitis


def get_total_loss(logitis, labels):
    """
        计算loss
    """
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logitis, labels=labels, name='cross_entropy_per_sample'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def main():
    """
        主函数
    """
    # 下载cifar10数据集并解压
    cifar10.maybe_download_and_extract()

    # distorted_inputs产生训练数据
    images_train, labels_train = cifar10_input.distorted_inputs(data_dir=dataset_dir,
                                                                batch_size=batch_size)

    # 产生测试数据
    images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                    data_dir=dataset_dir,
                                                    batch_size=batch_size)

    # 为特征和label创建placeholder
    image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, [batch_size])

    # 创建CNN网络，并得到输出
    logitis = build_cnn_network(image_holder)

    # 计算loss
    total_loss = get_total_loss(logitis, label_holder)

    # 设置优化算法
    train_op = tf.train.AdamOptimizer(1e-3).minimize(total_loss)
    top_k_op = tf.nn.in_top_k(logitis, label_holder, 1)

    # 创建session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 启动线程操作，因为cifar10_input.distorted_inputs需要线程操作
    tf.train.start_queue_runners()

    # 训练模型
    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, total_loss],
                                 feed_dict={image_holder: image_batch,
                                            label_holder: label_batch})
        duration = time.time() - start_time

        if step % disp_step == 0:
            sample_per_sec = batch_size / duration
            sec_per_batch = float(duration)

            print('step %d, loss=%.2f (%.1f sample/sec; %.3f sec/batch)' % (
                step, loss_value, sample_per_sec, sec_per_batch
            ))

    # 测试模型
    n_test_samples = 10000
    n_iter = int(math.ceil(n_test_samples / batch_size))
    true_count = 0
    total_sample_count = n_iter * batch_size
    step = 0
    while step < n_iter:
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                      label_holder: label_batch})
        true_count += np.sum(predictions)

        step += 1

    precision = true_count / total_sample_count
    print('top 1 precision: %.3f' % precision)


if __name__ == '__main__':
    main()
