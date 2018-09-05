import os

import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.data import Iterator

test_file = '/home/iair-p05/Projects/Dataset/MNIST/TestImageList.txt'
weight_file = './checkpoints/model_epoch2.ckpt'

filewriter_path = "./tensorboard"
checkpoint_path = "./checkpoints"
batch_size = 16
num_classes = 10
dropout_rate = 0

with tf.device("/cpu:0"):
    test_data = ImageDataGenerator(test_file,
                                   mode="inference",
                                   batch_size=batch_size,
                                   num_classes=num_classes,
                                   shuffle=False)

    iterator = Iterator.from_structure(test_data.data.output_types,
                                        test_data.data.output_shapes)
    next_batch = iterator.get_next()

test_init_op = iterator.make_initializer(test_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, None)

# Link variable to model output
score = model.fc8

softmax_in = tf.nn.softmax(score)
softmax = tf.Print(softmax_in, [softmax_in], summarize=10, message='output')

with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score,
                                                                     labels = y))

tf.summary.histogram(score.name, score)

with tf.name_scope("Input_Image"):
    tf.summary.image('Input Image', x, 16)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    saver.restore(sess, weight_file)
    print('Model restored')
    sess.run(test_init_op)
    for step in range(test_batches_per_epoch):
        img_batch, label_batch = sess.run(next_batch)
        sess.run(softmax, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
        s = sess.run(merged_summary, feed_dict={x: img_batch, y:label_batch, keep_prob:1})
