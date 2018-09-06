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
img = tf.placeholder(tf.float32, [10, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, None)

# Link variable to model output
score = model.fc8

softmax = tf.nn.softmax(score)

infResult = tf.argmax(softmax, axis=1)
labelResult = tf.argmax(y, axis=1)
infResult = tf.Print(infResult, [infResult],       message="Inference: ", summarize=batch_size)
labelResult = tf.Print(labelResult, [labelResult], message="Label:     ", summarize=batch_size)

with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score,
                                                                     labels = y))

tf.summary.histogram(score.name, score)

with tf.name_scope("Input_Image"):
    for i in range(batch_size):
        if labelResult[i] != infResult[i]:
            tf.summary.image('Wrong Image', x[i:i+1,:,:,:], 1)
    #tf.summary.image('Input Image', x, 10)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

#test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))
test_batches_per_epoch = 5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    saver.restore(sess, weight_file)
    print('Model restored')
    sess.run(test_init_op)
    wrongImage = [[] for i in range(10)]
    for step in range(test_batches_per_epoch):
        img_batch, label_batch = sess.run(next_batch)
        infResultOut, labelResultOut = sess.run([infResult, labelResult], feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
        s = sess.run(merged_summary, feed_dict={x: img_batch, y:label_batch, keep_prob:1})
        writer.add_summary(s)
        for i in range(batch_size):
            if labelResultOut[i] != infResultOut[i]:
                wrongImage[infResultOut[i]].append(img_batch[i])

    wrongImageSummary = [tf.summary.image("Wrong"+str(i), wrongImage[i], max_outputs=10) for i in range(10)]
    wrongImageSummary = tf.summary.merge(wrongImageSummary)
    writer.add_summary(sess.run(wrongImageSummary))

