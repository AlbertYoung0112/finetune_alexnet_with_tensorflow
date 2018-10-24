import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datagenerator import IMAGENET_MEAN
from datetime import datetime
from tensorflow.data import Iterator
import scipy.misc

test_file = '/home/iair-p05/Projects/Dataset/MNIST/TestImageList.txt'
#weight_file = './checkpoints/2018-09-09 22:48:06.238454/model_epoch4.ckpt'
weight_file = './checkpoints/2018-10-16 21:25:10.327862/model_epoch13.ckpt'

filewriter_path = "./tensorboard/Test/" + str(datetime.now())
checkpoint_path = "./checkpoints"
wrongImage_path = "./WrongImage/Test" + str(datetime.now()) + '/'
SAVE_WRONG_IMAGE = True
os.makedirs(wrongImage_path)
batch_size = 32
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
model = AlexNet(x, keep_prob, num_classes, None, training=False)

# Link variable to model output
score = model.fc8

softmax = tf.nn.softmax(score)

infResult = tf.argmax(softmax, axis=1)
infResult = tf.Print(infResult, [infResult], message="Inference: ", summarize=batch_size)

with tf.name_scope("CrossEntropy"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score,
                                                                     labels = y))

tf.summary.histogram(score.name, score)

with tf.name_scope("TestAccuracy"):
    #correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    scaledScore = tf.div(score, tf.reshape(tf.reduce_max(score, 1), (batch_size, 1)))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    accuracy = tf.placeholder(tf.float16, name='TestAccuracy')


tf.summary.scalar('TestAccuracy', accuracy)
tf.summary.histogram('TestErrorConfidence', scaledScore)


merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))
#test_batches_per_epoch = 5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    saver.restore(sess, weight_file)
    print('Model restored')
    sess.run(test_init_op)
    wrongImage = [[] for i in range(10)]
    wrongOutput = [[] for i in range(10)]
    imageCount = 0
    wrongImageCount = [0 for i in range(10)]
    totalWrongImageCount = 0
    lalala = []
    tf.add_to_collection('update_op', infResult)
    tf.add_to_collection('update_op', score)
    update_op = tf.get_collection('update_op')
    for step in range(test_batches_per_epoch):
        print("Batch: ", step)
        img_batch, label_batch = sess.run(next_batch)
        _, infResultOut, scoreOut = sess.run([update_op, infResult, score], feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})
        if step <= 2:
            continue
        label = np.argmax(label_batch, axis=1)
        print("Label:    ", label)
        imageCount += batch_size
        for i in range(batch_size):
            if label[i] != infResultOut[i]:
                wrongImage[infResultOut[i]].append((img_batch[i], label[i]))
                wrongOutput[infResultOut[i]].append(score[i])
                wrongImageCount[infResultOut[i]] += 1
                totalWrongImageCount += 1
                lalala.append(label[i])
        acc = 1 - totalWrongImageCount / imageCount
        s = sess.run(merged_summary, feed_dict={x: img_batch, y:label_batch,
                                                keep_prob:1,
                                                accuracy: acc})
        writer.add_summary(s, global_step=step)

    aaaa = tf.summary.histogram("WrongImageStat", lalala)
    aaa = sess.run(aaaa)
    writer.add_summary(aaa, global_step=test_batches_per_epoch)
    print("Acc: ", (imageCount - totalWrongImageCount) / imageCount * 100, " %")
    plt.plot(wrongImageCount)
    if(SAVE_WRONG_IMAGE):
        for i in range(num_classes):
            no = 0
            for img, label in wrongImage[i]:
                wrongImageName = 'Wrong' + str(i) + \
                                 '_Label' + str(label) + \
                                 '_' + str(no) + '.png'
                wrongImageSaveRGB = (img[:, :, ::-1] + IMAGENET_MEAN) / 255
                wrongImageSaveRGB = wrongImageSaveRGB.eval()
                plt.imsave(wrongImage_path + wrongImageName, wrongImageSaveRGB)
                no += 1
    plt.show()



