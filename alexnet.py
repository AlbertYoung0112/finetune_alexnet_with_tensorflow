"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from BinaryOp import *

SET_BINARY = True


class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT', training=False):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.training = tf.convert_to_tensor(training, dtype='bool',
                                             name='is_training')

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        #norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        norm1 = bn(conv1, is_training=self.training, name='bn1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
        bin1 = pool1
        if(SET_BINARY):
            bin1 = Binarize(pool1)

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(bin1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        #norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        norm2 = bn(conv2, is_training=self.training, name='bn2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
        bin2 = pool2
        if(SET_BINARY):
            bin2 = Binarize(pool2)

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(bin2, 3, 3, 384, 1, 1, name='conv3')
        norm3 = bn(conv3, is_training=self.training, name='bn3')
        bin3 = norm3
        if(SET_BINARY):
            bin3 = Binarize(norm3)

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(bin3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        norm4 = bn(conv4, is_training=self.training, name='bn4')
        bin4 = norm4
        if(SET_BINARY):
            bin4 = Binarize(norm4)

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(bin4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        norm5 = bn(pool5, is_training=self.training, name='bn5')
        bin5 = norm5
        if(SET_BINARY):
            bin5 = Binarize(norm5)

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(bin5, [-1, 6*6*256])
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')
        #self.fc8 = tf.Print(fc8, [fc8[0,:]], message="FC8", summarize=10)

    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        #weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()
        weights_dict = {}

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if(SET_BINARY):
        weights = Binarize(weights)

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if(SET_BINARY):
        weights = Binarize(weights)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def bn(x, name, use_bias=False, is_training=False):
    MOVING_AVERAGE_DECAY = 0.9997
    BN_DECAY = MOVING_AVERAGE_DECAY
    BN_EPSILON = 0.001
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if use_bias:
        bias = tf.get_variable('bias', shape=params_shape,
                                  initializer=tf.zeros_initializer)
        return x + bias

    axis = list(range(len(x_shape) - 1))

    with tf.variable_scope(name) as scope:
        beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
        moving_mean = tf.get_variable('moving_mean', params_shape,
                                      initializer=tf.zeros_initializer, trainable=False)
        moving_variance = tf.get_variable('moving_variance', params_shape,
                                      initializer=tf.ones_initializer, trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection('update_op', update_moving_mean)
    tf.add_to_collection('update_op', update_moving_variance)
    #moving_mean = tf.Print(moving_mean, [moving_mean], message='MovMean')

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    # x.set_shape(inputs.get_shape()) ??

    return x


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)
