import tensorflow as tf
from BinaryOp import *

a = tf.constant(-1., dtype=tf.float32)
b = 5 * a
c = 6 * a
binarized1 = Binarize(b)
binarized2 = Binarize(c)
g = tf.gradients(b, a)
gb = tf.gradients(binarized1, a)
gc = tf.gradients(binarized2, a)
print(g, gb, gc)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run([b, binarized1, binarized2]))
    print(sess.run([g, gb, gc]))