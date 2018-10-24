import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

"""
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def ShiftBasedMul(x, y):
    return tf.multiply(x, y)

def _binarize(x):
    return np.sign(x)


def Binarize(x, name=None):
    with ops.op_scope([x], name, "binarize") as name:
        y = py_func(_binarize, [x], [tf.float32], name="binarize", grad=lambda op, grad:grad)
        return y
"""


def __binarize(x, grad):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.sign(x)

def Binarize(x):
    g = tf.get_default_graph()
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e8))

    with g.gradient_override_map({"Sign": "BinarizeGrad"}):
        return tf.sign(x, name = "Sign")




@ops.RegisterGradient("BinarizeGrad")
def _binarize_grad(op, grad):
    return tf.clip_by_value(grad, -1, 1)

"""
__binarizeOp = tf.load_op_library('libbinarizeOp.so')

def Binarize(x):
    #return __binarizeOp.binarize(x)
    return tf.sign(x)

@ops.RegisterGradient("sign")
def _binarize_grad(op, grad):
    return grad
"""


def ap2(input_tensor):
    sign = tf.sign(input_tensor)
    r = tf.round(tf.log(tf.abs(input_tensor)) / tf.log(2.))
    approximate = tf.multiply(sign, tf.pow(2., r))
    return approximate


def ShiftBasedBatchNormalization(input_tensor, offset, scale, eps=1e-5):
    offset, scale, eps = map(float, (offset, scale, eps))
    avg = tf.reduce_mean(input_tensor, axis = 0)
    # avg = tf.Print(avg, [avg], message='SBN Input avg')
    centeredInput = input_tensor - avg
    approxCenteredInput = ap2(centeredInput)
    variance = tf.reduce_mean(
        ShiftBasedMul(centeredInput, approxCenteredInput),
        axis = 0
        )
    # variance = tf.Print(variance, [variance], message='SBN Input Var')
    div = tf.rsqrt(variance + eps)
    normalized = ShiftBasedMul(centeredInput, div)
    scaleAndShifted = ShiftBasedMul(ap2(scale), normalized) + offset
    return scaleAndShifted


def ShiftBasedAdaMax(prev_param, grad, prev_moment, perv_velocity,
    learning_rate, alpha, beta1, beta2):
    # Update biased 1st and 2nd moment estimates
    moment = tf.multiply(beta1, prev_moment) + \
        tf.multiply((1 - beta1), grad)
    velocity = tf.maximum(
        tf.multiply(beta2, perv_velocity), 
        tf.abs(grad))

    # Update parameters
    param = prev_param - tf.multiply(
        ShiftBasedMul(alpha, 1 - beta1), 
        ShiftBasedMul(moment, tf.pow(velocity, -1))
        )

    return param, moment, velocity
