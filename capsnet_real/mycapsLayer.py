import numpy as np
import tensorflow as tf
from myconfig import cfg
epsilon = 1e-9
import time
import logging




def primary_caps_layer(X, caps_w, num_outputs, vec_len, strides, padding):
    # fixme assert shapes
    # the PrimaryCaps layer, a convolutional layer
    capsules = tf.nn.conv2d(X, caps_w, strides=strides, padding=padding)
    capsules = tf.nn.relu(capsules)
    #print "capsules:", capsules
    #time.sleep(100)
    capsules = tf.reshape(capsules, (cfg.batch_size, -1, vec_len, 1))
    # [batch_size, 1152, 8, 1]
    capsules = squash(capsules)
    return capsules


def digit_caps_layer(X,caps_w,num_outputs):
        # the DigitCaps layer, a fully connected layer
        # Reshape the input into [batch_size, 1152, 1, 8, 1]
        X = tf.reshape(X, shape=(cfg.batch_size, -1, 1, X.shape[-2].value, 1))
        with tf.variable_scope('routing'):
            # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
            # about the reason of using 'batch_size', see issue #21
            b_IJ = tf.constant(np.zeros([cfg.batch_size, X.shape[1].value, num_outputs, 1, 1], dtype=np.float32))
            capsules = routing(X, caps_w, b_IJ)
            capsules = tf.squeeze(capsules, axis=1)
        return capsules



def routing(input, W, b_IJ):
    ''' The routing algorithm.

    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    print "running routing algorithm with args:",input, W, b_IJ

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    print "tiled input to :",input
    W = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1])
    assert input.get_shape() == [cfg.batch_size, 3200, 10, 8, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    u_hat = tf.matmul(W, input, transpose_a=True)
    print "u_hat:",u_hat
    assert u_hat.get_shape() == [cfg.batch_size, 3200, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(cfg.iter_routing):
        print "in dynamic routing iteration",r_iter
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [1, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)
            print "c_IJ", c_IJ
            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == cfg.iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < cfg.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                print "tf * c_IJ u_hat_stopped, sj:", c_IJ,u_hat_stopped,s_J
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                print "s_J",s_J
                v_J = squash(s_J)
                print "v_J",v_J
                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 10, 1152, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, 3200, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                print "u_produce_v = uhat x v_J_tiled", u_produce_v,u_hat_stopped,v_J_tiled
                assert u_produce_v.get_shape() == [cfg.batch_size, 3200, 10, 1, 1]
                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v
    return v_J


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return (vec_squashed)


