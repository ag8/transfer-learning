import numpy as np

import tensorflow as tf

from config import cfg

epsilon = 1e-9

def primary_caps_layer(X, caps_w, num_outputs, vec_len, strides, padding):
    """
    The PrimaryCaps layer, a convolutional layer
    :param X: The input
    :param caps_w: The capsule weights
    :param num_outputs: The number of outputs (not used)
    :param vec_len: The length of the capsule vectors (dimensions)
    :param strides: Convolution strides
    :param padding: Convolution padding
    :return: The updated capsules
    """

    # First, convolve over the input with the given kernel, strides, and padding
    capsules = tf.nn.conv2d(X, filter=caps_w, strides=strides, padding=padding)

    # Apply a ReLU nonlinearity to the capsules
    capsules = tf.nn.relu(capsules)

    # Reshape the capsules into the appropriate shape:
    # [BATCH_SIZE, 1152, 8, 1]
    # Where 8 is the number of dimensions the capsule vectors are watching
    # And 1152 is 9216/8
    capsules = tf.reshape(capsules, shape=[X.get_shape()[0], -1, vec_len, 1])

    # Apply the squash nonlinearity to the capsules
    # so that they represent a probability
    # and have other nice features
    capsules = squash(capsules)

    # Return the capsules
    return capsules


def digit_caps_layer(X, caps_w, num_outputs):
    """
    The DigitCaps layer, a fully-connected layer
    :param X:
    :param caps_w:
    :param num_outputs:
    :return:
    """

    # Reshape the input into [BATCH_SIZE, 1152, 1, 8, 1]
    # Where 8 is the number of dimensions of the capsule vectors
    X = tf.reshape(X, shape=[X.get_shape()[0], -1, 1, X.shape[-2].value, 1])


    with tf.variable_scope('routing'):
        # b_IJ has shape [BATCH_SIZE, number of capsules in layer, number of capsules in next layer, 1, 1]
        # Initialize it to all zeroes at the beginning
        b_IJ = tf.constant(np.zeros([X.get_shape()[0], X.shape[1].value, num_outputs, 1, 1], dtype=np.float32))

        # Apply routing
        capsules = routing(X, caps_w, b_IJ)

        # Remove the extra dimensions
        capsules = tf.squeeze(capsules, axis=1)

    # Return the capsules
    return capsules


def routing(input, W, b_IJ):
    """
    The routing algorithm.

    :param input: A tensor of shape [BATCH_SIZE, num_caps_l=1152, 1, length(u_i)=8, 1]
                    (num_caps_l refers to the number of capsules in layer l).
    :param W: The capsule weights, of shape [1, num_caps_i=1152, num_caps_j=10, len_u_i=8, len_v_j=16]
    :param b_IJ: The current activations, of shape [BATCH_SIZE, num_caps_i, num_caps_j, 1, 1]
    :return: A tensor of shape [BATCH_SIZE, num_caps_l_plus_1, length(v_j)=16, 1],
             representing the vector output `v_j` in the layer l+1.


    Notes:
        u_i represents the vector output of capsule `i` in the layer `l`, and
        v_j the vector output of capsule `j` in the layer `l+1`.
    """

    # Get the batch size from the input
    batch_size = input.get_shape()[0]


    # Currently, the shapes are:
    # input: [BATCH_SIZE, 1152, 1, 8, 1]
    # W: [1, 1152, 10, 8, 16]

    # Now, calculate u_hat according to Eq. 2
    # Tile the input and W matrices before multiplication:
    # input => [BATCH_SIZE, 1152, 10, 8, 1]
    # W => [BATCH_SIZE, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [batch_size, 1, 1, 1, 1])

    assert input.get_shape() == [batch_size, 1152, 10, 8, 1]


    # Now, do the matrix multiplication: u_hat = W * u_i
    # (where u_i is input, in this case).
    # In the last two dimensions, we have
    # [8, 16].T x [8, 1] => [16, 1] => [BATCH_SIZE, 1152, 10, 16, 1].
    #
    # Benchmark: tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    u_hat = tf.matmul(W, input, transpose_a=True)

    assert u_hat.get_shape() == [batch_size, 1152, 10, 16, 1]


    # In forward, u_hat_stopped = u_hat;
    # In backward, no gradient is passed back from u_hat_stopped to u_hat
    # Basically, the reason for this is that we don't want to apply
    #   backpropagation within the inner loop of the routing algorithm;
    #   however, it's still calculated via tensorflow, so we just use a
    #   variable (u_hat_stopped) that's masked from the gradient calculator.
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # Line 3: for `r` iterations do:
    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # Line 4: for all capsule `i` in layer `l`, c_i <-- softmax(b_i)
            # c_IJ has shape [1, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)


            # At the last routing iteration, use `u_hat` in order to receive
            #   the gradients from the following graph
            if r_iter == cfg.iter_routing - 1:

                # Line 5: for all capsule `j` in layer `l+1`: s_j <-- sum over c_ij * u_hat
                # That is, weigh u_hat with c_IJ, element-wise, in the last two dims.
                # s_J has shape [BATCH_SIZE, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)

                # Then sum the second dimension, resulting in
                # s_J having a shape of [BATCH_SIZE, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)


                assert s_J.get_shape() == [batch_size, 1, 10, 16, 1]


                # Line 6: for all capsule `j` in layer `l+1`, v_j <-- squash(s_j)
                # So, simply squash s_J using Eq. 1.
                # v_J retains the shape of s_J; namely, [BATCH_IZE, 1, 10, 16, 1]
                v_J = squash(s_J)


                assert v_J.get_shape() == [batch_size, 1, 10, 16, 1]

            elif r_iter < cfg.iter_routing - 1:  # Inner iterations; do not apply backpropagation.

                # Lines 5 and 6, but calculated with u_hat_stopped instead of u_hat
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)


                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 10, 1152, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]


                # Line 7: for all capsule `i` in layer `l` and capsule `j` in layer `l+1`: b_ij <-- b_ij + u_hat dot v_j
                # The way we do this is reshape and tile v_j
                # from [BATCH_SIZE, 1, 10, 16, 1] to [BATCH_SIZE, 10, 1152, 16, 1]
                # Then, matmul in the last two dimensions: [16, 1].T x [16, 1] => [1, 1],
                # then reduce mean in the BATCH_SIZE dimension, resulting in a
                # b_IJ of shape [1, 1152, 10, 1, 1].
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)

                assert u_produce_v.get_shape() == [batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v


    # Return the next layer of capsules
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
