import numpy as np

import tensorflow as tf

import utils as u

from capslayer import primary_caps_layer, digit_caps_layer
from config import cfg


def weight_variable(name, shape):
    """
    Create a weight variable of a certain shape using the Xavier initialization.

    :param name: The name of the weight variable
    :param shape: The shape of the weight tensor
    :return: The Xavier-initialized weights
    """
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    """
    Create bias variables of a certain shape.

    :param shape: The shape of the biases tensor
    :return: The initialized biases
    """
    initial = tf.constant(0.005, shape=shape)
    return tf.Variable(initial)


class CapsNet():

    def __init__(self):
        self.batch_size = cfg.batch_size

        # Conv1: a 9x9 filter of depth 1, extracts 128 features.
        self.conv1_weights = weight_variable("conv1_weights", shape=[9, 9, 1, 128])
        self.conv1_biases = bias_variable(shape=[128])

        # Conv2: a 9x9 filter of depth 128, extracts 256 features.
        self.conv2_weights = weight_variable("conv2_weights", shape=[9, 9, 128, 256])
        self.conv2_biases = bias_variable(shape=[256])

        # Capsule
        # First, a 9x9 filter with depth 256 that extacts 32*8 features
        # Then, the weights for the DigitCaps, a fully-connected layer,
        #   of size [1, num_caps_i=1152, num_caps_j=10, channels_in=8, channels_out=16]
        self.capsule_conv_weights = weight_variable("capsule_conv_weights", shape=[9, 9, 256, 32 * 8])
        self.capsule_fc_weights = weight_variable("capsule_fc_weights", shape=[1, 1152, 10, 8, 16])

        # Decoder
        # Takes the 10x16 digit capsules, and does FC layers of sizes
        # 512 -> 1024 -> 2048 -> 1296 = 36*36, the size of the reconstructed image.
        self.reconstruction_fc_1_weights = weight_variable("reconstruction_fc_1_weights", shape=[10 * 16, 512])
        self.reconstruction_fc_1_bias = bias_variable(shape=[512])

        self.reconstruction_fc_2_weights = weight_variable("reconstruction_fc_2_weights", shape=[512, 1024])
        self.reconstruction_fc_2_bias = bias_variable(shape=[1024])

        self.reconstruction_fc_3_weights = weight_variable("reconstruction_fc_3_weights", shape=[1024, 2048])
        self.reconstruction_fc_3_bias = bias_variable(shape=[2048])

        self.reconstruction_fc_4_weights = weight_variable("reconstruction_fc_4_weights", shape=[2048, 36 * 36])
        self.reconstruction_fc_4_bias = bias_variable(shape=[1296])




        # Memo weights
        # Conv1: a 9x9 filter of depth 1, extracts 128 features.
        self.memo_conv1_weights = weight_variable("memo_conv1_weights", shape=[9, 9, 11, 128])
        self.memo_conv1_biases = bias_variable(shape=[128])

        # Conv2: a 9x9 filter of depth 128, extracts 256 features.
        self.memo_conv2_weights = weight_variable("memo_conv2_weights", shape=[9, 9, 128, 256])
        self.memo_conv2_biases = bias_variable(shape=[256])

        # Capsule
        # First, a 9x9 filter with depth 256 that extacts 32*8 features
        # Then, the weights for the DigitCaps, a fully-connected layer,
        #   of size [1, num_caps_i=1152, num_caps_j=10, channels_in=8, channels_out=16]
        self.memo_capsule_conv_weights = weight_variable("memo_capsule_conv_weights", shape=[9, 9, 256, 32 * 8])
        self.memo_capsule_fc_weights = weight_variable("memo_capsule_fc_weights", shape=[1, 1152, 10, 8, 16])


    def _image_to_digitcaps(self, X):
        # Apply the first convolution.
        with tf.variable_scope('Conv1_layer'):
            # The input is X, an image of size [batch_size, 36, 36, 1].
            # Apply a 9x9 filter of depth 1 that extracts 128 features,
            # so the output becomes a tensor of size [batch_size, 28, 28, 128].
            conv1 = tf.nn.conv2d(input=X, filter=self.conv1_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='VALID') + \
                    self.conv1_biases

            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            assert conv1.get_shape() == [self.batch_size, 28, 28, 128]

        # Apply the second convolution.
        with tf.variable_scope('Conv2_layer'):
            # The input is conv1, a feature matrix of size [batch_size, 28, 28, 128].
            # Apply a 9x9 filter of depth 128 that extracts 256 features,
            # so the output becomes a tensor of size [batch_size, 20, 20, 256].
            conv2 = tf.nn.conv2d(input=conv1, filter=self.conv2_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='VALID') + \
                    self.conv2_biases

            conv2 = tf.nn.leaky_relu(conv2, alpha=0.15)

            assert conv2.get_shape() == [self.batch_size, 20, 20, 256]

        # Apply the primary caps layer.
        with tf.variable_scope('PrimaryCaps_layer'):
            # The input is conv2, a feature matrix of size [batch_size, 20, 20, 256].
            # Apply a 9x9 filter with depth 256 and stride 2 that extracts 32*8 features,
            # which gives us a result of shape [batch_size, 6, 6, 32*8].
            # After applying the proper ReLU activation, this gets reshaped into [batch_size, 1152, 8, 1]
            caps1 = primary_caps_layer(X=conv2, caps_w=self.capsule_conv_weights,
                                       num_outputs=32, vec_len=8,
                                       strides=[1, 2, 2, 1],
                                       padding='VALID')

            # TODO: Activation function?

            assert caps1.get_shape() == [self.batch_size, 1152, 8, 1]

        # Apply the digit caps layer.
        with tf.variable_scope('DigitCaps_layer'):
            # The input is caps1, the capsule tensor of size [batch_size, 6x6x8x4, 8, 1]
            # The digit caps layer applies routing, and returns the digit capsules
            # of size [BATCH_SIZE, 10, 16, 1].
            caps2 = digit_caps_layer(caps1, self.capsule_fc_weights, num_outputs=10)

        return caps2

    def _digitcaps_to_image(self, digitcaps, Y):
        def _decode_image(Y):
            """
            Decodes an image from the digitcaps given a label,
            using two fully-connected layers

            (fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> sigmoid -> ??? -> Profit!)

            :param Y: a one-hot encoding of the label
            :return: the
            """
            # Get the masked vector from the digit capsules based on what vector we're considering
            masked_v = tf.multiply(digitcaps[:, :, :, 0], tf.expand_dims(Y, 2))

            # Reshape into [batch_size, 160]
            vector_j = tf.reshape(masked_v, shape=[self.batch_size, 160])

            # Apply fully connected layers to go from 160 -> 512 -> 1024 = 32*32 = decoded image!
            fc1 = tf.contrib.layers.fully_connected(inputs=vector_j, num_outputs=512,
                                                    biases_initializer=tf.constant_initializer(0.005))

            fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1024,
                                                    biases_initializer=tf.constant_initializer(0.005))

            fc3 = tf.contrib.layers.fully_connected(inputs=fc2, num_outputs=1296,
                                                    activation_fn=tf.nn.sigmoid,
                                                    biases_initializer=tf.constant_initializer(0.005))

            decoded_image = fc3

            return decoded_image

        # Y is a [batch_size, 2, 10] tensor that contains one-hot encodings of the labels.
        # For instance, let's say our image contains a 3 and a 7.
        # Then Y would be the following tensor:
        # [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]

        # Get each label separately
        label_1 = Y[:, 0, :]
        label_2 = Y[:, 1, :]

        reconstruction_1 = _decode_image(label_1)
        recontsruction_2 = _decode_image(label_2)

        reconstructed_image = reconstruction_1 + recontsruction_2

        return reconstructed_image, reconstruction_1, recontsruction_2


    def _memo_to_digitcaps(self, X, keep_prob):
        # Apply the first convolution.
        with tf.variable_scope('memo_Conv1_layer'):
            # The input is X, an image of size [batch_size, 36, 36, 1].
            # Apply a 9x9 filter of depth 1 that extracts 128 features,
            # so the output becomes a tensor of size [batch_size, 28, 28, 128].
            conv1 = tf.nn.conv2d(input=X, filter=self.memo_conv1_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='VALID') + \
                    self.conv1_biases

            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

            assert conv1.get_shape() == [self.batch_size, 28, 28, 128]

        # Apply the second convolution.
        with tf.variable_scope('memo_Conv2_layer'):
            # The input is conv1, a feature matrix of size [batch_size, 28, 28, 128].
            # Apply a 9x9 filter of depth 128 that extracts 256 features,
            # so the output becomes a tensor of size [batch_size, 20, 20, 256].
            conv2 = tf.nn.conv2d(input=conv1, filter=self.memo_conv2_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='VALID') + \
                    self.conv2_biases

            conv2 = tf.nn.leaky_relu(conv2, alpha=0.15)

            assert conv2.get_shape() == [self.batch_size, 20, 20, 256]

        # Apply the primary caps layer.
        with tf.variable_scope('memo_Caps_layer'):
            # The input is conv2, a feature matrix of size [batch_size, 20, 20, 256].
            # Apply a 9x9 filter with depth 256 and stride 2 that extracts 32*8 features,
            # which gives us a result of shape [batch_size, 6, 6, 32*8].
            # After applying the proper ReLU activation, this gets reshaped into [batch_size, 1152, 8, 1]
            caps1 = primary_caps_layer(X=conv2, caps_w=self.memo_capsule_conv_weights,
                                       num_outputs=32, vec_len=8,
                                       strides=[1, 2, 2, 1],
                                       padding='VALID')

            assert caps1.get_shape() == [self.batch_size, 1152, 8, 1]

            flattened = tf.reshape(caps1, shape=[self.batch_size, 9216])
            fc1 = tf.contrib.layers.fully_connected(inputs=flattened, num_outputs=1024,
                                                    biases_initializer=tf.constant_initializer(0.005),
                                                    scope='fc1',
                                                    reuse=tf.AUTO_REUSE)
            fc1_drop = tf.nn.dropout(fc1,keep_prob=keep_prob)
            fc2 = tf.contrib.layers.fully_connected(inputs=fc1_drop, num_outputs=160,
                                                    biases_initializer=tf.constant_initializer(0.005),
                                                    scope='fc2',
                                                    reuse=tf.AUTO_REUSE)

            reshaped_capsules = tf.reshape(fc2, shape=[self.batch_size, 10, 16, 1])

        return reshaped_capsules

    def _digitcaps_to_memo(self, X, digitcaps):


        def _decode_memo(digitcaps, Y):
            """
            Decodes an image from the digitcaps given a label,
            using two fully-connected layers

            (fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> sigmoid -> ??? -> Profit!)

            :param Y: a one-hot encoding of the label
            :return: the
            """
            # Get the masked vector from the digit capsules based on what vector we're considering
            masked_v = tf.multiply(digitcaps[:, :, :, 0], tf.expand_dims(Y, 2))

            # Reshape into [batch_size * 10, 160]
            vector_j = tf.reshape(masked_v, shape=[self.batch_size * 10, 160])

            # Apply fully connected layers to go from 160 -> 512 -> 1024 = 32*32 = decoded image(s)!
            fc1 = tf.contrib.layers.fully_connected(inputs=vector_j, num_outputs=512,
                                                    biases_initializer=tf.constant_initializer(0.005))

            fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1024,
                                                    biases_initializer=tf.constant_initializer(0.005))

            fc3 = tf.contrib.layers.fully_connected(inputs=fc2, num_outputs=1296,
                                                    activation_fn=tf.nn.sigmoid,
                                                    biases_initializer=tf.constant_initializer(0.005))

            decoded_image = fc3

            return decoded_image


        # Tile the digitcaps ten times into shape [batch_size * 10, 10, 16, 1]
        digitcaps = tf.tile(digitcaps, [10, 1, 1, 1])

        # Decode all ten images at the same time
        memo = _decode_memo(digitcaps, u.generate_Ys_hot(self.batch_size))

        # reshape [1280,1296] into [batch_size, 36, 36, 10]
        memo = tf.reshape(memo, [self.batch_size, 10, 36, 36])
        memo = tf.transpose(memo, [0, 2, 3, 1])

        # stop gradient
        # memo_stopped = tf.stop_gradient(memo) #FIXME

        # concatenate the initial image
        memo_hat = tf.concat([memo * X, X], axis=3)  # FIXME test - not giving X

        return memo_hat


    def compute_output(self, X, Y, keep_prob=cfg.keep_prob, regularization_scale=cfg.regularization_scale):

        print("Size of input:")
        print(X.get_shape())

        # 1. Convolve the input image up to the digit capsules.
        digit_caps = self._image_to_digitcaps(X)

        # 2. Get the margin loss
        margin_loss = u.margin_loss(digit_caps, Y)

        # 3. Reconstruct the images
        reconstructed_image, reconstruction_1, reconstruction_2 = self._digitcaps_to_image(digit_caps, Y)

        # 4. Get the reconstruction loss
        reconstruction_loss = u.reconstruction_loss(reconstructed_image, X)

        # 5. Get the total loss
        total_loss = margin_loss + regularization_scale * reconstruction_loss

        # 6. Get the batch accuracy
        batch_accuracy = u.acc(digit_caps, Y)

        # 7. Reconstruct all possible images
        memo = self._digitcaps_to_memo(X, digit_caps)

        # 8. Get the memo capsules
        memo_caps = self._memo_to_digitcaps(memo, keep_prob=keep_prob)

        # 9. Get the memo margin loss
        memo_margin_loss = u.margin_loss(memo_caps, Y)

        # 10. Get the memo accuracy
        memo_accuracy = u.acc(memo_caps, Y)

        # 11. Return all of the losses and reconstructions
        return (total_loss,
                margin_loss,
                reconstruction_loss,
                reconstructed_image,
                reconstruction_1,
                reconstruction_2,
                batch_accuracy,
                memo,
                memo_margin_loss,
                memo_accuracy)

