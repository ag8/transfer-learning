import tensorflow as tf

from myconfig import cfg
from mycapsLayer import primary_caps_layer, digit_caps_layer
import time


# Tell TensorFlow how we want to randomly initialize the weights
def weight_variable(name, shape):
    # initial = tf.truncated_normal(shape, stddev=0.005)
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    initial = tf.constant(0.005, shape=shape)
    return tf.Variable(initial)


epsilon = 1e-9


class CapsNet:
    """
    The capsule network.
    """

    def __init__(self, b_size=cfg.batch_size):
        # Initialize encoder weights
        self.enc_conv_w1 = weight_variable("enc_conv_w1", [9, 9, 1, 128])  # 10386
        self.enc_conv_b1 = bias_variable([128])
        self.enc_conv_w2 = weight_variable("enc_conv_w2", [9, 9, 128, 256])  # 2654208
        self.enc_conv_b2 = bias_variable([256])
        self.enc_caps_w1 = weight_variable("enc_caps_w1", [9, 9, 256, 32 * 8])  # 5308416
        self.enc_caps_fc1 = weight_variable("enc_caps_fc1", [1, 1152, 10, 8, 16])  # 1474560

        # Initialize decoder weights
        self.dec_fc1 = weight_variable("dec_fc1", [160, 512])
        self.dec_b1 = bias_variable([512])
        self.dec_fc2 = weight_variable("dec_fc2", [512, 1024])
        self.dec_b2 = bias_variable([1024])
        self.dec_fc3 = weight_variable("dec_fc3", [1024, 1296])
        self.dec_b3 = bias_variable([1296])
        self.b_size = b_size

        # Initialize classifier weights
        self.cls_conv_w1 = weight_variable("cls_conv_w1", [9, 9, 2, 128])
        self.cls_conv_b1 = bias_variable([128])
        self.cls_conv_w2 = weight_variable("cls_conv_w2", [9, 9, 128, 256])
        self.cls_conv_b2 = bias_variable([256])
        self.cls_conv_w3 = weight_variable("cls_conv_w3", [9, 9, 256, 256])
        self.cls_conv_b3 = bias_variable([256])

        # Initialize decoder weights
        self.cls_fc1 = weight_variable("cls_decod_fc1", [9216, 1024])
        self.cls_b1 = bias_variable([1024])
        self.cls_fc2 = weight_variable("cls_decod_fc2", [1024, 16])
        self.cls_b2 = bias_variable([16])

    def _img_to_digitcaps(self, X):
        """
        convolve up to the digit caps level
        :return: logit, reconstructed_image, loss, reconstruction_loss
        """
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 28, 28, 256]
            conv1 = tf.nn.conv2d(X, self.enc_conv_w1, strides=[1, 1, 1, 1], padding='VALID') + self.enc_conv_b1
            conv1 = tf.nn.relu(conv1)
            assert conv1.get_shape() == [self.b_size, 28, 28, 128]

        with tf.variable_scope('Conv2_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv2 = tf.nn.conv2d(conv1, self.enc_conv_w2, strides=[1, 1, 1, 1], padding='VALID') + self.enc_conv_b2
            conv2 = tf.nn.relu(conv2)
            assert conv2.get_shape() == [self.b_size, 20, 20, 256]

        with tf.variable_scope('PrimaryCaps_layer'):
            # input: [batch_size, 20, 20, 256]
            # TODO: I guess this needs act function
            caps1 = primary_caps_layer(conv2, caps_w=self.enc_caps_w1, num_outputs=32, vec_len=8, strides=[1, 2, 2, 1],
                                       padding='VALID')
            assert caps1.get_shape() == [self.b_size, 1152, 8, 1]

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            caps2 = digit_caps_layer(caps1, self.enc_caps_fc1, num_outputs=10)
        return caps2

    def _digitcaps_to_img(self, digitcaps, Y, keep_prob):
        # tiled caps [b_size,n_class,n_class,16]
        tiled_caps = tf.tile(tf.expand_dims(digitcaps[:, :, :, 0], 1), [1, 10, 1, 1])
        # masked_v [b_size,n_class,n_class,16]
        # fakeY [b_size,n_class] # repeats are [0,0,0,0 .... 9,9,9,9]
        maskY = tf.tile(tf.expand_dims(tf.range(10), 0), [self.b_size, 1])
        # fakeY [b_size,n_class,n_class,1]
        maskY = tf.expand_dims(tf.one_hot(maskY, 10), 3)
        # maskedV [b_size*n_class,n_class*16]
        maskedV = tf.reshape(tiled_caps * maskY, [self.b_size * 10, -1])
        # make images out of the masked vector
        fc1 = tf.matmul(maskedV, self.dec_fc1) + self.dec_b1
        fc1 = tf.nn.relu(fc1)
        # fc1 = tf.nn.dropout(fc1,keep_prob)
        assert fc1.get_shape() == [self.b_size * 10, 512]
        fc2 = tf.matmul(fc1, self.dec_fc2) + self.dec_b2
        fc2 = tf.nn.relu(fc2)
        assert fc2.get_shape() == [self.b_size * 10, 1024]
        fc3 = tf.matmul(fc2, self.dec_fc3) + self.dec_b3
        fc3 = tf.nn.dropout(fc3, keep_prob=keep_prob)
        img = tf.nn.sigmoid(fc3)
        img = tf.reshape(img, [self.b_size * 10, 36, 36])
        # realV [b_size,2,n_class*16]
        imgY0 = tf.gather(img, tf.range(self.b_size) * 10 + Y[:, 0])
        imgY1 = tf.gather(img, tf.range(self.b_size) * 10 + Y[:, 1])
        imgY = tf.stack([imgY0, imgY1], axis=1)
        return img, imgY

    def _img_to_pred(self, img, X, Y, keep_prob):
        # Conv1, [batch_size, 28, 28, 256]
        img = tf.expand_dims(img, 3)
        # concatenate the full picture to each of the images
        X_init = tf.reshape(tf.tile(tf.expand_dims(X, 1), [1, 10, 1, 1, 1]), [self.b_size * 10, 36, 36, 1])
        img = tf.concat([img, X_init], axis=3)

        # concatenate the real images as a second layer
        conv1 = tf.nn.conv2d(img, self.cls_conv_w1, strides=[1, 1, 1, 1], padding='VALID') + self.cls_conv_b1
        conv1 = tf.nn.relu(conv1)
        assert conv1.get_shape() == [self.b_size * 10, 28, 28, 128]

        # Conv1, [batch_size, 20, 20, 256]
        conv2 = tf.nn.conv2d(conv1, self.cls_conv_w2, strides=[1, 1, 1, 1], padding='VALID') + self.cls_conv_b2
        conv2 = tf.nn.relu(conv2)
        assert conv2.get_shape() == [self.b_size * 10, 20, 20, 256]

        # input: [batch_size, 20, 20, 256]
        conv3 = tf.nn.conv2d(conv2, self.cls_conv_w3, strides=[1, 2, 2, 1], padding='VALID') + self.cls_conv_b3
        flat = tf.reshape(conv3, [self.b_size * 10, -1])
        fc1 = tf.matmul(flat, self.cls_fc1) + self.cls_b1
        fc1 = tf.nn.relu(fc1)
        fc1_drop = tf.nn.dropout(fc1, keep_prob=keep_prob)
        nocaps = tf.matmul(fc1_drop, self.cls_fc2) + self.cls_b2
        nocaps = tf.reshape(nocaps, [self.b_size, 10, 16, 1])
        labels = tf.to_int32(tf.reshape(tf.one_hot(Y[:, 0], 10) + tf.one_hot(Y[:, 1], 10), [-1]))
        return nocaps, img, labels

        # labels = tf.to_int32(tf.reshape(tf.one_hot(Y[:,0],10) + tf.one_hot(Y[:,1],10),[-1]))
        # cross_entr = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
        # acc = tf.contrib.metrics.streaming_accuracy(tf.to_int32(tf.nn.softmax(logits)[:,1] > 0.5),labels)
        # return cross_entr,acc,img,labels

    def _margin_loss(self, digitcaps, Y, m_plus=cfg.m_plus, m_minus=cfg.m_minus, lambda_val=cfg.lambda_val):
        # The margin loss
        # [batch_size, 10, 1, 1]
        # FIXME: I am not very sure about the lambda_val 0.5 weight (in the article did they mean 0.5/num_abs_class)
        Y_hot = tf.one_hot(Y, 10)

        v_length = tf.sqrt(tf.reduce_sum(tf.square(digitcaps), axis=2, keep_dims=True) + epsilon)

        # clip the length of the digitcaps activation vector from the top and from the bottom
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., m_plus - v_length))[:, :, 0, 0]
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., v_length - m_minus))[:, :, 0, 0]
        assert max_l.get_shape() == [self.b_size, 10]

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = Y_hot[:, 0, :] + Y_hot[:, 1, :]
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r
        margin_err = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        return margin_err

    def _reconstruction_loss(self, imgY, X):
        # 2. The reconstruction loss
        imgY = imgY[:, 0, :, :] + imgY[:, 1, :, :]
        assert imgY.get_shape() == X[:, :, :, 0].get_shape()
        rec_err = tf.reduce_mean(tf.clip_by_value((imgY - X[:, :, :, 0]) ** 2, clip_value_min=0, clip_value_max=1))
        return rec_err

    def _acc(self, digitcaps, Y):
        # FIXME: I am not sure about the combinatin of of length and softmax here (should it be squash)
        v_length = tf.sqrt(tf.reduce_sum(tf.square(digitcaps), axis=2, keep_dims=True) + epsilon)
        softmax_v = tf.nn.softmax(v_length, dim=1)
        assert softmax_v.get_shape() == [cfg.batch_size, 10, 1, 1]
        softmax_v = softmax_v[:, :, 0, 0]

        # take two highest activations
        act2 = tf.nn.top_k(softmax_v, k=2)
        act2 = act2.indices
        Y2 = tf.stack([tf.argmax(Y[:, 0, :], 1), tf.argmax(Y[:, 1, :], 1)], axis=1)
        Y2 = tf.to_int32(Y2)
        b_acc1 = tf.to_float(tf.equal(Y2[:, 0], act2[:, 0]))
        b_acc2 = tf.to_float(tf.equal(Y2[:, 1], act2[:, 1]))
        b_acc3 = tf.to_float(tf.equal(Y2[:, 0], act2[:, 1]))
        b_acc4 = tf.to_float(tf.equal(Y2[:, 1], act2[:, 0]))
        b_acc = tf.reduce_sum(tf.concat([b_acc1, b_acc2, b_acc3, b_acc4], 0)) / (2 * self.b_size)
        return b_acc

    def comp_output(self, X, Y, keep_prob, reg_scale=cfg.regularization_scale):
        # 1. convolve image to digitcaps
        digitcaps = self._img_to_digitcaps(X)
        # 2. compute cross entropy with margin loss
        margin_err = self._margin_loss(digitcaps, Y)
        # 3. reconstruct image
        img, imgY = self._digitcaps_to_img(digitcaps, Y, keep_prob)
        # 4. compute reconstruction loss
        rec_err = self._reconstruction_loss(imgY, X)
        # 5. Total loss
        # # The paper uses sum of squared error as reconstruction error, but we
        # # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # # mean squared error. In order to keep in line with the paper,the
        # # regularization scale should be 0.0005*784=0.392
        total_err = margin_err + reg_scale * rec_err
        # 6. compute batch accuracy
        b_acc = self._acc(digitcaps, tf.one_hot(Y, 10))

        img = tf.stop_gradient(img)
        cls_caps, cls_img, cls_labels = self._img_to_pred(img, X, Y, keep_prob)
        cls_margin_err = self._margin_loss(cls_caps, Y)
        cls_acc = self._acc(cls_caps, tf.one_hot(Y, 10))

        return total_err, margin_err, rec_err, img, imgY, b_acc, cls_margin_err, cls_acc, cls_img, cls_labels
        # return total_err,margin_err,rec_err,img,imgY,b_acc,cls_loss,clsacc,cls_img,cls_label
