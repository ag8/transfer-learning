import tensorflow as tf

from myconfig import cfg
from mycapsLayer import primary_caps_layer,digit_caps_layer
import time


# # telling tensorflow how we want to randomly initialize weights
def weight_variable(name,shape):
    #initial = tf.truncated_normal(shape, stddev=0.005)
    return tf.get_variable(name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
    initial = tf.constant(0.005, shape=shape)
    return tf.Variable(initial)



epsilon = 1e-9
class CapsNet:

    def __init__(self,b_size=cfg.batch_size):
        # initialize convolution and caps weights
        self.conv_w1 = weight_variable("conv_w1", [9,9,1,128])
        self.conv_b1 = bias_variable([128])
        self.conv_w2 = weight_variable("conv_w2", [9,9,128,256])
        self.conv_b2 = bias_variable([256])
        self.conv_w3 = weight_variable("conv_w3", [9,9,256,256])
        self.conv_b3 = bias_variable([256])

        self.caps_w1 = weight_variable("caps_w1", [9,9,256,32*8])
        self.caps_fc1 = weight_variable("caps_fc1", [1, 1152, 10, 8, 16])
        # initialize decoder weights
        self.decod_fc1 = weight_variable("decod_fc1", [9216, 1024])
        self.decod_b1 = bias_variable([1024])
        self.decod_fc2 = weight_variable("decod_fc2", [1024, 10])
        #self.decod_b2 = bias_variable([10])
        #self.decod_fc3 = weight_variable("decod_fc3", [1024, 1296])
        #self.decod_b3 = bias_variable([1296])

        self.b_size = b_size

    def _img_to_digitcaps(self,X,keep_prob):
        """
        convolve up to the digit caps level
        :return: logit, reconstructed_image, loss, reconstruction_loss
        """
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 28, 28, 256]
            conv1 = tf.nn.conv2d(X, self.conv_w1, strides=[1,1,1,1], padding='VALID') + self.conv_b1
            conv1 = tf.nn.relu(conv1)
            assert conv1.get_shape() == [self.b_size, 28, 28, 128]

        with tf.variable_scope('Conv2_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv2 = tf.nn.conv2d(conv1, self.conv_w2, strides=[1,1,1,1], padding='VALID') + self.conv_b2
            conv2 = tf.nn.relu(conv2)
            assert conv2.get_shape() == [self.b_size, 20, 20, 256]

        with tf.variable_scope('PrimaryCaps_layer'):
            # input: [batch_size, 20, 20, 256]
            # TODO: I guess this needs act function
            conv3 = tf.nn.conv2d(conv2,self.conv_w3, strides=[1,2,2,1],padding='VALID') + self.conv_b3
            #print conv3
            #time.sleep(100)
            flat = tf.reshape(conv3,[self.b_size,-1])
            fc1 = tf.matmul(flat, self.decod_fc1) + self.decod_b1
            fc1 = tf.nn.relu(fc1)
            fc1_drop = tf.nn.dropout(fc1,keep_prob=keep_prob)
            logits = tf.matmul(fc1_drop, self.decod_fc2)
            #caps1 = primary_caps_layer(conv2, caps_w=self.caps_w1, num_outputs=32, vec_len=8, strides=[1,2,2,1], padding='VALID')
            #assert caps1.get_shape() == [self.b_size, 1152, 8, 1]

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        #with tf.variable_scope('DigitCaps_layer'):
            #caps2 = digit_caps_layer(caps1, self.caps_fc1, num_outputs=10)
        return logits

    def _digitcaps_to_img(self,digitcaps,Y):
        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            #v_length = tf.sqrt(tf.reduce_sum(tf.square(digitcaps), axis=2, keep_dims=True) + epsilon)
            #softmax_v = tf.nn.softmax(v_length, dim=1)[:,:,0,0]
            # FIXME - I did not find a place where the softmax was used ( Should it be squash here?)
            #assert softmax_v.get_shape() == [self.b_size, 10]
            # select two activations
            T_c = Y[:, 0, :] + Y[:, 1, :]
            masked_v = tf.multiply(digitcaps[:,:,:,0], tf.expand_dims(T_c,2))

        # 2. Reconstruct MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        #with tf.variable_scope('Decoder'):
        #    vector_j = tf.reshape(masked_v, shape=(self.b_size, -1))#
        #
        #    fc1 = tf.matmul(vector_j, self.decod_fc1) + self.decod_b1
        #    fc1 = tf.nn.relu(fc1)
        #    assert fc1.get_shape() == [self.b_size, 512]
        #    fc2 = tf.matmul(fc1,self.decod_fc2) + self.decod_b2
        #    fc2 = tf.nn.relu(fc2)
        #    assert fc2.get_shape() == [self.b_size, 1024]
        #    fc3 = tf.matmul(fc2,self.decod_fc3) + self.decod_b3
        #    rec_img = tf.nn.sigmoid(fc3)
        #    return rec_img


    def _acc(self,preds,Y):

        # FIXME: I am not sure about the combinatin of of length and softmax here (should it be squash)
        #v_length = tf.sqrt(tf.reduce_sum(tf.square(digitcaps), axis=2, keep_dims=True) + epsilon)
        #softmax_v = tf.nn.softmax(v_length, dim=1)
        #assert softmax_v.get_shape() == [cfg.batch_size,10, 1, 1]
        #softmax_v = softmax_v[:,:,0,0]

        # take two highest activations
        act2 = tf.nn.top_k(preds, k=2)
        act2 = act2.indices
        Y2 = tf.stack([tf.argmax(Y[:,0,:],1),tf.argmax(Y[:,1,:],1)],axis=1)
        Y2 = tf.to_int32(Y2)
        b_acc1 = tf.to_float(tf.equal(Y2[:, 0], act2[:, 0]))
        b_acc2 = tf.to_float(tf.equal(Y2[:, 1], act2[:, 1]))
        b_acc3 = tf.to_float(tf.equal(Y2[:, 0], act2[:, 1]))
        b_acc4 = tf.to_float(tf.equal(Y2[:, 1], act2[:, 0]))
        b_acc = tf.reduce_sum(tf.concat([b_acc1,b_acc2,b_acc3,b_acc4],0)) / (2*self.b_size)
        return b_acc

    def comp_output(self,X,Y,keep_prob,reg_scale=cfg.regularization_scale):

        # 1. convolve image to digitcaps
        #digitcaps = self._img_to_digitcaps(X)
        # 2. compute cross entropy with margin loss
        #margin_err = self._margin_loss(digitcaps, Y)
        # 3. reconstruct image
        #rec_img = self._digitcaps_to_img(digitcaps, Y)
        # 4. compute reconstruction loss
        #rec_err = self._reconstruction_loss(rec_img,X)
        # 5. Total loss
        # # The paper uses sum of squared error as reconstruction error, but we
        # # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # # mean squared error. In order to keep in line with the paper,the
        # # regularization scale should be 0.0005*784=0.392
        #total_err = margin_err + reg_scale * rec_err
        # 6. compute batch accuracy
        #b_acc = self._acc(digitcaps, Y)
        #return total_err,margin_err,rec_err,rec_img,b_acc
        logits = self._img_to_digitcaps(X,keep_prob)
        labels = (Y[:,0,:] + Y[:,1,:])/2
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        b_acc = self._acc(tf.nn.softmax(logits=logits,dim=-1), Y)
        #print cost,b_acc
        #time.sleep(100)
        return tf.reduce_mean(cost),b_acc
