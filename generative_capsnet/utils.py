from __future__ import print_function
import collections
import numpy as np

import tensorflow as tf

from config import cfg

from input_utils import load_mmnist, load_submmnist

epsilon = 1e-9


def create_training_and_testing_batches(batch_size=cfg.batch_size, num_threads=cfg.num_threads):
    """
    Load the multiMNIST dataset, and create X and Y training batches, and a testing batch.

    :param batch_size: (Optional) the batch size
    :param num_threads: (Optional) number of threads for preprocessing

    :return: the training-X batch, the training-Y batch, and the test batch.
    """

    # Load the multiMNIST dataset
    dataset = cfg.dataset_full
    mmnist = load_mmnist(dataset)

    # Create a batch of training examples
    training_queue = tf.train.slice_input_producer([tf.convert_to_tensor(mmnist["trX"], tf.float32),
                                                    tf.one_hot(tf.convert_to_tensor(mmnist["trY"], tf.int32),
                                                               depth=10)])

    training_batch_X, training_batch_Y = tf.train.shuffle_batch(training_queue,
                                                                num_threads=cfg.num_threads,
                                                                batch_size=cfg.batch_size,
                                                                capacity=cfg.batch_size * 64,
                                                                min_after_dequeue=cfg.batch_size * 32,
                                                                allow_smaller_final_batch=False)

    test_queue = tf.train.slice_input_producer([tf.convert_to_tensor(mmnist["tes0X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes0Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes2X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes2Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes4X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes4Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes6X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes6Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes8X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes8Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tesR30RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tesR30RY"], tf.int32),
                                                           depth=10),
                                                tf.convert_to_tensor(mmnist["tesR60RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tesR60RY"], tf.int32),
                                                           depth=10),
                                                tf.convert_to_tensor(mmnist["tef0X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef0Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef2X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef2Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef4X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef4Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef6X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef6Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef8X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef8Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tefR30RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tefR30RY"], tf.int32),
                                                           depth=10),
                                                tf.convert_to_tensor(mmnist["tefR60RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tefR60RY"], tf.int32), depth=10)
                                                ])

    test_batch = tf.train.shuffle_batch(test_queue,
                                        num_threads=cfg.num_threads,
                                        batch_size=cfg.batch_size,
                                        capacity=cfg.batch_size * 64,
                                        min_after_dequeue=cfg.batch_size * 32,
                                        allow_smaller_final_batch=False)

    return training_batch_X, training_batch_Y, test_batch


def create_training_and_testing_batches_for_sub_MMNIST(batch_size=cfg.batch_size, num_threads=cfg.num_threads):
    """
    Load the multiMNIST dataset, and create X and Y training batches, and a testing batch.

    :param batch_size: (Optional) the batch size
    :param num_threads: (Optional) number of threads for preprocessing

    :return: the training-X batch, the training-Y batch, and the test batch.
    """

    # Load the multiMNIST dataset
    mmnist = load_submmnist(cfg.dataset)

    # Create a batch of training examples
    training_queue = tf.train.slice_input_producer([tf.convert_to_tensor(mmnist["trX"], tf.float32),
                                                    tf.one_hot(tf.convert_to_tensor(mmnist["trY"], tf.int32),
                                                               depth=10)])

    training_batch_X, training_batch_Y = tf.train.shuffle_batch(training_queue,
                                                                num_threads=cfg.num_threads,
                                                                batch_size=cfg.batch_size,
                                                                capacity=cfg.batch_size * 64,
                                                                min_after_dequeue=cfg.batch_size * 32,
                                                                allow_smaller_final_batch=False)

    test_queue = tf.train.slice_input_producer([tf.convert_to_tensor(mmnist["tes0X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes0Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes2X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes2Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes4X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes4Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes6X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes6Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes8X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes8Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tesR30RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tesR30RY"], tf.int32),
                                                           depth=10),
                                                tf.convert_to_tensor(mmnist["tesR60RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tesR60RY"], tf.int32),
                                                           depth=10),
                                                tf.convert_to_tensor(mmnist["tef0X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef0Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef2X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef2Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef4X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef4Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef6X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef6Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef8X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef8Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tefR30RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tefR30RY"], tf.int32),
                                                           depth=10),
                                                tf.convert_to_tensor(mmnist["tefR60RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tefR60RY"], tf.int32), depth=10)
                                                ])

    test_batch = tf.train.shuffle_batch(test_queue,
                                        num_threads=cfg.num_threads,
                                        batch_size=cfg.batch_size,
                                        capacity=cfg.batch_size * 64,
                                        min_after_dequeue=cfg.batch_size * 32,
                                        allow_smaller_final_batch=False)

    return training_batch_X, training_batch_Y, test_batch


def create_training_and_testing_batches_for_ld_MMNIST(batch_size=cfg.batch_size, num_threads=cfg.num_threads):
    """
    Load the multiMNIST dataset, and create X and Y training batches, and a testing batch.

    :param batch_size: (Optional) the batch size
    :param num_threads: (Optional) number of threads for preprocessing

    :return: the training-X batch, the training-Y batch, and the test batch.
    """

    # Load the multiMNIST dataset
    mmnist = load_submmnist(cfg.dataset_ld)

    # Create a batch of training examples
    training_queue = tf.train.slice_input_producer([tf.convert_to_tensor(mmnist["trX"], tf.float32),
                                                    tf.one_hot(tf.convert_to_tensor(mmnist["trY"], tf.int32),
                                                               depth=10)])

    training_batch_X, training_batch_Y = tf.train.shuffle_batch(training_queue,
                                                                num_threads=cfg.num_threads,
                                                                batch_size=cfg.batch_size,
                                                                capacity=cfg.batch_size * 64,
                                                                min_after_dequeue=cfg.batch_size * 32,
                                                                allow_smaller_final_batch=False)

    test_queue = tf.train.slice_input_producer([tf.convert_to_tensor(mmnist["tes0X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes0Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes2X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes2Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes4X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes4Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes6X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes6Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tes8X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tes8Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tesR30RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tesR30RY"], tf.int32),
                                                           depth=10),
                                                tf.convert_to_tensor(mmnist["tesR60RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tesR60RY"], tf.int32),
                                                           depth=10),
                                                tf.convert_to_tensor(mmnist["tef0X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef0Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef2X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef2Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef4X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef4Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef6X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef6Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tef8X"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tef8Y"], tf.int32), depth=10),
                                                tf.convert_to_tensor(mmnist["tefR30RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tefR30RY"], tf.int32),
                                                           depth=10),
                                                tf.convert_to_tensor(mmnist["tefR60RX"], tf.float32),
                                                tf.one_hot(tf.convert_to_tensor(mmnist["tefR60RY"], tf.int32), depth=10)
                                                ])

    test_batch = tf.train.shuffle_batch(test_queue,
                                        num_threads=cfg.num_threads,
                                        batch_size=cfg.batch_size,
                                        capacity=cfg.batch_size * 64,
                                        min_after_dequeue=cfg.batch_size * 32,
                                        allow_smaller_final_batch=False)

    return training_batch_X, training_batch_Y, test_batch


def init(sess):
    # Create a queue coordinator (necessary for correct input loading)
    coord = tf.train.Coordinator()

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Finalize the graph, making sure that no rogue threads are still adding operations
    tf.get_default_graph().finalize()

    # Start the queue runners, and send them off a-loadin' examples
    tf.train.start_queue_runners(sess=sess, coord=coord)

    # Create the training monitor
    tm = TrainingMonitor()

    return coord, tm


class TrainingMonitor:
    """
    A class that monitors training performance over time.
    """

    def __init__(self):
        self._hist_records = collections.OrderedDict()

    def add(self, name, value, history_length=20):
        """
        Add a value to track to the monitor.

        :param name: the name of the value to track
        :param value: the current value
        :param history_length: the length of the history to store for this variable (default: 20)
        :return:
        """

        if name not in self._hist_records:
            self._hist_records[name] = []

        self._hist_records[name].append(value)

        return np.average(self._hist_records[name][-history_length:])

    def prints(self, file, step):
        print("--------------------------  training monitor  --------------------------------------")

        i = 0

        test_accuracies = []

        for key in self._hist_records:
            i = i + 1
            print(key, self._hist_records[key][-1], "ave:", np.average(self._hist_records[key][-20:]))

            if i in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
                test_accuracies.append(np.average(self._hist_records[key][-20:]))

        file.write(str(step) + "," + str(test_accuracies[0]) + "," + str(test_accuracies[1]) + "," + str(
            test_accuracies[2]) + "," + str(test_accuracies[3]) + "," + str(test_accuracies[4]) + "\n")

        print("==========================  *************** ========================================")

    def addsix(self, curr_train_total_error, curr_train_margin_error, curr_train_reconstruction_error,
               curr_train_accuracy, curr_memo_margin_loss, curr_memo_accuracy):
        """
        Add the six common train errors.

        :param curr_train_total_error: the current total training error
        :param curr_train_margin_error: the current training margin error
        :param curr_train_reconstruction_error:  the current training reconstruction error
        :param curr_train_accuracy: the current training accuracy
        :param curr_memo_margin_loss: the current memo margin loss
        :param curr_memo_accuracy: the current memo accuracy
        :return:
        """
        self.add("train_total_error", curr_train_total_error)
        self.add("train_margin_error", curr_train_margin_error)
        self.add("train_reconstruction_error", curr_train_reconstruction_error)
        self.add("train_accuracy", curr_train_accuracy)
        self.add("train_memo_margin_loss", curr_memo_margin_loss)
        self.add("train_memo_accuracy", curr_memo_accuracy)

    def addsixtest(self, curr_total_error, curr_margin_error, curr_reconstruction_error, curr_accuracy,
                   curr_memo_margin_error, curr_memo_accuracy,
                   curr_accuracy_2px,
                   curr_accuracy_4px,
                   curr_accuracy_6px,
                   curr_accuracy_8px):
        """
        Add the six common test errors + four more accuracies.

        :param curr_total_error: the current total error
        :param curr_margin_error: the current margin error
        :param curr_reconstruction_error: the current reconstruction error
        :param curr_accuracy: the current accuracy
        :param curr_memo_margin_error: the current memo margin loss
        :param curr_memo_accuracy: the current memo accuracy

        :param curr_accuracy_2px: the current accuracy on the 2px-shifted multiMNIST subset
        :param curr_accuracy_4px: the current accuracy on the 4px-shifted multiMNIST subset
        :param curr_accuracy_6px: the current accuracy on the 6px-shifted multiMNIST subset
        :param curr_accuracy_8px: the current accuracy on the 8px-shifted multiMNIST subset
        :return:
        """

        self.add("test_0px_total_error", curr_total_error)
        self.add("test_0px_margin_error", curr_margin_error)
        self.add("test_0px_reconstruction_error", curr_reconstruction_error)
        self.add("test_0px_accuracy", curr_accuracy)
        self.add("test_0px_memo_margin_loss", curr_memo_margin_error)
        self.add("test_0px_memo_accuracy", curr_memo_accuracy)
        self.add("test_2px_accuracy", curr_accuracy_2px)
        self.add("test_4px_accuracy", curr_accuracy_4px)
        self.add("test_6px_accuracy", curr_accuracy_6px)
        self.add("test_8px_accuracy", curr_accuracy_8px)

    def addeleventest(self, curr_total_error, curr_margin_error, curr_reconstruction_error,
                      curr_memo_margin_error, curr_memo_accuracy,
                      curr_accuracy,
                      curr_accuracy_2px,
                      curr_accuracy_4px,
                      curr_accuracy_6px,
                      curr_accuracy_8px,
                      curr_sub_accuracy,
                      curr_sub_accuracy_2px,
                      curr_sub_accuracy_4px,
                      curr_sub_accuracy_6px,
                      curr_sub_accuracy_8px
                      ):
        """
        Add the six common test errors + four more accuracies.

        :param curr_total_error: the current total error
        :param curr_margin_error: the current margin error
        :param curr_reconstruction_error: the current reconstruction error
        :param curr_accuracy: the current accuracy
        :param curr_memo_margin_error: the current memo margin loss
        :param curr_memo_accuracy: the current memo accuracy

        :param curr_accuracy_2px: the current accuracy on the 2px-shifted multiMNIST subset
        :param curr_accuracy_4px: the current accuracy on the 4px-shifted multiMNIST subset
        :param curr_accuracy_6px: the current accuracy on the 6px-shifted multiMNIST subset
        :param curr_accuracy_8px: the current accuracy on the 8px-shifted multiMNIST subset
        :return:
        """

        self.add("test_0px_total_error", curr_total_error)
        self.add("test_0px_margin_error", curr_margin_error)
        self.add("test_0px_reconstruction_error", curr_reconstruction_error)
        self.add("test_0px_memo_margin_loss", curr_memo_margin_error)
        self.add("test_0px_memo_accuracy", curr_memo_accuracy)
        self.add("test_full_0px_accuracy", curr_accuracy)
        self.add("test_full_2px_accuracy", curr_accuracy_2px)
        self.add("test_full_4px_accuracy", curr_accuracy_4px)
        self.add("test_full_6px_accuracy", curr_accuracy_6px)
        self.add("test_full_8px_accuracy", curr_accuracy_8px)
        self.add("test_sub_0px_accuracy", curr_sub_accuracy)
        self.add("test_sub_2px_accuracy", curr_sub_accuracy_2px)
        self.add("test_sub_4px_accuracy", curr_sub_accuracy_4px)
        self.add("test_sub_6px_accuracy", curr_sub_accuracy_6px)
        self.add("test_sub_8px_accuracy", curr_sub_accuracy_8px)


def generate_Ys_hot(batch_size=cfg.batch_size):
    """
    Create Ys_hot encodings for memo reconstruction.

    For instance, a batch size of 128 will create a 1280x10 tensor, which will look like
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    ...
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

     where the 10-row 1-hot encodings repeat 128 times.

    :param batch_size: the batch size for the hot encodings generation
    :return: a tensor of one-hot encodings with shape [batch_size * 10, 10]
    """

    Ys = np.tile(np.expand_dims(np.arange(10), 1), [batch_size, 1]).reshape(-1)
    Ys_hot = np.zeros([batch_size * 10, 10], np.float32)
    Ys_hot[np.arange(batch_size * 10), Ys] = 1
    return Ys_hot


# LOSSES

def margin_loss(digitcaps, Y, batch_size=cfg.batch_size, m_plus=cfg.m_plus, m_minus=cfg.m_minus,
                lambda_val=cfg.lambda_val):
    # The margin loss
    # [batch_size, 10, 1, 1]
    # Note: a lambda weight of 0.5 is not necessarily optimal, 0.5/[number of classes] might be better

    v_length = tf.sqrt(tf.reduce_sum(tf.square(digitcaps), axis=2, keep_dims=True) + epsilon)

    # clip the length of the digitcaps activation vector from the top and from the bottom
    # max_l = max(0, m_plus-||v_c||)^2
    max_l = tf.square(tf.maximum(0., m_plus - v_length))[:, :, 0, 0]
    # max_r = max(0, ||v_c||-m_minus)^2
    max_r = tf.square(tf.maximum(0., v_length - m_minus))[:, :, 0, 0]
    assert max_l.get_shape() == [batch_size, 10]

    # calc T_c: [batch_size, 10]
    # T_c = Y, is my understanding correct? Try it.
    T_c = Y[:, 0, :] + Y[:, 1, :]
    # [batch_size, 10], element-wise multiply
    L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r
    margin_err = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

    return margin_err


def reconstruction_loss(rec_img, X, batch_size=cfg.batch_size):
    # 2. The reconstruction loss
    orgin = tf.reshape(X, shape=(batch_size, -1))
    squared = tf.square(rec_img - orgin)
    reconstruction_err = tf.reduce_mean(squared)
    return reconstruction_err


def acc(digitcaps, Y, batch_size=cfg.batch_size):
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
    b_acc = tf.reduce_sum(tf.concat([b_acc1, b_acc2, b_acc3, b_acc4], 0)) / (2 * batch_size)
    return b_acc
