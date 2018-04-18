from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
import sys
import time

import tensorflow as tf

sys.path.append('../')
from myconfig import cfg
from myutils import load_mmnist, TrainingMonitor
from mycapsnet import CapsNet
from input_utils import load_mmnist, load_submmnist
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main(_):
    # Read the images in the MultiMNIST dataset
    # mmnist = load_mmnist(cfg.dataset)
    mmnist = load_submmnist(cfg.dataset)

    num_batches = int(600000 / cfg.batch_size) * cfg.num_epochs
    print("Number of batches: " + str(num_batches))

    # Create batches of training examples
    training_queue = tf.train.slice_input_producer([tf.convert_to_tensor(mmnist["trX"], tf.float32),
                                                    tf.convert_to_tensor(mmnist["trY"], tf.int32)])

    batch_trainX, batch_trainY = tf.train.shuffle_batch(training_queue,
                                                        num_threads=cfg.num_threads,
                                                        batch_size=cfg.batch_size,
                                                        capacity=cfg.batch_size * 64,
                                                        min_after_dequeue=cfg.batch_size * 32,
                                                        allow_smaller_final_batch=False)
    # Create batches of test examples
    test_queue = tf.train.slice_input_producer([tf.convert_to_tensor(mmnist["tef0X"], tf.float32),
                                                tf.convert_to_tensor(mmnist["tef0Y"], tf.int32),
                                                tf.convert_to_tensor(mmnist["tef2X"], tf.float32),
                                                tf.convert_to_tensor(mmnist["tef2Y"], tf.int32),
                                                tf.convert_to_tensor(mmnist["tef4X"], tf.float32),
                                                tf.convert_to_tensor(mmnist["tef4Y"], tf.int32),
                                                tf.convert_to_tensor(mmnist["tef6X"], tf.float32),
                                                tf.convert_to_tensor(mmnist["tef6Y"], tf.int32),
                                                tf.convert_to_tensor(mmnist["tef8X"], tf.float32),
                                                tf.convert_to_tensor(mmnist["tef8Y"], tf.int32),
                                                tf.convert_to_tensor(mmnist["tes0X"], tf.float32),
                                                tf.convert_to_tensor(mmnist["tes0Y"], tf.int32),
                                                tf.convert_to_tensor(mmnist["tes2X"], tf.float32),
                                                tf.convert_to_tensor(mmnist["tes2Y"], tf.int32),
                                                tf.convert_to_tensor(mmnist["tes4X"], tf.float32),
                                                tf.convert_to_tensor(mmnist["tes4Y"], tf.int32),
                                                tf.convert_to_tensor(mmnist["tes6X"], tf.float32),
                                                tf.convert_to_tensor(mmnist["tes6Y"], tf.int32),
                                                tf.convert_to_tensor(mmnist["tes8X"], tf.float32),
                                                tf.convert_to_tensor(mmnist["tes8Y"], tf.int32)])

    batch_test = tf.train.shuffle_batch(test_queue,
                                        num_threads=cfg.num_threads,
                                        batch_size=cfg.batch_size,
                                        capacity=cfg.batch_size * 64,
                                        min_after_dequeue=cfg.batch_size * 32,
                                        allow_smaller_final_batch=False)

    # Initialize the capsule network, and compute the test and train errors
    capsnet = CapsNet()

    train_total_error, train_marginal_error, train_reconstruction_error, train_img, train_imgY, train_accuracy, train_class_error, train_class_accuracy, train_classimg, train_classlabel = capsnet.comp_output(
        batch_trainX, batch_trainY, keep_prob=0.5)

    train_GAN = tf.train.AdamOptimizer().minimize(train_reconstruction_error)
    train_CLS = tf.train.AdamOptimizer().minimize(train_class_error)

    test0_total_error, test0_marginal_error, test0_reconstruction_error, _, _, test0_accuracy, test0_classerror, test0_classaccuracy, _, _ = capsnet.comp_output(
        batch_test[0],
        batch_test[1],
        keep_prob=1)
    te2_total_err, te2_margin_err, te2_rec_err, _, _, te2_acc, te2_ce, te2_clsacc, _, _ = capsnet.comp_output(
        batch_test[2],
        batch_test[3],
        keep_prob=1)
    te4_total_err, te4_margin_err, te4_rec_err, _, _, te4_acc, te4_ce, te4_clsacc, _, _ = capsnet.comp_output(
        batch_test[4],
        batch_test[5],
        keep_prob=1)
    te6_total_err, te6_margin_err, te6_rec_err, _, _, te6_acc, te6_ce, te6_clsacc, _, _ = capsnet.comp_output(
        batch_test[6],
        batch_test[7],
        keep_prob=1)
    te8_total_err, te8_margin_err, te8_rec_err, _, _, te8_acc, te8_ce, te8_clsacc, _, _ = capsnet.comp_output(
        batch_test[8],
        batch_test[9],
        keep_prob=1)

    saver = tf.train.Saver()

    sess = tf.Session()
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tf.get_default_graph().finalize()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    training_monitor = TrainingMonitor()

    # saver.restore(sess, "/home/urops/andrewg/capsule-b/test-1c-benchmarks/saved/model.ckpt")
    # print("Model restored.")

    # First, pre-train the GAN
    for i in range(1500):
        # Report generated images
        _clsimg, _clslabel, _ = sess.run([train_classimg, train_classlabel, train_GAN])
        print("--pretraining--", i)

    # Now, actually train the networks
    for batch_num in range(num_batches):
        start = time.time()

        # Training step
        _total_err, _margin_err, _rec_err, _acc, _tr_ce, _tr_clsacc, _ = \
            sess.run(
                [train_total_error, train_marginal_error, train_reconstruction_error, train_accuracy, train_class_error,
                 train_class_accuracy, train_CLS])

        # Add and display all the errors
        training_monitor.add("train_total_error", _total_err)
        training_monitor.add("train_marginal_error", _margin_err)
        training_monitor.add("train_reconstruction_error", _rec_err)
        training_monitor.add("train_accuracy", _acc)
        training_monitor.add("train_class_error", _tr_ce)
        training_monitor.add("train_class_accuracy", _tr_clsacc)
        print("step:", batch_num, "\ttotal loss:", _total_err, "\tmargin loss:", _margin_err)
        print(
            "\treconstruct error:", _rec_err, "\tacc:", _acc, "\ttrain_class_error:", _tr_ce, "\ttrain_class_accuracy",
            _tr_clsacc)
        print("\texps:", "%.3f" % (cfg.batch_size / (time.time() - start)))




        # Every 100 batches, give information on testing performance
        if batch_num % 100 == 9:
            _total_err, _margin_err, _rec_err, _acc, _tr_ce, _tr_clsacc, = \
                sess.run([test0_total_error, test0_marginal_error, test0_reconstruction_error, test0_accuracy,
                          test0_classerror, test0_classaccuracy])
            training_monitor.add("test0_total_error", _total_err)
            training_monitor.add("test0_marginal_error", _margin_err)
            training_monitor.add("test0_reconstruction_error", _rec_err)
            training_monitor.add("test0_accuracy", _acc)
            training_monitor.add("test0_classerror", _tr_ce)
            training_monitor.add("test0_classaccuracy", _tr_clsacc)
            _total_err, _margin_err, _rec_err, _acc, _tr_ce, _tr_clsacc, = \
                sess.run([te2_total_err, te2_margin_err, te2_rec_err, te2_acc, te2_ce, te2_clsacc])
            training_monitor.add("te2_total_err", _total_err)
            training_monitor.add("te2_margin_err", _margin_err)
            training_monitor.add("te2_rec_err", _rec_err)
            training_monitor.add("te2_acc", _acc)
            training_monitor.add("te2_ce", _tr_ce)
            training_monitor.add("te2_clsacc", _tr_clsacc)
            _total_err, _margin_err, _rec_err, _acc, _tr_ce, _tr_clsacc, = \
                sess.run([te4_total_err, te4_margin_err, te4_rec_err, te4_acc, te4_ce, te4_clsacc])
            training_monitor.add("te4_total_err", _total_err)
            training_monitor.add("te4_margin_err", _margin_err)
            training_monitor.add("te4_rec_err", _rec_err)
            training_monitor.add("te4_acc", _acc)
            training_monitor.add("te4_ce", _tr_ce)
            training_monitor.add("te4_clsacc", _tr_clsacc)
            _total_err, _margin_err, _rec_err, _acc, _tr_ce, _tr_clsacc, = \
                sess.run([te6_total_err, te6_margin_err, te6_rec_err, te6_acc, te6_ce, te6_clsacc])
            training_monitor.add("te6_total_err", _total_err)
            training_monitor.add("te6_margin_err", _margin_err)
            training_monitor.add("te6_rec_err", _rec_err)
            training_monitor.add("te6_acc", _acc)
            training_monitor.add("te6_ce", _tr_ce)
            training_monitor.add("te6_clsacc", _tr_clsacc)
            _total_err, _margin_err, _rec_err, _acc, _tr_ce, _tr_clsacc, = \
                sess.run([te8_total_err, te8_margin_err, te8_rec_err, te8_acc, te8_ce, te8_clsacc])
            training_monitor.add("te8_total_err", _total_err)
            training_monitor.add("te8_margin_err", _margin_err)
            training_monitor.add("te8_rec_err", _rec_err)
            training_monitor.add("te8_acc", _acc)
            training_monitor.add("te8_ce", _tr_ce)
            training_monitor.add("te8_clsacc", _tr_clsacc)
            training_monitor.prints()


            save_path = saver.save(sess, "saved/model" + str(batch_num) + ".ckpt")
            print("Model saved in path: %s" % save_path)


        # Every 100 batches, generate the images of how it's doing right now
        if (batch_num % 500 == 11):
            plt.figure()
            plt.imshow(_clsimg[0, :, :, 1], cmap='gray')
            plt.savefig('./imgs/' + str(batch_num) + '_backg0.png')
            plt.close()
            plt.figure()
            plt.imshow(_clsimg[0, :, :, 0], cmap='gray')
            plt.savefig('./imgs/' + str(batch_num) + "_" + str(_clslabel[0]) + '_gan0.png')
            plt.close()
            plt.figure()
            plt.imshow(_clsimg[1, :, :, 0], cmap='gray')
            plt.savefig('./imgs/' + str(batch_num) + "_" + str(_clslabel[1]) + '_gan1.png')
            plt.close()
            plt.figure()
            plt.imshow(_clsimg[2, :, :, 0], cmap='gray')
            plt.savefig('./imgs/' + str(batch_num) + "_" + str(_clslabel[2]) + '_gan2.png')
            plt.close()
            plt.figure()
            plt.imshow(_clsimg[3, :, :, 0], cmap='gray')
            plt.savefig('./imgs/' + str(batch_num) + "_" + str(_clslabel[3]) + '_gan3.png')
            plt.close()
            plt.figure()
            plt.imshow(_clsimg[4, :, :, 0], cmap='gray')
            plt.savefig('./imgs/' + str(batch_num) + "_" + str(_clslabel[4]) + '_gan4.png')
            plt.close()
            plt.figure()
            plt.imshow(_clsimg[5, :, :, 0], cmap='gray')
            plt.savefig('./imgs/' + str(batch_num) + "_" + str(_clslabel[5]) + '_gan5.png')
            plt.close()
            plt.figure()
            plt.imshow(_clsimg[6, :, :, 0], cmap='gray')
            plt.savefig('./imgs/' + str(batch_num) + "_" + str(_clslabel[6]) + '_gan6.png')
            plt.close()
            plt.figure()
            plt.imshow(_clsimg[7, :, :, 0], cmap='gray')
            plt.savefig('./imgs/' + str(batch_num) + "_" + str(_clslabel[7]) + '_gan7.png')
            plt.close()
            plt.figure()
            plt.imshow(_clsimg[8, :, :, 0], cmap='gray')
            plt.savefig('./imgs/' + str(batch_num) + "_" + str(_clslabel[8]) + '_gan8.png')
            plt.close()
            plt.figure()
            plt.imshow(_clsimg[9, :, :, 0], cmap='gray')
            plt.savefig('./imgs/' + str(batch_num) + "_" + str(_clslabel[9]) + '_gan9.png')
            plt.close()

            # if (batch_num % 100 == 11):
        #     #save reconstructed images (just for fun)
        #     _trX,_tr_imgY = sess.run([batch_trainX,train_imgY])
        #     plt.figure()
        #     plt.imshow(_trX[0, :, :, 0], cmap='gray')
        #     plt.savefig('./imgs/' + str(batch_num) + '_real.png')
        #     plt.close()
        #     plt.figure()
        #     plt.imshow(_tr_imgY[0,0,:,:]+_tr_imgY[0,1,:,:], cmap='gray')
        #     plt.savefig('./imgs/' + str(batch_num) + '_recstr.png')
        #     plt.close()
        #     plt.figure()
        #     plt.imshow(_tr_imgY[0, 0, :, :], cmap='gray')
        #     plt.savefig('./imgs/' + str(batch_num) + '_recstr0.png')
        #     plt.close()
        #     plt.figure()
        #     plt.imshow(_tr_imgY[0, 0, :, :] + _tr_imgY[0, 1, :, :], cmap='gray')
        #     plt.savefig('./imgs/' + str(batch_num) + '_recstr1.png')
        #     plt.close()


if __name__ == "__main__":
    tf.app.run()
