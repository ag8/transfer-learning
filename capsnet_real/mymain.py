import os,sys,time
import tensorflow as tf
sys.path.append('../')

from myconfig import cfg
from myutils import load_mmnist,TrainingMonitor
from mycapsnet import CapsNet
from matplotlib import pyplot as plt
import numpy as np


def main(_):

    # read the images in the dataset
    #num_test_batch = 10000 // cfg.batch_size
    mmnist = load_mmnist(cfg.dataset)


    num_batches = int(6000000 / cfg.batch_size) * cfg.num_epochs
    # make bathes of training examples
    tr_q = tf.train.slice_input_producer([tf.convert_to_tensor(mmnist["trX"],tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["trY"],tf.int32),10)])
    b_trX, b_trY = tf.train.shuffle_batch(tr_q,
                                          num_threads=cfg.num_threads,
                                          batch_size=cfg.batch_size,
                                          capacity=cfg.batch_size * 64,
                                          min_after_dequeue=cfg.batch_size * 32,
                                          allow_smaller_final_batch=False)
    # make bathes of test examples
    te_q = tf.train.slice_input_producer([tf.convert_to_tensor(mmnist["tes0X"],tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["tes0Y"],tf.int32),10),
                                          tf.convert_to_tensor(mmnist["tes2X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["tes2Y"], tf.int32), 10),
                                          tf.convert_to_tensor(mmnist["tes4X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["tes4Y"], tf.int32), 10),
                                          tf.convert_to_tensor(mmnist["tes6X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["tes6Y"], tf.int32), 10),
                                          tf.convert_to_tensor(mmnist["tes8X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["tes8Y"], tf.int32), 10),
                                          tf.convert_to_tensor(mmnist["tef0X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["tef0Y"], tf.int32), 10),
                                          tf.convert_to_tensor(mmnist["tef2X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["tef2Y"], tf.int32), 10),
                                          tf.convert_to_tensor(mmnist["tef4X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["tef4Y"], tf.int32), 10),
                                          tf.convert_to_tensor(mmnist["tef6X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["tef6Y"], tf.int32), 10),
                                          tf.convert_to_tensor(mmnist["tef8X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["tef8Y"], tf.int32), 10)
                                          ])

    b_te = tf.train.shuffle_batch(te_q,
                                  num_threads=cfg.num_threads,
                                  batch_size=cfg.batch_size,
                                  capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32,
                                  allow_smaller_final_batch=False)
    # initialize the Capsule network, compute train and test errors
    capsnet = CapsNet()
    tr_total_err, tr_margin_err, tr_rec_err, tr_rec_img, tr_acc = capsnet.comp_output(b_trX, b_trY)
    train_op = tf.train.AdamOptimizer().minimize(tr_total_err)
    tes0_total_err, tes0_margin_err, tes0_rec_err, tes0_rec_img, tes0_acc = capsnet.comp_output(b_te[0], b_te[1])
    tes2_total_err, tes2_margin_err, tes2_rec_err, tes2_rec_img, tes2_acc = capsnet.comp_output(b_te[2], b_te[3])
    tes4_total_err, tes4_margin_err, tes4_rec_err, tes4_rec_img, tes4_acc = capsnet.comp_output(b_te[4], b_te[5])
    tes6_total_err, tes6_margin_err, tes6_rec_err, tes6_rec_img, tes6_acc = capsnet.comp_output(b_te[6], b_te[7])
    tes8_total_err, tes8_margin_err, tes8_rec_err, tes8_rec_img, tes8_acc = capsnet.comp_output(b_te[8], b_te[9])
    tef0_total_err, tef0_margin_err, tef0_rec_err, tef0_rec_img, tef0_acc = capsnet.comp_output(b_te[10], b_te[11])
    tef2_total_err, tef2_margin_err, tef2_rec_err, tef2_rec_img, tef2_acc = capsnet.comp_output(b_te[12], b_te[13])
    tef4_total_err, tef4_margin_err, tef4_rec_err, tef4_rec_img, tef4_acc = capsnet.comp_output(b_te[14], b_te[15])
    tef6_total_err, tef6_margin_err, tef6_rec_err, tef6_rec_img, tef6_acc = capsnet.comp_output(b_te[16], b_te[17])
    tef8_total_err, tef8_margin_err, tef8_rec_err, tef8_rec_img, tef8_acc = capsnet.comp_output(b_te[18], b_te[19])

    saver = tf.train.Saver()

    # For output data
    f1 = open('out_caps_train_on_sub.csv', 'w+', 0)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    tf.get_default_graph().finalize()
    tf.train.start_queue_runners(sess=sess,coord=coord)
    tm = TrainingMonitor()


    for b_num in range(num_batches):#(cfg.epoch):
        start = time.time()
        _total_err, _margin_err, _rec_err,_acc,_ = sess.run([tr_total_err, tr_margin_err, tr_rec_err,tr_acc, train_op])
        tm.add("tr_total_err",_total_err)
        tm.add("tr_margin_err", _margin_err)
        tm.add("tr_rec_err", _rec_err)
        tm.add("tr_acc", _acc)
        print "step:", b_num, "\ttotal loss:", _total_err, "\tmargin loss:", _margin_err,
        print "\treconstruct error:", _rec_err, "\tacc:", _acc,
        print "\texps:", "%.3f" % (cfg.batch_size / (time.time() - start))

        if (b_num % 100 == 9):
            _total_err, _margin_err, _rec_err,_acc = sess.run([tes0_total_err, tes0_margin_err, tes0_rec_err, tes0_acc])
            tm.add("tes0_total_err", _total_err)
            tm.add("tes0_margin_err", _margin_err)
            tm.add("tes0_rec_err", _rec_err)
            tm.add("tes0_acc", _acc)
            _total_err, _margin_err, _rec_err,_acc = sess.run([tes2_total_err, tes2_margin_err, tes2_rec_err, tes2_acc])
            tm.add("tes2_total_err", _total_err)
            tm.add("tes2_margin_err", _margin_err)
            tm.add("tes2_rec_err", _rec_err)
            tm.add("tes2_acc", _acc)
            _total_err, _margin_err, _rec_err,_acc = sess.run([tes4_total_err, tes4_margin_err, tes4_rec_err, tes4_acc])
            tm.add("tes4_total_err", _total_err)
            tm.add("tes4_margin_err", _margin_err)
            tm.add("tes4_rec_err", _rec_err)
            tm.add("tes4_acc", _acc)
            _total_err, _margin_err, _rec_err,_acc = sess.run([tes6_total_err, tes6_margin_err, tes6_rec_err, tes6_acc])
            tm.add("tes6_total_err", _total_err)
            tm.add("tes6_margin_err", _margin_err)
            tm.add("tes6_rec_err", _rec_err)
            tm.add("tes6_acc", _acc)
            _total_err, _margin_err, _rec_err,_acc = sess.run([tes8_total_err, tes8_margin_err, tes8_rec_err, tes8_acc])
            tm.add("tes8_total_err", _total_err)
            tm.add("tes8_margin_err", _margin_err)
            tm.add("tes8_rec_err", _rec_err)
            tm.add("tes8_acc", _acc)

            tm.add("tef0_total_err", _total_err)
            tm.add("tef0_margin_err", _margin_err)
            tm.add("tef0_rec_err", _rec_err)
            tm.add("tef0_acc", _acc)
            _total_err, _margin_err, _rec_err, _acc = sess.run(
                [tef2_total_err, tef2_margin_err, tef2_rec_err, tef2_acc])
            tm.add("tef2_total_err", _total_err)
            tm.add("tef2_margin_err", _margin_err)
            tm.add("tef2_rec_err", _rec_err)
            tm.add("tef2_acc", _acc)
            _total_err, _margin_err, _rec_err, _acc = sess.run(
                [tef4_total_err, tef4_margin_err, tef4_rec_err, tef4_acc])
            tm.add("tef4_total_err", _total_err)
            tm.add("tef4_margin_err", _margin_err)
            tm.add("tef4_rec_err", _rec_err)
            tm.add("tef4_acc", _acc)
            _total_err, _margin_err, _rec_err, _acc = sess.run(
                [tef6_total_err, tef6_margin_err, tef6_rec_err, tef6_acc])
            tm.add("tef6_total_err", _total_err)
            tm.add("tef6_margin_err", _margin_err)
            tm.add("tef6_rec_err", _rec_err)
            tm.add("tef6_acc", _acc)
            _total_err, _margin_err, _rec_err, _acc = sess.run(
                [tef8_total_err, tef8_margin_err, tef8_rec_err, tef8_acc])
            tm.add("tef8_total_err", _total_err)
            tm.add("tef8_margin_err", _margin_err)
            tm.add("tef8_rec_err", _rec_err)
            tm.add("tef8_acc", _acc)

            tm.prints(file=f1, step=b_num)

            save_path = saver.save(sess, "saved/model" + str(b_num) + ".ckpt")
            print("Model saved in path: %s" % save_path)




        # if (b_num % 500 == 9):
        #     #save reconstructed images (just for fun)
        #     _trX,_tr_rec_img = sess.run([b_trX,tr_rec_img])
        #     plt.figure()
        #     plt.imshow(_trX[0, :, :, 0], cmap='gray')
        #     plt.savefig('../imgs/' + str(b_num) + '_real.png')
        #     plt.figure()
        #     #print "shape of the rec image:", _tr_rec_img.shape
        #     plt.imshow(np.reshape(_tr_rec_img[0, :],[36,36]), cmap='gray')
        #     plt.savefig('../imgs/' + str(b_num) + '_recstr.png')



if __name__ == "__main__":
    tf.app.run()
