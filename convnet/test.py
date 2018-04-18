import os,time
import tensorflow as tf
#from tqdm import tqdm

from myconfig import cfg
from myutils import load_mmnist,TrainingMonitor
from input_utils import load_submmnist, load_mmnist
from mycapsnet import CapsNet
#from matplotlib import pyplot as plt
import numpy as np

def main(_):

    # read the images in the dataset
    #num_test_batch = 10000 // cfg.batch_size
    mmnist = load_mmnist("/media/data4/affnist/mmnist")

    num_batches = int(60000 / cfg.batch_size) * cfg.num_epochs
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
    te_q = tf.train.slice_input_producer([tf.convert_to_tensor(mmnist["te0X"],tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["te0Y"],tf.int32),10),
                                          #tf.convert_to_tensor(mmnist["te1X"],tf.float32),
                                          #tf.one_hot(tf.convert_to_tensor(mmnist["te1Y"],tf.int32),10),
                                          tf.convert_to_tensor(mmnist["te2X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["te2Y"], tf.int32), 10),
                                          #tf.convert_to_tensor(mmnist["te3X"], tf.float32),
                                          #tf.one_hot(tf.convert_to_tensor(mmnist["te3Y"], tf.int32), 10),
                                          tf.convert_to_tensor(mmnist["te4X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["te4Y"], tf.int32), 10),
                                          #tf.convert_to_tensor(mmnist["te5X"], tf.float32),
                                          #tf.one_hot(tf.convert_to_tensor(mmnist["te5Y"], tf.int32), 10),
                                          tf.convert_to_tensor(mmnist["te6X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["te6Y"], tf.int32), 10),
                                          #tf.convert_to_tensor(mmnist["te7X"], tf.float32),
                                          #tf.one_hot(tf.convert_to_tensor(mmnist["te7Y"], tf.int32), 10),
                                          tf.convert_to_tensor(mmnist["te8X"], tf.float32),
                                          tf.one_hot(tf.convert_to_tensor(mmnist["te8Y"], tf.int32), 10)]
                                          # tf.convert_to_tensor(mmnist["teR30X"], tf.float32),
                                          # tf.one_hot(tf.convert_to_tensor(mmnist["teR30Y"], tf.int32), 10),
                                          # tf.convert_to_tensor(mmnist["teR60X"], tf.float32),
                                          # tf.one_hot(tf.convert_to_tensor(mmnist["teR60Y"], tf.int32), 10),
                                          # tf.convert_to_tensor(mmnist["teR90X"], tf.float32),
                                          # tf.one_hot(tf.convert_to_tensor(mmnist["teR90Y"], tf.int32), 10)])
                                         )

    b_te = tf.train.shuffle_batch(te_q,
                                  num_threads=cfg.num_threads,
                                  batch_size=cfg.batch_size,
                                  capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32,
                                  allow_smaller_final_batch=False)
    # initialize the Capsule network, compute train and test errors
    capsnet = CapsNet()
    tr_err, tr_acc = capsnet.comp_output(b_trX, b_trY,keep_prob=0.5)
    train_op = tf.train.AdamOptimizer().minimize(tr_err)
    tr0_err, tr0_acc = capsnet.comp_output(b_trX, b_trY, keep_prob=1)
    te0_err, te0_acc = capsnet.comp_output(b_te[0], b_te[1],keep_prob=1)
    #te1_err, te1_acc = capsnet.comp_output(b_te[2], b_te[3],keep_prob=1)
    te2_err, te2_acc = capsnet.comp_output(b_te[2], b_te[3],keep_prob=1)
    #te3_err, te3_acc = capsnet.comp_output(b_te[6], b_te[7],keep_prob=1)
    te4_err, te4_acc = capsnet.comp_output(b_te[4], b_te[5],keep_prob=1)
    #te5_err, te5_acc = capsnet.comp_output(b_te[10], b_te[11],keep_prob=1)
    te6_err, te6_acc = capsnet.comp_output(b_te[6], b_te[7],keep_prob=1)
    #te7_err, te7_acc = capsnet.comp_output(b_te[14], b_te[15],keep_prob=1)
    te8_err, te8_acc = capsnet.comp_output(b_te[8], b_te[9],keep_prob=1)
    # teR30_err, teR30_acc = capsnet.comp_output(b_te[10], b_te[11],keep_prob=1)
    # teR60_err, teR60_acc = capsnet.comp_output(b_te[12], b_te[13],keep_prob=1)
    # teR90_err, teR90_acc = capsnet.comp_output(b_te[14], b_te[15],keep_prob=1)
    # teR90_err, teR90_acc = capsnet.comp_output(b_te[14], b_te[15], keep_prob=1)

    saver = tf.train.Saver()

    # For output data
    f1 = open('out_test.csv', 'w+', 0)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    tf.get_default_graph().finalize()
    tf.train.start_queue_runners(sess=sess,coord=coord)
    tm = TrainingMonitor()

    saver.restore(sess, "/home/urops/andrewg/capsule-b/test-1c-benchmarks/saved/model423509.ckpt")
    print("Model restored.")

    print("Total trainable parameters:")
    print(np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
    print("***")


    for b_num in range(num_batches):#(cfg.epoch):
        start = time.time()
        _total_err,_acc,_ = sess.run([tr_err,tr_acc, train_op])
        tm.add("tr_total_err_drop",_total_err)
        #tm.add("tr_margin_err", _margin_err)
        #tm.add("tr_rec_err", _rec_err)
        tm.add("tr_acc_drop", _acc)
        print "step:", b_num, "\ttotal loss:", _total_err, "\tacc:", _acc,
        print "\texps:", "%.3f" % (cfg.batch_size / (time.time() - start))

        if (b_num % 100 == 9):
            _total_err,_acc = sess.run([tr0_err, tr0_acc])
            tm.add("tr0_total_err", _total_err)
            tm.add("tr0_acc",_acc)

            _total_err,_acc = sess.run([te0_err, te0_acc])
            tm.add("te0_total_err", _total_err)
            #tm.add("te0_margin_err", _margin_err)
            #tm.add("te0_rec_err", _rec_err)
            tm.add("te0_acc", _acc)
            #_total_err,_acc = sess.run([te1_err, te1_acc])
            #tm.add("te1_total_err", _total_err)
            #tm.add("te1_margin_err", _margin_err)
            ##tm.add("te1_rec_err", _rec_err)
            #tm.add("te1_acc", _acc)
            _total_err,_acc = sess.run([te2_err, te2_acc])
            tm.add("te2_total_err", _total_err)
            #tm.add("te2_margin_err", _margin_err)
            #tm.add("te2_rec_err", _rec_err)
            tm.add("te2_acc", _acc)
            #_total_err,_acc = sess.run([te3_err, te3_acc])
            #tm.add("te3_total_err", _total_err)
            #tm.add("te3_margin_err", _margin_err)
            #tm.add("te3_rec_err", _rec_err)
            #tm.add("te3_acc", _acc)
            _total_err,_acc = sess.run([te4_err, te4_acc])
            tm.add("te4_total_err", _total_err)
            #tm.add("te4_margin_err", _margin_err)
            #tm.add("te4_rec_err", _rec_err)
            tm.add("te4_acc", _acc)
            #_total_err,_acc = sess.run([te5_err, te5_acc])
            #tm.add("te5_total_err", _total_err)
            #tm.add("te5_margin_err", _margin_err)
            #tm.add("te5_rec_err", _rec_err)
            #tm.add("te5_acc", _acc)
            _total_err,_acc = sess.run([te6_err, te6_acc])
            tm.add("te6_total_err", _total_err)
            #tm.add("te6_margin_err", _margin_err)
            #tm.add("te6_rec_err", _rec_err)
            tm.add("te6_acc", _acc)
            #_total_err,_acc = sess.run([te7_err, te7_acc])
            #tm.add("te7_total_err", _total_err)
            #tm.add("te7_margin_err", _margin_err)
            #tm.add("te7_rec_err", _rec_err)
            #tm.add("te7_acc", _acc)
            _total_err,_acc = sess.run([te8_err, te8_acc])
            tm.add("te8_total_err", _total_err)
            #tm.add("te8_margin_err", _margin_err)
            #tm.add("te8_rec_err", _rec_err)
            tm.add("te8_acc", _acc)
            # _total_err, _acc = sess.run([teR30_err, teR30_acc])
            # tm.add("teR30_total_err", _total_err)
            # tm.add("teR30_acc", _acc)
            # _total_err, _acc = sess.run([teR60_err, teR60_acc])
            # tm.add("teR60_total_err", _total_err)
            # tm.add("teR60_acc", _acc)
            # _total_err, _acc = sess.run([teR60_err, teR60_acc])
            # tm.add("teR60_total_err", _total_err)
            # tm.add("teR60_acc", _acc)

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

    f1.close()



if __name__ == "__main__":
    tf.app.run()
