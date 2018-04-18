import tensorflow as tf
import os,time
flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('num_epochs', 50, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')

flags.DEFINE_float('keep_prob', 0.5, 'keep probability for dropout')


############################
#   environment setting    #
############################
#home = os.path.dirname(os.path.abspath(__file__))
flags.DEFINE_string('dataset_full', '/home/urops/andrewg/transfer-learning/data/mmnist', 'the path for the full MMNIST dataset')
flags.DEFINE_string('dataset_ld', '/home/urops/andrewg/transfer-learning/data/ld', 'the path for the low-data MMNIST dataset')
flags.DEFINE_string('dataset', '/home/urops/andrewg/transfer-learning/data/submmnist', 'the path for the subMMNIST dataset')
# flags.DEFINE_string('dataset', '/mas/u/mkkr/andrewg/mmnist', 'the path for dataset')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 256, 'number of threads of enqueueing examples')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 100, 'the frequency of saving train summary(step)') # 50
flags.DEFINE_integer('test_sum_freq', 200, 'the frequency of saving test summary(step)') # 500
flags.DEFINE_integer('save_freq', 2, 'the frequency of saving model(epoch)')
flags.DEFINE_string('results', 'results', 'path for saving results')

############################
#   distributed setting    #
############################
#flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
#flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
#flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
